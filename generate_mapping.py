import re
import mujoco
import mujoco.viewer
import numpy as np
import time

def get_dominant_axis(vector):
    """Return the dominant signed axis label for a 3-element vector.

    Finds the component with the largest absolute value and returns a
    string such as ``'+X'``, ``'-Y'``, or ``'+Z'``.

    Args:
        vector (array-like): A 3-element numeric vector.

    Returns:
        str: Signed axis label, e.g. ``'+X'`` or ``'-Z'``.
    """
    axes = ["X", "Y", "Z"]
    max_idx = np.argmax(np.abs(vector))
    sign = "+" if vector[max_idx] > 0 else "-"
    return f"{sign}{axes[max_idx]}"

def invert_axis(axis_str):
    """Flip the sign of a signed axis string.

    Args:
        axis_str (str): A signed axis label such as ``'+X'`` or ``'-Z'``.

    Returns:
        str: The inverted axis label, e.g. ``'+X'`` becomes ``'-X'``.
    """
    return axis_str.replace('+', 'TEMP').replace('-', '+').replace('TEMP', '-')

def string_to_array(axis_str):
    """Converts a string like '+X' into a numpy array [1.0, 0.0, 0.0]."""
    mapping = {
        "+X": np.array([1.0, 0.0, 0.0]), "-X": np.array([-1.0, 0.0, 0.0]),
        "+Y": np.array([0.0, 1.0, 0.0]), "-Y": np.array([0.0, -1.0, 0.0]),
        "+Z": np.array([0.0, 0.0, 1.0]), "-Z": np.array([0.0, 0.0, -1.0])
    }
    return mapping[axis_str.upper()]

def array_to_string(arr):
    """Formats a numpy array into a clean string for code generation."""
    return f"np.array([{arr[0]}, {arr[1]}, {arr[2]}])"

def prompt_global_environment():
    """Prompts the user for Up/Forward axes to support both real and simulated robots."""
    print(f"\n{'='*60}")
    print("1. GLOBAL ENVIRONMENT CALIBRATION")
    print(f"{'='*60}")
    print("Define the coordinate system of your physical lab or simulation world.\n")
    print("IF USING MUJOCO:")
    print(" 1. Open the left-hand control panel in the viewer.")
    print(" 2. Expand the 'Rendering' tab.")
    print(" 3. Check the 'Frame' box under the 'World' section.")
    print(" 4. Look at the floor to find the arrows: Red (+X), Green (+Y), Blue (+Z).\n")
    
    # Prompt user for Up and Forward
    up_axis = input("1. Which World Arrow points UP? (e.g., +Z, +X): ").strip().upper()
    fwd_axis = input("2. Which World Arrow points FORWARD? (e.g., +X, -Y): ").strip().upper()
    
    # Calculate all 6 cardinal directions dynamically
    vec_up = string_to_array(up_axis)
    vec_down = -vec_up
    vec_fwd = string_to_array(fwd_axis)
    vec_back = -vec_fwd
    
    # Physical Right is always Forward x Up in a right-handed coordinate system
    vec_right = np.cross(vec_fwd, vec_up)
    vec_left = -vec_right

    # Generate the Python code string for the dictionaries
    dict_code = f"""    # 1. DYNAMIC GLOBAL TARGET VECTORS (Auto-calibrated)
    finger_vectors = {{
        "forward": {array_to_string(vec_fwd)},
        "backward": {array_to_string(vec_back)},
        "up": {array_to_string(vec_up)},
        "down": {array_to_string(vec_down)},
        "left": {array_to_string(vec_left)},
        "right": {array_to_string(vec_right)}
    }}
    
    back_of_hand_vectors = {{
        "palms_up": {array_to_string(vec_down)},
        "palms_down": {array_to_string(vec_up)},
        "palms_forward": {array_to_string(vec_back)},
        "palms_backward": {array_to_string(vec_fwd)}
    }}

    if is_left:
        back_of_hand_vectors["palms_in"] = {array_to_string(vec_left)}
        back_of_hand_vectors["palms_out"] = {array_to_string(vec_right)}
    else:
        back_of_hand_vectors["palms_in"] = {array_to_string(vec_right)}
        back_of_hand_vectors["palms_out"] = {array_to_string(vec_left)}"""
        
    return dict_code

def prompt_user_for_anchors(hand_name):
    """Interactively prompt the user to describe the physical orientation of a hand.

    Instructs the user to look at the robot in the simulator or physically
    and enter which world axis each anatomical direction (fingers, back of
    hand, thumb) currently points toward.

    Args:
        hand_name (str): A descriptive label for the hand (e.g. ``'LEFT HAND'``).

    Returns:
        dict: With keys ``'t_fingers'``, ``'t_boh'``, and ``'t_thumb'``,
            each containing a signed axis string such as ``'+X'``.
    """
    print(f"\n{'='*60}")
    print(f"VISUAL INSPECTION: {hand_name.upper()}")
    print(f"{'='*60}")
    fingers = input("1. Which way do the FINGERS point? (e.g., +X, -Y): ").strip().upper()
    boh = input("2. Which way does the BACK OF THE HAND point? (e.g., +Y, -Z): ").strip().upper()
    thumb = input("3. Which way does the THUMB point? (e.g., -X, +Z): ").strip().upper()
    return {"t_fingers": fingers, "t_boh": boh, "t_thumb": thumb}

def generate_safe_mapping(data, body_name, user_anatomy, indent="    "):
    """Generate Python column-mapping code for a URDF body's rotation matrix.

    Reads the current world-frame rotation matrix of *body_name* from
    MuJoCo, identifies which world axis each column points along, and
    cross-references it against the user-supplied anatomical directions to
    produce ready-to-paste ``col_x / col_y / col_z`` assignment code.

    Args:
        data (mujoco.MjData): Live MuJoCo data object (robot at rest pose).
        body_name (str): Name of the MuJoCo body to inspect (e.g. ``'l_wrist'``).
        user_anatomy (dict): Mapping of anatomical direction name to signed axis
            string, e.g. ``{'t_fingers': '+X', 't_boh': '+Z', 't_thumb': '-Y'}``.
        indent (str): Indentation prefix for the generated lines (default 4 spaces).

    Returns:
        str: Multi-line Python code fragment assigning ``col_x``, ``col_y``,
            and ``col_z``, or an error comment if the body was not found.
    """
    try:
        xmat = data.body(body_name).xmat.reshape(3, 3)
    except Exception:
        return f"{indent}# [ERROR] Body not found."

    math_cols = {
        "col_x": get_dominant_axis(xmat[:, 0]),
        "col_y": get_dominant_axis(xmat[:, 1]),
        "col_z": get_dominant_axis(xmat[:, 2])
    }

    generated_code = []
    
    for col_name, math_axis in math_cols.items():
        matched = False
        for target_name, user_axis in user_anatomy.items():
            # If the URDF exactly matches the biological part
            if math_axis == user_axis:
                generated_code.append(f"{indent}{col_name} = {target_name}")
                matched = True
                break
            # If the URDF axis is the exact opposite (e.g., pointing to the Pinky instead of Thumb)
            elif math_axis == invert_axis(user_axis):
                generated_code.append(f"{indent}{col_name} = -{target_name}")
                matched = True
                break
                
        if not matched:
            generated_code.append(f"{indent}# [ERROR] Could not map {col_name} (points to {math_axis}). Check inputs.")

    return "\n".join(generated_code)

def run_integrated_wizard(left_hand_link, right_hand_link, urdf_path=None, package_path=None, mjcf_path=None):
    """Run the full interactive calibration wizard for a robot's hand orientation.

    Loads a robot model from either a MJCF or URDF file, optionally opens a
    MuJoCo passive viewer, and walks the user through:

    1. Global environment calibration (which world axis is Up/Forward).
    2. Visual inspection of each hand's anatomical axes.
    3. Auto-generation of a ``get_robot_orientation`` function body to paste
       into ``robot_profiles.py``.

    Args:
        left_hand_link (str): Name of the left-hand body in the model
            (e.g. ``'l_wrist'``).
        right_hand_link (str): Name of the right-hand body in the model
            (e.g. ``'r_wrist'``).
        urdf_path (str | None): Path to the robot URDF file. Mutually
            exclusive with *mjcf_path*.
        package_path (str | None): Filesystem path used to resolve
            ``package://`` URIs inside the URDF.
        mjcf_path (str | None): Path to a MuJoCo MJCF XML file. Takes
            priority over *urdf_path* when both are supplied.
    """
    model = None
    data = None

    if mjcf_path is not None:
        print(f"Loading MJCF from: {mjcf_path}")
        try:
            model = mujoco.MjModel.from_xml_path(mjcf_path)
            data = mujoco.MjData(model)
        except Exception as e:
            print(f"Failed to load MJCF: {e}")
            return

    elif urdf_path is not None:
        print(f"Loading simulator from: {urdf_path}")
        try:
            with open(urdf_path, "r") as f:
                urdf_xml = f.read()

            repo_path = package_path.replace("\\", "/")
            # Replace any package:// prefix found in the URDF generically
            pkg_names = re.findall(r'package://([^/]+)/', urdf_xml)
            for pkg_name in set(pkg_names):
                urdf_xml = urdf_xml.replace(f"package://{pkg_name}", repo_path)

            def make_compiler_config(discard_visual):
                flag = "true" if discard_visual else "false"
                return f"""
  <mujoco>
    <compiler balanceinertia="true" discardvisual="{flag}"/>
  </mujoco>
"""
            def try_load(xml, discard_visual):
                patched = re.sub(
                    r'(<robot[^>]*>)',
                    r'\1' + make_compiler_config(discard_visual),
                    xml
                )
                return mujoco.MjModel.from_xml_string(patched)

            try:
                model = try_load(urdf_xml, discard_visual=False)
            except Exception:
                print("Visual meshes failed to load (unsupported format e.g. .dae). Retrying without visuals...")
                model = try_load(urdf_xml, discard_visual=True)

            data = mujoco.MjData(model)
        except Exception as e:
            print(f"Failed to load URDF: {e}")
            return

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

    def _run_prompts_and_print():
        global_dicts_code = prompt_global_environment()
        left_input = prompt_user_for_anchors("LEFT HAND (l_hand)")
        right_input = prompt_user_for_anchors("RIGHT HAND (r_hand)")

        left_code = generate_safe_mapping(data, left_hand_link, left_input, indent="        ") if data else "        # No simulator — fill manually"
        right_code = generate_safe_mapping(data, right_hand_link, right_input, indent="        ") if data else "        # No simulator — fill manually"

        print(f"\n{'='*60}")
        print("SUCCESS! COPY THIS ENTIRE FUNCTION")
        print(f"{'='*60}\n")

        full_function = f"""import numpy as np

def get_robot_orientation(orientation_string, finger_string, is_left):
    \"\"\"
    Auto-generated URDF orientation matrix for MuJoCo / Physical Robots.
    Fully adapted to the global physics frame.
    \"\"\"
{global_dicts_code}

    t_fingers = finger_vectors.get(finger_string, finger_vectors["forward"])
    t_boh = back_of_hand_vectors.get(orientation_string, back_of_hand_vectors["palms_in"])

    if np.abs(np.dot(t_fingers, t_boh)) > 0.1:
        if t_fingers[2] == 0:  
            t_boh = finger_vectors["up"]  
        else:                       
            t_boh = back_of_hand_vectors["palms_in"]

    if is_left:
        t_thumb = np.cross(t_fingers, t_boh)
    else:
        t_thumb = np.cross(t_boh, t_fingers)

    # 2. URDF PERMUTATION MAP
    if is_left:
{left_code}
    else:
{right_code}

    # 3. MATRIX BUILD
    rotation_matrix = np.column_stack((col_x, col_y, col_z))
    return rotation_matrix
"""
        print(full_function)
        print("="*60)

    if model is not None:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
            viewer.cam.azimuth = 180
            viewer.sync()
            _run_prompts_and_print()
            while viewer.is_running():
                time.sleep(0.1)
    else:
        print("No simulator loaded — running in physical robot mode.")
        _run_prompts_and_print()

if __name__ == "__main__":
    # --- USER CONFIGURATION Examples ---
    # from robot_descriptions import pepper_description as desc
    # URDF_PATH = desc.URDF_PATH      # Set to None for a physical robot or different simulator
    # PACKAGE_PATH = desc.PACKAGE_PATH  # Set to None for a physical robot or different simulator
    # LEFT_HAND_LINK = "l_wrist"
    # RIGHT_HAND_LINK = "r_wrist"

    from robot_descriptions import ergocub_description as desc
    URDF_PATH = desc.URDF_PATH
    PACKAGE_PATH = desc.PACKAGE_PATH
    LEFT_HAND_LINK = "l_hand_palm"
    RIGHT_HAND_LINK = "r_hand_palm"

    run_integrated_wizard(LEFT_HAND_LINK, RIGHT_HAND_LINK, urdf_path=URDF_PATH, package_path=PACKAGE_PATH)