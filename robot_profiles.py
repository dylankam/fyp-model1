import os
import numpy as np
from robot_descriptions import pepper_description, icub_description, ergocub_description

# Robot specific functions to compute rotation matrices based on orientation and finger direction
def get_nao_orientation(orientation_string, finger_string, is_left):
    """Construct a 3x3 end-effector rotation matrix for the NAO / Pepper robot.

    Builds the matrix from two biological anchor vectors — the finger
    direction (X column) and the back-of-hand direction (Z column) — then
    derives the Y column via the cross product.  A singularity guard
    substitutes a safe fallback Z axis when the two anchors are parallel.

    Args:
        orientation_string (str): Palm orientation key, one of
            ``'palms_forward'``, ``'palms_backward'``, ``'palms_up'``,
            ``'palms_down'``, ``'palms_in'``, ``'palms_out'``.
        finger_string (str): Finger direction key, one of
            ``'forward'``, ``'backward'``, ``'up'``, ``'down'``,
            ``'left'``, ``'right'``.
        is_left (bool): ``True`` for the left hand, ``False`` for the right.
            Controls the lateral palm-in / palm-out vectors.

    Returns:
        numpy.ndarray: A ``(3, 3)`` rotation matrix.
    """
    # 1. Map Finger String to Global X-Axis Vector
    finger_vectors = {
        "forward": np.array([1.0, 0.0, 0.0]),
        "backward": np.array([-1.0, 0.0, 0.0]),
        "up": np.array([0.0, 0.0, 1.0]),
        "down": np.array([0.0, 0.0, -1.0]),
        "left": np.array([0.0, 1.0, 0.0]),
        "right": np.array([0.0, -1.0, 0.0])
    }
    
    # 2. Map Palm String to Global Z-Axis Vector (Back of the Hand)
    back_of_hand_vectors = {
        "palms_up": np.array([0.0, 0.0, -1.0]),       
        "palms_down": np.array([0.0, 0.0, 1.0]),      
        "palms_forward": np.array([-1.0, 0.0, 0.0]),  
        "palms_backward": np.array([1.0, 0.0, 0.0])   
    }

    if is_left:
        back_of_hand_vectors["palms_in"] = np.array([0.0, 1.0, 0.0])
        back_of_hand_vectors["palms_out"] = np.array([0.0, -1.0, 0.0])
    else:
        back_of_hand_vectors["palms_in"] = np.array([0.0, -1.0, 0.0])
        back_of_hand_vectors["palms_out"] = np.array([0.0, 1.0, 0.0])

    X_axis = finger_vectors.get(finger_string, np.array([1.0, 0.0, 0.0]))
    Z_axis = back_of_hand_vectors.get(orientation_string, back_of_hand_vectors["palms_in"])

    if np.abs(np.dot(X_axis, Z_axis)) > 0.1:
        if X_axis[2] == 0:  
            Z_axis = np.array([0.0, 0.0, 1.0])
        else:               
            Z_axis = back_of_hand_vectors["palms_in"]

    Y_axis = np.cross(Z_axis, X_axis)
    rotation_matrix = np.column_stack((X_axis, Y_axis, Z_axis))
    
    return rotation_matrix


def get_icub_orientation(orientation_string, finger_string, is_left):
    """Construct a 3x3 end-effector rotation matrix for the iCub robot.

    Uses the back-of-hand vector as the Y column and the derived thumb
    vector as the Z column, letting X be computed dynamically so that
    the left/right URDF axis mirroring is absorbed automatically.
    Tailored to MuJoCo's global frame where ``+X`` is backward,
    ``+Y`` is right, and ``+Z`` is up.

    A determinant check ensures the resulting matrix is right-handed;
    if it is not, the X column is negated to correct it.

    Args:
        orientation_string (str): Palm orientation key — see
            :func:`get_nao_orientation` for valid values.
        finger_string (str): Finger direction key — see
            :func:`get_nao_orientation` for valid values.
        is_left (bool): ``True`` for the left hand, ``False`` for the right.

    Returns:
        numpy.ndarray: A ``(3, 3)`` rotation matrix.
    """
    # 1. GLOBAL TARGET VECTORS (MuJoCo Reality)
    finger_vectors = {
        "forward": np.array([-1.0, 0.0, 0.0]),  # -X is Forward
        "backward": np.array([1.0, 0.0, 0.0]),  # +X is Backward
        "up": np.array([0.0, 0.0, 1.0]),        # +Z is Up
        "down": np.array([0.0, 0.0, -1.0]),     # -Z is Down
        "left": np.array([0.0, -1.0, 0.0]),     # -Y is Left
        "right": np.array([0.0, 1.0, 0.0])      # +Y is Right
    }
    
    back_of_hand_vectors = {
        "palms_up": np.array([0.0, 0.0, -1.0]),       # Palm up = BoH down
        "palms_down": np.array([0.0, 0.0, 1.0]),      # Palm down = BoH up
        "palms_forward": np.array([1.0, 0.0, 0.0]),   # Palm fwd = BoH backward (+X)
        "palms_backward": np.array([-1.0, 0.0, 0.0])  # Palm back = BoH forward (-X)
    }

    if is_left:
        back_of_hand_vectors["palms_in"] = np.array([0.0, -1.0, 0.0]) # BoH points Left (-Y) 
        back_of_hand_vectors["palms_out"] = np.array([0.0, 1.0, 0.0]) # BoH points Right (+Y)
    else:
        back_of_hand_vectors["palms_in"] = np.array([0.0, 1.0, 0.0])  # BoH points Right (+Y) #
        back_of_hand_vectors["palms_out"] = np.array([0.0, -1.0, 0.0])  # BoH points Left (-Y)

    # Fetch primary targets
    t_fingers = finger_vectors.get(finger_string, np.array([-1.0, 0.0, 0.0]))
    t_boh = back_of_hand_vectors.get(orientation_string, back_of_hand_vectors["palms_in"])

    # Singularity Check
    if np.abs(np.dot(t_fingers, t_boh)) > 0.1:
        if t_fingers[2] == 0:  
            t_boh = np.array([0.0, 0.0, 1.0])  
        else:                       
            t_boh = back_of_hand_vectors["palms_in"] 

    # ---------------------------------------------------------
    # 2. CALCULATE THE THUMB VECTOR (The new anchor)
    # ---------------------------------------------------------
    # The physical thumb direction depends on handedness. 
    # Right Hand: Thumb is Cross(Back of Hand, Fingers)
    # Left Hand: Thumb is Cross(Fingers, Back of Hand)
    if is_left:
        t_thumb = np.cross(t_fingers, t_boh)
    else:
        t_thumb = np.cross(t_boh, t_fingers)

    # ---------------------------------------------------------
    # 3. THE PERMUTATION MAP (Anchored on Y and Z)
    # ---------------------------------------------------------
    # Based on your observation: Y = Back of Palm, Z = Thumb
    col_y = t_boh
    col_z = t_thumb
    
    # Calculate X (the normal/fingers) dynamically. 
    # This automatically absorbs the left/right hardware inversion!
    col_x = np.cross(col_y, col_z)

    # Build the 3x3 Matrix
    rotation_matrix = np.column_stack((col_x, col_y, col_z))
    
    # SAFETY: Determinant Check to prevent IK spazzing
    if np.linalg.det(rotation_matrix) < 0:
        col_x = -col_x 
        rotation_matrix = np.column_stack((col_x, col_y, col_z))
        
    return rotation_matrix


def get_ergocub_orientation(orientation_string, finger_string, is_left):
    """Construct a 3x3 end-effector rotation matrix for the ergoCub robot.

    Auto-generated by the ``generate_mapping.py`` wizard and calibrated to
    the ergoCub URDF frame.  Uses thumb (X), back-of-hand (Y / -Y), and
    fingers (-Z) as the three matrix columns, with handedness applied per
    arm.

    Args:
        orientation_string (str): Palm orientation key — see
            :func:`get_nao_orientation` for valid values.
        finger_string (str): Finger direction key — see
            :func:`get_nao_orientation` for valid values.
        is_left (bool): ``True`` for the left hand, ``False`` for the right.

    Returns:
        numpy.ndarray: A ``(3, 3)`` rotation matrix.
    """
    # 1. DYNAMIC GLOBAL TARGET VECTORS (Auto-calibrated)
    finger_vectors = {
        "forward": np.array([1.0, 0.0, 0.0]),
        "backward": np.array([-1.0, -0.0, -0.0]),
        "up": np.array([0.0, 0.0, 1.0]),
        "down": np.array([-0.0, -0.0, -1.0]),
        "left": np.array([-0.0, 1.0, -0.0]),
        "right": np.array([0.0, -1.0, 0.0])
    }
    
    back_of_hand_vectors = {
        "palms_up": np.array([-0.0, -0.0, -1.0]),
        "palms_down": np.array([0.0, 0.0, 1.0]),
        "palms_forward": np.array([-1.0, -0.0, -0.0]),
        "palms_backward": np.array([1.0, 0.0, 0.0])
    }

    if is_left:
        back_of_hand_vectors["palms_in"] = np.array([-0.0, 1.0, -0.0])
        back_of_hand_vectors["palms_out"] = np.array([0.0, -1.0, 0.0])
    else:
        back_of_hand_vectors["palms_in"] = np.array([0.0, -1.0, 0.0])
        back_of_hand_vectors["palms_out"] = np.array([-0.0, 1.0, -0.0])

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
        col_x = t_thumb
        col_y = t_boh
        col_z = -t_fingers
    else:
        col_x = t_thumb
        col_y = -t_boh
        col_z = -t_fingers

    # 3. MATRIX BUILD
    rotation_matrix = np.column_stack((col_x, col_y, col_z))
    return rotation_matrix

ROBOT_PROFILES = {
    "nao": {
        "urdf_path": "nao_clean.urdf",
        "end_effectors": {
            "left": "l_wrist",
            "right": "r_wrist"
        },
        "rest_pose": {
            "left_pos": [0.0, 0.15, -0.1],  
            "right_pos": [0.0, -0.15, -0.1]
        },
        "limits": {
            "pitch_joints": ["LShoulderPitch", "RShoulderPitch"],
            "shoulder_pitch_max": -1.5 
        },
        "scale": {
            "x_max": 0.20,   
            "y_max": 0.20,   
            "z_head": 0.25,  
            "z_waist": -0.05 
        },
        "axis_inversion": {"x": 1.0, "y": 1.0, "z": 1.0},
        "get_orientation": get_nao_orientation
    },
    "pepper": {
        "urdf_path": pepper_description.URDF_PATH,
        "package_dirs": [os.path.dirname(pepper_description.REPOSITORY_PATH)],
        "end_effectors": {
            "left": "l_wrist",
            "right": "r_wrist"
        },
        "rest_pose": {
            "left_pos": [0.0, 0.20, 0.60],  
            "right_pos": [0.0, -0.20, 0.60]
        },
        "limits": {
            "pitch_joints": ["LShoulderPitch", "RShoulderPitch"],
            "shoulder_pitch_max": -1.5 
        },
        "stiff_joints": [
            "HipRoll", "HipPitch", "KneePitch",
        ],
        "scale": {
            "x_max": 0.55,   

            "y_max": 0.55,   
            "z_head": 1.15,  
            "z_waist": 0.60
        },
        "axis_inversion": {"x": 1.0, "y": 1.0, "z": 1.0},
        "get_orientation": get_nao_orientation
    },
    "icub": {
        "urdf_path": icub_description.URDF_PATH,
        "package_dirs": [os.path.dirname(icub_description.PACKAGE_PATH)],
        "end_effectors": {
            "left": "l_hand",
            "right": "r_hand"
        },
        "rest_pose": {
            "left_pos": [-0.048, -0.090, -0.126],
            "right_pos": [-0.048,  0.090, -0.126]
        },
        "limits": {
            "pitch_joints": ["l_shoulder_pitch", "r_shoulder_pitch"],
            "shoulder_pitch_max": -1.5
        },
        "stiff_joints": [
            "l_hip_pitch", "r_hip_pitch", "l_hip_roll", "r_hip_roll",
            "torso_yaw", "torso_roll", "torso_pitch"
        ],
        "fingers": [
            "l_hand_finger", "l_thumb_oppose", "l_thumb_proximal", "l_thumb_distal",
            "l_index_proximal", "l_index_distal", "l_middle_proximal", "l_middle_distal",
            "l_ring_pinky",
            "r_hand_finger", "r_thumb_oppose", "r_thumb_proximal", "r_thumb_distal",
            "r_index_proximal", "r_index_distal", "r_middle_proximal", "r_middle_distal",
            "r_ring_pinky"
        ],
        "scale": {
            "x_max": 0.40,
            "y_max": 0.40,
            "z_head": 0.37,
            "z_waist": -0.10
        },
        "axis_inversion": {"x": -1.0, "y": -1.0, "z": 1.0},
        "get_orientation": get_icub_orientation
    },
    "ergocub": {
        "urdf_path": ergocub_description.URDF_PATH,
        "package_dirs": [os.path.dirname(ergocub_description.PACKAGE_PATH)],
        "end_effectors": {
            "left": "l_hand_palm",
            "right": "r_hand_palm"
        },
        "rest_pose": {
            "left_pos":  [0.052,  0.246, 0.008],
            "right_pos": [0.052, -0.246, 0.008]
        },
        "limits": {
            "pitch_joints": ["l_shoulder_pitch", "r_shoulder_pitch"],
            "shoulder_pitch_max": -1.5
        },
        "stiff_joints": [
            "torso_roll", "torso_pitch", "torso_yaw",
            "l_hip_pitch", "l_hip_roll", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",
            "r_hip_pitch", "r_hip_roll", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll"
        ],
        "scale": {
            "x_max": 0.40,
            "y_max": 0.35,
            "z_head": 0.45,   # shoulder height
            "z_waist": 0.00  # below pelvis
        },
        "axis_inversion": {"x": 1.0, "y": 1.0, "z": 1.0},
        "get_orientation": get_ergocub_orientation
    }
}


