import os
import re
import time
import base64
import requests
import tempfile
import mujoco
import mujoco.viewer

try:
    import pygame
except ImportError:
    print("Pygame not found. Audio will not play.")

from robot_profiles import ROBOT_PROFILES

# --- USER CONFIGURATION ---
ROBOT_NAME = "icub"
# --------------------------

profile = ROBOT_PROFILES[ROBOT_NAME]

# ==========================================
# 1. LOAD MODEL (MJCF or URDF)
# ==========================================
def load_model(profile):
    """Load a MuJoCo model from a robot profile dict.

    Supports both MJCF (``.xml``) and URDF sources.  For URDFs, resolves
    ``package://`` URIs using the ``package_dirs`` entry in *profile*, injects
    a ``<compiler balanceinertia="true"/>`` element, and retries without visual
    meshes if the first load attempt fails (e.g. unsupported ``.dae`` files).

    Args:
        profile (dict): A robot profile from ``ROBOT_PROFILES``, expected to
            contain either a ``'mjcf_path'`` key or a ``'urdf_path'`` key
            (plus optional ``'package_dirs'``).

    Returns:
        mujoco.MjModel: Compiled MuJoCo model ready for simulation.
            Calls ``exit()`` on unrecoverable load failure.
    """
    if "mjcf_path" in profile:
        print(f"Loading {ROBOT_NAME} from MJCF...")
        model = mujoco.MjModel.from_xml_path(profile["mjcf_path"])
        print(f"{ROBOT_NAME} loaded successfully.")
        return model

    urdf_path = profile["urdf_path"]
    print(f"Loading {ROBOT_NAME} from URDF...")
    with open(urdf_path, "r") as f:
        urdf_xml = f.read()

    pkg_names = re.findall(r'package://([^/]+)/', urdf_xml)
    if pkg_names:
        base_dir = profile["package_dirs"][0].replace("\\", "/")
        for pkg_name in set(pkg_names):
            repo_path = f"{base_dir}/{pkg_name}"
            urdf_xml = urdf_xml.replace(f"package://{pkg_name}", repo_path)

    def inject_compiler(xml, discard_visual):
        flag = "true" if discard_visual else "false"
        cfg = f'\n  <mujoco>\n    <compiler balanceinertia="true" discardvisual="{flag}"/>\n  </mujoco>\n'
        return re.sub(r'(<robot[^>]*>)', r'\1' + cfg, xml)

    try:
        model = mujoco.MjModel.from_xml_string(inject_compiler(urdf_xml, False))
        print(f"{ROBOT_NAME} loaded successfully.")
        return model
    except Exception:
        print("Visual meshes unsupported (e.g. .dae), retrying without visuals...")
    try:
        model = mujoco.MjModel.from_xml_string(inject_compiler(urdf_xml, True))
        print(f"{ROBOT_NAME} loaded successfully (no visuals).")
        return model
    except Exception as e:
        print(f"MuJoCo Load Error: {e}")
        exit()


model = load_model(profile)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# Pre-calculate joint name -> qpos address mapping
joint_name_to_qposadr = {}
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name:
        joint_name_to_qposadr[name] = model.jnt_qposadr[i]

# ==========================================
# 2. LAUNCH VIEWER
# ==========================================
scale = profile["scale"]
lookat_z = (scale["z_head"] + scale["z_waist"]) / 2
cam_distance = max(scale["x_max"], scale["y_max"]) * 6

viewer = mujoco.viewer.launch_passive(model, data)
with viewer.lock():
    viewer.cam.distance = cam_distance
    viewer.cam.lookat[2] = lookat_z
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 180

# ==========================================
# 3. SERVER CALL
# ==========================================
user_input = input("Enter text for the LLM: ")

SERVER_URL = "http://127.0.0.1:65432/process"
print("Sending to Brain Server...")

try:
    response = requests.post(SERVER_URL, json={"text": user_input, "robot": ROBOT_NAME})
    response.raise_for_status()
    payload = response.json()
except Exception as e:
    print("Failed to connect to server:", e)
    viewer.close()
    exit()

# ==========================================
# 4. EXECUTE PAYLOAD (Audio + Motion)
# ==========================================
print("Payload received! Executing...")

for i, item in enumerate(payload):
    sentence = item.get("sentence", "")
    audio_b64 = item.get("audio_b64", "")
    traj = item.get("trajectory", {})

    print("Robot says:", sentence)

    audio_path = os.path.join(tempfile.gettempdir(), f"{ROBOT_NAME}_audio_{i}.mp3")
    if audio_b64 and 'pygame' in globals():
        pygame.mixer.init()
        with open(audio_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))
        pygame.mixer.music.load(audio_path)

    if traj and traj.get("names"):
        names = traj["names"]
        angles = traj["angles"]
        num_frames = len(angles[0])

        if audio_b64 and 'pygame' in globals():
            pygame.mixer.music.play()

        for frame in range(num_frames):
            for idx, name in enumerate(names):
                if name in joint_name_to_qposadr:
                    data.qpos[joint_name_to_qposadr[name]] = angles[idx][frame]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.04)

    if audio_b64 and 'pygame' in globals():
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()
        try:
            os.remove(audio_path)
        except Exception:
            pass

print("Finished execution.")

data.qvel[:] = 0.0

while viewer.is_running():
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(0.01)

viewer.close()
