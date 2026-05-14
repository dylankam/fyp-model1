import os
import re
import json
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

# 1. Load the iCub model using MuJoCo
from robot_descriptions import icub_description
from robot_profiles import ROBOT_PROFILES

profile = ROBOT_PROFILES["icub"]

# Read the URDF file as a raw text string
with open(profile["urdf_path"], "r") as f:
    urdf_xml = f.read()

# 1. Swap the ROS 'package://' prefix using the native PACKAGE_PATH
repo_path = icub_description.PACKAGE_PATH.replace("\\", "/")

# The iCub URDF explicitly requests "package://iCub/..." so we replace that exact string
fixed_urdf_xml = urdf_xml.replace("package://iCub", repo_path)

# Inject the MuJoCo compiler directive to balance inertia and keep visuals.
# iCub's URDF has near-zero inertia links that would crash MuJoCo without this.
mj_compiler_config = """
  <mujoco>
    <compiler balanceinertia="true" discardvisual="false"/>
  </mujoco>
"""

# Find the end of the opening <robot ... > tag and insert our config
fixed_urdf_xml = re.sub(r'(<robot[^>]*>)', r'\1' + mj_compiler_config, fixed_urdf_xml)

# 2. Load the model into MuJoCo
try:
    model = mujoco.MjModel.from_xml_string(fixed_urdf_xml)
    data = mujoco.MjData(model)
    print("iCub successfully loaded with balanced inertia!")
except Exception as e:
    print("MuJoCo Load Error: ", e)
    exit()

# iCub has no floating base — all 32 qpos entries are hinge joints.
# Set a neutral standing pose: legs straight, arms at sides.
# Joint index map: r_hip_pitch=0, r_knee=3, r_ankle_pitch=4,
#                  l_hip_pitch=26, l_knee=29, l_ankle_pitch=30,
#                  r_shoulder_pitch=12, l_shoulder_pitch=19
# At q=0 the knees are slightly bent and torso hunches due to URDF zero pose.
# r_knee and l_knee ranges end at +0.07 rad (nearly 0), minimum is ~-2.16.
# Setting them to 0 (max extension) straightens legs.
# Shoulder pitch: range [-1.67, +0.17]. At 0 arms point forward; set slightly down.
data.qpos[3]  = -0.05   # r_knee — nearly straight
data.qpos[29] = -0.05   # l_knee — nearly straight
data.qpos[12] = 0.15    # r_shoulder_pitch — arms slightly forward/down
data.qpos[19] = 0.15    # l_shoulder_pitch
mujoco.mj_forward(model, data)

# Pre-calculate the joint name to qpos mapping for fast lookup during animation.
# iCub has a highly complex kinematic chain — dynamic lookup is essential.
joint_name_to_qposadr = {}
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name:
        joint_name_to_qposadr[name] = model.jnt_qposadr[i]

# 3. Start the Viewer with camera settings suited to a ~1m tall robot
# iCub faces world -Y, so azimuth=90 positions the camera along world -Y (robot's front)
viewer = mujoco.viewer.launch_passive(model, data)
with viewer.lock():
    viewer.cam.distance = 2.0      # Closer than TALOS (which needed ~3m)
    viewer.cam.lookat[2] = 0.3     # Look at roughly chest height on iCub
    viewer.cam.elevation = -15     # Slight downward angle for a natural view
    viewer.cam.azimuth = 0         # Camera at world +X, looking at robot's face (+X direction)

# ==========================================
# 4. THE END-TO-END SERVER CALL
# ==========================================
user_input = input("Enter text for the LLM: ")

SERVER_URL = "http://127.0.0.1:65432/process"  # Ensure this matches your server port
print("Sending to Brain Server...")

try:
    # Send the request to your live LLM/TTS server
    response = requests.post(SERVER_URL, json={
        "text": user_input,
        "robot": "icub",
    })
    response.raise_for_status()
    payload = response.json()
except Exception as e:
    print("Failed to connect to server:", e)
    viewer.close()
    exit()

# ==========================================
# 5. EXECUTE THE PAYLOAD (Audio + Motion)
# ==========================================
print("Payload received! Executing...")

for i, item in enumerate(payload):
    sentence = item.get("sentence", "")
    audio_b64 = item.get("audio_b64", "")
    traj = item.get("trajectory", {})

    print("Robot says:", sentence)

    # Setup Audio
    audio_path = os.path.join(tempfile.gettempdir(), "icub_audio_{}.mp3".format(i))
    if audio_b64 and 'pygame' in globals():
        pygame.mixer.init()
        with open(audio_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))
        pygame.mixer.music.load(audio_path)

    # Setup Motion
    if traj and traj.get("names"):
        names = traj["names"]
        angles = traj["angles"]
        num_frames = len(angles[0])

        # Play Audio
        if audio_b64 and 'pygame' in globals():
            pygame.mixer.music.play()

        # Play Motion Frame-by-Frame at ~25fps, matching brain-server dt=0.04.
        # Uses dynamic name-to-ID lookup — safe for iCub's complex kinematic chain.
        for frame in range(num_frames):
            for idx, name in enumerate(names):
                if name in joint_name_to_qposadr:
                    data.qpos[joint_name_to_qposadr[name]] = angles[idx][frame]

            mujoco.mj_forward(model, data)  # Recompute kinematics from qpos
            viewer.sync()
            time.sleep(0.04)

    # Wait for audio to finish before moving to the next sentence
    if audio_b64 and 'pygame' in globals():
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()
        try:
            os.remove(audio_path)
        except Exception:
            pass

print("Finished execution.")

# Zero velocities so the physics integrator inside mj_step doesn't drift the pose.
# The whole pipeline is kinematic (mj_forward), so residual qvel must not carry over.
data.qvel[:] = 0.0

# Keep viewer open until the user closes the window.
# Use mj_forward (not mj_step) to hold the last kinematic pose without running physics.
# mj_step applies gravity to a free-floating base and would immediately collapse the robot.
while viewer.is_running():
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(0.01)

viewer.close()
