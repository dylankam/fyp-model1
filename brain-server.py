import socket
import json
import os
import re
import base64
from gtts import gTTS
from mutagen.mp3 import MP3
import google.genai as genai
from google.genai import types
import numpy as np
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
import time
from robot_profiles import ROBOT_PROFILES

# --- CONFIGURATION ---
HOST = '0.0.0.0'
PORT = 65432
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
print("Gemini Client Initialized:", "Yes" if client else "No")

LLM1_PROMPT = (
    "You are the linguistic intent analyzer for a humanoid robot.\n"
    "Input: A spoken sentence and its audio duration in seconds.\n"
    "Output: A JSON block containing the original text, the duration, the required handedness, and a clear physical description of the gesture.\n"
    "\n"
    "HARDWARE CONSTRAINTS:\n"
    "- The robot has simple mitten-like grippers. Do not describe individual finger movements (e.g., pointing with an index finger, making a fist). Describe the hand and wrist as a single unit.\n"
    "\n"
    "RULES:\n"
    "- 'use_hand' MUST be exactly one of: 'left', 'right', 'both', or 'none'.\n"
    "- If the sentence does not require a gesture (e.g., short transition words, simple factual statements), output 'none' for use_hand.\n"
    "- SUBTLETY RULE: Keep gestures low and restrained (waist to lower-chest height). Do NOT describe hands going to shoulder or head height unless the sentence explicitly demands extreme excitement, pointing high, or a specific tall pose.\n"
    "- If the gesture is unilateral (e.g., waving), default to 'right' unless the text implies left.\n"
    "- 'description' MUST clearly describe the physical motion and final pose of the arms and hands in natural language. Be descriptive enough that an animator could easily visualize it (e.g., mention general height, arm extension, palm direction and finger direction (perpendicular to palms)).\n"
    "- TEMPORAL SEQUENCES: If the sentence implies sequential actions (e.g., 'On one hand... but on the other...'), explicitly describe the timing (e.g., 'First, the right hand... Then, halfway through, the left hand...').\n"
    "\n"
    "Example Output 1 (Static Target):\n"
    "```json\n"
    '{"text": "Stop right there!", "use_hand": "right", "description": "The robot raises its right arm, bending the elbow to bring the hand to chest height with the palm facing forward in a halting motion.", "duration": 1.6}\n'
    "```\n"
    "Example Output 2 (No Gesture):\n"
    "```json\n"
    '{"text": "Blue.", "use_hand": "none", "description": "The robot remains still with no hand movements.", "duration": 0.6}\n'
    "```\n"
    "Example Output 3 (Static Target):\n"
    "```json\n"
    '{"text": "Welcome to this demonstration.", "use_hand": "both", "description": "The robot holds both arms down near its waist, bending the elbows slightly to extend the hands forward and outward with palms facing up.", "duration": 2.5}\n'
    "```\n"
    "Example Output 4 (Sequential Action):\n"
    "```json\n"
    '{"text": "On one hand it is expensive, but on the other it saves time.", "use_hand": "both", "description": "First, the right hand is raised to waist level with the palm up. Then, halfway through the sentence, the left hand is raised to match it, both palms facing up.", "duration": 4.2}\n'
    "```"
)

LLM2_PROMPT = (
    "You are a Cartesian mapping model for a humanoid robot \n"
    "Input: A JSON intent containing 'use_hand' ('left', 'right', or 'both'), a description, and duration.\n"
    "Output: A JSON block containing a 'keyframes' array. Each keyframe represents a waypoint in the animation.\n"
    "\n"
    "KEYFRAME & use_hand RULES:\n"
    "- You MUST output a list of 1 or more keyframes inside a 'keyframes' array.\n"
    "- Only include more than 1 keyframe if the description implies a clear sequential action (e.g., 'first', 'then', 'afterwards').\n"
    "- Each keyframe MUST have a 'time_fraction' (a float between 0.1 and 1.0) indicating when in the audio this pose is reached.\n"
    "- TIMING & INTERPOLATION (CRITICAL): The robot moves sequentially. Movement toward Keyframe 2 only begins exactly when Keyframe 1's time_fraction is reached.\n"
    "- ANCHORING (HOLDING POSES): If a hand is NOT included in a keyframe, it will completely FREEZE and hold its last known position up to that time_fraction.\n"
    "- 'use_hand' dictates which hands are given new target coordinates at this specific moment.\n"
    "- If use_hand is 'right': Output ONLY right_hand keys. OMIT left_hand keys (the left arm will freeze in place).\n"
    "- If use_hand is 'left': Output ONLY left_hand keys. OMIT right_hand keys (the right arm will freeze in place).\n"
    "- If use_hand is 'both': Output keys for both hands (both arms move to new targets).\n"
    "\n"
    "NORMALIZED SPATIAL CONSTRAINTS (CRITICAL):\n"
    "- Do NOT output meters. You must output normalized float coordinates between -1.0 and 1.0.\n"
    "- X (Forward/Back): 0.0 is the lower torso. 1.0 is maximum reach forward.\n"
    "- Y (Left/Right): 0.0 is the center of the lower torso. +/- 1.0 is maximum reach outward. Left hand uses positive Y, Right hand uses negative Y.\n"
    "- Z (Up/Down): -1.0 is resting at the waist. 0.0 is the center of the lower torso. 1.0 is the height of the face/head.\n"
    "- Orientations: MUST be one of ['palms_up', 'palms_down', 'palms_in', 'palms_out', 'palms_forward', 'palms_backward'].\n"
    "- Fingers: MUST be one of ['forward', 'backward', 'up', 'down', 'left', 'right'].\n"
    "\n"
    "ORTHOGONALITY RULE (ANATOMY LIMITS):\n"
    "Palms and fingers CANNOT point along the same axis. They must be 90 degrees apart.\n"
    "- If palm is 'forward'/'backward', fingers MUST be 'up', 'down', 'left', or 'right'.\n"
    "- If palm is 'up'/'down', fingers MUST be 'forward', 'backward', 'left', or 'right'.\n"
    "- If palm is 'in'/'out', fingers MUST be 'forward', 'backward', 'up', or 'down'.\n"
    "\n"
    "SPATIAL GUIDELINES:\n"
    "- 'Stop': X=0.15, Z=0.5, orientation: 'palms_forward', fingers: 'up'.\n"
    "- 'Welcome' or 'Present': X=0.05, active Y=+/- 0.1, Z=-1.0, orientation: 'palms_up', fingers: 'forward'.\n"
    "\n"
    "Example Output 1 (Single / Static Target):\n"
    "```json\n"
    '{"keyframes": [\n'
    '  {"time_fraction": 1.0, "use_hand": "right", "right_hand_pos": [0.15, -0.07, 0.05], "right_orientation": "palms_forward", "right_fingers": "up"}\n'
    '], "duration": 1.6}\n'
    "```\n"
    "Example Output 2 (Sequential Action):\n"
    "```json\n"
    '{"keyframes": [\n'
    '  {"time_fraction": 0.5, "use_hand": "right", "right_hand_pos": [0.15, -0.07, 0.05], "right_orientation": "palms_forward", "right_fingers": "up"},\n'
    '  {"time_fraction": 1.0, "use_hand": "both", "right_hand_pos": [0.15, -0.07, 0.05], "right_orientation": "palms_forward", "right_fingers": "up", "left_hand_pos": [0.15, 0.07, 0.05], "left_orientation": "palms_forward", "left_fingers": "up"}\n'
    '], "duration": 4.2}\n'
    "```"
)

def extract_json(text_response):
    try:
        start = text_response.find("```json") + 7
        end = text_response.find("```", start)
        return json.loads(text_response[start:end].strip())
    except:
        return None

def call_llm(prompt_instruction, input_data):
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=str(input_data),
        config=types.GenerateContentConfig(
            system_instruction=prompt_instruction,
            temperature=0.7
        )
    )
    return extract_json(response.text)

def generate_tts_and_duration(sentence, index):
    filename = f"temp_{index}.mp3"
    tts = gTTS(text=sentence, lang='en')
    tts.save(filename)
    
    audio = MP3(filename)
    duration = audio.info.length
    
    with open(filename, "rb") as f:
        encoded_audio = base64.b64encode(f.read()).decode('utf-8')
        
    os.remove(filename)
    return encoded_audio, duration



def generate_pink_trajectory(cartesian_target, duration, active_robot="nao", current_angles=None, dt=0.04):
    try:
        # 1. Load Robot Profile
        profile = ROBOT_PROFILES.get(active_robot)
        if not profile:
            raise ValueError(f"Robot profile '{active_robot}' not found.")
        
        pkg_dirs = profile.get("package_dirs", [])
        robot = pin.RobotWrapper.BuildFromURDF(profile["urdf_path"], package_dirs=pkg_dirs)
        q_initial = robot.q0.copy()

        if current_angles:
            for i in range(1, robot.model.njoints):
                name = robot.model.names[i]
                if name in current_angles:
                    q_idx = robot.model.joints[i].idx_q
                    q_initial[q_idx] = current_angles[name]

        q_initial = np.maximum(q_initial, robot.model.lowerPositionLimit)
        q_initial = np.minimum(q_initial, robot.model.upperPositionLimit)

        # 2. Dynamic Anti-Spin / Height Limits
        pitch_joints = profile["limits"].get("pitch_joints", [])
        pitch_max = profile["limits"].get("shoulder_pitch_max", -1.5)
        
        for joint_name in pitch_joints:
            if robot.model.existJointName(joint_name):
                joint_id = robot.model.getJointId(joint_name)
                idx_q = robot.model.joints[joint_id].idx_q
                robot.model.lowerPositionLimit[idx_q] = max(robot.model.lowerPositionLimit[idx_q], pitch_max)

        configuration = pink.Configuration(robot.model, robot.data, q_initial)

        # 3. Extract Keyframes and Global Usage
        keyframes = cartesian_target.get("keyframes", []) if cartesian_target else []
        keyframes = sorted(keyframes, key=lambda k: k.get("time_fraction", 1.0))

        global_use_left = any(kf.get("use_hand", "none") in ["left", "both"] for kf in keyframes)
        global_use_right = any(kf.get("use_hand", "none") in ["right", "both"] for kf in keyframes)

        # 4. Define the Absolute Rest Pose from Profile
        REST_L_POS = np.array(profile["rest_pose"]["left_pos"])
        REST_R_POS = np.array(profile["rest_pose"]["right_pos"])
        REST_R_LEFT = profile["get_orientation"]("palms_in", "down", is_left=True)
        REST_R_RIGHT = profile["get_orientation"]("palms_in", "down", is_left=False)

       # 5. Initialize Tasks
        tasks = []
        
        # Weighted posture task to stiffen the 'stiff joints' defined in robot profiles.
        nv = robot.model.nv
        cost_vector = np.full(nv, 0.01)
        
        # Apply a massive cost penalty to the stiff joints
        stiff_joints = profile.get("stiff_joints", [])
        for joint_name in stiff_joints:
            if robot.model.existJointName(joint_name):
                joint_id = robot.model.getJointId(joint_name)
                idx_v = robot.model.joints[joint_id].idx_v
                nv_joint = robot.model.joints[joint_id].nv
                
                # Apply the heavy cost to all degrees of freedom for this joint to makes it 100x more "expensive" for the solver to move that joint.
                cost_vector[idx_v : idx_v + nv_joint] = 1.0 
                
        # 3. Feed the custom cost array into the task
        posture_task = PostureTask(cost=cost_vector)
        posture_task.set_target(q_initial)
        tasks.append(posture_task)
        # ----------------------------------

        initial_l_se3 = configuration.get_transform_frame_to_world(profile["end_effectors"]["left"])
        initial_r_se3 = configuration.get_transform_frame_to_world(profile["end_effectors"]["right"])

        l_waypoints = [(0.0, initial_l_se3)]
        r_waypoints = [(0.0, initial_r_se3)]

        # Debug helpers to track normalized inputs and their converted absolute positions
        l_debug = [(0.0, None, initial_l_se3.translation.tolist())]
        r_debug = [(0.0, None, initial_r_se3.translation.tolist())]

        l_wrist_task = FrameTask(profile["end_effectors"]["left"], position_cost=1.0, orientation_cost=0.1)
        r_wrist_task = FrameTask(profile["end_effectors"]["right"], position_cost=1.0, orientation_cost=0.1)
        tasks.append(l_wrist_task)
        tasks.append(r_wrist_task)

        # --- LEFT HAND TIMELINE ---
        if global_use_left:
            for kf in keyframes:
                frac = kf.get("time_fraction", 1.0)
                if kf.get("use_hand", "none") in ["left", "both"]:
                    # Check if normalized pos exists, otherwise use last known absolute position
                    if "left_hand_pos" in kf:
                        norm_pos = kf["left_hand_pos"]
                        # Scaling Math: Normalized -> Absolute Meters
                        abs_x = max(0.0, norm_pos[0]) * profile["scale"]["x_max"]
                        abs_y = norm_pos[1] * profile["scale"]["y_max"]
                        abs_z = ((norm_pos[2] + 1.0) / 2.0) * (profile["scale"]["z_head"] - profile["scale"]["z_waist"]) + profile["scale"]["z_waist"]
                        l_pos = np.array([abs_x, abs_y, abs_z])
                        l_debug.append((frac, list(norm_pos), l_pos.tolist()))
                    else:
                        l_pos = l_waypoints[-1][1].translation
                        l_debug.append((frac, None, l_pos.tolist()))
                        
                    # Fetch hardware-specific orientation
                    R_left = profile["get_orientation"](kf.get("left_orientation", "palms_in"), kf.get("left_fingers", "down"), is_left=True)
                    l_waypoints.append((frac, pin.SE3(R_left, l_pos)))
                else:
                    l_waypoints.append((frac, l_waypoints[-1][1]))
        else:
            # AUTO-REST
            l_waypoints.append((0.3, pin.SE3(REST_R_LEFT, REST_L_POS)))
            l_waypoints.append((1.0, pin.SE3(REST_R_LEFT, REST_L_POS)))

        # --- RIGHT HAND TIMELINE ---
        if global_use_right:
            for kf in keyframes:
                frac = kf.get("time_fraction", 1.0)
                if kf.get("use_hand", "none") in ["right", "both"]:
                    if "right_hand_pos" in kf:
                        norm_pos = kf["right_hand_pos"]
                        # Scaling Math: Normalized -> Absolute Meters
                        abs_x = max(0.0, norm_pos[0]) * profile["scale"]["x_max"]
                        abs_y = norm_pos[1] * profile["scale"]["y_max"]
                        abs_z = ((norm_pos[2] + 1.0) / 2.0) * (profile["scale"]["z_head"] - profile["scale"]["z_waist"]) + profile["scale"]["z_waist"]
                        r_pos = np.array([abs_x, abs_y, abs_z])
                        r_debug.append((frac, list(norm_pos), r_pos.tolist()))
                    else:
                        r_pos = r_waypoints[-1][1].translation
                        r_debug.append((frac, None, r_pos.tolist()))
                        
                    R_right = profile["get_orientation"](kf.get("right_orientation", "palms_in"), kf.get("right_fingers", "down"), is_left=False)
                    r_waypoints.append((frac, pin.SE3(R_right, r_pos)))
                else:
                    r_waypoints.append((frac, r_waypoints[-1][1]))
        else:
            # AUTO-REST
            r_waypoints.append((0.3, pin.SE3(REST_R_RIGHT, REST_R_POS)))
            r_waypoints.append((1.0, pin.SE3(REST_R_RIGHT, REST_R_POS)))

        def get_interpolated_se3(progress, waypoints):
            if progress <= waypoints[0][0]: return waypoints[0][1]
            if progress >= waypoints[-1][0]: return waypoints[-1][1]
            for i in range(len(waypoints) - 1):
                t1, pose1 = waypoints[i]
                t2, pose2 = waypoints[i+1]
                if t1 <= progress <= t2:
                    local_progress = (progress - t1) / (t2 - t1) if (t2 - t1) > 0 else 1.0
                    return pin.SE3.Interpolate(pose1, pose2, local_progress)
            return waypoints[-1][1]

        # Print normalized -> absolute waypoint conversions for debugging
        try:
            print("\n[DEBUG] Left waypoints (time_fraction, normalized, absolute_meters):")
            for entry in l_debug:
                print(f"  {entry}")
            print("[DEBUG] Right waypoints (time_fraction, normalized, absolute_meters):")
            for entry in r_debug:
                print(f"  {entry}")
        except Exception as _e:
            print(f"[DEBUG] Error printing waypoint debug info: {_e}")

        # 6. Simulation Loop
        time_steps = np.arange(0, duration, dt)
        if len(time_steps) == 0: time_steps = [duration]

        # Extract joints securely
        joint_names = [n for i, n in enumerate(robot.model.names) if i > 0 and "Thumb" not in n and "Finger" not in n]
        joint_indices = [robot.model.joints[i].idx_q for i in range(1, robot.model.njoints) if "Thumb" not in robot.model.names[i] and "Finger" not in robot.model.names[i]]
        
        trajectory_angles = {name: [] for name in joint_names}
        trajectory_times = {name: [] for name in joint_names}

        for t in time_steps:
            progress = min(t / duration, 1.0)
            
            l_wrist_task.set_target(get_interpolated_se3(progress, l_waypoints))
            r_wrist_task.set_target(get_interpolated_se3(progress, r_waypoints))

            velocity = solve_ik(configuration, tasks, dt, solver="quadprog")
            configuration.integrate_inplace(velocity, dt)

            for name, q_idx in zip(joint_names, joint_indices):
                trajectory_angles[name].append(float(configuration.q[q_idx]))
                trajectory_times[name].append(float(t + dt))

        return {"names": joint_names, "times": [trajectory_times[n] for n in joint_names], "angles": [trajectory_angles[n] for n in joint_names]}
        
    except Exception as e:
        print(f"PINK IK Error: {e}")
        return None

def process_paragraph(paragraph, current_angles=None, active_robot="nao"):
    sentences = re.split(r'(?<=[.!?]) +', paragraph)
    payload = []
    
    # Initialize a running tracker starting from the physical reality
    running_angles = current_angles.copy() if current_angles else {}
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence: continue
            
        print(f"\nProcessing: {sentence}")
        audio_b64, duration = generate_tts_and_duration(sentence, i)
        
        intent_json = call_llm(LLM1_PROMPT, f"Sentence: {sentence}, Duration: {duration}")
        print(f"LLM 1 Intent: {intent_json}")
        time.sleep(0.1) 
        
        use_hand = intent_json.get("use_hand", "none")
        
        # Bypass LLM 2 if no gesture is needed
        if use_hand == "none":
            print("No active gesture needed. Triggering Auto-Rest.")
            cartesian_json = {"keyframes": [], "duration": duration}
        else:
            cartesian_json = call_llm(LLM2_PROMPT, intent_json)
            print(f"LLM 2 Cartesian: {cartesian_json}")
        
        # Generate the trajectory using the running state, not the start state
        trajectory = generate_pink_trajectory(cartesian_json, duration, active_robot, running_angles)
        
        # CONTINUOUS STATE UPDATE:
        # Extract the very last frame of this trajectory and update our running tracker
        if trajectory and trajectory["angles"]:
            for name, angles in zip(trajectory["names"], trajectory["angles"]):
                if len(angles) > 0:
                    running_angles[name] = angles[-1]
        
        payload.append({
            "sentence": sentence,
            "audio_b64": audio_b64,
            "trajectory": trajectory
        })
        time.sleep(0.1) 
        
    return payload

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Add SO_REUSEADDR so rapid server restarts don't trigger "Address already in use"
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"BRAIN SERVER: Listening on {HOST}:{PORT}...")

    while True:
        try:
            print("\n[Waiting for Robot to connect and report state...]")
            conn, addr = server_socket.accept()
            print(f"Connected to Robot at {addr}")
            
            # Increased buffer size to cleanly receive the entire physical state JSON
            data = conn.recv(4096).decode('utf-8')
            if not data: 
                conn.close()
                continue
                
            # Parse the robot's physical joint angles
            current_angles = {}
            try:
                robot_state = json.loads(data)
                current_angles = robot_state.get("angles", {})
                print(f"Received physical state for {len(current_angles)} joints.")
            except json.JSONDecodeError:
                print("Warning: Could not parse robot state, defaulting to URDF resting pose.")
            active_robot = "nao"
            text_to_process = input(f"Enter paragraph for {active_robot} to execute: ")
            
            final_payload = process_paragraph(text_to_process, current_angles, active_robot)
            
            json_string = json.dumps(final_payload) + "<EOF>"
            conn.sendall(json_string.encode('utf-8'))
            print("Data sent to robot.")
            conn.close()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            break
        except Exception as e:
            print(f"Server Error: {e}")

if __name__ == "__main__":
    start_server()