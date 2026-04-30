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
    "You are a Cartesian mapping model for a NAO humanoid robot (0.5m tall).\n"
    "Input: A JSON intent containing 'use_hand' ('left', 'right', or 'both'), a description, and duration.\n"
    "Output: A JSON block containing a 'keyframes' array. Each keyframe represents a waypoint in the animation.\n"
    "\n"
    "KEYFRAME & use_hand RULES:\n"
    "- You MUST output a list of 1 or more keyframes inside a 'keyframes' array.\n"
    "- Each keyframe MUST have a 'time_fraction' (a float between 0.1 and 1.0) indicating when in the audio this pose is reached.\n"
    "- TIMING & INTERPOLATION (CRITICAL): The robot moves sequentially. Movement toward Keyframe 2 only begins exactly when Keyframe 1's time_fraction is reached.\n"
    "- ANCHORING (HOLDING POSES): If a hand is NOT included in a keyframe, it will completely FREEZE and hold its last known position up to that time_fraction.\n"
    "- 'use_hand' dictates which hands are given new target coordinates at this specific moment.\n"
    "- If use_hand is 'right': Output ONLY right_hand keys. OMIT left_hand keys (the left arm will freeze in place).\n"
    "- If use_hand is 'left': Output ONLY left_hand keys. OMIT right_hand keys (the right arm will freeze in place).\n"
    "- If use_hand is 'both': Output keys for both hands (both arms move to new targets).\n"
    "\n"
    "PHYSICAL CONSTRAINTS (CRITICAL):\n"
    "- The Origin (0.0, 0.0, 0.0) is located at the center of NAO's lower torso.\n"
    "- X (Forward/Back): Max 0.25 meters. 0.0 is the lower torso.\n"
    "- Y (Left/Right): Max 0.25 metres. 0.0 is the lower torso. Left hand side positive Y. Right hand side negative Y.\n"
    "- Z (Up/Down): Min -0.05 metres. 0.0 is the lower torso. +0.15 is the upper chest. +0.25 is the face.\n"
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
    "- 'Stop': X=0.15, Z=0.05, orientation: 'palms_forward', fingers: 'up'.\n"
    "- 'Welcome' or 'Present': X=0.05, active Y=+/- 0.1, Z=-0.05, orientation: 'palms_up', fingers: 'forward'.\n"
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

def get_dynamic_rotation(orientation_string, finger_string, is_left):
    """
    Dynamically constructs a 3x3 rotation matrix from finger and palm vectors.
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

def generate_pink_trajectory(cartesian_target, duration, current_angles=None, dt=0.04):
    try:
        # 1. Load Robot
        urdf_filename = "nao_clean.urdf"
        robot = pin.RobotWrapper.BuildFromURDF(urdf_filename)
        q_initial = robot.q0.copy()

        if current_angles:
            for i in range(1, robot.model.njoints):
                name = robot.model.names[i]
                if name in current_angles:
                    q_idx = robot.model.joints[i].idx_q
                    q_initial[q_idx] = current_angles[name]

        q_initial = np.maximum(q_initial, robot.model.lowerPositionLimit)
        q_initial = np.minimum(q_initial, robot.model.upperPositionLimit)
        configuration = pink.Configuration(robot.model, robot.data, q_initial)

        # 2. Extract Keyframes and Global Usage
        keyframes = cartesian_target.get("keyframes", []) if cartesian_target else []
        keyframes = sorted(keyframes, key=lambda k: k.get("time_fraction", 1.0))

        global_use_left = any(kf.get("use_hand", "none") in ["left", "both"] for kf in keyframes)
        global_use_right = any(kf.get("use_hand", "none") in ["right", "both"] for kf in keyframes)

        # 3. Define the Absolute Rest Pose (Arms down by sides)
        REST_L_POS = np.array([0.0, 0.15, -0.1])
        REST_R_POS = np.array([0.0, -0.15, -0.1])
        REST_R_LEFT = get_dynamic_rotation("palms_in", "down", is_left=True)
        REST_R_RIGHT = get_dynamic_rotation("palms_in", "down", is_left=False)

        # 4. Initialize Tasks (ALWAYS track both arms now)
        tasks = []
        posture_task = PostureTask(cost=0.01)
        posture_task.set_target(q_initial)
        tasks.append(posture_task)

        initial_l_se3 = configuration.get_transform_frame_to_world("l_wrist")
        initial_r_se3 = configuration.get_transform_frame_to_world("r_wrist")

        l_waypoints = [(0.0, initial_l_se3)]
        r_waypoints = [(0.0, initial_r_se3)]

        l_wrist_task = FrameTask("l_wrist", position_cost=1.0, orientation_cost=0.1)
        r_wrist_task = FrameTask("r_wrist", position_cost=1.0, orientation_cost=0.1)
        tasks.append(l_wrist_task)
        tasks.append(r_wrist_task)

        # --- LEFT HAND TIMELINE ---
        if global_use_left:
            for kf in keyframes:
                frac = kf.get("time_fraction", 1.0)
                if kf.get("use_hand", "none") in ["left", "both"]:
                    l_pos = np.array(kf.get("left_hand_pos", l_waypoints[-1][1].translation))
                    R_left = get_dynamic_rotation(kf.get("left_orientation", "palms_in"), kf.get("left_fingers", "down"), True)
                    l_waypoints.append((frac, pin.SE3(R_left, l_pos)))
                else:
                    l_waypoints.append((frac, l_waypoints[-1][1]))
        else:
            # AUTO-REST: Arm is unused. Smoothly lower it to rest over 30% of the duration.
            l_waypoints.append((0.3, pin.SE3(REST_R_LEFT, REST_L_POS)))
            l_waypoints.append((1.0, pin.SE3(REST_R_LEFT, REST_L_POS)))

        # --- RIGHT HAND TIMELINE ---
        if global_use_right:
            for kf in keyframes:
                frac = kf.get("time_fraction", 1.0)
                if kf.get("use_hand", "none") in ["right", "both"]:
                    r_pos = np.array(kf.get("right_hand_pos", r_waypoints[-1][1].translation))
                    R_right = get_dynamic_rotation(kf.get("right_orientation", "palms_in"), kf.get("right_fingers", "down"), False)
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

        # 5. Simulation Loop
        time_steps = np.arange(0, duration, dt)
        if len(time_steps) == 0: time_steps = [duration]

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

        # 6. Full Output (Masking Removed)
        # We now send all upper body joints so the physical robot knows how to lower its arms.
        return {"names": joint_names, "times": [trajectory_times[n] for n in joint_names], "angles": [trajectory_angles[n] for n in joint_names]}
        
    except Exception as e:
        print(f"PINK IK Error: {e}")
        return None

def process_paragraph(paragraph, current_angles=None):
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
        trajectory = generate_pink_trajectory(cartesian_json, duration, running_angles)
        
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

            text_to_process = input("Enter paragraph for NAO to execute: ")
            
            final_payload = process_paragraph(text_to_process, current_angles)
            
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