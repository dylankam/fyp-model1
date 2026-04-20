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

# --- PROMPTS ---
LLM1_PROMPT = (
    "You are the linguistic intent analyzer for a humanoid robot.\n"
    "Input: A spoken sentence and its audio duration in seconds.\n"
    "Output: A JSON block containing the original text, the duration, the required handedness, and a clear physical description of the gesture.\n"
    "\n"
    "HARDWARE CONSTRAINTS:\n"
    "- The robot has simple mitten-like grippers. Do not describe individual finger movements (e.g., pointing with an index finger, making a fist). Describe the hand and wrist as a single unit.\n"
    "\n"
    "RULES:\n"
    "- 'use_hand' MUST be exactly one of: 'left', 'right', or 'both'.\n"
    "- If the gesture is unilateral (e.g., stopping, refusing), default to 'right' unless the text implies left.\n"
    "- 'description' MUST clearly describe the physical motion and final pose of the arms and hands in natural language. Be descriptive enough that an animator could easily visualize it (e.g., mention general height, arm extension, palm direction and finger direction (perpendicular to palms)).\n"
    "\n"
    "Example Output 1:\n"
    "```json\n"
    '{"text": "Stop right there!", "use_hand": "right", "description": "The robot raises its right arm, bending the elbow to bring the hand to chest height with the palm facing forward in a halting motion.", "duration": 1.6}\n'
    "```\n"
    "Example Output 2:\n"
    "```json\n"
    '{"text": "Hi there!", "use_hand": "right", "description": "The robot raises its right arm, bending the elbow to hold the hand near head height, and waves it side to side with the palm facing forward.", "duration": 1.5}\n'
    "```\n"
    "Example Output 3:\n"
    "```json\n"
    '{"text": "Welcome to this demonstration.", "use_hand": "both", "description": "The robot holds both arms down near its waist, bending the elbows slightly to extend the hands forward and outward with palms facing up.", "duration": 2.5}\n'
    "```"
)

LLM2_PROMPT = (
    "You are a Cartesian mapping model for a NAO humanoid robot (0.5m tall).\n"
    "Input: A JSON intent containing 'use_hand' ('left', 'right', or 'both'), a description, and duration.\n"
    "Output: A JSON block containing 'use_hand', target (x, y, z), orientation, AND fingers direction ONLY for the active hand(s).\n"
    "\n"
    "THE use_hand RULE:\n"
    "You MUST include the 'use_hand' key in your output.\n"
    "- If use_hand is 'right': Output ONLY right_hand keys. OMIT ALL left_hand keys entirely.\n"
    "- If use_hand is 'left': Output ONLY left_hand keys. OMIT ALL right_hand keys entirely.\n"
    "- If use_hand is 'both': Output keys for both hands.\n"
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
    "Example Output (If input says use_hand: 'right'):\n"
    "```json\n"
    '{"use_hand": "right", "right_hand_pos": [0.15, -0.07, 0.05], "right_orientation": "palms_forward", "right_fingers": "up", "duration": 1.6}\n'
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

# def get_rotation_matrix(orientation_string, is_left_hand):
#     """Maps qualitative strings to 3x3 rotation matrices using Roll, Pitch, Yaw."""
#     # Base roll adjustments (might need flipping depending on NAO's exact URDF frame)
#     roll_flip = 1.0 if is_left_hand else -1.0 
    
#     # Defaults: Assuming 0,0,0 is palms facing inward (towards the legs)
#     orientations = {
#         "palms_in": pin.rpy.rpyToMatrix(0.0, 0.0, 0.0),
#         "palms_down": pin.rpy.rpyToMatrix(1.57 * roll_flip, 0.0, 0.0),
#         "palms_up": pin.rpy.rpyToMatrix(-1.57 * roll_flip, 0.0, 0.0),
#         "palms_forward": pin.rpy.rpyToMatrix(0.0, -1.57, 0.0),
#         "palms_out": pin.rpy.rpyToMatrix(3.14, 0.0, 0.0)
#     }
    
#     # Fallback to palms_in if the LLM hallucinates a weird string
#     return orientations.get(orientation_string, orientations["palms_in"])

import numpy as np

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
    # Remember: Z is the BACK of the hand, so it is the opposite of the palm direction.
    back_of_hand_vectors = {
        "palms_up": np.array([0.0, 0.0, -1.0]),       # Palm is +Z, Back is -Z
        "palms_down": np.array([0.0, 0.0, 1.0]),      # Palm is -Z, Back is +Z
        "palms_forward": np.array([-1.0, 0.0, 0.0]),  # Palm is +X, Back is -X
        "palms_backward": np.array([1.0, 0.0, 0.0])   # Palm is -X, Back is +X
    }

    # Handle Chirality for In/Out
    if is_left:
        # Left arm: Palm in faces RIGHT (-Y). Back of hand faces LEFT (+Y).
        back_of_hand_vectors["palms_in"] = np.array([0.0, 1.0, 0.0])
        back_of_hand_vectors["palms_out"] = np.array([0.0, -1.0, 0.0])
    else:
        # Right arm: Palm in faces LEFT (+Y). Back of hand faces RIGHT (-Y).
        back_of_hand_vectors["palms_in"] = np.array([0.0, -1.0, 0.0])
        back_of_hand_vectors["palms_out"] = np.array([0.0, 1.0, 0.0])

    # Fetch the vectors (with safe fallbacks)
    X_axis = finger_vectors.get(finger_string, np.array([1.0, 0.0, 0.0]))
    Z_axis = back_of_hand_vectors.get(orientation_string, back_of_hand_vectors["palms_in"])

    # 3. Orthogonality Check (Prevent Physics Crashes)
    # The fingers (X) and the back of the hand (Z) MUST be perpendicular (dot product = 0).
    # If the LLM hallucinates (e.g., fingers "forward" AND palms "forward"), fix it.
    if np.abs(np.dot(X_axis, Z_axis)) > 0.1:
        # Force a valid perpendicular Z-axis depending on the finger direction
        if X_axis[2] == 0:  # If fingers are horizontal, force palms down
            Z_axis = np.array([0.0, 0.0, 1.0])
        else:               # If fingers are vertical, force palms in
            Z_axis = back_of_hand_vectors["palms_in"]

    # 4. Calculate Y-Axis via Cross Product
    Y_axis = np.cross(Z_axis, X_axis)

    # 5. Stack the vectors into a 3x3 matrix
    # np.column_stack takes 1D arrays and turns them into the columns of a 2D matrix
    rotation_matrix = np.column_stack((X_axis, Y_axis, Z_axis))
    
    return rotation_matrix

def generate_pink_trajectory(cartesian_target, duration, dt=0.04):
    """
    Solves IK over time using PINK and the nao_clean.urdf.
    Outputs the exact dictionary structure expected by ALMotion.angleInterpolation.
    """
    try:
        # 1. Load Robot
        urdf_filename = "nao_clean.urdf"
        robot = pin.RobotWrapper.BuildFromURDF(urdf_filename)
        robot.q0 = np.maximum(robot.q0, robot.model.lowerPositionLimit)
        robot.q0 = np.minimum(robot.q0, robot.model.upperPositionLimit)

        configuration = pink.Configuration(robot.model, robot.data, robot.q0)

        # 2. Get the active hand from the JSON (default to both for safety)
        use_hand = cartesian_target.get("use_hand", "both")

        # 3. Initialize Tasks List
        tasks = []

        # Global Posture Task (keeps the torso/legs from collapsing)
        q_ref = robot.q0.copy()
        posture_task = PostureTask(cost=0.01)
        posture_task.set_target(q_ref)
        tasks.append(posture_task)

        # 4. Conditionally Setup Hand Tasks (Task Masking)
        if use_hand in ["left", "both"]:
            l_wrist_task = FrameTask("l_wrist", position_cost=1.0, orientation_cost=0.0)
            initial_l_pos = configuration.get_transform_frame_to_world("l_wrist").translation
            target_l_pos = np.array(cartesian_target.get("left_hand_pos", initial_l_pos))
            
            l_orient = cartesian_target.get("left_orientation", "palms_in")
            l_fingers = cartesian_target.get("left_fingers", "down")
            R_left = get_dynamic_rotation(l_orient, l_fingers, is_left=True)
            
            tasks.append(l_wrist_task)

        if use_hand in ["right", "both"]:
            r_wrist_task = FrameTask("r_wrist", position_cost=1.0, orientation_cost=0.0)
            initial_r_pos = configuration.get_transform_frame_to_world("r_wrist").translation
            target_r_pos = np.array(cartesian_target.get("right_hand_pos", initial_r_pos))
            
            r_orient = cartesian_target.get("right_orientation", "palms_in")
            r_fingers = cartesian_target.get("right_fingers", "down")
            R_right = get_dynamic_rotation(r_orient, r_fingers, is_left=False)
            
            tasks.append(r_wrist_task)

        # 5. Prepare Trajectory Storage
        time_steps = np.arange(0, duration, dt)
        if len(time_steps) == 0: 
            time_steps = [duration]

        joint_names = []
        joint_indices = []
        for i in range(1, robot.model.njoints):
            name = robot.model.names[i]
            q_idx = robot.model.joints[i].idx_q
            if "Thumb" not in name and "Finger" not in name and q_idx >= 0:
                joint_names.append(name)
                joint_indices.append(q_idx)

        trajectory_angles = {name: [] for name in joint_names}
        trajectory_times = {name: [] for name in joint_names}

        # 6. Simulation Loop
        for t in time_steps:
            progress = min(t / duration, 1.0)
            
            # Conditionally update targets only for the active hand(s)
            if use_hand in ["left", "both"]:
                cur_l_target = initial_l_pos + progress * (target_l_pos - initial_l_pos)
                l_wrist_task.set_target(pin.SE3(R_left, cur_l_target))
                
            if use_hand in ["right", "both"]:
                cur_r_target = initial_r_pos + progress * (target_r_pos - initial_r_pos)
                r_wrist_task.set_target(pin.SE3(R_right, cur_r_target))

            # Solve IK using ONLY the tasks inside the dynamic list
            velocity = solve_ik(configuration, tasks, dt, solver="quadprog")
            configuration.integrate_inplace(velocity, dt)

            for name, q_idx in zip(joint_names, joint_indices):
                trajectory_angles[name].append(float(configuration.q[q_idx]))
                trajectory_times[name].append(float(t + dt))

        # 7. Output Masking (Filter out the inactive arm)
        final_names = []
        final_times = []
        final_angles = []

        for name in joint_names:
            is_left_arm = name.startswith("LShoulder") or name.startswith("LElbow") or name.startswith("LWrist")
            is_right_arm = name.startswith("RShoulder") or name.startswith("RElbow") or name.startswith("RWrist")

            # Drop the joints of the arm that isn't being used
            if use_hand == "right" and is_left_arm:
                continue
            if use_hand == "left" and is_right_arm:
                continue

            final_names.append(name)
            final_times.append(trajectory_times[name])
            final_angles.append(trajectory_angles[name])

        return {
            "names": final_names,
            "times": final_times,
            "angles": final_angles
        }
        
    except Exception as e:
        print(f"PINK IK Error: {e}")
        return None

def process_paragraph(paragraph):
    sentences = re.split(r'(?<=[.!?]) +', paragraph)
    payload = []
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        print(f"\nProcessing: {sentence}")
        
        audio_b64, duration = generate_tts_and_duration(sentence, i)
        print(f"calculated duration: {duration} seconds")
        intent_json = call_llm(LLM1_PROMPT, f"Sentence: {sentence}, Duration: {duration}")
        print(f"LLM 1 Intent: {intent_json}")
        time.sleep(2)  # Small delay to avoid overwhelming the LLM with rapid calls
        cartesian_json = call_llm(LLM2_PROMPT, intent_json)
        print(f"LLM 2 Cartesian: {cartesian_json}")
        
        if cartesian_json:
            trajectory = generate_pink_trajectory(cartesian_json, duration)
        else:
            trajectory = None
            print("Skipping IK generation due to missing Cartesian data.")
        
        payload.append({
            "sentence": sentence,
            "audio_b64": audio_b64,
            "trajectory": trajectory
        })
        time.sleep(1)  # Small delay before processing the next sentence
        
    return payload  

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"BRAIN SERVER: Listening on {HOST}:{PORT}...")

    while True:
        try:
            print("\n[Waiting for Robot to request next instruction...]")
            conn, addr = server_socket.accept()
            print(f"Connected to Robot at {addr}")
            
            data = conn.recv(1024).decode('utf-8')
            if not data: 
                conn.close()
                continue
                
            text_to_process = input("Enter paragraph for NAO to execute: ")
            
            final_payload = process_paragraph(text_to_process)
            
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