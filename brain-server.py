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

# --- CONFIGURATION ---
HOST = '0.0.0.0'
PORT = 65432
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
print("Gemini Client Initialized:", "Yes" if client else "No")

# --- PROMPTS ---
LLM1_PROMPT = (
    "You are a high-level gesture intent engine for a humanoid robot.\n"
    "Input: A sentence and the exact duration it takes to speak.\n"
    "Output: A JSON block containing the text, a high-level gesture description, and the duration.\n"
    "Keep the gesture simple and expressive.\n"
    "Example Output:\n"
    "```json\n"
    '{"text": "Hello everyone.", "gesture": {"amplitude": 0.8, "type": "open_arms"}, "duration": 1.5}\n'
    "```"
)

LLM2_PROMPT = (
    "You are a Cartesian mapping model for a NAO humanoid robot.\n"
    "Input: A JSON intent and duration.\n"
    "Output: A JSON block containing the target (x, y, z) coordinates for the left and right hand end-effectors, "
    "a qualitative wrist orientation, and the duration.\n"
    "\n"
    "PHYSICAL CONSTRAINTS & SYMMETRY (CRITICAL):\n"
    "- X (Forward/Back): Max 0.3 meters. 0.0 is chest. Do not use negative X.\n"
    "- Y (Left/Right): 0.0 is center. Left hand positive Y. Right hand negative Y.\n"
    "- Z (Up/Down): ORIGIN (0.0) IS THE MID-TORSO. +0.15 is face height. 0.0 is chest height. -0.1 is the waist/belt level. -0.2 is resting down by the hips.\n"
    "- Wrist Orientation: MUST be one of ['palms_up', 'palms_down', 'palms_in', 'palms_out', 'palms_forward', 'palms_backward'].\n"
    "- SYMMETRY: Most gestures (stop, point, refuse) are UNILATERAL (one-handed). To do a one-handed gesture, "
    "move the dominant hand to the target, and force the non-dominant hand to REST exactly at [0.0, +/-0.1, 0.0].\n"
    "\n"
    "GESTURE DESIGN:\n"
    "- 'Stop' or 'Halt': One hand ONLY. X=0.15 (close to chest), Z=0.0 (chest height), orientation: 'palms_forward'.\n"
    "- 'Welcome' or 'Present': Both hands. X=0.15, Y=+/- 0.25, Z=-0.15 (waist level), orientation: 'palms_up'.\n"
    "\n"
    "Example Output:\n"
    "```json\n"
    '{"left_hand_pos": [0.15, 0.25, -0.15], "right_hand_pos": [0.15, -0.25, -0.15], "orientation": "palms_up", "duration": 1.5}\n'
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
        model="gemini-2.0-flash",
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

def get_hardcoded_rotation(orientation_string, is_left):
    """
    Returns a 3x3 numpy rotation matrix explicitly mapped for Left/Right chirality.
    Global Frame: +X is Forward, +Y is Left, +Z is Up.
    Local Hand: +X is Fingertips, +Z is Back of Hand.
    """
    
    # ---------------- RIGHT HAND MATRICES ----------------
    # +Y is the Thumb side.
    right_orientations = {
        # Palms Down: Back of hand UP (+Z), Thumb LEFT (+Y)
        "palms_down": np.array([
            [1.0,  0.0,  0.0],
            [0.0,  1.0,  0.0],
            [0.0,  0.0,  1.0]
        ]),
        # Palms Up: Back of hand DOWN (-Z), Thumb RIGHT (-Y)
        "palms_up": np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ]),
        # Palms In: Palm faces LEFT (+Y), Back of hand RIGHT (-Y), Thumb UP (+Z)
        "palms_in": np.array([
            [1.0,  0.0,  0.0],
            [0.0,  0.0,  1.0],
            [0.0, -1.0,  0.0]
        ]),
        # Palms Forward (Stop): Fingers UP (+Z), Back of hand BACKWARD (-X)
        "palms_forward": np.array([
            [0.0,  0.0, -1.0],
            [0.0,  1.0,  0.0],
            [1.0,  0.0,  0.0]
        ]),
        # Palms Backward: Fingers UP (+Z), Back of hand FORWARD (+X)
        "palms_backward": np.array([
            [0.0,  0.0,  1.0],
            [0.0, -1.0,  0.0],
            [1.0,  0.0,  0.0]
        ])
    }

    # ---------------- LEFT HAND MATRICES ----------------
    # +Y is the Pinky side (Thumb is -Y).
    left_orientations = {
        # Palms Down: Back of hand UP (+Z), Pinky LEFT (+Y)
        "palms_down": np.array([
            [1.0,  0.0,  0.0],
            [0.0,  1.0,  0.0],
            [0.0,  0.0,  1.0]
        ]),
        # Palms Up: Back of hand DOWN (-Z), Pinky RIGHT (-Y)
        "palms_up": np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ]),
        # Palms In: Palm faces RIGHT (-Y), Back of hand LEFT (+Y), Thumb UP (-Z local? No, Y local is DOWN)
        "palms_in": np.array([
            [1.0,  0.0,  0.0],
            [0.0,  0.0, -1.0],
            [0.0,  1.0,  0.0]
        ]),
        # Palms Forward (Stop): Fingers UP (+Z), Back of hand BACKWARD (-X)
        "palms_forward": np.array([
            [0.0,  0.0, -1.0],
            [0.0,  1.0,  0.0],
            [1.0,  0.0,  0.0]
        ]),
        # Palms Backward: Fingers UP (+Z), Back of hand FORWARD (+X)
        "palms_backward": np.array([
            [0.0,  0.0,  1.0],
            [0.0, -1.0,  0.0],
            [1.0,  0.0,  0.0]
        ])
    }

    # Fetch the corresponding dictionary
    active_dict = left_orientations if is_left else right_orientations
    
    # Return the requested matrix, default to palms_in if the LLM hallucinates
    return active_dict.get(orientation_string, active_dict["palms_in"])

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

        # 2. Setup Tasks
        l_wrist_task = FrameTask("l_wrist", position_cost=1.0, orientation_cost=0.2)
        r_wrist_task = FrameTask("r_wrist", position_cost=1.0, orientation_cost=0.2)

        q_ref = robot.q0.copy()
        posture_task = PostureTask(cost=0.01)
        posture_task.set_target(q_ref)

        tasks = [l_wrist_task, r_wrist_task, posture_task]

        # 3. Get Initial vs Target Positions
        initial_l_pos = configuration.get_transform_frame_to_world("l_wrist").translation
        initial_r_pos = configuration.get_transform_frame_to_world("r_wrist").translation

        target_l_pos = np.array(cartesian_target.get("left_hand_pos", initial_l_pos))
        target_r_pos = np.array(cartesian_target.get("right_hand_pos", initial_r_pos))

        # 4. Prepare Trajectory Storage
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

        # Grab the desired orientation from the LLM, default to palms_in
        intent_orientation = cartesian_target.get("orientation", "palms_in")
        R_left = get_hardcoded_rotation(intent_orientation, is_left=True)
        R_right = get_hardcoded_rotation(intent_orientation, is_left=False)

        # 5. Simulation Loop
        for t in time_steps:
            progress = min(t / duration, 1.0)
            cur_l_target = initial_l_pos + progress * (target_l_pos - initial_l_pos)
            cur_r_target = initial_r_pos + progress * (target_r_pos - initial_r_pos)

            # Apply BOTH the moving translation and the static rotation matrix
            l_wrist_task.set_target(pin.SE3(R_left, cur_l_target))
            r_wrist_task.set_target(pin.SE3(R_right, cur_r_target))

            velocity = solve_ik(configuration, tasks, dt, solver="quadprog")
            configuration.integrate_inplace(velocity, dt)

            for name, q_idx in zip(joint_names, joint_indices):
                trajectory_angles[name].append(float(configuration.q[q_idx]))
                trajectory_times[name].append(float(t + dt))

        # 6. Format for ALMotion Client
        return {
            "names": joint_names,
            "times": [trajectory_times[name] for name in joint_names],
            "angles": [trajectory_angles[name] for name in joint_names]
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
        
        intent_json = call_llm(LLM1_PROMPT, f"Sentence: {sentence}, Duration: {duration}")
        print(f"LLM 1 Intent: {intent_json}")
        
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