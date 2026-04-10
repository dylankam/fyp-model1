import socket
import json
import base64
import os
import tempfile 
from naoqi import ALProxy

# --- CONFIGURATION ---
SERVER_IP = '127.0.0.1' # Change to the IP of your Python 3 machine
SERVER_PORT = 65432
ROBOT_IP = "127.0.0.1"  # Change to physical robot IP if not using simulator
ROBOT_PORT = 9559

def connect_to_server():
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_IP, SERVER_PORT))
        client_socket.sendall("Requesting execution payload\n")
        
        # Receive the data chunks until the delimiter
        buffer_string = ""
        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            buffer_string += chunk
            if "<EOF>" in buffer_string:
                buffer_string = buffer_string.replace("<EOF>", "")
                break
                
        client_socket.close()
        return json.loads(buffer_string)
    except Exception as e:
        print("Socket Error:", e)
        return None

def execute_payload(payload):
    try:
        motion = ALProxy("ALMotion", ROBOT_IP, ROBOT_PORT)
        audio_player = ALProxy("ALAudioPlayer", ROBOT_IP, ROBOT_PORT)
        memory = ALProxy("ALMemory", ROBOT_IP, ROBOT_PORT)
        posture = ALProxy("ALRobotPosture", ROBOT_IP, ROBOT_PORT)
    except Exception as e:
        print("Could not create proxies:", e)
        return

    # Wake up robot and set stiffness
    motion.wakeUp()
    
    for item in payload:
        sentence = item["sentence"]
        audio_b64 = item["audio_b64"]
        traj = item["trajectory"]
        
        print("Executing:", sentence)
        
        # 1. Decode audio and save to local OS temp folder safely
        audio_path = os.path.join(tempfile.gettempdir(), "current_sentence.mp3")
        with open(audio_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))
            
        # 2. Trigger Choregraphe Speech Bubble
        memory.raiseEvent("ALTextToSpeech/CurrentSentence", str(sentence))
        
        if traj:
            ascii_names = [str(name) for name in traj["names"]]
            
            # Extract the very first angle for each joint from the trajectory
            start_angles = [angles_list[0] for angles_list in traj["angles"]]
            
            # Gently move to PINK's starting position over 1.5 seconds
            print("Moving to prep pose...")
            motion.angleInterpolation(ascii_names, start_angles, 1.5, True)

            # --- THE TIME DILATION FIX ---
            # Stretch the timestamps by 5% to absorb quintic spline rounding errors
            safe_times = []
            for time_list in traj["times"]:
                safe_times.append([t * 1.05 for t in time_list])
            # -----------------------------

            # 3. Start Audio (Non-blocking using 'post')
            audio_task = audio_player.post.playFile(audio_path)
            
            # 4. Execute Main Trajectory using the dilated times
            motion.angleInterpolation(ascii_names, traj["angles"], safe_times, True)
            
        # Wait for audio to finish if motion ended slightly early
        try:
            audio_player.wait(audio_task, 0)
        except:
            pass
        
        # 5. Clear Speech Bubble and Cleanup
        memory.raiseEvent("ALTextToSpeech/CurrentSentence", "")
        try:
            os.remove(audio_path)
        except:
            pass

    # Return to neutral pose
    print("Returning to neutral standing pose...")
    posture.goToPosture("Stand", 0.5)

import time # Add this to your imports at the top of the file if not already there

if __name__ == "__main__":
    print("Starting NAO Client Loop...")
    
    # Wrap the execution in an infinite loop
    while True:
        print("\nConnecting to Brain Server...")
        payload = connect_to_server()
        
        if payload:
            print("Payload received. Beginning execution.")
            execute_payload(payload)
        else:
            print("Server not ready or disconnected. Retrying in 3 seconds...")
            time.sleep(3)