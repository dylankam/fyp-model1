import requests
import json
import base64
import os
import tempfile 
import time
from naoqi import ALProxy
import pygame

# --- CONFIGURATION ---
SERVER_URL = 'http://127.0.0.1:65432/process'  # Change IP to the Brain Server machine if remote
ROBOT_IP = "127.0.0.1"  # Change to physical robot IP if not using simulator
ROBOT_PORT = 60031

def connect_to_server(motion, text):
    try:
        # 1. READ PHYSICAL REALITY
        joint_names = motion.getBodyNames("Body")
        joint_angles = motion.getAngles("Body", True)
        current_state_dict = dict(zip(joint_names, joint_angles))

        # 2. SEND via HTTP POST
        response = requests.post(SERVER_URL, json={
            "text": text,
            "robot": "nao",
            "angles": current_state_dict
        }, timeout=300)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print("Request Error:", e)
        return None

def execute_payload(payload, motion, audio_player, memory, posture):
    """
    Executes the payload. Currently toggled for Pygame local audio testing.
    Retains audio_player argument for instant transition to physical hardware.
    """
    for i, item in enumerate(payload):
        pygame.mixer.init()
        
        sentence = item["sentence"]
        audio_b64 = item["audio_b64"]
        traj = item["trajectory"]
        
        print("Executing:", sentence)
        
        # 1. Decode audio and save to local OS temp folder safely
        timestamp = int(time.time())
        audio_path = os.path.join(tempfile.gettempdir(), "nao_sent_{}_{}.mp3".format(i, timestamp))
        with open(audio_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))
            
        # 2. Trigger Choregraphe Speech Bubble
        memory.raiseEvent("ALTextToSpeech/CurrentSentence", str(sentence))
        
        pygame.mixer.music.load(audio_path)
        
        if traj and traj.get("names") and len(traj["names"]) > 0:
            ascii_names = [str(name) for name in traj["names"]]

            # --- THE TIME DILATION FIX ---
            safe_times = []
            for time_list in traj["times"]:
                safe_times.append([(t * 1.05) + 0.35 for t in time_list])
            # -----------------------------

            # 3. Start Audio (Non-blocking)
            
            # --- VIRTUAL TESTING TOGGLE ---
            pygame.mixer.music.play() 
            
            # --- PHYSICAL HARDWARE TOGGLE ---
            # audio_task = audio_player.post.playFile(audio_path)
            
            # 4. Execute Main Trajectory
            motion.angleInterpolation(ascii_names, traj["angles"], safe_times, True)
            
        else:
            # Fallback for empty trajectory
            pygame.mixer.music.play()
            # audio_task = audio_player.post.playFile(audio_path)
            
        # Wait for audio to finish if motion ended slightly early
        # --- VIRTUAL TESTING TOGGLE ---
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        # --- PHYSICAL HARDWARE TOGGLE ---
        # try:
        #     audio_player.wait(audio_task, 0)
        # except:
        #     pass
            
        # 5. Clear Speech Bubble and Cleanup
        memory.raiseEvent("ALTextToSpeech/CurrentSentence", "")
        
        pygame.mixer.quit()
        
        try:
            os.remove(audio_path)
        except Exception as e:
            pass

    print("Returning to neutral standing pose...")
    posture.goToPosture("Stand", 0.5)

if __name__ == "__main__":
    print("Starting NAO Client Loop...")
    
    # Initialize Proxies once globally so both functions can use them
    try:
        motion = ALProxy("ALMotion", ROBOT_IP, ROBOT_PORT)
        audio_player = ALProxy("ALAudioPlayer", ROBOT_IP, ROBOT_PORT)
        memory = ALProxy("ALMemory", ROBOT_IP, ROBOT_PORT)
        posture = ALProxy("ALRobotPosture", ROBOT_IP, ROBOT_PORT)
    except Exception as e:
        print("Could not create proxies:", e)
        exit(1)

    # Wake up robot and set stiffness
    motion.wakeUp()
    
    # Wrap the execution in an infinite loop
    while True:
        text_to_speak = raw_input("\nEnter text for NAO to speak (or Ctrl+C to quit): ").strip()
        if not text_to_speak:
            continue

        print("\nConnecting to Brain Server...")
        # Pass motion and text into the server connection
        payload = connect_to_server(motion, text_to_speak)
        
        if payload:
            print("Payload received. Beginning execution.")
            # Pass all initialized proxies into the execution function
            execute_payload(payload, motion, audio_player, memory, posture)
        else:
            print("Server not ready or disconnected. Retrying in 3 seconds...")
            time.sleep(3)