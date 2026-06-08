import requests
import json
import base64
import os
import tempfile
import time

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARN] pygame not installed — audio playback will be skipped.")

# --- CONFIGURATION ---
SERVER_URL = "http://127.0.0.1:65432/process"
ROBOT_PROFILE = "pepper"  # Change to match the profile the server should use

# Neutral standing pose angles for Pepper/NAO (radians).
# These mirror the values the real client would read from the robot at rest.
DEFAULT_JOINT_ANGLES = {
    "HeadYaw": 0.0,
    "HeadPitch": 0.0,
    "LShoulderPitch": 1.5708,
    "LShoulderRoll": 0.2094,
    "LElbowYaw": -1.2217,
    "LElbowRoll": -0.5236,
    "LWristYaw": 0.0,
    "LHand": 0.0,
    "RShoulderPitch": 1.5708,
    "RShoulderRoll": -0.2094,
    "RElbowYaw": 1.2217,
    "RElbowRoll": 0.5236,
    "RWristYaw": 0.0,
    "RHand": 0.0,
    "HipRoll": 0.0,
    "HipPitch": 0.0,
    "KneePitch": 0.0,
}


def send_request(text):
    """Send text + neutral pose to the Brain Server and return the payload."""
    try:
        print("\n[INFO] Sending request to Brain Server...")
        t_start = time.time()
        response = requests.post(
            SERVER_URL,
            json={
                "text": text,
                "robot": ROBOT_PROFILE,
                "angles": DEFAULT_JOINT_ANGLES,
            },
            timeout=300,
        )
        response.raise_for_status()
        elapsed = time.time() - t_start
        print("[INFO] Round-trip time: {:.2f}s".format(elapsed))
        return response.json()
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to Brain Server at {}".format(SERVER_URL))
        print("        Make sure brain-server.py is running.")
        return None
    except Exception as e:
        print("[ERROR] Request failed:", e)
        return None


def play_audio(audio_b64, index):
    """Decode base64 audio and play it with pygame, blocking until done."""
    if not PYGAME_AVAILABLE:
        print("        (audio skipped — pygame not available)")
        return

    timestamp = int(time.time())
    audio_path = os.path.join(
        tempfile.gettempdir(), "test_client_{}_{}.mp3".format(index, timestamp)
    )
    try:
        with open(audio_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))

        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


def print_trajectory_summary(traj):
    """Print a human-readable summary of a trajectory dict."""
    if not traj or not traj.get("names"):
        print("        Trajectory: (empty / no motion)")
        return
    names = traj["names"]
    times = traj["times"]
    angles = traj["angles"]
    print("        Trajectory joints ({} total):".format(len(names)))
    for i, name in enumerate(names):
        waypoint_count = len(times[i]) if i < len(times) else "?"
        print("          {} — {} waypoints".format(name, waypoint_count))


def execute_payload(payload):
    """Print each sentence and play its audio. No robot execution."""
    for i, item in enumerate(payload):
        sentence = item.get("sentence", "")
        audio_b64 = item.get("audio_b64", "")
        traj = item.get("trajectory", {})

        print("\n[{}] Sentence: {}".format(i + 1, sentence))
        print_trajectory_summary(traj)
        print("     Playing audio...")
        play_audio(audio_b64, i)

    print("\n[INFO] Payload execution complete.")


if __name__ == "__main__":
    print("=== Brain Server Test Client ===")
    print("Robot profile : {}".format(ROBOT_PROFILE))
    print("Server URL    : {}".format(SERVER_URL))
    print("(No robot connection required)\n")

    while True:
        try:
            text_input = input("Enter text to send (Ctrl+C to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not text_input:
            continue

        payload = send_request(text_input)
        if payload:
            print("[INFO] Payload received ({} sentence(s)).".format(len(payload)))
            execute_payload(payload)
        else:
            print("[WARN] No payload returned.")
