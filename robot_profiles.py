import os
import numpy as np
from robot_descriptions import pepper_description, talos_description, icub_description

# Robot specific functions to compute rotation matrices based on orientation and finger direction
def get_nao_orientation(orientation_string, finger_string, is_left):
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


def get_icub_orientation(orientation_string, finger_string, is_left):
    """
    Constructs a 3x3 rotation matrix for the iCub physical l_hand / r_hand body.

    The robot faces world +X (pinocchio world frame).
    Left arm at world -Y, right arm at world +Y, up = world +Z.
    Forward = +X, backward = -X, left = -Y, right = +Y.

    Used with end-effectors 'l_hand' and 'r_hand' (physical bodies, no DH offset).
    The matrix columns represent the desired finger direction (col0), an
    orthogonal in-plane axis (col1), and the back-of-hand direction (col2).
    """
    # Robot faces world +X. Left arm at world -Y, right arm at world +Y, up = +Z.
    # forward=+X, backward=-X, left=-Y, right=+Y
    finger_vectors = {
        "forward":  np.array([ 1.0,  0.0,  0.0]),
        "backward": np.array([-1.0,  0.0,  0.0]),
        "up":       np.array([ 0.0,  0.0, -1.0]),
        "down":     np.array([ 0.0,  0.0,  1.0]),
        "left":     np.array([ 0.0, -1.0,  0.0]),
        "right":    np.array([ 0.0,  1.0,  0.0]),
    }

    # Store palm NORMAL (Y-axis = col1). Z (thumb) is derived as cross(X_finger, Y_palm).
    # Storing Y instead of Z means palm direction stays consistent for ALL finger directions.
    # Derived from confirmed empirical cases — same Y_axis regardless of finger direction.
    # IK world: robot faces +X, left arm at +Y, right arm at -Y, up = +Z.
    palm_normals = {
        "palms_forward":  np.array([ 1.0,  0.0,  0.0]),  # confirmed
        "palms_backward": np.array([-1.0,  0.0,  0.0]),  # confirmed
        "palms_up":       np.array([ 0.0,  0.0, -1.0]),  # confirmed
        "palms_down":     np.array([ 0.0,  0.0,  1.0]),  # confirmed
    }

    if is_left:
        finger_vectors["forward"]  = np.array([-1.0,  0.0,  0.0])
        finger_vectors["backward"] = np.array([ 1.0,  0.0,  0.0])
        finger_vectors["up"]       = np.array([ 0.0,  0.0,  1.0])
        finger_vectors["down"]     = np.array([ 0.0,  0.0, -1.0])
        palm_normals["palms_in"]  = np.array([ 0.0, -1.0,  0.0])  # confirmed
        palm_normals["palms_out"] = np.array([ 0.0,  1.0,  0.0])
    else:
        palm_normals["palms_in"]  = np.array([ 0.0,  1.0,  0.0])  # confirmed
        palm_normals["palms_out"] = np.array([ 0.0, -1.0,  0.0])

    X_axis = finger_vectors.get(finger_string, finger_vectors["forward"])
    Y_axis = palm_normals.get(orientation_string, palm_normals["palms_in"])

    # If palm normal is parallel to finger direction, pick an orthogonal fallback
    if np.abs(np.dot(X_axis, Y_axis)) > 0.9:
        for candidate in [np.array([0,0,1.0]), np.array([0,1.0,0]), np.array([1.0,0,0])]:
            if np.abs(np.dot(X_axis, candidate)) < 0.9:
                Y_axis = candidate
                break

    Z_axis = np.cross(X_axis, Y_axis)
    Z_axis = Z_axis / np.linalg.norm(Z_axis)
    Y_axis = np.cross(Z_axis, X_axis)  # reorthogonalize
    Y_axis = Y_axis / np.linalg.norm(Y_axis)
    return np.column_stack((X_axis, Y_axis, Z_axis))

   
# Define profiles for each robot
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
            "x_max": 0.25,   
            "y_max": 0.25,   
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
            "x_max": 0.45,   
            "y_max": 0.40,   
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
        "axis_inversion": {"x": 1.0, "y": -1.0, "z": 1.0},
        "get_orientation": get_icub_orientation
    }
}


