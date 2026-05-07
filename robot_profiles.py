import os
import numpy as np
from robot_descriptions import pepper_description

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
        "get_orientation": get_nao_orientation
    }
}
