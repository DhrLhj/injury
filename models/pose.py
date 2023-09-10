import numpy as np
import mediapipe as mp

class pose():
    """pose estimation
    1. open camera and get image frame
    2. pose estimation
    3. return key points according the requests
    """
    def __init__(self):
        self.camera = 0 
        self.pose_model = None
        self.keypoints = None

    def get_frame(self):
        return None
