from side_face.utils import get_angle, get_center_eye, get_visualize
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from imutils import face_utils

"""
Note: Nếu quá nhạy, thì sử dụng nhiều frame để quyết định.
"""

class SideFaceDetection(object):
    def __init__(self, landmarks_path):
        self.landmarks_predictor = dlib.shape_predictor(landmarks_path)
        self.centerPoints = []
    
    def __call__(self, frame, face_bbox, visualize=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = self.landmarks_predictor(gray, face_bbox)
        landmarks = face_utils.shape_to_np(landmarks)
        
        (lEyeStart, lEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rEyeStart, rEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        leftEye = landmarks[lEyeStart:lEyeEnd]
        rightEye = landmarks[rEyeStart:rEyeEnd]
        noseCenter = landmarks[30]
        
        leftCenterEye = get_center_eye(leftEye)
        rightCenterEye = get_center_eye(rightEye)
        
        self.centerPoints.append(leftCenterEye)
        self.centerPoints.append(rightCenterEye)
        self.centerPoints.append(noseCenter)
        
        angR = get_angle(rightCenterEye, leftCenterEye, noseCenter)
        angL = get_angle(leftCenterEye, rightCenterEye, noseCenter)
        
        if ((int(angR) in range(35, 70)) and (int(angL) in range(35, 71))):
            label = 'frontal'
        else:
            if angR < angL:
                label = 'left'
            else:
                label = 'right'
        
        if visualize:
            get_visualize(frame, self.centerPoints, label)
                
        return label