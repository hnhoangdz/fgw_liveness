from side_face.utils import get_angle, get_center_eye, get_visualize
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from imutils import face_utils

"""
Note: Nếu quá nhạy, thì sử dụng nhiều frame để quyết định.
"""
fontScale = 2
fontThickness = 3

class SideFaceDetection(object):
    def __init__(self, landmarks_path, require_side):
        self.landmarks_predictor = dlib.shape_predictor(landmarks_path)
        self.require_side = require_side
        
    def __call__(self, frame, face_bbox, visualize=True):
        label = ''
        centerPoints = []
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
        
        centerPoints.append(leftCenterEye)
        centerPoints.append(rightCenterEye)
        centerPoints.append(noseCenter)
        
        angR = get_angle(rightCenterEye, leftCenterEye, noseCenter)
        angL = get_angle(leftCenterEye, rightCenterEye, noseCenter)
        
        if ((int(angR) in range(30, 60)) and (int(angL) in range(30, 61))):
            label = 'frontal'
        else:
            if angR < angL:
                label = 'left'
            else:
                label = 'right'
        
        if visualize:
            if label == 'frontal':
                color = (0, 0, 0)
            elif label == 'right':
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            cv2.putText(frame, f'side: {label}', (10, 140),
                        cv2.FONT_HERSHEY_PLAIN, fontScale, color, 1, cv2.LINE_AA)
            for (x,y) in landmarks:
                cv2.circle(frame, (int(x), int(y)), radius=1, color=(
                        0, 255, 125), thickness=2)
            for (x,y) in centerPoints:
                cv2.circle(frame, (int(x), int(y)), radius=1, color=(
                        125, 255, 125), thickness=2)
                
        return label == self.require_side