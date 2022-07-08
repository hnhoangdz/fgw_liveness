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
    def __init__(self, landmarks_path):
        self.landmarks_predictor = dlib.shape_predictor(landmarks_path)
        self.label = ''
    
    def __call__(self, frame, face_bbox, visualize=True):
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
        
        if ((int(angR) in range(35, 70)) and (int(angL) in range(35, 71))):
            self.label = 'frontal'
        else:
            if angR < angL:
                self.label = 'left'
            else:
                self.label = 'right'
        
        if visualize:
            if self.label == 'frontal':
                color = (0, 0, 0)
            elif self.label == 'right':
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            cv2.putText(frame, f'pred: {self.label}', (10, 70),
                        cv2.FONT_HERSHEY_PLAIN, fontScale, color, fontThickness, cv2.LINE_AA)
            for (x,y) in landmarks:
                cv2.circle(frame, (int(x), int(y)), radius=1, color=(
                        0, 255, 125), thickness=2)
            for (x,y) in centerPoints:
                cv2.circle(frame, (int(x), int(y)), radius=1, color=(
                        125, 255, 125), thickness=2)
                
        return self.label