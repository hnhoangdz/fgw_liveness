from imutils import face_utils
import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import requests
import argparse
import time

print(face_utils.FACIAL_LANDMARKS_IDXS)

parser = argparse.ArgumentParser('Face side detection')
parser.add_argument('-p', '--path', help='To use image path', type=str)
parser.add_argument('-u', '--url', help="To use image url", type=str)
parser.add_argument('-t', '--trained_model_path', 
                    default='../trained_models/shape_predictor_68_face_landmarks.dat', type=str)
args = parser.parse_args()

path = args.path
url = args.url
trained_model = args.trained_model_path

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(trained_model)

prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
fontThickness = 3

(lEyeStart, lEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rEyeStart, rEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(lNoseStart, rNoseEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

def get_angle(a, b, c):
    ba = a - b
    bc = c - b
    
    cosine = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(cosine)
    
    return np.degrees(angle)

def predFaceSide(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    label = None
    centerPoints = []
    
    for rect in rects:
        landmarks = predictor(gray, rect)
        landmarks = face_utils.shape_to_np(landmarks)
        leftEye = landmarks[lEyeStart:lEyeEnd]
        rightEye = landmarks[rEyeStart:rEyeEnd]
        leftCenterEye = np.mean(leftEye, axis=0)
        rightCenterEye = np.mean(rightEye, axis=0)
        noseCenter = landmarks[30]
        
        angR = get_angle(rightCenterEye, leftCenterEye, noseCenter)
        angL = get_angle(leftCenterEye, rightCenterEye, noseCenter)
        
        if ((int(angR) in range(35, 70)) and (int(angL) in range(35, 71))):
            label = 'frontal'
        else:
            if angR < angL:
                label = 'left'
            else:
                label = 'right'
        
        centerPoints.append(leftCenterEye)
        centerPoints.append(rightCenterEye)
        centerPoints.append(noseCenter)
        
    return centerPoints, label

def visualize(frame, centerPoints, label):
    
    if label == 'frontal':
        color = (0, 0, 0)
    elif label == 'right':
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    cv2.putText(frame, f'pred: {label}', (10, 70),
                cv2.FONT_HERSHEY_PLAIN, fontScale, color, fontThickness, cv2.LINE_AA)
    
    for (x,y) in centerPoints:
        cv2.circle(frame, (int(x), int(y)), radius=1, color=(
                0, 255, 255), thickness=-1)
        
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))
    cv2.putText(frame, f'FPS: {fps}', (10, 140), font, 1,
                (100, 255, 0), 3, cv2.LINE_AA)
    centerPoints, label = predFaceSide(frame)
    visualize(frame, centerPoints, label)

    cv2.imshow("Output", frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
