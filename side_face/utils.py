import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
fontThickness = 3

def get_angle(a, b, c):
    ba = a - b
    bc = c - b
    
    cosine = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(cosine)
    
    return np.degrees(angle)

def get_center_eye(eye):
    return np.mean(eye, axis=0)

def get_visualize(frame, centerPoints, label):
    
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
                125, 255, 125), thickness=-1)