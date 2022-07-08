from facenet_pytorch import MTCNN
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import math
import requests
import argparse
import torch
import cv2
import time

parser = argparse.ArgumentParser('Face side detection')
parser.add_argument('-p', '--path', help='To use image path', type=str)
parser.add_argument('-u', '--url', help="To use image url", type=str)
args = parser.parse_args()

path = args.path
url = args.url

prev_frame_time = 0
new_frame_time = 0
left_offset = 20
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
fontThickness = 3
text_color = (0, 0, 255)
lineColor = (255, 255, 0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

mtcnn = MTCNN(image_size=240,
              margin=0,
              min_face_size=20,
              thresholds=[0.3, 0.4, 0.4],  # MTCNN thresholds
              factor=0.709,
              post_process=True,
              device=device  # If you don't have GPU
              )


def get_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(cosine)

    return np.degrees(angle)


def predFaceSide(frame):
    bbox_, prob_, landmarks_ = mtcnn.detect(frame, landmarks=True)
    label = None

    for bbox, landmarks, prob in zip(bbox_, landmarks_, prob_):
        if bbox is not None:
            if prob > 0.8:
                angR = get_angle(landmarks[0], landmarks[1], landmarks[2])
                angL = get_angle(landmarks[1], landmarks[0], landmarks[2])
                if ((int(angR) in range(35, 65)) and (int(angL) in range(35, 66))):
                    label = 'Frontal'
                else:
                    if angR < angL:
                        label = 'Left'
                    else:
                        label = 'Right'
            else:
                print('The detected face is Less then the detection threshold')
        else:
            print('No face detected in the image')
    return landmarks_, label


def visualize(frame, centerPoints, label):

    if label == 'frontal':
        color = (0, 0, 0)
    elif label == 'right':
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    cv2.putText(frame, f'pred: {label}', (10, 70),
                cv2.FONT_HERSHEY_PLAIN, fontScale, color, fontThickness, cv2.LINE_AA)
    print(centerPoints)
    for (x, y) in centerPoints[0]:
        cv2.circle(frame, (int(x), int(y)), radius=1, color=(
            0, 125, 125), thickness=-1)


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
