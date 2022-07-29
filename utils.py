import cv2
import dlib
import numpy as np
from imutils import face_utils
from cfg import *
from facenet_pytorch import MTCNN
import torch


def detect_face(frame, name_detector, detector):
    """Face detection
        using: dlib or haar cascade or mtcnn Pytorch
    Args:
        gray (numpyarray): gray image
        type (str, optional): Face detector type. Defaults to 'dlib'.

    Returns:
        list: bounding box of face
              x, y, w, h
    """
    # assert type == 'dlib' or type == 'opencv', 'Error face detector object'
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    if name_detector == 'dlib':
        faces = detector(gray, 0)
        faces = [face_utils.rect_to_bb(face) for face in faces]
        return faces
    elif name_detector == 'opencv':
        faces = detector.detectMultiScale3(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30),
                                           flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels=True)
        faces = faces[0]
        return faces


def ioa(boxA, boxB):
    """Calculate intersection over original area

    Args:
        boxA (numpy array/list): bounding box A
        boxB (numpy array/list): bounding box B

    Returns:
        float: 0 <= ioa_score <= 1
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    ioa_score = interArea / float(boxAArea)
    return ioa_score

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou_score = interArea/(boxAArea + boxBArea - interArea)
    return iou_score

def default_center_box(frame_width, frame_height):
    center_x = frame_width // 4
    center_y = frame_height // 4
    center_box = [center_x, center_y,
                  center_x + center_x*2, center_y + center_y*2]
    return center_box

def calculate_fps(prev_frame_time, curr_frame_time):
    return int(1/(curr_frame_time-prev_frame_time))


if __name__ == '__main__':
    img = cv2.imread('images/1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_face(gray, type='opencv')
    print(faces)
    for rect in faces:
        print(rect)
        x, y, w, h = rect[0], rect[1], rect[2], rect[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv2.imshow('img', img)
    cv2.waitKey(0)
