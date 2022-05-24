import cv2
import dlib
import numpy as np
from eye_blink.eye_blink_detection import EyeBlink
from orientation.face_orientation import FaceOrientation
from utils import detect_face
from cfg import *

eye_blink = EyeBlink(facial_landmarks)
face_orientation = FaceOrientation(facial_orientation_haar)

def detect_liveness(frame, gray, rect, COUNTER=0, TOTAL=0):
    COUNTER, TOTAL = eye_blink.detect_eye_blink(frame, gray, rect, COUNTER=COUNTER, TOTAL=TOTAL)
    box_orientation, name_orientation = face_orientation.detect_orientation(gray)
    out = {
        'total_blinks': TOTAL,
        'count_blinks_consecutives': COUNTER,
        'box_orientation': box_orientation,
        'name_orientation': name_orientation
    }
    return out