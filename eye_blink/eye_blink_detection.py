from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from utils import eye_aspect_ratio

def EyeBlink(object):
    def __init__(self, eye_threshold, counter_consecutive, path_landmarks):
        self.eye_threshold = eye_threshold
        self.counter_consecutive = counter_consecutive
        self.predictor = dlib.shape_predictor(path_landmarks)
    
    def detect_eye_blink(self, gray, rect, COUNTER, TOTAL):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]