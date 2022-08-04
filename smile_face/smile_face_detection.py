from matplotlib.pyplot import get
from smile_face.model import Model
from smile_face.utils import predict, get_visualize
import cv2
import dlib
from PIL import Image
from imutils.face_utils import shape_to_np
import numpy as np

class SmileFaceDetection(object):
    def __init__(self, model_path, num_classes=2):
        self.model = Model(num_classes)
        self.model.eval()
        self.model_path = model_path
        
    def __call__(self, frame, face_bbox, visualize=True):
        face_bbox = frame[face_bbox.top(): face_bbox.bottom(), face_bbox.left():face_bbox.right()]
        gray = cv2.cvtColor(face_bbox, cv2.COLOR_BGR2GRAY)
        label = predict(gray, self.model_path, self.model)
        
        if visualize:
            get_visualize(frame, label)
        
        return label