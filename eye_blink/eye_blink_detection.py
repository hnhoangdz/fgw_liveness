from imutils import face_utils
import dlib
import cv2
from eye_blink.model import Model
from eye_blink.utils import eye_aspect_ratio, eye_landmarks_to_bbox, predict
from PIL import Image

"""
Note: Nếu quá nhạy, thì sử dụng nhiều frame để quyết định.
"""

class EyeBlinkDetection(object):
    def __init__(self, model_path, landmarks_path, num_classes=4):
        self.landmarks_predictor = dlib.shape_predictor(landmarks_path)
        self.model_path = model_path
        self.model = Model(num_classes)
        self.model.eval()
        self.is_blinked = False
    
    def __call__(self, frame, face_bbox, visualize=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = self.landmarks_predictor(gray, face_bbox)
        landmarks = face_utils.shape_to_np(landmarks)
        
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        leftEye = landmarks[lStart:lEnd]
        rightEye = landmarks[rStart:rEnd]
        
        leftEar = eye_aspect_ratio(leftEye)
        rightEar = eye_aspect_ratio(rightEye)
        EAR = (leftEar + rightEar)/2.0
        
        xmin_l, ymin_l, xmax_l, ymax_l = eye_landmarks_to_bbox(leftEye)
        left_eye_bbox = Image.fromarray(gray[ymin_l:ymax_l, xmin_l:xmax_l])
        left_eye_label = predict(left_eye_bbox, self.model_path, self.model)
        
        xmin_r, ymin_r, xmax_r, ymax_r = eye_landmarks_to_bbox(rightEye)
        right_eye_bbox = Image.fromarray(gray[ymin_r:ymax_r, xmin_r:xmax_r])
        right_eye_label = predict(right_eye_bbox, self.model_path, self.model)
        
        if left_eye_label == 'left_close' and right_eye_label == 'right_close':
            if EAR < 0.2:
                print('Blinking')
                self.is_blinked = True
                
        if visualize:
            for (x,y) in landmarks:
                cv2.circle(frame, (x,y), 1, (255,255,0), 1, cv2.LINE_4)
        return self.is_blinked
    
