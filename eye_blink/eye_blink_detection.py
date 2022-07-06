from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from eye_blink.utils import eye_aspect_ratio

def eye_landmarks_to_bbox(eyes, padding=10):
    p0, p1, p2, p3, p4, p5 = [x for x in eyes]
    xmin = p0[0]
    ymin = min(p1[1], p2[1])
    xmax = p3[0]
    ymax = max(p4[1], p5[1])
    return xmin - padding, ymin - padding, xmax + padding, ymax + padding

def eye_aspect_ratio(eyes):
    A = dist.euclidean(eyes[1],eyes[5])
    B = dist.euclidean(eyes[2],eyes[4])
    C = dist.euclidean(eyes[0],eyes[3])
    ear = (A + B) / (2.0 * C)
    return ear


class EyeBlink(object):
    def __init__(self, path_landmarks, eye_threshold=0.2, counter_consecutive=5):
        self.eye_threshold = eye_threshold
        self.counter_consecutive = counter_consecutive
        self.predictor = dlib.shape_predictor(path_landmarks)
    
    def detect_eye_blink(self, frame, gray, rect, COUNTER, TOTAL):
        """_summary_

        Args:
            gray (numpy): gray image
            rect: bx, by, bw, bh 
            rect (numpy/list/tuple): bounding box of face
            COUNTER (int): count number of eye blinks in consecutive frames
            TOTAL (int): count number of eye blinks in whole video
        """
        
        # Indexes of eye in facial_lankmarks model (68 points)
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # Facial points 
        facial_points = self.predictor(gray, rect)
        facial_points = face_utils.shape_to_np(facial_points)
        
        # Eye points
        left_eye = facial_points[lStart:lEnd]
        right_eye = facial_points[rStart:rEnd]
        for (i, (x, y)) in enumerate(left_eye):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                    
        for (i, (x, y)) in enumerate(right_eye):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            
        # Eye aspect ratio
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        eye_ear = (left_ear + right_ear)/2.0
        print('left ear: ', left_ear)
        print('right ear: ', right_ear)
        print('=========================')
        # Check eye blink
        if left_ear < self.eye_threshold:
            COUNTER += 1
        else:
            # print('counter: ', COUNTER)
            if COUNTER >= self.counter_consecutive:
                TOTAL += 1
            COUNTER = 0
        return COUNTER, TOTAL
    
if __name__ == '__main__':
    font = cv2.FONT_HERSHEY_SIMPLEX
    landmarks_path = 'landmarks68/shape_predictor_68_face_landmarks.dat'
    eye_blink = EyeBlink(landmarks_path)
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor()

    print("[INFO] camera sensor warming up...")
    video = cv2.VideoCapture(0)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('result.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size)
    # used to record the time when we processed last frame
    prev_frame_time = 0
    
    # used to record the time at which we processed current frame
    new_frame_time = 0

    COUNTER, TOTAL = 0,0
    while True:
        ret, frame = video.read()
        if ret == True:
            frame = cv2.flip(frame,1)
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = detector(gray, 0)
            
            # time when we finish processing for this frame
            new_frame_time = time.time()

            # Tính fps
            # fps sẽ là số khung hình được x
            # vì sẽ có hầu hết sai số thời gian là 0,001 giây, 
            # trừ nó để có kết quả chính xác hơn
            fps = 1/(new_frame_time-prev_frame_time)
        
            prev_frame_time = new_frame_time

            fps = str(int(fps))

            # putting the FPS count on the frame
            cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            
            # loop over the face detections
            for rect in rects:
                (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
                    (0, 255, 0), 1)
                
                COUNTER, TOTAL = eye_blink.detect_eye_blink(frame, gray, rect, COUNTER, TOTAL)
                cv2.putText(frame, 'Blink Left: ' + str(TOTAL), (7, 140), font, 1, (0, 255, 0), 3, cv2.LINE_AA)
            result.write(frame)
            # show the frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        else:
            print('Camera device is not recognized!!!')
            break
        
    result.release()
    cv2.destroyAllWindows()
        
        