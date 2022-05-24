from eye_blink.eye_blink_detection import EyeBlink
from orientation.face_orientation import FaceOrientation
from liveness_detection import detect_liveness
from utils import detect_face, ioa
from rules import make_rules
from imutils.video import FPS
import cv2
import matplotlib.pyplot as plt
import dlib
import numpy as np
import argparse
from cfg import *
# ==================== Input parameters ==========================
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
                help="path to input video file")

ap.add_argument("-t", "--tracker", type=str, default="kcf",
                help="opencv object tracking type")

ap.add_argument("-d", "--detector", type=str, default="opencv",
                help="face detector module (dlib/opencv)")

ap.add_argument("-tc", "--threshold_center", type=float, default=0.75,
                help="threshold center to decide tracking face or not")
args = vars(ap.parse_args())

# ====================== TRACKING OBJECT =========================
(major, minor) = cv2.__version__.split(".")[:2]
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())
else:
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.legacy.TrackerCSRT_create,
        "kcf": cv2.legacy.TrackerKCF_create,
        "boosting": cv2.legacy.TrackerBoosting_create,
        "mil": cv2.legacy.TrackerMIL_create,
        "tld": cv2.legacy.TrackerTLD_create,
        "medianflow": cv2.legacy.TrackerMedianFlow_create,
        "mosse": cv2.legacy.TrackerMOSSE_create
    }
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# Check Video or Webcam
if args.get("video", False) == None:
    video = cv2.VideoCapture(0)
else:
    video = cv2.VideoCapture(args["video"])

# Check video is opened or not
if video.isOpened() == False:
    print('Error when reading video')

# Width and height of frame
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

# Font
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize the FPS
fps = None

# Write video
result = cv2.VideoWriter('test_system.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

# Default center box
center_x = frame_width // 4
center_y = frame_height // 4
center_box = [center_x, center_y,
              center_x + center_x*2, center_y + center_y*2]

# Matching between face bounding box with default center box
is_match = False

ioa_list = []
rects = []
face_box = None
info = None

COUNTER, TOTAL = 0, 0
eye_blink = EyeBlink(facial_landmarks)
while True:
    # Start calculating fps
    fps = FPS().start()
    # Read video
    ret, frame = video.read()
    # Flip frame 
    frame = cv2.flip(frame, 1)
    if ret:
        # Convert image to gray 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if is_match == False:
            rects = detect_face(gray, type=args["detector"])
            # If there are more than one person
            # Don't allow to pass
            if len(rects) > 1:
                rects = []
                cv2.putText(frame, "WARNING: MUST HAVE 1 PERSON HERE!!!", \
                            (7, 140), font, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
            # If there is only one person
            elif len(rects) == 1:
                # Face bounding box
                (x, y, w, h) = rects[0]
                xmin, ymin, xmax, ymax = (x, y, x+w, y+h)
                rects = [xmin, ymin, xmax, ymax]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 1)
                
                # IOA score
                ioa_score = ioa(rects, center_box)
                ioa_list.append(ioa_score)
                print('IOA: ', ioa_score)
                
                # IOA score to determine face matched center box or not
                if ioa_score < args["threshold_center"]:
                    rects = []
                    cv2.putText(frame, "WARNING: FACE MUST BE IN CENTER",\
                                (7, 140), font, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    face_box = (x, y, w, h)
                    cv2.imwrite('b.jpg',frame[y: y + h,x: x + w])
                    tracker.init(frame, face_box)
                    is_match = True
                    
        elif face_box is not None:
            success, box = tracker.update(frame)
            if success:
                (x_, y_, w_, h_) = [int(v) for v in box]
                face_bbox = dlib.rectangle(x_, y_, x_ + w_, y_ + h_)
                cv2.rectangle(frame, (x_, y_), (x_ + w_ + 20, y_ + h_ ),
                              (0, 255, 0), 2)
                cv2.imwrite('a.jpg',frame[y_: y_ + h_,x_: x_ + w_])
                c, t = eye_blink.detect_eye_blink(frame, gray, face_bbox, COUNTER, TOTAL)
                if TOTAL == 1:
                    print('BLINK EYE')
                    exit()
                COUNTER = c
                TOTAL = t
                print('Total Blinking Eye', TOTAL)
                # out = detect_liveness(frame, gray, face_bbox, COUNTER, TOTAL)
                # COUNTER = out['count_blinks_consecutives']
                # TOTAL = out['total_blinks']
                # if TOTAL == 1:
                #     print('BLINK EYE')
                #     exit()
        fps.update()
        fps.stop()
        info = [
            ("Tracker", args["tracker"]),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, frame_height - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for (xmin, ymin, xmax, ymax) in [center_box]:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), 
                        (255, 255, 0), 1)            
        cv2.imshow("Frame", frame)
        result.write(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print('Stop camera!!!')
            break
    else:
        print('Camera device is not working!!!')
        break
result.release()
plt.plot(ioa_list)
plt.show()
cv2.destroyAllWindows()
print(face_box)