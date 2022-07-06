from eye_blink.eye_blink_detection import EyeBlink
from orientation.face_orientation import FaceOrientation
from liveness_detection import detect_liveness
from utils import detect_face, ioa, default_center_box, calculate_fps
from rules import make_rules
from imutils.video import FPS
import cv2
import matplotlib.pyplot as plt
import dlib
import numpy as np
import argparse
from cfg import *
import time

# ====================== Input parameters ======================
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str,
                    help="path to input video file")

    ap.add_argument("-t", "--tracker", type=str, default="kcf",
                    help="opencv object tracking type")

    ap.add_argument("-d", "--detector", type=str, default="dlib",
                    help="face detector module (dlib/opencv/mtcnn)")

    ap.add_argument("-tc", "--threshold_center", type=float, default=0.75,
                    help="threshold center to decide tracking face or not")
    args = vars(ap.parse_args())
    return args

# ====================== TRACKING OBJECT =======================
def get_tracker(args):
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
    return tracker

# ======================= Video/Webcam =========================
def get_video(args):
    if args.get("video", False) == None:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(args["video"])
    # Kiểm tra mở video/webcam có được không
    if video.isOpened() == False:
        print('Error when reading video')
        exit()
    return video

if __name__ == '__main__':
    args = get_args()
    tracker = get_tracker(args)
    video = get_video(args)

    # Chiều rộng, chiều dài của frame
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)

    # Khởi tạo default center box (dcb)
    # để bắt người dùng đưa khuôn mặt vào
    center_box = default_center_box(frame_width, frame_height)

    # Font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # FPS
    fps = None

    # Write video
    result = cv2.VideoWriter('videos/test_system.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size)

    # Match giữa dcb và facebox
    is_match = False

    ioa_list = []
    rects = []
    face_box = None
    info = None

    COUNTER, TOTAL = 0, 0

    # time của frame trước
    prev_frame_time, curr_frame_time = 0, 0

    while True:
        # Tính fps
        curr_frame_time = time.time()
        # Đọc video
        ret, frame = video.read()
        # Flip frame 
        frame = cv2.flip(frame, 1)
        if ret:
            # Nếu chưa match giữa
            # face bounding box và dcb
            if is_match == False:
                rects = detect_face(frame, type=args["detector"])
                # Nếu có nhiều hơn 1 face
                # không cho phép tiếp tục
                if len(rects) > 1:
                    rects = []
                    cv2.putText(frame, "WARNING: MUST HAVE 1 PERSON HERE!!!", \
                                (7, 140), font, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                # Nếu chỉ có 1 face
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
                    
                    # Xét ngưỡng IOA
                    # để quyết định cho phép 
                    # tiếp tục hay k
                    if ioa_score < args["threshold_center"]:
                        rects = []
                        cv2.putText(frame, "WARNING: FACE MUST BE IN CENTER",\
                                    (7, 140), font, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        face_box = (x, y, w, h)
                        cv2.imwrite('b.jpg',frame[y: y + h,x: x + w])
                        # Khởi tạo tracking
                        tracker.init(frame, face_box)
                        is_match = True
            
            # Nếu đã match
            # và lấy được face bounding box            
            elif face_box is not None:
                # Cập nhật tracking
                success, box = tracker.update(frame)
                # Toi uu code o day
                if success:
                    # my_rules = make_rules()
                    # print(my_rules)
                    (x_, y_, w_, h_) = [int(v) for v in box]
                    face_bbox = dlib.rectangle(x_, y_, x_ + w_, y_ + h_)
                    cv2.rectangle(frame, (x_, y_), (x_ + w_ + 20, y_ + h_ ),(0, 255, 0), 2)
                    cv2.imwrite('a.jpg',frame[y_: y_ + h_,x_: x_ + w_])
                    pass
                                    
            fps = calculate_fps(prev_frame_time, curr_frame_time)
            prev_frame_time = curr_frame_time
            
            info = [
                ("Tracker", args["tracker"]),
                ("FPS", "{:.2f}".format(fps)),
            ]
            
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, frame_height - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # vẽ default center box
            for (xmin, ymin, xmax, ymax) in [center_box]:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), 
                            (255, 255, 0), 1)            
            cv2.imshow("Frame", frame)
            result.write(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print('Stopping camera!!!')
                break
        else:
            print('Camera device is not working!!!')
            break
        
    result.release()
    plt.plot(ioa_list)
    plt.show()
    cv2.destroyAllWindows()
    print(face_box)