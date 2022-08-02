from eye_blink.eye_blink_detection import EyeBlinkDetection
from side_face.side_face_detection import SideFaceDetection
from smile_face.smile_face_detection import SmileFaceDetection
from utils import detect_face, ioa, default_center_box, calculate_fps, iou
import cv2
import matplotlib.pyplot as plt
import dlib
import numpy as np
import argparse
from cfg import *
from rules import make_rules, solve_rule, convert_rule2require
import time

# ====================== Input parameters ======================
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str,
                    help="path to input video file")

    ap.add_argument("-d", "--detector", type=str, default="dlib",
                    help="face detector module (dlib/opencv/mtcnn)")

    ap.add_argument("-tc", "--threshold_match", type=float, default=0.5,
                    help="threshold center to decide tracking face or not")
    ap.add_argument("-c", "--case", type=str, default="eye_blink")
    ap.add_argument("-nr", "--num_rules", type=int, default=3)
    args = vars(ap.parse_args())
    return args

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
    rules = make_rules(args["num_rules"])
    
    banks = {
        'eye_blink': EyeBlinkDetection(eye_blink_model_cew_v3, facial_landmarks, 2),
        'side_face_left': SideFaceDetection(facial_landmarks, require_side='left'),
        'side_face_right': SideFaceDetection(facial_landmarks, require_side='right'),
        'smile_face': SmileFaceDetection(smile_face_model_fer)
    }
        
    # face detector    
    if args["detector"] == "opencv":
        detector = cv2.CascadeClassifier(facial_haar)
    elif args["detector"] == "dlib":
        detector = dlib.get_frontal_face_detector()
    
    # parameters
    font = cv2.FONT_HERSHEY_SIMPLEX # font character
    fps = None # fps
    is_match = False # match between dcb & fb 
    iou_list = [] # store iou values of dcb & fb
    face_box = None # face bounding box (fb)
    info = None # information to display
    n_frames = 0 # number of frames
    challenge = 'fail' # 
    rule_ith = 0
    t0 = 0
    
    # init video & write
    video = get_video(args) # video object
    frame_width = int(video.get(3)) # width
    frame_height = int(video.get(4)) # height
    size = (frame_width, frame_height) # size
    result = cv2.VideoWriter('videos/test_system3.avi',
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size) # write video
    
    # default center box (dcb)
    center_box = default_center_box(frame_width, frame_height)
    
    # prev time & curr time of frame
    prev_frame_time, curr_frame_time = 0, 0
    message = None
    time_step = None
    
    while True:
        # start time of frame 
        curr_frame_time = time.time()
        
        # read video
        ret, frame = video.read()
        
        # flip frame
        frame = cv2.flip(frame, 1)
        
        # delay 10 frames in starting
        if n_frames > 10 and message is None:
            
            # detect face
            rects = detect_face(frame, args["detector"], detector)
            
            # draw dcb
            for (xmin, ymin, xmax, ymax) in [center_box]:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                            (255, 255, 0), 1)
                
            # more than one face
            # require one
            if len(rects) > 1:
                del rects 
                cv2.putText(frame, "WARNING: MUST HAVE 1 PERSON HERE!!!",
                            (7, 140), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            
            # only one face
            elif len(rects) == 1:
                # face bounding box
                (x, y, w, h) = rects[0]
                xmin, ymin, xmax, ymax = (x, y, x+w, y+h)
                cv2.rectangle(frame, (xmin, ymin),
                                (xmax, ymax), (0, 255, 255), 1)

                # IOU score
                rects = [xmin, ymin, xmax, ymax]
                iou_score = iou(rects, center_box)
                ioa_score = ioa(rects, center_box)
                # iou_list.append(iou_score)
                # print('IOU: ', iou_score)
                # print('IOA: ', ioa_score)

                # if IOU < threshold_match
                if iou_score < args["threshold_match"] and ioa_score < 0.7:
                    del rects
                    cv2.putText(frame, "WARNING: FACE MUST BE IN CENTER",
                                (7, 70), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    face_box = (x, y, w, h)
                    is_match = True
                    
                    # draw fb
                    (x, y, w, h) = [int(v) for v in face_box]
                    face_bbox = dlib.rectangle(x, y, x + w, y + h)
                    cv2.rectangle(frame, (x, y),
                                (x + w, y + h), (0, 255, 0), 2)
                                        
                    if rule_ith == 3:
                        message = 'You did good'
                        time_break = time.time()
                        # cv2.putText(frame, , (7, 70), 1, 1, (0,0,255), 1)
                        # cv2.waitKey(1)
                        # time.sleep(5)
                        # print("break")
                        # exit()
                    else:
                        # do-rule

                        if time_step is None:
                            require = convert_rule2require(rules[rule_ith])
                            cv2.putText(frame, require,
                                        (7, 70), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                            label = banks[rules[rule_ith]](frame, face_bbox)
                            # print('label: ', label)
                            challenge = solve_rule(rules[rule_ith], label)
                            
                            if challenge == 'pass':
                                time_step = time.time()
                            
                        else:
                            cv2.putText(frame, 'wait 3s', (7, 70), 1, 2, (0,255,0), 3)
                            if time.time() - time_step > 3:
                                print(11111111111)
                                rule_ith += 1
                                time_step = None
                        
                            
            # calculate fps
            fps = calculate_fps(prev_frame_time, curr_frame_time)
            prev_frame_time = curr_frame_time

            info = [
                (args["case"],'is working'),
                ("FPS", "{:.2f}".format(fps))
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, frame_height - ((i * 20) + 20)),
                            font, 0.6, (0, 0, 255), 2)
        if message:
            cv2.putText(frame, message, (7, 70), 1, 2, (0,255,0), 3)
            if time.time()-time_break>3:
                exit()
        cv2.imshow("Frame", frame)
        result.write(frame)
        key = cv2.waitKey(1) & 0xFF
        n_frames += 1
        if key == ord("q"):
            print('Stopping camera!!!')
            break

    result.release()
    plt.show()
    cv2.destroyAllWindows()
    print(face_box)
