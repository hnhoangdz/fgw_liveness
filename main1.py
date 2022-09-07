from eye_blink.eye_blink_detection import EyeBlinkDetection
from side_face.side_face_detection import SideFaceDetection
from smile_face.smile_face_detection import SmileFaceDetection
from utils import *
import cv2
import matplotlib.pyplot as plt
import dlib
import numpy as np
import argparse
from cfg import *
from rules import make_rules, solve_rule, convert_rule2require
import time

def run():
    
    # =========================== PARAMETERS ==============================
    font = cv2.FONT_HERSHEY_SIMPLEX # font character
    line_type = cv2.LINE_AA # line type of text
    fps = None # fps
    is_match = False # match between dcb & fb 
    iou_list = [] # store iou values of dcb & fb
    face_box = None # face bounding box (fb)
    info = None # information to display
    n_frames = 0 # number of frames
    challenge = 'fail' # pass or fail of each rule
    rule_ith = 0 # ith-rule
    prev_frame_time = 0 # prev time of frame
    curr_frame_time = 0 # curr time of frame
    message = None # if pass all
    time_step = None # if pass one
    time_per_rule = None
    require = None # requirement (text)
    consecutive_pass_counter = 0
    colors = {
        'blue': (0, 0, 255), 
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'orange': (255, 128, 0),
        'pink': (255, 153, 255)
    }
    
    while True:
        # start time of frame 
        curr_frame_time = time.time()
        _, frame = video.read()
        frame = cv2.flip(frame, 1)
        
        # delay 15 frames in starting
        if n_frames > 15 and message is None:
            # face detection
            rects = detect_face(frame, args["detector"], detector)
            # draw dcb
            for (xmin, ymin, xmax, ymax) in [center_box]:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                            colors['pink'], 1)
                
            # if more than one face (require one) 
            if len(rects) > 1:
                del rects 
                cv2.putText(frame, "WARNING: MUST HAVE 1 PERSON HERE!!!",
                            (7, 140), font, 0.5, colors['red'], 2, line_type)
                
            # if only one face 
            elif len(rects) == 1:
                (x, y, w, h) = rects[0]
                # face bounding box
                face_bbox = (x, y, x+w, y+h)
                # face area
                face_bbox_area = area(face_bbox)
                cv2.rectangle(frame, (x, y),(x+w, y+h), 
                              colors['orange'], 2, line_type)

                # IOU score
                # rects = [xmin, ymin, xmax, ymax]
                iou_score, ioo_score, ioc_score = overlap_rate_box(face_bbox, center_box)

                # if IOU < threshold_match
                if ioc_score < args["threshold_match"] or face_bbox_area/center_box_area < 0.6:
                    del face_bbox
                    cv2.putText(frame, "WARNING: FACE MUST BE IN CENTER BOX",
                                (7, 70), font, 0.5, colors['red'], 2, line_type)
                else:
                    face_bbox = dlib.rectangle(x, y, x+w, y+h)
                    cv2.rectangle(frame, (x, y),
                                (x + w, y + h), colors['green'], 2)
                                        
                    if rule_ith == 3:
                        message = 'You did good'
                        time_break = time.time()

                    else:
                        if time_per_rule is None:
                            time_per_rule = time.time()
                        # do-rule
                        if time_step is None:
                            
                            require = convert_rule2require(rules[rule_ith])
                            cv2.putText(frame, require,
                                        (7, 70), font, 0.5, colors['red'], 2, line_type)
                            label = questions[rules[rule_ith]](frame, face_bbox)
                            # label = questions["side_face_left"](frame, face_bbox)
                            challenge = solve_rule(rules[rule_ith], label)

                            if challenge == 'pass':
                                print(f'passed {label}')
                                consecutive_pass_counter += 1
                                if consecutive_pass_counter >= 5:
                                    time_step = time.time()
                                    consecutive_pass_counter = 0 

                            # if challenge == 'fail':
                            #     if time.time() - time_per_rule >= 15:
                            #         print('eeeeeeeee')
                            #         cv2.putText(frame, 'Time is expired', (7, 70), 1, 2, colors['red'], 3)
                            #         continue
                            # else:
                            #     time_step = time.time()
                            #     time_per_rule = None
                            
                        else:
                            cv2.putText(frame, 'Wait 3s', (7, 70), 1, 2, colors['pink'], 3)
                            if time.time() - time_step > 3:
                                print(11111111111)
                                rule_ith += 1
                                challenge = 'fail'
                                time_step = None
                                
            # calculate fps
            fps = calculate_fps(prev_frame_time, curr_frame_time)
            prev_frame_time = curr_frame_time

            info = [
                (require,'is working'),
                ("FPS", "{:.2f}".format(fps))
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, frame_height - ((i * 20) + 20)),
                            font, 0.6, (0, 0, 255), 2)
                
        if message is not None:
            cv2.putText(frame, message, (7, 70), 1, 2, (0,255,0), 3)
            if time.time() - time_break>3:
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


if __name__ == '__main__':  
    # Input parameters
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str,
                    help="path to input video file")
    ap.add_argument("-d", "--detector", type=str, default="dlib",
                    help="face detector module (dlib/opencv/mtcnn)")

    ap.add_argument("-tc", "--threshold_match", type=float, default=0.5,
                    help="threshold center to decide tracking face or not")
    ap.add_argument("-nr", "--num_rules", type=int, default=3)
    args = vars(ap.parse_args())

    # Face detector    
    if args["detector"] == "opencv":
        detector = cv2.CascadeClassifier(facial_haar)
    elif args["detector"] == "dlib":
        detector = dlib.get_frontal_face_detector()

    # Require questions
    questions = {
        'eye_blink': EyeBlinkDetection(eye_blink_model_cew_v3, facial_landmarks, 2),
        'side_face_left': SideFaceDetection(facial_landmarks, require_side='left'),
        'side_face_right': SideFaceDetection(facial_landmarks, require_side='right'),
        'smile_face': SmileFaceDetection(smile_face_model_fer)
    }

    # Require rules
    rules = make_rules(args["num_rules"])

    # Video/Webcam
    if args.get("video", False) == None:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(args["video"])
        
    # Check Video/Webcam is working or not
    if video.isOpened() == False:
        print('Error when reading video')
        exit()

    # spatial dimension of frame
    frame_width = int(video.get(3))
    frame_height = int(video.get(4)) 
    size = (frame_width, frame_height)
    print('size frame: ', size)
    
    # write video
    result = cv2.VideoWriter('videos/test_system4.avi',
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size) 

    # default center box (dcb)
    center_box = default_center_box(frame_width, frame_height) 
    center_box_area = area(center_box)
    print(center_box_area)
    run()    
     
 