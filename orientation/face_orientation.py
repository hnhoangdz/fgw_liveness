from orientation.utils import detect, convert_rightbox, get_areas
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

class FaceOrientation(object):
    def __init__(self, face_orientation_path):
        # face_orientation_path: path to pre-trained Face Orientation model
        self.face_orientation_path = face_orientation_path
        
    def detect_orientation(self, gray):
        # Init instance Face Orientation
        face_orientation_obj = cv2.CascadeClassifier(self.face_orientation_path)
        
        # Left side detect
        box_left = detect(gray,face_orientation_obj)
        if len(box_left) == 0:
            box_left = []
            name_left = []
        else:
            name_left = ["left"]
            
        # Right side detect
        gray_flip = cv2.flip(gray, 1)
        box_right = detect(gray_flip,face_orientation_obj)
        if len(box_right) == 0:
            box_right = []
            name_right = []
        else:
            box_right = convert_rightbox(gray, box_right)
            name_right = ["right"]
            
        # Find orientation
        boxes = list(box_left)+list(box_right)
        names = list(name_left)+list(name_right)
        if len(boxes)==0:
            return boxes, names
        else:
            index = np.argmax(get_areas(boxes))
            boxes = [boxes[index].tolist()]
            names = [names[index]]
        return boxes, names
    
if __name__ == '__main__':
    # Init webcam
    video = cv2.VideoCapture(0)
   
    # Check camera is opened or not
    if (video.isOpened() == False): 
        print("Error reading video file")
    
    # Size of frame (convert to int)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    
    # Write video
    result = cv2.VideoWriter('test_face_orientation.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
    
    while(True):
        ret, frame = video.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x = FaceOrientation('haar_cascade/haarcascade_profileface.xml')
        rects, confidence = x.detect_orientation(gray)
        print('name: ', confidence)
        if ret == True: 
            for (xmin, ymin, xmax, ymax) in rects:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            
            # Write frame
            result.write(frame)
    
            # Display the frame
            cv2.imshow('Frame', frame)
    
            # Press S on keyboard 
            # to stop the process
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
    
        # Break the loop
        else:
            break
    
    # When everything done, release 
    # the video capture and video 
    # write objects
    video.release()
    result.release()
        
    # Closes all the frames
    cv2.destroyAllWindows()
    
    print("The video was successfully saved")