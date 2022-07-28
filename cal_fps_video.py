import cv2
from utils import calculate_fps
import time
scale_factor = 1.2
min_neighbors = 3
min_size = (50, 50)
webcam=True #if working with video file then make it 'False'

def detect(path):

    cascade = cv2.CascadeClassifier(path)
    if webcam:
        video_cap = cv2.VideoCapture(0) # use 0,1,2..depanding on your webcam
        frame_height = int(video_cap.get(4))
    else:
        video_cap = cv2.VideoCapture("videoFile.mp4")
    prev_frame_time, curr_frame_time = 0, 0
    while True:
        curr_frame_time = time.time()
        # Capture frame-by-frame
        ret, img = video_cap.read()

        #converting to gray image for faster video processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                         minSize=min_size)
        # if at least 1 face detected
        if len(rects) >= 0:
            # Draw a rectangle around the faces
            for (x, y, w, h) in rects:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the resulting frame
            
            
            
        fps = calculate_fps(prev_frame_time, curr_frame_time)
        prev_frame_time = curr_frame_time

        info = [
            ("FPS", "{:.2f}".format(fps))
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(img, text, (10, frame_height - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow('Face Detection on Video', img)
        #wait for 'q' to close the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_cap.release()

def main():
    cascadeFilePath="trained_models/haarcascade_frontalface_alt.xml"
    detect(cascadeFilePath)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()