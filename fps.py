# import the necessary packages
import datetime
from threading import Thread
import cv2
import argparse
import imutils
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=200,
                help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=1,
                help="Whether or not frames should be displayed")
args = vars(ap.parse_args())


class FPS(object):
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class WebcamVideoStream(object):
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


print("[INFO] sampling THREADED frames from webcam...")
# khởi tạo stream và khởi tạo Thread
vs = WebcamVideoStream(src=0).start()
# khởi tạo fps và bắt đầu tính thời điểm bắt đầu
fps = FPS().start() 
print(fps)

while True:
    # lấy ra frame
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # check to see if the frame should be displayed to our screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    # cập nhật số lương frame
    fps.update() 
    if key == ord("q"):
        print('Stopping camera!!!')
        print('end at: ', datetime.datetime.now())
        break
    
# kết thúc và tính thời điểm kết thúc
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()
