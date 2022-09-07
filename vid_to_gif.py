import cv2
import imageio
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v","--video_path", required=True, type=str)
ap.add_argument("-g","--gif_path", required=True, type=str)
args = vars(ap.parse_args())

video_path = args["video_path"]
gif_path = args["gif_path"]

cap = cv2.VideoCapture(video_path)
image_lst = []

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_lst.append(frame_rgb)

    cv2.imshow('a', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert to gif using the imageio.mimsave method
imageio.mimsave(gif_path, image_lst, fps=25)
