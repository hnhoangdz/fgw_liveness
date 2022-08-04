import cv2
import imageio

video_path = '/home/hoangdinhhuy/Hoang/project_fgw/main_action/videos/test_system4.avi'
gif_path = '/home/hoangdinhhuy/Hoang/project_fgw/main_action/gifs/test_system4.gif'
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
