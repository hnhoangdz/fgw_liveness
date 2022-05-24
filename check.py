import cv2
import dlib
from imutils import face_utils
predictor = dlib.shape_predictor('eye_blink/landmarks68/shape_predictor_68_face_landmarks.dat')
img = cv2.imread('b.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
rects = [[0, 0, gray.shape[1], gray.shape[0]]]
for rect in rects:
    rect = dlib.rectangle(rect[0], rect[1], rect[2], rect[3])
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    for (i, (x, y)) in enumerate(leftEye):
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(img, str(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    for (i, (x, y)) in enumerate(rightEye):
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
cv2.imshow('img', img)
cv2.waitKey(0)