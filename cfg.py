import cv2

# ========================= TRACKING ============================
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.legacy.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create
}

# ======================== PATH TO PRE-TRAINED ====================
facial_landmarks = 'eye_blink/landmarks68/shape_predictor_68_face_landmarks.dat'
facial_haar = 'orientation/haar_cascade/haarcascade_frontalface_alt.xml'
facial_orientation_haar = 'orientation/haar_cascade/haarcascade_profileface.xml'