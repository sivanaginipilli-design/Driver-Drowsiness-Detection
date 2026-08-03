import os
import urllib.request
import cv2
import dlib
import numpy as np
import streamlit as st
from scipy.spatial import distance as dist
from imutils import face_utils

# -------------------------------------------------------------
# 1. HUGGING FACE NUNDI MODEL DOWNLOAD CHESE LOGIC
# -------------------------------------------------------------
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
MODEL_URL = "https://huggingface.co/datasets/sivanagini-p/dlib-model/resolve/main/shape_predictor_68_face_landmarks.dat"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading shape predictor model from Hugging Face... Please wait..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return dlib.get_frontal_face_detector(), dlib.shape_predictor(MODEL_PATH)

# -------------------------------------------------------------
# 2. EYE ASPECT RATIO (EAR) CALCULATE CHESE FUNCTION
# -------------------------------------------------------------
def eye_aspect_ratio(eye):
    # Vertical landmarks distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Horizontal landmark distance
    C = dist.euclidean(eye[0], eye[3])
    # EAR calculation
    ear = (A + B) / (2.0 * C)
    return ear

# -------------------------------------------------------------
# 3. STREAMLIT APP UI & LOGIC
# -------------------------------------------------------------
st.title("🚗 Driver Drowsiness Detection System")
st.text("Real-time webcam feed through OpenCV & Dlib")

# Load model and detectors
detector, predictor = load_model()

# Eye landmarks indices (68-point model)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
COUNTER = 0

run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera access cheyaleka potunnam!")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw contours around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check drowsiness threshold
        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Convert BGR image to RGB for Streamlit display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

else:
    camera.release()
