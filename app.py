import os
import urllib.request
import cv2
import dlib
import numpy as np
import streamlit as st
from scipy.spatial import distance as dist
from imutils import face_utils

st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")
st.text("Webcam Image-based Detection System")

# -------------------------------------------------------------
# 1. HUGGING FACE NUNDI MODEL DOWNLOAD LOGIC
# -------------------------------------------------------------
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
# Direct resolve URL with follow redirects
MODEL_URL = "https://huggingface.co/datasets/sivanagini-p/dlib-model/resolve/main/shape_predictor_68_face_landmarks.dat?download=true"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading shape predictor model... Please wait..."):
            req = urllib.request.Request(
                MODEL_URL, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
                
    return dlib.get_frontal_face_detector(), dlib.shape_predictor(MODEL_PATH)

# -------------------------------------------------------------
# 2. EYE ASPECT RATIO (EAR) FUNCTION
# -------------------------------------------------------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

try:
    detector, predictor = load_model()
except Exception as e:
    st.error(f"Model Load Cheyadam lo Error vachindi: {e}")
    st.stop()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EAR_THRESHOLD = 0.25

# Streamlit Camera Input (Works seamlessly on Browser/Cloud)
img_file_buffer = st.camera_input("Take a photo to check Drowsiness")

if img_file_buffer is not None:
    # Convert image buffer to OpenCV format
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) == 0:
        st.warning("Face detect avvaledu! Memory clear ga unte malli try cheyandi.")
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(cv2_img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(cv2_img, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check Drowsiness
        if ear < EAR_THRESHOLD:
            cv2.putText(cv2_img, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            st.error(f"⚠️ ALERT! Drowsiness Detected! EAR: {ear:.2f}")
        else:
            st.success(f"✅ Driver Active. EAR: {ear:.2f}")

        cv2.putText(cv2_img, f"EAR: {ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Convert to RGB for Streamlit Display
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    st.image(cv2_img, caption="Processed Image", use_container_width=True)
