import os
import cv2
import dlib
import gdown
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from scipy.spatial import distance as dist

DAT_FILE = "shape_predictor_68_face_landmarks.dat"

# --- 1. సగం డౌన్‌లోడ్ అయిన పాడైపోయిన ఫైల్స్ ఉంటే డిలీట్ చేసి, Fresh గా డౌన్‌లోడ్ చేయడం ---
if os.path.exists(DAT_FILE):
    # ఫైల్ సైజ్ 90MB (90,000,000 bytes) కన్నా తక్కువ ఉంటే అది సగం డౌన్‌లోడ్ అయిన ఫైల్ అని అర్థం
    if os.path.getsize(DAT_FILE) < 90000000:
        os.remove(DAT_FILE)

if not os.path.exists(DAT_FILE):
    with st.spinner("Downloading shape predictor model (~99MB)... Please wait 15-20 seconds..."):
        # Google Drive Direct File ID
        file_id = '1eT-jHlyT9hN125d0P
        gdown.download(id=file_id, output=DAT_FILE, quiet=False)

# --- 2. Dlib Detector & Predictor లోడ్ చేయడం ---
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DAT_FILE)
except Exception as err:
    st.error(f"Error loading shape predictor file: {err}")

# --- 3. Eye Aspect Ratio (EAR) గణన ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# --- 4. Streamlit WebRTC వీడియో ప్రొసెసింగ్ ---
class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape_np = np.zeros((68, 2), dtype="int")
            for i in range(0, 68):
                shape_np[i] = (shape.part(i).x, shape.part(i).y)

            leftEye = shape_np[lStart:lEnd]
            rightEye = shape_np[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                self.counter += 1
                if self.counter >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.counter = 0

            cv2.putText(img, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return img

# --- 5. UI డిజైన్ ---
st.title("🚗 Driver Drowsiness Detection System")
st.write("Click **START** below to enable your camera and check for drowsiness in real-time.")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="drowsiness-detection",
    video_transformer_factory=DrowsinessTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
