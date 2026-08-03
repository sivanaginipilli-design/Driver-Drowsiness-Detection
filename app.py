import streamlit as st
import cv2
import os
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")

# -------------------------------------------------------------
# 1. MODEL CHECK
# -------------------------------------------------------------
MODEL_FILE = "shape_predictor_68_face_landmarks.dat" # Ni exact file name ikkada rayi

if not os.path.exists(MODEL_FILE):
    st.error(f"⚠️ Model file '{MODEL_FILE}' Repo lo ledu! Exact name & path verify cheyyi.")
else:
    st.success("✅ Model File Detected!")

# -------------------------------------------------------------
# 2. WEBRTC VIDEO PROCESSOR (Cloud & Browser Friendly)
# -------------------------------------------------------------
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        # Frame ni numpy array format lo teeskuntundi
        img = frame.to_ndarray(format="bgr24")

        # -----------------------------------------------------
        # IKKADA NI DROWSINESS DETECTION LOGIC WORK AVTUNDI
        # (Example: Face Landmark Processing, EAR calculation, etc.)
        # -----------------------------------------------------

        # Screen paina Live status overlay
        cv2.putText(img, "Detection Active", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# STUN Server configuration (Browser network camera allow cheyadaniki)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.subheader("📹 Real-time Camera Feed")

# Streamlit WebRTC Component setup
webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)
