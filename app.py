import streamlit as st
import os
import urllib.request
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")
st.title("🚗 Driver Drowsiness Detection System")

# -------------------------------------------------------------
# 1. AUTOMATIC MODEL DOWNLOAD (GitIgnore Fix)
# -------------------------------------------------------------
MODEL_FILE = "shape_predictor_68_face_landmarks.dat"
# Landmark file direct download URL:
URL = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("⏳ Model file Cloud server loki download avthundi (1-2 mins pattachu)..."):
            urllib.request.urlretrieve(URL, MODEL_FILE)
        st.success("✅ Model file successfully downloaded!")
    else:
        st.success("✅ Model file detected!")
    
    # Ekkada predictor/model initialize chesko
    # import dlib
    # predictor = dlib.shape_predictor(MODEL_FILE)
    return MODEL_FILE

# Execute model download/load
model_file_path = download_and_load_model()

# -------------------------------------------------------------
# 2. CAMERA FEED (WebRTC)
# -------------------------------------------------------------
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # --- NI DROWSINESS DETECTION LOGIC IKKADA RASI PROCESS CHEYYI ---
        
        cv2.putText(img, "System Active", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.subheader("📹 Real-time Camera Feed")

webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)
