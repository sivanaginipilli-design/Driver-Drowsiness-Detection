import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode
)

# 1. Page Title
st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗")
st.title("🚗 Driver Drowsiness Detection System")

# 2. Fast OpenCV Haar Cascades for Eyes Detection (No MediaPipe Heavy Load)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 3. Video Processing Class
class DrowsinessTransformer(VideoProcessorBase):
    def __init__(self):
        self.counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Face Detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # Eye Detection inside Face
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

            # Drowsiness Logic (Eyes closed / not detected)
            if len(eyes) == 0:
                self.counter += 1
                if self.counter >= 8:  # 8 continuous frames with no eyes visible
                    cv2.putText(img, "DROWSINESS ALERT!", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                self.counter = 0
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. Fast WebRTC Connection Config
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="fast-drowsiness-v3",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DrowsinessTransformer,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={
        "video": {"width": {"ideal": 480}, "height": {"ideal": 360}, "frameRate": {"ideal": 15}},
        "audio": False
    },
    async_processing=True,
)
