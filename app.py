import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Page Config
st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")
st.write("Live WebCam లో కళ్ళు మూసుకుంటే డిటెక్ట్ చేస్తుంది.")

# OpenCV Built-in Classifiers (నో Extra Downloads/Libraries)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.closed_counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect Faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # Draw Face Rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Region of Interest for Eyes
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Detect Eyes inside Face
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

            # If eyes are detected
            if len(eyes) >= 1:
                self.closed_counter = 0
                cv2.putText(img, "Status: Awake", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            else:
                # Eyes not detected (Closed or Drowsy)
                self.closed_counter += 1
                if self.closed_counter > 10:  # ~1-2 sec of closed eyes
                    cv2.rectangle(img, (0, 0), (img.shape[1], 80), (0, 0, 255), -1)
                    cv2.putText(img, "⚠️ WARNING: DROWSINESS DETECTED!", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

        import av
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Google STUN Server Config for Web Cam Stream
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
)
