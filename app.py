import os
import cv2
import urllib.request
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# 1. Download required Cascade files if they don't exist
def download_cascade_if_missing(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{filename}"
        urllib.request.urlretrieve(url, filename)

download_cascade_if_missing("haarcascade_frontalface_default.xml")
download_cascade_if_missing("haarcascade_eye.xml")

# Load OpenCV Classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

st.set_page_config(page_title="Drowsiness Detection", layout="centered")
st.title("Driver Drowsiness Detection System 🚗💤")
st.write("Live Camera feed start avvadaniki **START** button click cheyandi.")

# 2. WebRTC Video Processing Frame-by-Frame
class DrowsinessDetector(VideoProcessorBase):
    def __init__(self):
        self.drowsy_frames = 0  # Counter for consecutive frames without eyes

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Check if eyes are open or closed
            if len(eyes) == 0:
                self.drowsy_frames += 1
            else:
                self.drowsy_frames = 0
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Trigger alert on frame if eyes closed for > 5 continuous frames
            if self.drowsy_frames >= 5:
                cv2.putText(img, "ALERT: DROWSINESS DETECTED!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        return frame.from_ndarray(img, format="bgr24")

# 3. WebRTC Stream Setup with STUN server for cloud deployment
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="drowsiness-detection-stream",
    video_processor_factory=DrowsinessDetector,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
)
