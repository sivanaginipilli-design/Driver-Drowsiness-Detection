import os
import cv2
import numpy as np
import urllib.request
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Haar cascade XML files auto download function
def download_cascade_if_missing(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{filename}"
        urllib.request.urlretrieve(url, filename)

download_cascade_if_missing("haarcascade_frontalface_default.xml")
download_cascade_if_missing("haarcascade_eye.xml")

# Load Classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Streamlit App UI
st.title("Driver Drowsiness Detection System")
st.write("Live Camera feed start avvadaniki kinda unna **START** button click cheyandi.")

# Session state to track drowsiness for sound trigger
if "drowsy" not in st.session_state:
    st.session_state.drowsy = False

# Video Processor Class for WebRTC
class VideoTransformer(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        drowsy_detected = False

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            # If face is found but eyes are NOT found (Drowsiness indication)
            if len(eyes) == 0:
                drowsy_detected = True
                cv2.putText(img, "DROWSINESS ALERT!", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        st.session_state.drowsy = drowsy_detected
        return frame.from_ndarray(img, format="bgr24")

# WebRTC STUN Server Config
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=VideoTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# JavaScript Alarm Trigger if Drowsy
if st.session_state.drowsy:
    alarm_html = """
    <audio autoplay>
      <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
    </audio>
    <p style="color:red; font-weight:bold; font-size:20px;">🚨 Drowsiness Alert! Wake up! 🚨</p>
    """
    st.markdown(alarm_html, unsafe_allow_html=True)
