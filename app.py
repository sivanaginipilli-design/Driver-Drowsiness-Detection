import streamlit as st
import cv2
import av
import urllib.request
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Page Configuration
st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")
st.write("Live Camera feed start avvadaniki kinda unna **'START'** button click cheyandi.")

# Haar Cascade XML files load cheyadaniki helper function
@st.cache_resource
def load_cascades():
    face_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    eye_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
    
    if not os.path.exists("haarcascade_frontalface_default.xml"):
        urllib.request.urlretrieve(face_url, "haarcascade_frontalface_default.xml")
    if not os.path.exists("haarcascade_eye.xml"):
        urllib.request.urlretrieve(eye_url, "haarcascade_eye.xml")
        
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    return face_cascade, eye_cascade

face_cascade, eye_cascade = load_cascades()

# STUN server configuration for Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Video Processing Function
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    if img is None or img.size == 0:
        return frame

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Green box around face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Eyes detection inside face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit WebRTC Component
webrtc_streamer(
    key="drowsiness",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
