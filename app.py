import streamlit as st
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Page Configuration
st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")
st.write("Live Camera feed start avvadaniki kinda unna **'START'** button click cheyandi.")

# Haar Cascade / Face & Eye Detection models load cheyadam
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# STUN servers ni use chesthe network/black screen issues raavu
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Video Processing Class (Black screen raakunda frame by frame handle chestundi)
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # AV frame ni OpenCV image (numpy array) loki convert cheyadam
    img = frame.to_ndarray(format="bgr24")
    
    # Safety Check: Frame blank ga unte original pampinchali
    if img is None or img.size == 0:
        return frame

    # Grayscale conversion for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Face Detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Face chuttu green box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Region of Interest (ROI) for eyes inside face (Indentation corrected)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

    # Processed frame ni malli WebRTC ki pampinchadam
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC Streamer
webrtc_streamer(
    key="drowsiness",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
