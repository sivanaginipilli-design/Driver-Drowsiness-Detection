import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Page Configuration
st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Driver Drowsiness Detection System")
st.caption("Live WebCam లో కళ్ళు మూసుకుంటే డిటెక్ట్ చేస్తుంది.")

# MediaPipe Face Mesh Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Left and Right Eye Landmarks in MediaPipe
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices, width, height):
    # Get Eye Coordinates
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append(np.array([lm.x * width, lm.y * height]))
    
    # Calculate Distances
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h = np.linalg.norm(coords[0] - coords[3])
    
    # Eye Aspect Ratio (EAR)
    ear = (v1 + v2) / (2.0 * h)
    return ear

class DrowsinessVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Calculate EAR for both eyes
                left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
                right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # EAR Threshold (0.22 కంటే తక్కువ ఉంటే కళ్ళు మూసుకున్నట్లు)
                if avg_ear < 0.22:
                    cv2.putText(
                        img, 
                        "DROWSINESS ALERT!", 
                        (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, 
                        (0, 0, 255), 
                        3
                    )
                else:
                    cv2.putText(
                        img, 
                        "EYES OPEN", 
                        (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, 
                        (0, 255, 0), 
                        2
                    )

        return frame.from_ndarray(img, format="bgr24")

# STUN Servers (Black Screen నివారించడానికి అతి ముఖ్యమైనది)
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ]
})

# Streamlit WebRTC Component
webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=DrowsinessVideoProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
        "audio": False
    },
    async_processing=True,
)
