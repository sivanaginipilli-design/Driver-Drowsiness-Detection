import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode
)

# 1. Page Configuration
st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗")
st.title("🚗 Driver Drowsiness Detection System")
st.write("Click **START** below to enable your camera and check for drowsiness in real-time.")

# 2. MediaPipe FaceMesh Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3. Constants & Thresholds
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices, w, h):
    pts = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_indices])
    
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    
    ear = (A + B) / (2.0 * C)
    return ear, pts

# 4. Streamlit WebRTC Video Processing Class
class DrowsinessTransformer(VideoProcessorBase):
    def __init__(self):
        self.counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear, left_pts = calculate_ear(face_landmarks.landmark, LEFT_EYE, w, h)
                right_ear, right_pts = calculate_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
                
                ear = (left_ear + right_ear) / 2.0

                cv2.polylines(img, [left_pts], True, (0, 255, 0), 1)
                cv2.polylines(img, [right_pts], True, (0, 255, 0), 1)

                if ear < EYE_AR_THRESH:
                    self.counter += 1
                    if self.counter >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(img, "DROWSINESS ALERT!", (30, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                else:
                    self.counter = 0

                cv2.putText(img, f"EAR: {ear:.2f}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 5. STUN Server Setup for WebRTC
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}
        ]
    }
)

# 6. WebRTC Streamer
webrtc_streamer(
    key="drowsiness-detection-final",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DrowsinessTransformer,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={
        "video":{
            "width": {"ideal": 640},
            "height":{"ideal": 480},
            "frameRate": {"ideal":15}
        },
        "audio": False,
    },
    async_processing=True,
)
