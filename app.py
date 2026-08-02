import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Streamlit Page Config
st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")
st.write("Live Camera ద్వారా Drowsiness ని డిటెక్ట్ చేస్తుంది.")

# Mediapipe Face Mesh Initialization
mp_face_mesh = mp.solutions.face_mesh

# Eye Landmarks Indices (Left and Right Eyes)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def get_ear(landmarks, eye_indices, img_w, img_h):
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * img_w), int(lm.y * img_h)))
    
    v1 = euclidean_distance(coords[1], coords[5])
    v2 = euclidean_distance(coords[2], coords[4])
    h = euclidean_distance(coords[0], coords[3])
    
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Changed from VideoTransformerBase to VideoProcessorBase
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.EAR_THRESHOLD = 0.21
        self.CLOSED_TIME_THRESHOLD = 1.5
        self.eye_closed_start_time = None

    def recv(self, frame):
        # Using recv() instead of transform() for streamlit-webrtc compatibility
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear = get_ear(face_landmarks.landmark, LEFT_EYE, w, h)
                right_ear = get_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < self.EAR_THRESHOLD:
                    if self.eye_closed_start_time is None:
                        self.eye_closed_start_time = time.time()
                    else:
                        elapsed_time = time.time() - self.eye_closed_start_time
                        if elapsed_time >= self.CLOSED_TIME_THRESHOLD:
                            cv2.rectangle(img, (0, 0), (w, 80), (0, 0, 255), -1)
                            cv2.putText(img, "WARNING: DROWSINESS DETECTED!", (20, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                else:
                    self.eye_closed_start_time = None

                cv2.putText(img, f"EAR: {avg_ear:.2f}", (30, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        import av
        return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Changed video_transformer_factory to video_processor_factory
webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
