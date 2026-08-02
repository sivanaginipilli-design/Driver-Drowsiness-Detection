import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from scipy.spatial import distance as dist
import av

# --- 1. MediaPipe Face Mesh Init --
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 2. Eye Aspect Ratio (EAR) ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15

# --- 3. Streamlit WebRTC Video Processing ---
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
                landmarks = face_landmarks.landmark
                
                left_eye = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in LEFT_EYE_IDXS])
                right_eye = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in RIGHT_EYE_IDXS])

                leftEAR = eye_aspect_ratio(left_eye)
                rightEAR = eye_aspect_ratio(right_eye)
                ear = (leftEAR + rightEAR) / 2.0

                cv2.polylines(img, [left_eye], True, (0, 255, 0), 1)
                cv2.polylines(img, [right_eye], True, (0, 255, 0), 1)

                if ear < EYE_AR_THRESH:
                    self.counter += 1
                    if self.counter >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(img, "DROWSINESS ALERT!", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                else:
                    self.counter = 0

                cv2.putText(img, f"EAR: {ear:.2f}", (w - 150, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. UI Design ---
st.title("🚗 Driver Drowsiness Detection System")
st.write("Click **START** below to enable your camera and check for drowsiness in real-time.")

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
}

webrtc_streamer(
    key="drowsiness",
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    video_processor_factory=DrowsinessTransformer,
    async_processing=True,
)
