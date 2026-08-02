import cv2
import dlib
import numpy as np
import streamlit as st
from scipy.spatial import distance as dist
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# Page title and setup
st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")
st.title("🚗 Driver Drowsiness Detection System")
st.write("Click **START** below to enable your camera and check for drowsiness in real-time.")

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Eye landmarks indices for 68-point model
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

# Load Face Detector and Predictor
@st.cache_resource
def load_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor

try:
    detector, predictor = load_models()
except Exception as e:
    st.error("Error loading shape predictor file. Make sure 'shape_predictor_68_face_landmarks.dat' is available.")

class DrowsinessDetector(VideoProcessorBase):
    def __init__(self):
        self.counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape_np = np.zeros((68, 2), dtype="int")
            for i in range(0, 68):
                shape_np[i] = (shape.part(i).x, shape.part(i).y)

            leftEye = shape_np[LEFT_EYE]
            rightEye = shape_np[RIGHT_EYE]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eyes contour
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                self.counter += 1
                if self.counter >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.counter = 0

            cv2.putText(img, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame.from_ndarray(img, format="bgr24")

# Start Webcam Stream
webrtc_streamer(key="drowsiness-detection", video_processor_factory=DrowsinessDetector)
