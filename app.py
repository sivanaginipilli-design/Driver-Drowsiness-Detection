
import os
import threading
from pathlib import Path

import av
import cv2
import dlib
import gdown
import streamlit as st
from imutils import face_utils
from scipy.spatial import distance as dist
from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent

PREDICTOR_PATH = APP_DIR / "shape_predictor_68_face_landmarks.dat"
ALARM_PATH = APP_DIR / "alaram.mp3"

EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 10

# Google Drive model file ID
MODEL_FILE_ID = "1Ftih_JTHoNA3LCwQ2qTvRg9XlCta0nT7"


# -------------------------------
# DOWNLOAD AND LOAD MODEL
# -------------------------------

@st.cache_resource
def load_face_models():

    if not PREDICTOR_PATH.exists():

        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

        downloaded = gdown.download(
            url,
            str(PREDICTOR_PATH),
            quiet=False
        )

        if not downloaded or not PREDICTOR_PATH.exists():
            raise RuntimeError(
                "Model download failed. Check Google Drive sharing "
                "permission: Anyone with the link - Viewer."
            )

    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor(
        str(PREDICTOR_PATH)
    )

    return detector, predictor


# -------------------------------
# EYE ASPECT RATIO
# -------------------------------

def calculate_ear(eye):

    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])

    if c == 0:
        return 0.0

    return (a + b) / (2.0 * c)


# -------------------------------
# DROWSINESS DETECTION
# -------------------------------

class DrowsinessProcessor(VideoProcessorBase):

    def __init__(self):

        self.detector, self.predictor = load_face_models()

        self.left_start, self.left_end = (
            face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        )

        self.right_start, self.right_end = (
            face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        )

        self.counter = 0
        self.alarm_on = False
        self.ear_value = None

        self.lock = threading.Lock()

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        image = cv2.resize(image, (640, 480))

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.detector(gray, 0)

        current_ear = None
        alert = False

        for face in faces:

            landmarks = self.predictor(gray, face)

            points = face_utils.shape_to_np(landmarks)

            left_eye = points[
                self.left_start:self.left_end
            ]

            right_eye = points[
                self.right_start:self.right_end
            ]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            current_ear = (left_ear + right_ear) / 2.0

            # Draw eye contours
            cv2.drawContours(
                image,
                [cv2.convexHull(left_eye)],
                -1,
                (0, 255, 0),
                1
            )

            cv2.drawContours(
                image,
                [cv2.convexHull(right_eye)],
                -1,
                (0, 255, 0),
                1
            )

            # Check eye closure
            if current_ear < EAR_THRESHOLD:

                self.counter += 1

            else:

                self.counter = 0

            alert = self.counter >= CONSECUTIVE_FRAMES

            # Display EAR
            cv2.putText(
                image,
                f"EAR: {current_ear:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Drowsiness alert
            if alert:

                cv2.putText(
                    image,
                    "DROWSINESS ALERT!",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

        # No face detected
        if not faces:

            self.counter = 0

            cv2.putText(
                image,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2
            )

        with self.lock:

            self.ear_value = current_ear
            self.alarm_on = alert

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# -------------------------------
# DASHBOARD UI
# -------------------------------

st.title("🚗 Driver Drowsiness Detection")

st.markdown(
    """
    ### AI-Based Driver Safety Monitoring System

    This project detects driver drowsiness using facial landmarks
    and Eye Aspect Ratio (EAR).
    """
)

st.divider()

# Sidebar
with st.sidebar:

    st.header("⚙️ Detection Settings")

    st.metric(
        "EAR Threshold",
        EAR_THRESHOLD
    )

    st.metric(
        "Consecutive Frames",
        CONSECUTIVE_FRAMES
    )

    st.info(
        "Allow camera permission when your browser asks. "
        "Keep your face visible and ensure good lighting."
    )

# Load model
try:

    load_face_models()

except Exception as exc:

    st.error(f"Face model loading failed: {exc}")

    st.stop()


# Camera section
st.subheader("📷 Live Camera Detection")

st.write(
    "Click START to begin monitoring. "
    "Click STOP to stop the camera."
)

ctx = webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

# Detection status
if ctx.state.playing and ctx.video_processor:

    processor = ctx.video_processor

    with processor.lock:

        is_alert = processor.alarm_on
        ear = processor.ear_value

    if is_alert:

        st.error(
            "⚠️ DROWSINESS ALERT! Please take a safe break."
        )

    elif ear is not None:

        st.success(
            f"Monitoring Active | EAR: {ear:.2f}"
        )

    else:

        st.info(
            "Monitoring active. Detecting face..."
        )

else:

    st.info(
        "Camera stopped. Press START to begin detection."
    )


# -------------------------------
# ALARM AUDIO
# -------------------------------

if ALARM_PATH.exists():

    st.subheader("🔊 Alarm Sound")

    st.audio(
        str(ALARM_PATH),
        format="audio/mp3"
    )

    st.caption(
        "If an alert appears, play the alarm using the audio control "
        "if your browser does not autoplay it."
    )

else:

    st.warning(
        "alaram.mp3 not found. Add alaram.mp3 to the repository root."
    )


# -------------------------------
# PROJECT INFORMATION
# -------------------------------

st.divider()

st.subheader("📌 Project Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Technology", "OpenCV + Dlib")

with col2:
    st.metric("Detection Method", "EAR")

with col3:
    st.metric("Landmark Points", "68")

st.divider()

st.caption(
    "Educational prototype only. This system may miss drowsiness "
    "or generate false alerts. Never rely on it as the sole driver "
    "safety system."
)
