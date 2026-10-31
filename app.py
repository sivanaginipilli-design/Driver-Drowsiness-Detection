import os
import threading
from pathlib import Path

import av
import cv2
import dlib
import gdown
import numpy as np
import streamlit as st
from imutils import face_utils
from scipy.spatial import distance as dist
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

APP_DIR = Path(__file__).resolve().parent
PREDICTOR_PATH = APP_DIR / "shape_predictor_68_face_landmarks.dat"
ALARM_PATH = APP_DIR / "alaram.mp3"

EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 10


def calculate_ear(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    return (a + b) / (2.0 * c) if c else 0.0


@st.cache_resource
def load_face_models():
    # Download the 68-point landmark model from the user's public Google Drive file.
    if not PREDICTOR_PATH.exists():
        file_id = "1Ftih_JTHoNA3LCwQ2qTvRg9XlCta0nT7"
        url = f"https://drive.google.com/uc?id={file_id}"
        downloaded = gdown.download(url, str(PREDICTOR_PATH), quiet=False)
        if not downloaded or not PREDICTOR_PATH.exists():
            raise RuntimeError(
                "Could not download the landmark model. Check that the Google Drive "
                "file is shared as Anyone with the link (Viewer)."
            )
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
    return detector, predictor


class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector, self.predictor = load_face_models()
        self.left_start, self.left_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.right_start, self.right_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.counter = 0
        self.alarm_on = False
        self.ear_value = None
        self.lock = threading.Lock()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.resize(image, (640, 480))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)

        current_ear = None
        alert = False

        for face in faces:
            landmarks = self.predictor(gray, face)
            points = face_utils.shape_to_np(landmarks)

            left_eye = points[self.left_start:self.left_end]
            right_eye = points[self.right_start:self.right_end]
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            current_ear = (left_ear + right_ear) / 2.0

            cv2.drawContours(image, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
            cv2.drawContours(image, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

            if current_ear < EAR_THRESHOLD:
                self.counter += 1
            else:
                self.counter = 0

            alert = self.counter >= CONSECUTIVE_FRAMES
            cv2.putText(
                image, f"EAR: {current_ear:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            if alert:
                cv2.putText(
                    image, "DROWSINESS ALERT!", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

        if not faces:
            self.counter = 0
            cv2.putText(
                image, "No face detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2
            )

        with self.lock:
            self.ear_value = current_ear
            self.alarm_on = alert

        return av.VideoFrame.from_ndarray(image, format="bgr24")


st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗", layout="wide")

st.title("🚗 Driver Drowsiness Detection")
st.caption("OpenCV + Dlib 68 facial landmarks + Eye Aspect Ratio (EAR)")

with st.sidebar:
    st.header("Detection settings")
    st.write(f"EAR threshold: **{EAR_THRESHOLD}**")
    st.write(f"Closed-eye frames: **{CONSECUTIVE_FRAMES}**")
    st.info(
        "Allow camera access when your browser asks. Use good lighting and keep "
        "your face visible to the camera."
    )

try:
    load_face_models()
except Exception as exc:
    st.error(f"Could not load the face detector: {exc}")
    st.stop()

st.subheader("Live camera")
st.write("Press START, allow browser camera access, and look toward the camera.")

ctx = webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.state.playing and ctx.video_processor:
    processor = ctx.video_processor
    with processor.lock:
        is_alert = processor.alarm_on
        ear = processor.ear_value

    if is_alert:
        st.error("⚠️ DROWSINESS ALERT — Please take a safe break.")
        st.warning("The original alarm file is included for manual playback below.")
    elif ear is not None:
        st.success(f"Monitoring active — EAR: {ear:.2f}")
    else:
        st.info("Monitoring active — detecting face...")
else:
    st.info("Camera is stopped. Press START to begin detection.")

if ALARM_PATH.exists():
    st.subheader("Alarm sound")
    st.audio(str(ALARM_PATH), format="audio/mp3")
    st.caption(
        "Browser autoplay may be blocked. If an alert appears, press play on the "
        "audio control. Keep this app as an assistive demo, not a substitute for "
        "safe driving or proper rest."
    )
else:
    st.caption("Optional: add `alaram.mp3` to the repository root for the alarm audio control.")

st.divider()
st.caption(
    "Safety note: This is an educational prototype. It can miss drowsiness or "
    "produce false alerts. Never use it as the sole driver-safety system."
)
