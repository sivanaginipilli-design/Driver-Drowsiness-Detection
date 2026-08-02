import cv2
import mediapipe as mp
import numpy as np
import scipy.spatial.distance as dist
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 Driver Drowsiness Detection System")
st.markdown("Real-time drowsiness detection using OpenCV & MediaPipe")

# Eye Landmark Indices for MediaPipe Face Mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def calculate_ear(eye_landmarks):
    # Calculate Eye Aspect Ratio (EAR)
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Sidebar Parameters
st.sidebar.header("⚙️ Configuration Settings")
EAR_THRESHOLD = st.sidebar.slider(
    "EAR Threshold", min_value=0.15, max_value=0.35, value=0.22, step=0.01
)
CONSEC_FRAMES = st.sidebar.slider(
    "Alert Frame Count", min_value=10, max_value=50, value=20, step=5
)

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Control State
run = st.checkbox("Start Webcam / కెమెరా ప్లే చేయి")
FRAME_WINDOW = st.image([])
alert_placeholder = st.empty()

counter = 0

if run:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Camera access failed.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        drowsy = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmark coordinates
                landmarks = np.array(
                    [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
                )

                left_eye = landmarks[LEFT_EYE]
                right_eye = landmarks[RIGHT_EYE]

                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)

                avg_ear = (left_ear + right_ear) / 2.0

                # Display EAR Value on Frame
                cv2.putText(
                    frame,
                    f"EAR: {avg_ear:.2f}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                # Check Drowsiness
                if avg_ear < EAR_THRESHOLD:
                    counter += 1
                    if counter >= CONSEC_FRAMES:
                        drowsy = True
                        cv2.putText(
                            frame,
                            "DROWSINESS ALERT!",
                            (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            3,
                        )
                else:
                    counter = 0

        # Convert back to RGB for Streamlit rendering
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        if drowsy:
            alert_placeholder.error("🚨 ALERT: Driver is feeling sleepy!")
        else:
            alert_placeholder.empty()

    cap.release()
else:
    st.info("Check the box above to start the webcam monitoring.")
