import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Driver Drowsiness Detection", page_icon="🚗", layout="centered"
)

st.title("🚗 Driver Drowsiness Detection System")
st.write(
    "ఈ అప్లికేషన్ వెబ్‌క్యామ్ ద్వారా మీ కళ్ళను గమనించి నిద్రమత్తును గుర్తిస్తుంది."
)

# Cascade Classifiers Load చేయడం
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# UI Controls
run = st.checkbox("Start Camera / కెమెరా ప్రారంభించు")
FRAME_WINDOW = st.image([])
alert_placeholder = st.empty()

counter = 0
CLOSED_LIMIT = 15  # కళ్ళు మూసుకున్న ఫ్రేమ్‌ల పరిమితి

if run:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error(
                "కెమెరా యాక్సెస్ కాలేదు. దయచేసి వెబ్‌క్యామ్ అనుమతులు చెక్ చేయండి."
            )
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes_found = False

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = frame[y : y + h, x : x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

            if len(eyes) > 0:
                eyes_found = True
                for ex, ey, ew, eh in eyes:
                    cv2.rectangle(
                        roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                    )

        # నిద్రమత్తు గుర్తింపు లాజిక్
        if len(faces) == 0 or not eyes_found:
            counter += 1
        else:
            counter = 0

        if counter >= CLOSED_LIMIT:
            cv2.putText(
                frame,
                "DROWSINESS ALERT!",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )
            alert_placeholder.error(
                "🚨 ALERT: డ్రైవర్ నిద్రమత్తులో ఉన్నారు! హెచ్చరిక!"
            )
        else:
            alert_placeholder.empty()

        # OpenCV Image ని Streamlit లో డిస్ప్లే చేయడం
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

    cap.release()
else:
    st.info("కెమెరా ఆన్ చేయడానికి పైన ఉన్న 'Start Camera' బాక్స్‌పై క్లిక్ చేయండి.")
