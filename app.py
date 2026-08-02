import cv2
import numpy as np
import scipy.spatial.distance as dist
import streamlit as st

# Webpage setup
st.set_page_config(
    page_title="Driver Drowsiness Detection System", layout="centered"
)
st.title("😴 Driver Drowsiness Detection System")
st.write(
    "ఈ అప్లికేషన్ కెమెరా ద్వారా మీ కళ్ళ చలనాన్ని గమనించి నిద్రమత్తును గుర్తిస్తుంది."
)


# Eye Aspect Ratio (EAR) Calculate చేయడానికి ఫంక్షన్
def calculate_ear(eye):
    # కంటి లంబ దూరాలు (Vertical distances)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # కంటి క్షితిజ సమాంతర దూరం (Horizontal distance)
    C = dist.euclidean(eye[0], eye[3])

    # EAR సూత్రం
    ear = (A + B) / (2.0 * C)
    return ear


# Control Buttons
run = st.checkbox("Start WebCam / కెమెరా ప్రారంభించు")
FRAME_WINDOW = st.image([])

# Parameters (అవసరానికి బట్టి మార్చుకోవచ్చు)
EAR_THRESHOLD = 0.25  # EAR ఈ విలువ కన్నా తగ్గితే కళ్ళు మూసుకున్నట్లు
CLOSED_FRAMES_LIMIT = 20  # వరుసగా ఎన్ని ఫ్రేమ్‌లు కళ్ళు మూస్తే అలర్ట్ రావాలి

# Face & Eye Cascade Detectors
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

camera = cv2.VideoCapture(0)
frame_counter = 0

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera access అవ్వడం లేదు!")
        break

    # OpenCV Image Processing
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    drowsy_detected = False

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)

        # కళ్ళు సరిగ్గా కనిపిస్తున్నాయా లేదా చెక్ చేయడం
        if len(eyes) == 0:
            frame_counter += 1
        else:
            frame_counter = 0

        for ex, ey, ew, eh in eyes:
            cv2.rectangle(
                roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
            )

    # డ్రైవర్ నిద్రపోతున్నాడని భావిస్తే అలర్ట్ ఇవ్వడం
    if frame_counter >= CLOSED_FRAMES_LIMIT:
        drowsy_detected = True
        cv2.putText(
            frame,
            "DROWSINESS ALERT!",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

    # Streamlit లో డిస్ప్లే చేయడానికి BGR to RGB మార్చడం
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)

    if drowsy_detected:
        st.warning("⚠️ ALERT: డ్రైవర్ నిద్రమత్తులో ఉన్నారు! హెచ్చరిక!")

else:
    camera.release()
    st.info("కెమెరా ఆఫ్ చేయబడింది. ప్రాసెస్ ప్రారంభించడానికి చెక్‌బాక్స్ పై క్లిక్ చేయండి.")
