import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

st.set_page_config(
    page_title="Driver Drowsiness Detection", page_icon="🚗", layout="centered"
)

st.title("🚗 Driver Drowsiness Detection System")
st.write(
    "ముందుగా బ్రౌజర్‌లో Camera Permission **Allow** చేసి, క్రింద START పై క్లిక్ చేయండి."
)

# STUN Server Settings (Cloud Connection సరిగ్గా అవ్వడానికి)
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


class VideoProcessor:

    def __init__(self):
        self.counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes_found = False

        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = img[y : y + h, x : x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

            if len(eyes) > 0:
                eyes_found = True
                for ex, ey, ew, eh in eyes:
                    cv2.rectangle(
                        roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                    )

        if len(faces) == 0 or not eyes_found:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= 15:
            cv2.putText(
                img,
                "DROWSINESS ALERT!",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )

        import av

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
