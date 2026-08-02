import cv2
import streamlit as st

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("🚗 Driver Drowsiness Detection System")

run = st.checkbox("Start Camera Feed")
FRAME_WINDOW = st.image([])

if run:
    # 0 ante default laptop camera
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error(
                "Camera access avvaledu! Index 1 try cheyyandi or camera check cheyyandi."
            )
            break

        # Color BGR to RGB conversion for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------------------------------------------------------------
        # MEERU DROWSINESS DETECTION LOGIC IKKADA ADD CHEYYANDI
        # -------------------------------------------------------------

        FRAME_WINDOW.image(frame_rgb)

    cap.release()
else:
    st.info("Camera off lo undi. Start cheyadaniki checkbox click cheyandi.")
