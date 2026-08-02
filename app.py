import cv2
import streamlit as st

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

st.title("🚗 Driver Drowsiness Detection System")
st.write("Camera start cheyadaniki kindha unna checkbox ni select cheyandi.")

# UI Checkbox
run_camera = st.checkbox("Start Camera Feed")
FRAME_WINDOW = st.image([])

if run_camera:
    # Camera initialize (0 ante integrated laptop camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera access cheyadam kudaratledhu. Device index check cheyandi.")

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera nundi frame capture avvatledhu.")
            break

        # Color Conversion (OpenCV BGR ni Streamlit RGB ki marustham)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------------------------------------------------------------
        # MEERU DRIVER DROWSINESS DETECTION MODEL LOGIC IKKADA ADDEYANDI
        # -------------------------------------------------------------

        # Process aina frame ni screen pyna chupisthundhi
        FRAME_WINDOW.image(frame_rgb)

    cap.release()
else:
    st.info("Camera off lo undhi. Start cheyadaniki checkbox click cheyandi.")
