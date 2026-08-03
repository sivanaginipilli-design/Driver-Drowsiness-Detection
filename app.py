import cv2
import numpy as np
import streamlit as st
import mediapipe.python.solutions.face_mesh as mp_face_mesh

# Page Title
st.title("🚗 Live Driver Drowsiness Detection System")
st.write("Take a picture or turn on camera to detect drowsiness using MediaPipe Mesh.")

# 1. Initialize MediaPipe Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Live Camera Input
img_file_buffer = st.camera_input("Take a photo / Live Feed")

if img_file_buffer is not None:
    # Convert image buffer to OpenCV format
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    # Process face mesh
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
        st.success("Face Detected Successfully! Processing status...")
        
        # Draw Mesh or Add EAR (Eye Aspect Ratio) logic here
        
    else:
        st.warning("No face detected! Please face the camera properly.")
