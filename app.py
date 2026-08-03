import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing

st.title("🚗 Live Driver Drowsiness Detection")
st.write("లైవ్ వెబ్‌క్యామ్ ద్వారా కళ్ళు మూసి ఉన్నాయో లేదో చెక్ చేయండి.")

# Initialize Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Video Processing Function
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert WebRTC frame to OpenCV array
    img = frame.to_ndarray(format="bgr24")
    
    # Convert to RGB for MediaPipe
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the frame for face landmarks
    results = face_mesh.process(rgb_img)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw Face Mesh on the face
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
            
            # ఇక్కడ మీరు మీ కళ్ళ (EAR) క్యాలిక్యులేషన్ లాజిక్ రాసుకోవచ్చు
            # ఉదాహరణకు: కళ్ళు మూసుకుంటే Drowsiness Alert ఇవ్వడం
            
    # Return the processed frame to display on screen
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start WebRTC Streamer
webrtc_streamer(
    key="drowsiness-detection",
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
