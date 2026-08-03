import cv2
import numpy as np
import streamlit as st
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing

st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")
st.text("Upload or Capture image to detect drowsiness using MediaPipe Mesh")

# -------------------------------------------------------------
# 1. MEDIAPIPE FACE MESH SETUP
# -------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye Landmark Indices for MediaPipe Face Mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# -------------------------------------------------------------
# 2. EYE ASPECT RATIO (EAR) FUNCTION
# -------------------------------------------------------------
def calculate_ear(landmarks, eye_indices, img_w, img_h):
    # Get 2D coordinates for landmarks
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))
    
    # Vertical distances
    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    # Horizontal distance
    C = np.linalg.norm(coords[0] - coords[3])
    
    # EAR Formula
    ear = (A + B) / (2.0 * C)
    return ear, coords

EAR_THRESHOLD = 0.22

# Camera Input
img_file_buffer = st.camera_input("Take a photo to check Drowsiness")

if img_file_buffer is not None:
    # Read Image
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    h, w, _ = cv2_img.shape
    
    # Convert BGR to RGB for MediaPipe
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # Calculate EAR for both eyes
            left_ear, left_coords = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear, right_coords = calculate_ear(landmarks, RIGHT_EYE, w, h)
            
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Draw contours on eyes
            cv2.polylines(cv2_img, [np.array(left_coords, dtype=np.int32)], True, (0, 255, 0), 1)
            cv2.polylines(cv2_img, [np.array(right_coords, dtype=np.int32)], True, (0, 255, 0), 1)

            # Check Drowsiness
            if avg_ear < EAR_THRESHOLD:
                cv2.putText(cv2_img, "DROWSINESS ALERT!", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                st.error(f"⚠️ ALERT! Drowsiness Detected! EAR: {avg_ear:.2f}")
            else:
                st.success(f"✅ Driver Active. EAR: {avg_ear:.2f}")

            cv2.putText(cv2_img, f"EAR: {avg_ear:.2f}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Convert back to RGB for Streamlit Display
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        st.image(cv2_img, caption="Detection Result", use_container_width=True)
    else:
        st.warning("Face detect avvaledu! Please face camera properly.")
