import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")
st.write("కెమెరా ఆన్ చేసి మీ ముఖాన్ని ఫోకస్ చేయండి.")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def get_ear(landmarks, eye_indices, img_w, img_h):
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * img_w), int(lm.y * img_h)))
    
    v1 = euclidean_distance(coords[1], coords[5])
    v2 = euclidean_distance(coords[2], coords[4])
    h = euclidean_distance(coords[0], coords[3])
    
    return (v1 + v2) / (2.0 * h)

# Built-in Streamlit Camera
img_file_buffer = st.camera_input("Take a photo / Check Drowsiness")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    h, w, _ = cv2_img.shape
    
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = get_ear(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = get_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < 0.21:
                st.error(f"⚠️ **ALERT: DROWSINESS DETECTED!** (EAR: {avg_ear:.2f})")
            else:
                st.success(f"✅ **DRIVER IS AWAKE!** (EAR: {avg_ear:.2f})")
            
            st.image(rgb_img, caption="Processed Image", use_column_width=True)
    else:
        st.warning("ముఖం స్పష్టంగా కనిపించడం లేదు. సరిగ్గా కెమెరా వైపు చూడండి.")
