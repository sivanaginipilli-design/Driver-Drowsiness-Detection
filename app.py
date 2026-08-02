import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Driver Drowsiness Detection", page_icon="🚗", layout="centered"
)

st.title("🚗 Driver Drowsiness Detection System")
st.write(
    "క్రింది కెమెరా ద్వారా ఫోటో క్యాప్చర్ చేసి డ్రైవర్ నిద్రమత్తును తనిఖీ చేయండి."
)

# Haar Cascade Classifiers
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Streamlit Native Camera Input (ఏ ఎర్రర్స్ రావు)
img_file_buffer = st.camera_input("Take a photo to check drowsiness")

if img_file_buffer is not None:
    # Image ని OpenCV format లోకి మార్చడం
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(
        np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
    )

    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_found = False

    for x, y, w, h in faces:
        cv2.rectangle(cv2_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = cv2_img[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        if len(eyes) > 0:
            eyes_found = True
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(
                    roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                )

    # Result చూపించడం
    if len(faces) == 0:
        st.warning("⚠️ ముఖం సరిగ్గా కనిపించడం లేదు. కెమెరా వైపు చూడండి.")
    elif not eyes_found:
        st.error(
            "🚨 **DROWSINESS ALERT!** కళ్ళు మూసుకుని ఉన్నాయి లేదా నిద్రమత్తులో ఉన్నారు!"
        )
    else:
        st.success("✅ **SAFE!** కళ్ళు తెరిచి ఉన్నాయి. డ్రైవర్ మెలకువగానే ఉన్నారు.")
