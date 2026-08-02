import cv2
import mediapipe as mp
import numpy as np
import pygame
from scipy.spatial import distance as dist

# 1. Pygame Audio Alert Setup (అలారం సౌండ్ సెటప్)
pygame.mixer.init()
try:
    pygame.mixer.music.load("alarm.wav")  # మీ అలారం సౌండ్ ఫైల్ పేరు
except:
    print(
        "⚠️ Warning: 'alarm.wav' ఫైల్ దొరకలేదు. సౌండ్ లేకుండా రన్ అవుతుంది."
    )

# 2. Eye Landmark Indices for MediaPipe
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# Eye Aspect Ratio (EAR) గెక్కించే ఫంక్షన్
def calculate_ear(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear


# 3. MediaPipe Face Mesh Initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Parameters (పరిమితులు)
EAR_THRESHOLD = (
    0.22  # EAR ఈ విలువ కంటే తగ్గితే కళ్ళు మూసుకున్నట్లు (Drowsy)
)
CONSEC_FRAMES = 20  # వరుసగా ఎన్ని ఫ్రేమ్‌లు కళ్ళు మూస్తే అలారం మోగాలి

COUNTER = 0
ALARM_ON = False

# 4. Camera Stream Initialization
cap = cv2.VideoCapture(0)

print("🚀 Driver Drowsiness Detection Start అయ్యింది. నిష్క్రమించడానికి 'q' నొక్కండి.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(" Camera access అవ్వడం లేదు.")
        break

    frame = cv2.flip(frame, 1)  # Selfie Mode (Mirroring)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face landmarks ని డిటెక్ట్ చేయడం
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array(
                [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
            )

            # కంటి ల్యాండ్‌మార్క్స్ తీయడం
            left_eye = landmarks[LEFT_EYE]
            right_eye = landmarks[RIGHT_EYE]

            # కంటి EAR లెక్కింపు
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # కంటి భాగాలను స్క్రీన్ పై గీయడం (Visual Confirmation)
            cv2.polylines(
                frame, [left_eye.astype(np.int32)], True, (0, 255, 0), 1
            )
            cv2.polylines(
                frame, [right_eye.astype(np.int32)], True, (0, 255, 0), 1
            )

            # EAR విలువ స్క్రీన్ పై చూపించడం
            cv2.putText(
                frame,
                f"EAR: {avg_ear:.2f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # నిద్రమత్తును తనిఖీ చేసే లాజిక్
            if avg_ear < EAR_THRESHOLD:
                COUNTER += 1

                if COUNTER >= CONSEC_FRAMES:
                    # నిద్రపోతున్నట్టు గుర్తిస్తే
                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT! WAKE UP!",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        3,
                    )

                    # అలారం మోగించడం
                    if not ALARM_ON:
                        ALARM_ON = True
                        try:
                            pygame.mixer.music.play(-1)  # Loop లో సౌండ్ వస్తుంది
                        except:
                            pass
            else:
                COUNTER = 0
                if ALARM_ON:
                    ALARM_ON = False
                    pygame.mixer.music.stop()  # అలారం ఆపివేయడం

    # Window స్క్రీన్ చూపించడం
    cv2.imshow("Driver Drowsiness Detection System", frame)

    # 'q' కీ నొక్కితే ఆగిపోతుంది
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
