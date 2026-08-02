import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import time

# Initialize Pygame Mixer for Alert Sound
pygame.mixer.init()
pygame.mixer.music.load("alaram.mp3")  # Ensure 'alarm.wav' is in the same folder

# Eye Aspect Ratio (EAR) Calculation Function
def calculate_EAR(eye):
    # Vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Horizontal distance
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold Values
EAR_THRESHOLD = 0.25  # EAR value drops below this when eyes are closed
CONSECUTIVE_FRAMES = 10  # Number of frames eyes must stay closed to trigger alert

COUNTER = 0
ALARM_ON = False

# Load Dlib face detector and shape predictor
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Get indices for left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start Webcam
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and convert to grayscale for faster processing
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray, 0)

    for face in faces:
        # Determine facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate EAR for both eyes
        leftEAR = calculate_EAR(leftEye)
        rightEAR = calculate_EAR(rightEye)

        # Average EAR
        ear = (leftEAR + rightEAR) / 2.0

        # Draw contour on eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check EAR threshold
        if ear < EAR_THRESHOLD:
            COUNTER += 1

            if COUNTER >= CONSECUTIVE_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    pygame.mixer.music.play(-1)  # Play alarm in loop

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            if ALARM_ON:
                ALARM_ON = False
                pygame.mixer.music.stop()

        # Display EAR score on frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show Output Frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Press 'q' to quit application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()