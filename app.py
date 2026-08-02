import time
import cv2

# కెమెరాను ఇనిషియలైజ్ చేయడం (0 అంటే డీఫాల్ట్ వెబ్‌క్యామ్)
cap = cv2.VideoCapture(0)

# కెమెరా సరిగ్గా ఓపెన్ అయిందో లేదో చెక్ చేయడం
if not cap.isOpened():
    print("Error: వెబ్‌క్యామ్ ఓపెన్ కాలేదు! దయచేసి వేరే యాప్స్ క్లోజ్ చేసి మళ్లీ ట్రై చేయండి.")
    exit()

print("🚀 Drowsiness Detection రన్ అవుతోంది... ఆపడానికి 'q' ప్రెస్ చేయండి.")

# వేరియబుల్స్
counter = 0
CLOSED_LIMIT = 20  # వరుసగా ఇన్ని ఫ్రేమ్‌లు కళ్ళు కనిపించకపోతే అలర్ట్ వస్తుంది

while True:
    ret, frame = cap.read()

    # ఒకవేళ ఫ్రేమ్ సరిగ్గా రీడ్ కాకపోతే (Black screen లేదా Disconnected)
    if not ret or frame is None:
        print("Warning: కెమెరా నుండి ఫ్రేమ్ అందడం లేదు...")
        time.sleep(0.1)
        continue

    # సెల్ఫీ మోడ్ లాగా ఫ్రేమ్‌ని రివర్స్ చేయడం (Mirror image)
    frame = cv2.flip(frame, 1)

    # గ్రే-స్కేల్ (Black & White) లోకి మార్చడం (ප්‍රോසెస్సింగ్ వేగంగా ఉండటానికి)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ఓపెన్‌సివి బేసిక్ ఫేస్ & ఐ డిటెక్షన్ క్యాస్కేడ్స్ లోడ్ చేయడం
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_found = False

    for x, y, w, h in faces:
        # ముఖం చుట్టూ రెక్టాంగుల్ గీయడం
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        if len(eyes) > 0:
            eyes_found = True
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(
                    roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                )

    # కళ్ళు కనిపించకపోతే లేదా మూసుకుంటే కౌంటర్ పెరుగుతుంది
    if len(faces) == 0 or not eyes_found:
        counter += 1
    else:
        counter = 0  # కళ్ళు తెరిచి ఉంటే కౌంటర్ రీసెట్ అవుతుంది

    # కౌంటర్ లిమిట్ దాటితే అలర్ట్ చూపించడం
    if counter >= CLOSED_LIMIT:
        cv2.putText(
            frame,
            "ALERT: DROWSY / SLEEPY!",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3,
        )

    # లైవ్ ఫ్రీమ్‌ని డిస్ప్లే చేయడం
    cv2.imshow("Driver Drowsiness Detection", frame)

    # కీబోర్డ్ లో 'q' నొక్కితే ప్రోగ్రామ్ ఆగిపోతుంది
ools
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# రిసోర్సెస్ ని క్లియర్ చేయడం
cap.release()
cv2.destroyAllWindows()
