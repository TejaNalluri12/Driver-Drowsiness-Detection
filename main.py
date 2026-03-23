import cv2
import winsound

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

counter = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        # Show number of eyes detected
        cv2.putText(frame, f"Eyes: {len(eyes)}", (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if len(eyes) >= 2:
            status = "Awake"
            counter = 0
        else:
            status = "Drowsy"
            counter += 1

        cv2.putText(frame, status, (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        if counter > 5:
            cv2.putText(frame, "DROWSY ALERT!", (50,150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            winsound.Beep(1000, 500)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()