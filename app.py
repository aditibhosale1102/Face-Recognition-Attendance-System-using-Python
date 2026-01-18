from flask import Flask, render_template, request
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, date
import sqlite3

app = Flask(__name__)

# =========================
# ROUTES
# =========================

@app.route('/new', methods=['GET', 'POST'])
def new():
    if request.method == "POST":
        return render_template('index.html')
    return "Everything is okay!"

@app.route('/name', methods=['GET', 'POST'])
def name():
    if request.method == "POST":
        name1 = request.form['name1']

        cam = cv2.VideoCapture(0)

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            cv2.imshow("Press SPACE to capture | ESC to exit", frame)
            k = cv2.waitKey(1)

            if k == 27:  # ESC
                break
            elif k == 32:  # SPACE
                img_name = name1 + ".png"
                path = "Training images"
                os.makedirs(path, exist_ok=True)
                cv2.imwrite(os.path.join(path, img_name), frame)
                print(f"✅ {img_name} saved in Training images")
                break

        cam.release()
        cv2.destroyAllWindows()
        return render_template('image.html')

    return "All is not well"


# =========================
# HOW PAGE
# =========================

@app.route('/how')
def how():
    return render_template('how.html')


# =========================
# FACE RECOGNITION
# =========================

@app.route("/", methods=["GET", "POST"])
def recognize():
    if request.method == "POST":

        path = "Training images"
        images = []
        classNames = []

        if not os.path.exists(path):
            return "❌ Training images folder not found"

        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            images.append(img)
            classNames.append(os.path.splitext(file)[0])

        def findEncodings(images):
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodes = face_recognition.face_encodings(img)
                if len(encodes) > 0:
                    encodeList.append(encodes[0])
            return encodeList

        encodeListKnown = findEncodings(images)

        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
            if not success:
                break

            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

                matches = face_recognition.compare_faces(
                    encodeListKnown,
                    encodeFace,
                    tolerance=0.6
                )

                faceDis = face_recognition.face_distance(
                    encodeListKnown,
                    encodeFace
                )

                if len(faceDis) == 0:
                    continue

                matchIndex = np.argmin(faceDis)

                # ✅ FIXED INDENTATION (THIS WAS THE ERROR)
                if matches[matchIndex] and faceDis[matchIndex] < 0.60:
                    name = classNames[matchIndex].upper()
                    markAttendance(name)
                    markData(name)
                else:
                    name = "UNKNOWN"

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0,255,0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

            cv2.imshow("Punch your Attendance (ESC to exit)", img)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        return render_template("first.html")

    return render_template("main.html")


# =========================
# ATTENDANCE FUNCTIONS
# =========================

def markAttendance(name):
    with open("attendance.csv", "a") as f:
        now = datetime.now()
        f.write(f"\n{name},{now.strftime('%H:%M')}")

def markData(name):
    now = datetime.now()
    today = date.today()

    conn = sqlite3.connect("information.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS Attendance
        (NAME TEXT, Time TEXT, Date TEXT)
    """)
    conn.execute(
        "INSERT INTO Attendance VALUES (?,?,?)",
        (name, now.strftime("%H:%M"), today)
    )
    conn.commit()
    conn.close()


# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(debug=True)
