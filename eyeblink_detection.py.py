import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify
import time
from threading import Thread

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
blink_count = 0
blink_detected = False
frame_counter = 0

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)

start_time = time.time()

def eye_aspect_ratio(eye_landmarks, landmarks):
    A = ((landmarks[eye_landmarks[1]].x - landmarks[eye_landmarks[5]].x) ** 2 +
         (landmarks[eye_landmarks[1]].y - landmarks[eye_landmarks[5]].y) ** 2) ** 0.5
    B = ((landmarks[eye_landmarks[2]].x - landmarks[eye_landmarks[4]].x) ** 2 +
         (landmarks[eye_landmarks[2]].y - landmarks[eye_landmarks[4]].y) ** 2) ** 0.5
    C = ((landmarks[eye_landmarks[0]].x - landmarks[eye_landmarks[3]].x) ** 2 +
         (landmarks[eye_landmarks[0]].y - landmarks[eye_landmarks[3]].y) ** 2) ** 0.5
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_average_blink_rate(total_blinks, start_time):
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        average_blink_rate = (total_blinks / elapsed_time) * 60     
    else:
        average_blink_rate = 0
    return average_blink_rate

@app.route('/blink', methods=['GET'])
def blink_detection():
    global blink_detected, blink_count, start_time
    avg_blink_rate = calculate_average_blink_rate(blink_count, start_time)
    return jsonify({
        'blink_detected': blink_detected, 
        'blink_count': blink_count,
        'average_blink_rate': avg_blink_rate
})

def generate_frames():
    global blink_count, frame_counter, blink_detected, start_time
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                left_eye_ear = eye_aspect_ratio(LEFT_EYE_IDX, landmarks)
                right_eye_ear = eye_aspect_ratio(RIGHT_EYE_IDX, landmarks)
                ear = (left_eye_ear + right_eye_ear) / 2.0
                if ear < EYE_AR_THRESH:
                    frame_counter += 1
                else:
                    if frame_counter >= EYE_AR_CONSEC_FRAMES:
                        blink_count += 1
                        blink_detected = True
                        print(f"Blink detected! Total blinks: {blink_count}")
                    frame_counter = 0
                    blink_detected = False
            avg_blink_rate = calculate_average_blink_rate(blink_count, start_time)
            cv2.putText(frame, f"Blinks: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Average Blink Rate: {avg_blink_rate:.2f} per min", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'debug': False})
    flask_thread.start()
