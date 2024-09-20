import cv2-OpenCV불러오기
import mediapipe as mp-mediapipe를 mp라는 약자로 불러오기
import time-시간 불러오기
from flask import Flask, jsonify-flask, jsonify불러오기

mp_face_mesh = mp.solutions.face_mesh-
mp_drawing = mp.solutions.drawing_utils-

app = Flask(__name__)-

EYE_AR_THRESH = 0.2-변수를 선언하고 초기값을 0.2로 정함
EYE_AR_CONSEC_FRAMES = 3-변수를 선언하고 초기값을 3으로 정함
blink_count = 0-변수를 선언하고 초기값을 0으로 정함
blink_detected = False-변수를 선언하고 초기값을 False로 정함
frame_counter = 0-변수를 선언하고 초기값을 0으로 정함

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]-기본적으로 mediapipe에서 제공되는 얼굴 랜드마크 포인트 left eye index값
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]-기본적으로 mediapipe에서 제공되는 얼굴 랜드마크 포인트 right eye index값

def eye_aspect_ratio(eye_landmarks, landmarks):-
    A = ((landmarks[eye_landmarks[1]].x - landmarks[eye_landmarks[5]].x) ** 2 +
         (landmarks[eye_landmarks[1]].y - landmarks[eye_landmarks[5]].y) ** 2) ** 0.5-EAR공식((p2-p6)+(p3-p5)/2(p1-p4))을 python코드로 나타냄
    B = ((landmarks[eye_landmarks[2]].x - landmarks[eye_landmarks[4]].x) ** 2 +
         (landmarks[eye_landmarks[2]].y - landmarks[eye_landmarks[4]].y) ** 2) ** 0.5-EAR공식((p2-p6)+(p3-p5)/2(p1-p4))을 python코드로 나타냄
    C = ((landmarks[eye_landmarks[0]].x - landmarks[eye_landmarks[3]].x) ** 2 +
         (landmarks[eye_landmarks[0]].y - landmarks[eye_landmarks[3]].y) ** 2) ** 0.5-EAR공식((p2-p6)+(p3-p5)/2(p1-p4))을 python코드로 나타냄
    ear = (A + B) / (2.0 * C)-ear이라는 변수를 선언하고 초기값을 (A값+B값)/(2곱하기 C값)으로 정함
    return ear-

@app.route('/blink', methods=['GET'])-
def blink_detection():-
    global blink_detected-
    return jsonify({'blink_detected': blink_detected})-

def detect_blink():-
    global blink_count, frame_counter, blink_detected-
    cap = cv2.VideoCapture(0)-
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:-
        while True:-참인 경우에
            ret, frame = cap.read()-
            if not ret:-
                break-기다리기
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)-
            results = face_mesh.process(frame_rgb)-
            if results.multi_face_landmarks:-
                landmarks = results.multi_face_landmarks[0].landmark-
                left_eye_ear = eye_aspect_ratio(LEFT_EYE_IDX, landmarks)-
                right_eye_ear = eye_aspect_ratio(RIGHT_EYE_IDX, landmarks)-
                ear = (left_eye_ear + right_eye_ear) / 2.0-ear변수의 값을 (변수+변수)나누기2-ear변수 값으로 정한다.
                if ear < EYE_AR_THRESH:-변수 ear의 값이 변수 EYE_AR_THRESH보다 작다면
                    frame_counter += 1-frame counter에 1을 추가한다.
                else:-아니라면
                    if frame_counter >= EYE_AR_CONSEC_FRAMES:-만약 frame_counter보다 EYE_AR_CONSEC_FRAMES가 작다면 
                        blink_count += 1-blink count 변수에 1을 더하고 그 값을 변수에 저장한다.
                        blink_detected = True-blink detected를 True로 지정한다.
                        print(f"Blink detected! Total blinks: {blink_count}")-Blink detected! Total blinks:{blink_count변수 값}을 출력한다. 
                    frame_counter = 0-
                    blink_detected = False-
            cv2.putText(frame, f"Blinks: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)-
            cv2.imshow('Blink Detection', frame)-
            if cv2.waitKey(1) & 0xFF == ord('q'):-
                break-기다리기
    cap.release()-
    cv2.destroyAllWindows()-

if __name__ == '__main__':-
    from threading import Thread-
    flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000})-
    flask_thread.start()-
    detect_blink()-
