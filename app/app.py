from flask import Flask, render_template
import cv2
import mediapipe as mp
import numpy as np
from requests import Response
app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle
    return angle


cap = cv2.VideoCapture(0)


def generate_frames():

    # Curl Counter variable
    curl_counter = 0
    curl_stage = ""
    # Squat Counter variable
    squat_counter = 0
    squat_stage = ""
    # Overhead Press Counter variable
    overhead_press_counter = 0
    overhead_press_stage = ""
    # Initiate Pose Model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # make detections
            results = pose.process(image)
            image.flags.writeable = True

            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Extract Landmarks
            try:
                # Getting Coordinates
                landmark = results.pose_landmarks.landmark
                shoulder = [landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                rshoulder = [landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                rhip = [landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                relbow = [landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                hip = [landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                # Calculate Angle
                curl_angle = calculate_angle(shoulder, elbow, wrist)
                squat_angle = calculate_angle(hip, knee, ankle)
                overhead_press_angle = calculate_angle(rhip, rshoulder, relbow)
                # Visualize
                cv2.putText(image, str(curl_angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                            255, 255), 2, cv2.LINE_AA
                            )
                cv2.putText(image, str(squat_angle),
                            tuple(np.multiply(knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                            255, 255), 2, cv2.LINE_AA
                            )
                cv2.putText(image, str(overhead_press_angle),
                            tuple(np.multiply(rshoulder, [
                                  640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                            255, 255), 2, cv2.LINE_AA
                            )
                # Curl counter logic
                if curl_angle > 150:
                    curl_stage = "down"
                if curl_angle < 30 and curl_stage == 'down':
                    curl_stage = "up"
                    curl_counter += 1

                # Squat counter logic
                if squat_angle > 160:
                    squat_stage = "up"
                if squat_angle < 90 and squat_stage == 'up':
                    squat_stage = "down"
                    squat_counter += 1

            # Overhead press counter logic
                if overhead_press_angle < 60:
                    overhead_press_stage = "down"
                if overhead_press_angle > 120 and overhead_press_stage == 'down':
                    overhead_press_stage = "up"
                    overhead_press_counter += 1

            except:
                pass
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (525, 73), (245, 117, 16), -1)

            # Rep data
            # for curls
            cv2.putText(image, 'Curls', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(curl_counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # for squats
            cv2.putText(image, 'Squats', (180, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(squat_counter),
                        (180, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # for overheadpress
            cv2.putText(image, 'Overhead Press', (320, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(overhead_press_counter),
                        (320, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # Stage data
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, curl_stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (240, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, squat_stage,
                        (240, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (460, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, overhead_press_stage,
                        (440, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Draw pose landmark
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Pose Model Detections', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                break
        return render_template("index.html")


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/redirect')
def redirected():
    return "You were redirected"


if __name__ == '__main__':
    app.run(debug=True)
