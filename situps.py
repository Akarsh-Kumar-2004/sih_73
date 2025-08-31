import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def situp_counter(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    state = "down"  

    # thresholds
    up_threshold = 0.15 
    down_threshold = 0.25 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

            d = distance(nose, hip)

            # Debug display
            cv2.putText(frame, f"Distance: {d:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            
            if state == "down" and d < up_threshold:
                state = "up"
            elif state == "up" and d > down_threshold:
                count += 1
                state = "down"

            cv2.putText(frame, f"Sit-ups: {count}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Sit-up Counter", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total Sit-ups: {count}")


situp_counter("D:\sih\MOST SIT UPS IN ONE MINUTE (online-video-cutter.com).mp4")
