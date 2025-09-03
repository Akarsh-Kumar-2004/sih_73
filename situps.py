import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def calculate_angle(a, b, c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
cap = cv2.VideoCapture("MOST SIT UPS IN ONE MINUTE (online-video-cutter.com).mp4")  

counter = 0
stage = None 

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            landmarks = results.pose_landmarks.landmark
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Take hip center as reference
            hip_center = [(left_hip[0] + right_hip[0]) / 2,
                          (left_hip[1] + right_hip[1]) / 2]

            # Reference vertical point
            vertical_point = [hip_center[0], hip_center[1] - 0.3]

            # Calculate angle (nose – hip – vertical ref)
            angle = calculate_angle(nose, hip_center, vertical_point)

            # Sit-up logic
            if angle > 32:  # going up
                stage = "up"
            if angle < 15 and stage == "up":  # came back down
                stage = "down"
                counter += 1

            # Display data
            cv2.putText(image, f'Angle: {int(angle)}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'Count: {counter}', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        except:
            pass

        # Show
        cv2.imshow('Sit-up Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
