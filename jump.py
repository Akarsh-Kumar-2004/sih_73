import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def mov_avg(x, k=7):
    if len(x) < 3: 
        return np.array(x, dtype=float)
    k = max(3, int(k) if int(k)%2==1 else int(k)+1) 
    pad = k//2
    xx = np.pad(x, (pad, pad), mode='edge')
    ker = np.ones(k)/k
    return np.convolve(xx, ker, mode='valid')

def lm_y(lm, idx, H, min_vis=0.5):
    if lm[idx].visibility < min_vis: 
        return None
    return float(lm[idx].y * H)

def pick_top_y(lm, H):
    cand = []
    for idx in [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    ]:
        y = lm_y(lm, idx.value, H)
        if y is not None:
            cand.append(y)
    return min(cand) if cand else None 

def pick_bottom_y(lm, H):
    cand = []
    for idx in [
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
        mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL,
        mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
    ]:
        y = lm_y(lm, idx.value, H)
        if y is not None:
            cand.append(y)
    return max(cand) if cand else None 

def pick_hip_y(lm, H):
    ys = []
    for idx in [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]:
        y = lm_y(lm, idx.value, H)
        if y is not None:
            ys.append(y)
    if not ys:
        return None
    return float(np.mean(ys))


def calculate_jump_height(video_path, known_height_cm, show_video=False, smooth_win=7):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open {video_path}")
        return None

    pose = mp_pose.Pose(min_detection_confidence=0.35, min_tracking_confidence=0.35)

    hip_y_raw = []
    body_h_px = []   
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1

        H, W = frame.shape[:2]
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            y_hip = pick_hip_y(lm, H)
            top_y = pick_top_y(lm, H)
            bot_y = pick_bottom_y(lm, H)

            if y_hip is not None:
                hip_y_raw.append(y_hip)
            else:
                hip_y_raw.append(np.nan)

            if (top_y is not None) and (bot_y is not None) and (bot_y > top_y):
                body_h_px.append(bot_y - top_y)
            else:
                body_h_px.append(np.nan)

            if show_video:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imshow("Jump Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            hip_y_raw.append(np.nan)
            body_h_px.append(np.nan)

    cap.release()
    if show_video:
        cv2.destroyAllWindows()

    hip_y_raw = np.array(hip_y_raw, dtype=float)
    body_h_px = np.array(body_h_px, dtype=float)

    # drop frames with no hip
    valid = ~np.isnan(hip_y_raw)
    if valid.sum() < 5:
        print("No sufficient pose detected.")
        return None

   
    hip_y_s = mov_avg(np.where(valid, hip_y_raw, np.interp(
        np.arange(len(hip_y_raw)),
        np.flatnonzero(valid),
        hip_y_raw[valid]
    )), k=smooth_win)

    
    valid_body = ~np.isnan(body_h_px)
    if valid_body.sum() >= 10:
        thresh = np.nanpercentile(body_h_px, 90)
        stand_idx = np.where((body_h_px >= thresh) & (~np.isnan(hip_y_s)))[0]
    else:
  
        stand_idx = np.arange(min(len(hip_y_s), 50))

    if len(stand_idx) == 0:
        stand_idx = np.arange(min(len(hip_y_s), 50))

    baseline_hip_y = float(np.mean(hip_y_s[stand_idx]))
    apex_hip_y = float(np.nanmin(hip_y_s))
    jump_px = max(0.0, baseline_hip_y - apex_hip_y)
    if len(stand_idx) > 0 and np.isfinite(body_h_px[stand_idx]).sum() > 0:
        stand_body_px = float(np.nanmean(body_h_px[stand_idx]))
        cm_per_px = known_height_cm / stand_body_px if stand_body_px > 0 else np.nan
    else:
        cm_per_px = np.nan

    jump_cm = jump_px * cm_per_px if np.isfinite(cm_per_px) else np.nan

    print(f"Video: {video_path}")
    print(f"Frames processed: {frames}")
    print(f"Baseline hip y (px): {baseline_hip_y:.1f}, Apex hip y (px): {apex_hip_y:.1f}")
    print(f"Jump height: {jump_px:.1f} px")
    if np.isfinite(jump_cm):
        print(f"Estimated jump height: {jump_cm:.2f} cm (height = {known_height_cm} cm)")
    else:
        print("Estimated jump height: N/A (could not calibrate; missing standing scale)")

   
    plt.figure(figsize=(10,5))
    plt.plot(hip_y_s, label="Hip Y (smoothed)")
    plt.axhline(baseline_hip_y, linestyle='--', label="Baseline (standing)")
    plt.axhline(apex_hip_y, linestyle='--', label="Apex (highest)")
    plt.gca().invert_yaxis()
    plt.title("Vertical Jump - Hip Trajectory")
    plt.xlabel("Frame")
    plt.ylabel("Y (pixels, inverted)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return jump_cm


if __name__ == "__main__":
    try:
        known_height_cm = float(input("Enter athlete height in cm (e.g., 170): ").strip())
    except Exception:
        known_height_cm = 170.0
        print("Couldn't parse height, defaulting to 170 cm.")

    video1 = r"videoplayback (1).mp4"
    video2 = r"videoplayback.mp4"

    print("\n--- Video 1 ---")
    calculate_jump_height(video1, known_height_cm=known_height_cm, show_video=False)

    print("\n--- Video 2 ---")
    calculate_jump_height(video2, known_height_cm=known_height_cm, show_video=False)
