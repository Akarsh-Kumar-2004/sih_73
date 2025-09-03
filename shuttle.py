from ultralytics import YOLO
import cv2
import json
from tqdm import tqdm

# Load YOLO model
model = YOLO("yolov8n.pt")

video_path = "Shuttle Run.mp4"
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Load cone distance from file
with open("cone_distance.json", "r") as f:
    data = json.load(f)
cone_distance = data["cone_distance"]
print(f"✅ Loaded cone distance: {cone_distance:.2f} meters")

positions = []
frame_count = 0

# Track crossings
crossings = 0
last_zone = None   # "left" or "right"

# Process video with progress bar
with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) > 0:
            # Assume largest detected box = player
            largest = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            x1, y1, x2, y2 = largest
            cx, cy = (x1+x2)/2, (y1+y2)/2
            positions.append((frame_count, cx, cy))

            # Shuttle crossing detection (simple: left/right halves)
            width = frame.shape[1]
            if cx < width * 0.3:
                zone = "left"
            elif cx > width * 0.7:
                zone = "right"
            else:
                zone = "middle"

            if last_zone in ["left", "right"] and zone in ["left", "right"] and zone != last_zone:
                crossings += 1
                print(f"Crossing detected at frame {frame_count} ({zone})")

            if zone in ["left", "right"]:
                last_zone = zone

        pbar.update(1)

cap.release()

total_time = frame_count / fps
total_distance = crossings * cone_distance
speed = total_distance / total_time if total_time > 0 else 0

print("\n📊 Results:")
print(f"Total time: {total_time:.2f} seconds")
print(f"Total crossings: {crossings}")
print(f"Total distance: {total_distance:.2f} m")
print(f"Average speed: {speed:.2f} m/s")
