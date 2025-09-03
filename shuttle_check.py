from ultralytics import YOLO
import cv2
import numpy as np
import json

model = YOLO("yolov8n.pt")

image_path = r"shuttle_run.png"
image = cv2.imread(image_path)

results = model.predict(image_path, conf=0.5)
detections = results[0].boxes.xyxy.cpu().numpy()

# Draw YOLO detections
for i, (x1, y1, x2, y2) in enumerate(detections):
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select 2 Points", image)

        if len(points) == 2:
            cv2.line(image, points[0], points[1], (255, 0, 0), 2)
            cv2.imshow("Select 2 Points", image)

            # Calculate distance
            (x1, y1), (x2, y2) = points
            pixel_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            print(f"Pixel Distance: {pixel_dist:.2f}")

            # Convert pixel -> meters
            scale = 0.01
            real_dist = pixel_dist * scale
            print(f"Real Distance: {real_dist:.2f} meters")

            # Save to JSON for shuttle.py
            with open("cone_distance.json", "w") as f:
                json.dump({"cone_distance": real_dist}, f)

            print("✅ Saved distance to cone_distance.json")

cv2.imshow("Select 2 Points", image)
cv2.setMouseCallback("Select 2 Points", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
