import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 model (nano for speed)
model = YOLO("yolov8n.pt")

# Define 4 lane zones (adjust these coordinates for your camera view)
lane_zones = {
    "A": np.array([[50, 400], [300, 400], [300, 720], [50, 720]]),
    "B": np.array([[320, 400], [560, 400], [560, 720], [320, 720]]),
    "C": np.array([[580, 400], [820, 400], [820, 720], [580, 720]]),
    "D": np.array([[840, 400], [1100, 400], [1100, 720], [840, 720]])
}

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


lane_states = {lane: "RED" for lane in lane_zones}
current_green = "A"
lane_states[current_green] = "GREEN"


green_duration = 10
last_switch_time = time.time()

vehicle_classes = {2: "Car", 3: "Motorbike", 5: "Bus", 7: "Truck"}


def is_in_zone(center, zone):
    return cv2.pointPolygonTest(zone, center, False) >= 0


def draw_zone(frame, zone, color):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone], color)
    return cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)


def draw_traffic_light(frame, position, state, timer):
    x, y = position
    cv2.rectangle(frame, (x, y), (x + 70, y + 160), (50, 50, 50), -1)
    red_center, green_center = (x + 35, y + 45), (x + 35, y + 115)

    cv2.circle(frame, red_center, 22, (0, 0, 255) if state == "RED" else (60, 0, 0), -1)
    cv2.circle(frame, green_center, 22, (0, 255, 0) if state == "GREEN" else (0, 60, 0), -1)

    cv2.putText(frame, str(timer), (x + 20, y + 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, stream=True)

    lane_counts = {lane: 0 for lane in lane_zones}

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                for lane, zone in lane_zones.items():
                    if is_in_zone((cx, cy), zone):
                        lane_counts[lane] += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 200), 2)
                        cv2.putText(frame, model.names[cls_id], (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
                        break

    current_time = time.time()
    elapsed = current_time - last_switch_time

    if elapsed >= green_duration:
        max_lane = max(lane_counts, key=lane_counts.get)
        if max_lane != current_green:
            for lane in lane_states:
                lane_states[lane] = "RED"
            lane_states[max_lane] = "GREEN"
            current_green = max_lane
            last_switch_time = current_time
            elapsed = 0
    remaining_green = int(green_duration - elapsed)

    lane_timers = {}
    order = list(lane_zones.keys())
    current_index = order.index(current_green)

    for i, lane in enumerate(order):
        if lane == current_green:
            lane_timers[lane] = max(remaining_green, 0)
        else:

            distance = (i - current_index) % len(order)
            wait_time = remaining_green + (distance * green_duration)
            lane_timers[lane] = wait_time

    zone_colors = {"A": (0, 255, 255), "B": (255, 0, 255), "C": (0, 128, 255), "D": (255, 128, 0)}
    for lane, zone in lane_zones.items():
        frame = draw_zone(frame, zone, zone_colors[lane])

    positions = {"A": (50, 50), "B": (350, 50), "C": (650, 50), "D": (950, 50)}
    for lane, pos in positions.items():
        draw_traffic_light(frame, pos, lane_states[lane], lane_timers[lane])

    cv2.imshow("Traffic AI - 4 Lanes with Timers", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
