import cv2
import numpy as np
from ultralytics import YOLO
import time
import cvzone


model = YOLO("yolov8n.pt")


lane_A_zone = np.array([[100, 400], [600, 400], [600, 720], [100, 720]])
lane_B_zone = np.array([[700, 400], [1200, 400], [1200, 720], [700, 720]])


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


lane_states = {"A": "RED", "B": "GREEN"}
last_switch_time = {"A": time.time(), "B": time.time()}
green_duration = 10
red_duration = 5


vehicle_classes = {2: "Car", 3: "Motorbike", 5: "Bus", 7: "Truck"}


def is_in_zone(center, zone):

    return cv2.pointPolygonTest(zone, center, False) >= 0


def draw_traffic_light(frame, position, state):

    x, y = position
    cv2.rectangle(frame, (x, y), (x + 80, y + 180), (50, 50, 50), -1)
    red_center, green_center = (x + 40, y + 45), (x + 40, y + 135)

    cv2.circle(frame, red_center, 25,
               (0, 0, 255) if state == "RED" else (60, 0, 0), -1)
    cv2.circle(frame, green_center, 25,
               (0, 255, 0) if state == "GREEN" else (0, 60, 0), -1)


while True:
    success, frame = cap.read()
    if not success:
        break

    # Object detection
    results = model(frame, stream=True)

    lane_counts = {"A": 0, "B": 0}
    lane_class_counts = {
        "A": {name: 0 for name in vehicle_classes.values()},
        "B": {name: 0 for name in vehicle_classes.values()},
    }

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2


                if is_in_zone((cx, cy), lane_A_zone):
                    lane_counts["A"] += 1
                    lane_class_counts["A"][vehicle_classes[cls_id]] += 1
                    color = (0, 255, 255)
                elif is_in_zone((cx, cy), lane_B_zone):
                    lane_counts["B"] += 1
                    lane_class_counts["B"][vehicle_classes[cls_id]] += 1
                    color = (255, 0, 255)
                else:
                    continue

                cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorC=color)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cvzone.putTextRect(frame, f"{model.names[cls_id]}", (x1, y1 - 10), scale=0.6)

    current_time = time.time()
    for lane in ["A", "B"]:
        if lane_states[lane] == "RED" and lane_counts[lane] > 3 and (current_time - last_switch_time[lane]) > red_duration:
            lane_states[lane] = "GREEN"
            last_switch_time[lane] = current_time
        elif lane_states[lane] == "GREEN" and (current_time - last_switch_time[lane]) > green_duration:
            lane_states[lane] = "RED"
            last_switch_time[lane] = current_time

    cv2.polylines(frame, [lane_A_zone], True, (0, 255, 255), 2)
    cv2.polylines(frame, [lane_B_zone], True, (255, 0, 255), 2)

    draw_traffic_light(frame, (50, 50), lane_states["A"])
    draw_traffic_light(frame, (1150, 50), lane_states["B"])


    panel_width = 350
    panel = np.zeros((frame.shape[0], panel_width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    cv2.putText(panel, "Traffic Stats", (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    y_off = 100
    for lane in ["A", "B"]:
        cv2.putText(panel, f"Lane {lane}: {lane_states[lane]}", (20, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if lane_states[lane] == "GREEN" else (0, 0, 255), 2)
        y_off += 40
        cv2.putText(panel, f"Total Vehicles: {lane_counts[lane]}", (20, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y_off += 30
        for cls, count in lane_class_counts[lane].items():
            cv2.putText(panel, f"{cls}: {count}", (40, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
            y_off += 25
        y_off += 20


    cv2.putText(panel, time.strftime("%I:%M:%S %p"), (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    combined = np.hstack((frame, panel))
    cv2.imshow("Traffic AI - 2 Lanes", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
