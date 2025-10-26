import cv2
import numpy as np
import time
from collections import OrderedDict
from math import hypot
from datetime import datetime

MIN_CONTOUR_AREA = 400
MAX_DISAPPEARED_FRAMES = 8
DISTANCE_THRESHOLD = 50

GREEN_HOLD_MIN = 4.0
COUNT_THRESHOLD_TO_FORCE = 2

COLOR_BOUNDARY = (255, 255, 0)
COLOR_ZONE_FILL = (40, 40, 40)
COLOR_TEXT = (245, 245, 245)
COLOR_LIGHT_BG = (40, 40, 40)


class CentroidTracker:
    def __init__(self, max_disappeared=MAX_DISAPPEARED_FRAMES, dist_thresh=DISTANCE_THRESHOLD):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.dist_thresh = dist_thresh

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            to_deregister = []
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    to_deregister.append(object_id)
            for oid in to_deregister:
                self.deregister(oid)
            return self.objects

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = hypot(oc[0] - ic[0], oc[1] - ic[1])

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows, assigned_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in assigned_rows or c in assigned_cols:
                continue
            if D[r, c] > self.dist_thresh:
                continue
            oid = object_ids[r]
            self.objects[oid] = input_centroids[c]
            self.disappeared[oid] = 0
            assigned_rows.add(r)
            assigned_cols.add(c)

        for i, oid in enumerate(object_ids):
            if i not in assigned_rows:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        for j, ic in enumerate(input_centroids):
            if j not in assigned_cols:
                self.register(ic)

        return self.objects


def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(np.array(poly, np.int32), pt, False) >= 0


def draw_traffic_light(frame, x, y, w=80, h=200, state='HORIZONTAL_GREEN'):
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_LIGHT_BG, -1, lineType=cv2.LINE_AA)
    lamp_r = int(w * 0.3)
    cx = x + w // 2
    top_y = y + 30
    gap = 60
    if state == 'HORIZONTAL_GREEN':
        cv2.circle(frame, (cx, top_y), lamp_r, (0, 200, 0), -1)
        cv2.circle(frame, (cx, top_y + gap), lamp_r, (0, 0, 180), -1)
    elif state == 'VERTICAL_GREEN':
        cv2.circle(frame, (cx, top_y), lamp_r, (0, 0, 180), -1)
        cv2.circle(frame, (cx, top_y + gap), lamp_r, (0, 200, 0), -1)
    else:
        cv2.circle(frame, (cx, top_y), lamp_r, (0, 0, 180), -1)
        cv2.circle(frame, (cx, top_y + gap), lamp_r, (0, 0, 180), -1)
    cv2.putText(frame, "H-ROW", (x - 4, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    h_zone = [(int(0.05*frame_w), int(0.45*frame_h)),
              (int(0.95*frame_w), int(0.45*frame_h)),
              (int(0.95*frame_w), int(0.60*frame_h)),
              (int(0.05*frame_w), int(0.60*frame_h))]

    v_zone = [(int(0.45*frame_w), int(0.05*frame_h)),
              (int(0.60*frame_w), int(0.05*frame_h)),
              (int(0.60*frame_w), int(0.95*frame_h)),
              (int(0.45*frame_w), int(0.95*frame_h))]

    backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=False)
    tracker = CentroidTracker()

    counted_in_h = set()
    counted_in_v = set()

    traffic_state = 'HORIZONTAL_GREEN'
    last_switch_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg = backsub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_centroids = []
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            detected_centroids.append((cx, cy))

        objects = tracker.update(detected_centroids)
        current_count_h = 0
        current_count_v = 0

        for oid, centroid in objects.items():
            cx, cy = int(centroid[0]), int(centroid[1])
            in_h = point_in_poly((cx, cy), h_zone)
            in_v = point_in_poly((cx, cy), v_zone)
            if in_h:
                current_count_h += 1
                if oid not in counted_in_h:
                    counted_in_h.add(oid)
                if oid in counted_in_v:
                    counted_in_v.discard(oid)
            elif in_v:
                current_count_v += 1
                if oid not in counted_in_v:
                    counted_in_v.add(oid)
                if oid in counted_in_h:
                    counted_in_h.discard(oid)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            cv2.putText(frame, f"ID {oid}", (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

        now = time.time()
        if now - last_switch_time >= GREEN_HOLD_MIN:
            if current_count_h - current_count_v >= COUNT_THRESHOLD_TO_FORCE and traffic_state != 'HORIZONTAL_GREEN':
                traffic_state = 'HORIZONTAL_GREEN'
                last_switch_time = now
            elif current_count_v - current_count_h >= COUNT_THRESHOLD_TO_FORCE and traffic_state != 'VERTICAL_GREEN':
                traffic_state = 'VERTICAL_GREEN'
                last_switch_time = now

        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(h_zone, np.int32)], COLOR_ZONE_FILL)
        cv2.fillPoly(overlay, [np.array(v_zone, np.int32)], COLOR_ZONE_FILL)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        cv2.polylines(frame, [np.array(h_zone, np.int32)], True, COLOR_BOUNDARY, 2, cv2.LINE_AA)
        cv2.polylines(frame, [np.array(v_zone, np.int32)], True, COLOR_BOUNDARY, 2, cv2.LINE_AA)

        cv2.putText(frame, f"H-Count: {current_count_h}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)
        cv2.putText(frame, f"V-Count: {current_count_v}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)

        now_dt = datetime.now()
        clock_text = now_dt.strftime("%I:%M:%S %p")
        cv2.putText(frame, clock_text, (frame.shape[1] - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)

        draw_traffic_light(frame, frame.shape[1] - 110, 60, w=90, h=140, state=traffic_state)

        status_text = "GREEN -> HORIZONTAL" if traffic_state == 'HORIZONTAL_GREEN' else ("GREEN -> VERTICAL" if traffic_state == 'VERTICAL_GREEN' else "ALL RED")
        cv2.putText(frame, status_text, (frame.shape[1] - 330, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)

        since_switch = int(now - last_switch_time)
        cv2.putText(frame, f"hold: {since_switch}s", (frame.shape[1] - 330, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

        cv2.imshow("Traffic Manager - Press 'q' to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
