import cv2
import time
from ultralytics import YOLO

# YOLO model
model = YOLO("yolov8n.pt")

def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        print("ERROR")
        return

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        frame = cv2.flip(frame, 1)


        results = model(frame, stream=True)


        for r in results:
            annotated_frame = r.plot()


        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time


        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        cv2.imshow("YOLO Webcam (Mirrored)", annotated_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
