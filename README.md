# Traffic-Management-System-using-Yolo-v8

📌 Project Overview

This project implements an intelligent traffic management system that uses computer vision and deep learning (YOLOv8) to detect and analyze real-time traffic density from video feeds. Based on the number of vehicles in each lane, the system dynamically controls traffic lights to optimize vehicle flow, reduce congestion, and minimize idle time at intersections.

Instead of relying on fixed-timer signals, this AI-driven approach adapts signal durations automatically depending on live traffic conditions, ensuring that high-density lanes get longer green signals, while less busy lanes remain on red or yellow.

⚙️ Instructions to Run the Project

Follow the steps below to set up and run the AI-Based Traffic Management System efficiently.

🧩 1. Install Python

Ensure that the latest version of Python (3.10 or higher) is installed on your system.
You can download it from the official Python website
.

📦 2. Install Dependencies

Open a terminal or command prompt in your project directory.

Install all the required Python libraries by running:

pip install -r requirements.txt


This will automatically install all the necessary modules, including OpenCV, Ultralytics YOLO, NumPy, and cvzone.

🎥 3. Test Webcam Access

Before running the full system, verify that your webcam is accessible and functioning properly:

python opening_cam.py


If the webcam window appears and displays live video, your camera setup is correct.

🚘 4. Run Object Detection Demo

Once the webcam test succeeds, run the object detection demo to ensure YOLOv8 is working correctly:

python demo_detect.py


This will open a window showing real-time object detection results (e.g., cars, trucks, buses, and motorcycles) from the live video feed.

📌 Note:
Make sure that the YOLO model file (yolov8n.pt) is located inside your main project folder.
If it is not present, the script will automatically download it from the official Ultralytics repository.

🚦 5. Run Traffic Management Simulation

After confirming object detection, you can run the full Traffic Management System by executing:

python main.py


This will start the adaptive traffic control interface where the system detects vehicles in each lane, calculates lane density, and dynamically changes the traffic light signals based on real-time traffic weight.

🧠 6. Testing and Verification

Run all the demo files (demo_detect.py, main.py, etc.) one by one to observe the system’s performance.

Ensure that your lane boundaries are correctly defined in the code to match your camera view.

You can adjust parameters like BASE_GREEN, MAX_GREEN, and YELLOW_DURATION in the script to fine-tune traffic timing behavior.

✅ Final Notes

Keep the YOLO model (yolov8n.pt) inside the project directory for smooth execution.

Ensure proper lighting for accurate vehicle detection.

The system works in real time and can be extended for IoT traffic signal integration in future versions.
