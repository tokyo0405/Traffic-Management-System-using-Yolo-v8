# Traffic-Management-System-using-Yolo-v8

📌 Project Overview

This project implements an intelligent traffic management system that uses computer vision and deep learning (YOLOv8) to detect and analyze real-time traffic density from video feeds. Based on the number of vehicles in each lane, the system dynamically controls traffic lights to optimize vehicle flow, reduce congestion, and minimize idle time at intersections.

Instead of relying on fixed-timer signals, this AI-driven approach adapts signal durations automatically depending on live traffic conditions, ensuring that high-density lanes get longer green signals, while less busy lanes remain on red or yellow.

🧠 Key Features

🔍 Real-time Vehicle Detection using YOLOv8 deep learning model.

🛣️ Lane-wise Traffic Density Estimation through bounding box mapping and centroid analysis.

🚦 Adaptive Signal Control that adjusts green, yellow, and red durations dynamically.

🎥 Webcam or Video Input Support for live or recorded footage.

📊 Performance Dashboard showing traffic counts, lane status, and current signal phase.

🌱 Sustainability-Oriented — reduces fuel wastage and emissions through efficient traffic flow.

⚙️ Tech Stack

Programming Language: Python 3.10

Libraries & Frameworks:

OpenCV
 for computer vision processing

Ultralytics YOLOv8
 for object detection

cvzone
 for visualization overlays

NumPy, time, and collections for data handling and logic control
