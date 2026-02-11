# Edge Vision and Local Control for RB-Kairos

This project is developed within the Robotics Laboratory at the School of Electrical Engineering, University of Belgrade. It implements an optimized edge computer vision pipeline on the NVIDIA Jetson Nano for autonomous agricultural robotics.

System Architecture

The project offloads computationally intensive vision tasks from the robot master to an edge node (Jetson Nano) to ensure low-latency control of the robotic arm.

- Perception: Real-time fruit detection using a custom YOLOv5n model trained via Roboflow and optimized via TensorRT.
- Runtime: C++ API implementation for GPU-accelerated inference.
- Communication: ROS Melodic integration with image_transport compression for efficient data transfer from Intel RealSense sensors.
- Application: Downstream coordinate generation for robotic arm trajectory planning in precision agriculture.

Technical Specifications

- Institution: University of Belgrade, School of Electrical Engineering, Robotics Laboratory.
- Hardware: NVIDIA Jetson Nano (Tegra X1), Intel RealSense D435.
- Platform: RB-Kairos Mobile Base with Franka-Emika Panda manipulator.
- Frameworks: ROS Melodic, CUDA 10.2, TensorRT, OpenCV.
- Training: Custom dataset annotated and augmented using Roboflow.
- Language: C++14.

Implementation Details

The vision node subscribes to compressed image streams, performs tensor-based inference, and publishes bounding box coordinates to the local control loop. The model is specifically optimized for Jetson Nano memory constraints and compute capabilities to maintain a stable 20 FPS throughput for fruit recognition and harvesting tasks.
