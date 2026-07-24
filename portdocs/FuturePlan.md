# C++ VIO Pipeline Future Roadmap

This document outlines the next engineering phases for deploying and optimizing the ported C++ VIO pipeline.

## 1. ROS2 Humble Wrapper Setup

To deploy the pipeline on autonomous robotic platforms, we will implement a ROS2 Humble wrapper package named `ov_msckf_ros2`:

```
ROS2 Node (ov_msckf_ros2)
   |
   +--> Subscription: sensor_msgs/msg/Imu -> feed_measurement_imu()
   |
   +--> Subscription: sensor_msgs/msg/Image (or custom FeatureTracks message) 
   |                   -> feed_measurement_camera_tracks()
   |
   +--> Publisher: nav_msgs/msg/Odometry (VIO EKF state estimate)
   +--> Publisher: sensor_msgs/msg/PointCloud2 (MSCKF triangulated landmarks)
```

### Key Tasks
* Setup standard `colcon` packaging with `rclcpp`.
* Implement a multi-threaded executor using ROS2 component nodes to feed camera frames and IMU messages concurrently.
* Utilize `image_transport` for efficient zero-copy image transport pointers when linking feature tracker nodes.

---

## 2. Jetson Orin Nano Optimizations

The NVIDIA Jetson Orin Nano features a 6-core ARM Cortex-A78AE CPU and an Ampere GPU. We can leverage specific hardware capabilities for optimized VIO execution:

### ARM Neon SIMD Vectorization
* **Action**: Configure Eigen to use ARM Neon vectorization by compiling with `-march=armv8-a+simd` or `-mcpu=cortex-a78ae`.
* **Impact**: Speeds up quaternion multiplications and $3 \times 3$ rotation matrix products by packing calculations into 128-bit SIMD registers.

### Single-Precision Floating Point (float)
* **Action**: Port state double variables to float (e.g. `Eigen::Vector4f`, `Eigen::MatrixXf`).
* **Impact**: Cuts EKF covariance storage and computation bandwidth in half. ARM Neon registers can execute four 32-bit floats simultaneously, doubling the SIMD throughput compared to 64-bit doubles.

### GPU Accelerated Visual Front-End
* **Action**: Offload visual feature tracking (such as Shi-Tomasi corner detection and KLT optical flow) to the Jetson Orin's GPU using OpenCV CUDA module bindings (`cv::cuda::CornernessHarris`, `cv::cuda::SparsePyrLKOpticalFlow`).
* **Impact**: Frees up CPU cores entirely for the EKF propagation and Schur update cycles.

---

## 3. Math & Algorithm Extensions

1. **Online Camera-IMU Time Offset Estimation**: Expand the EKF state vector to estimate the time delay ($t_{off}$) between the camera shutter and the IMU clock dynamically, correcting for temporal calibration errors during high-speed maneuvers.
2. **Online Extrinsic Calibration**: Refine the camera-IMU translation and rotation matrices ($^I\mathbf{p}_C, ^I\bar{q}_C$) inside the state vector to correct for mechanical changes or structural flexing on the robot.
3. **FEJ (First-Estimates Jacobian) Alignment**: Improve EKF consistency by evaluating state transition Jacobians at the first nominal estimate rather than the current state, preventing filter divergence on long-range trajectories.
