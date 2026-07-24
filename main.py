#!/usr/bin/env python3
import sys

import rospy
import cv2
import numpy as np
import message_filters
import tf
import os
import inspect
import threading
import time as pytime
from collections import deque
from typing import List

# ROS Messages
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

# --- Project Imports ---
from vio_pipeline_v2.config.config_loader import ConfigLoader
from vio_pipeline_v2.msckf.VioManager import VioManager
from vio_pipeline_v2.core.sensor_data import ImuData, CameraData
from cv_bridge import CvBridge

# Colors for terminal output
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class RosVioManager:
    """
    ROS Wrapper for the OpenVINS VIO Manager.
    
    Architecture matches C++ ROS1Visualizer:
    - IMU callback feeds IMU data AND triggers async camera processing
    - Camera callbacks only queue CameraData into a deque (no processing)
    - A worker thread (spawned from IMU callback) drains the camera queue
      once we have at least one IMU measurement newer than the oldest queued image
    - visualize_odometry is called from IMU callback at IMU rate (fast propagate)
    """

    def __init__(self):
        # 1. Initialize ROS Node
        rospy.init_node('openvins_python', anonymous=False)
        rospy.loginfo(f"{GREEN}[ROS] Initializing OpenVINS Python Node...{RESET}")

        # 2. Load Configuration
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_path = os.path.join(current_dir, "config/Custom_v1")
        config_path = rospy.get_param("~config_path", default_config_path)

        if not os.path.exists(config_path):
            rospy.logerr(f"[ROS] Config directory does not exist: {config_path}")
            sys.exit(1)

        rospy.loginfo(f"[ROS] Loading configuration from: {config_path}")
        self.options = ConfigLoader.load_all(config_path)

        # 3. Initialize Core VIO Logic
        self.vio_manager = VioManager(self.options)
        rospy.loginfo(f"[ROS] Using VioManager from: {inspect.getsourcefile(VioManager) or VioManager.__module__}")

        # 4. ROS Infrastructure Setup
        self.bridge = CvBridge()

        # --- Thread Synchronization (matching C++ ROS1Visualizer) ---
        # camera_queue: sorted deque of CameraData, protected by camera_queue_mtx
        # Camera callbacks only append to this queue under the lock.
        # The IMU callback spawns a worker thread that drains this queue.
        self.camera_queue = deque()
        self.camera_queue_mtx = threading.Lock()

        # Atomic-like flag: True while the update worker thread is running.
        # If True, IMU callback skips spawning a new worker (camera queue keeps filling).
        self.thread_update_running = False

        # Rate-limiter per camera: drop images faster than track_frequency
        self.camera_last_timestamp = {}

        # --- Publishers ---
        self.pub_odomimu = rospy.Publisher("~odomimu", Odometry, queue_size=10)
        self.pub_odom = rospy.Publisher("~odom", Odometry, queue_size=10)
        self.pub_path = rospy.Publisher("~path", Path, queue_size=10)
        self.pub_feature_img = rospy.Publisher("~feature_image", Image, queue_size=5)
        self.tf_broadcaster = tf.TransformBroadcaster()

        # Path message storage
        self.path_msg = Path()
        self.path_msg.header.frame_id = "global"

        # --- Subscribers ---
        imu_topic = rospy.get_param("~topic_imu", "/imu/data_raw")
        self.sub_imu = rospy.Subscriber(
            imu_topic, Imu, self.callback_imu,
            queue_size=2000, tcp_nodelay=True
        )
        rospy.loginfo(f"[ROS] Subscribed to IMU: {imu_topic}")

        cam0_topic = rospy.get_param("~topic_cam0", "/camera/infra1/image_rect_raw")

        if self.options.use_stereo:
            cam1_topic = rospy.get_param("~topic_cam1", "/camera/infra2/image_rect_raw")
            rospy.loginfo(f"[ROS] Stereo Mode: {cam0_topic} + {cam1_topic}")

            self.sub_cam0 = message_filters.Subscriber(cam0_topic, Image)
            self.sub_cam1 = message_filters.Subscriber(cam1_topic, Image)

            self.sync = message_filters.ApproximateTimeSynchronizer(
                [self.sub_cam0, self.sub_cam1], queue_size=20, slop=0.05
            )
            self.sync.registerCallback(self.callback_stereo)
        else:
            rospy.loginfo(f"[ROS] Monocular Mode: {cam0_topic}")
            self.sub_cam0 = rospy.Subscriber(
                cam0_topic, Image, self.callback_monocular, queue_size=50
            )

        rospy.loginfo(f"{GREEN}[ROS] Node Started. Waiting for messages...{RESET}")

    # ===================================================================
    #  IMU Callback — the MAIN driver (matches C++ callback_inertial)
    # ===================================================================
    def callback_imu(self, msg):
        """
        1. Convert ROS IMU msg → ImuData, feed to VIO.
        2. Publish fast-propagated odometry at IMU rate.
        3. If no update thread is running, spawn one to drain the camera queue.
        """
        timestamp = msg.header.stamp.to_sec()

        wm = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        am = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

        imu_data = ImuData(timestamp, wm, am)

        # Feed to VIO (propagator, initializer, ZUPT — all handled inside)
        self.vio_manager.feed_measurement_imu(imu_data)

        # Publish fast-propagated odometry at IMU rate
        self.visualize_odometry(timestamp)

        # If a processing thread is already running, just return.
        # The camera queue will keep accumulating and the thread will drain it.
        if self.thread_update_running:
            return

        self.thread_update_running = True

        # Capture the current IMU timestamp for the worker closure
        imu_timestamp = timestamp

        def update_worker():
            try:
                # Lock the queue — prevents new images from being appended
                # while we decide which ones to process
                with self.camera_queue_mtx:
                    # Count unique camera streams in queue
                    unique_cam_ids = set()
                    for cam_msg in self.camera_queue:
                        if cam_msg.sensor_ids:
                            unique_cam_ids.add(cam_msg.sensor_ids[0])

                    # Wait until we have one of each camera stream
                    params = self.vio_manager.get_params()
                    num_unique_needed = 1 if params.state_options.num_cameras == 2 else params.state_options.num_cameras

                    if len(unique_cam_ids) >= num_unique_needed:
                        # Convert IMU timestamp to camera clock
                        timestamp_imu_inC = imu_timestamp - self.vio_manager.get_state()._calib_dt_CAMtoIMU.value()[0]

                        # Process all camera messages whose timestamp < current IMU time (in cam clock)
                        while self.camera_queue and self.camera_queue[0].timestamp < timestamp_imu_inC:
                            rT0 = pytime.time()
                            update_dt = 100.0 * (timestamp_imu_inC - self.camera_queue[0].timestamp)

                            cam_msg = self.camera_queue.popleft()

                            self.vio_manager.feed_measurement_camera(cam_msg)
                            self.visualize()

                            rT1 = pytime.time()
                            time_total = rT1 - rT0
                            if time_total > 0:
                                rospy.loginfo(
                                    f"{BLUE}[TIME]: {time_total:.4f} seconds total "
                                    f"({1.0 / time_total:.1f} hz, {update_dt:.2f} ms behind){RESET}"
                                )
            finally:
                self.thread_update_running = False

        # Run synchronously or asynchronously based on config
        if not self.vio_manager.get_params().use_multi_threading_subs:
            update_worker()
        else:
            t = threading.Thread(target=update_worker, daemon=True)
            t.start()

    # ===================================================================
    #  Camera Callbacks — queue only, no processing
    # ===================================================================
    def callback_monocular(self, msg0, cam_id0=0):
        """
        Matches C++ callback_monocular: rate-limit, convert, queue.
        """
        timestamp = msg0.header.stamp.to_sec()

        # Rate-limit: drop if faster than track_frequency
        time_delta = 1.0 / self.vio_manager.get_params().track_frequency
        if cam_id0 in self.camera_last_timestamp:
            if timestamp < self.camera_last_timestamp[cam_id0] + time_delta:
                return
        self.camera_last_timestamp[cam_id0] = timestamp

        # Convert image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg0, desired_encoding="mono8")
        except Exception as e:
            rospy.logerr_throttle(1.0, f"cv_bridge exception: {e}")
            return

        # Build CameraData
        message = CameraData()
        message.timestamp = timestamp
        message.sensor_ids.append(cam_id0)
        message.images.append(cv_img.copy())

        # Mask
        if self.vio_manager.get_params().use_mask and cam_id0 in self.vio_manager.get_params().masks:
            message.masks.append(self.vio_manager.get_params().masks[cam_id0])
        else:
            message.masks.append(np.zeros(cv_img.shape[:2], dtype=np.uint8))

        # Append to queue (sorted) under lock
        with self.camera_queue_mtx:
            self.camera_queue.append(message)
            # Sort by timestamp (handles rare out-of-order arrivals)
            if len(self.camera_queue) > 1:
                sorted_list = sorted(self.camera_queue)
                self.camera_queue.clear()
                self.camera_queue.extend(sorted_list)

    def callback_stereo(self, msg0, msg1, cam_id0=0, cam_id1=1):
        """
        Matches C++ callback_stereo: rate-limit, convert both images, queue.
        """
        timestamp = msg0.header.stamp.to_sec()

        # Rate-limit
        time_delta = 1.0 / self.vio_manager.get_params().track_frequency
        if cam_id0 in self.camera_last_timestamp:
            if timestamp < self.camera_last_timestamp[cam_id0] + time_delta:
                return
        self.camera_last_timestamp[cam_id0] = timestamp

        # Convert images
        try:
            cv_img0 = self.bridge.imgmsg_to_cv2(msg0, desired_encoding="mono8")
        except Exception as e:
            rospy.logerr_throttle(1.0, f"cv_bridge exception (cam0): {e}")
            return

        try:
            cv_img1 = self.bridge.imgmsg_to_cv2(msg1, desired_encoding="mono8")
        except Exception as e:
            rospy.logerr_throttle(1.0, f"cv_bridge exception (cam1): {e}")
            return

        # Build CameraData
        message = CameraData()
        message.timestamp = timestamp
        message.sensor_ids.append(cam_id0)
        message.sensor_ids.append(cam_id1)
        message.images.append(cv_img0.copy())
        message.images.append(cv_img1.copy())

        # Masks
        if self.vio_manager.get_params().use_mask:
            masks = self.vio_manager.get_params().masks
            message.masks.append(masks.get(cam_id0, np.zeros(cv_img0.shape[:2], dtype=np.uint8)))
            message.masks.append(masks.get(cam_id1, np.zeros(cv_img1.shape[:2], dtype=np.uint8)))
        else:
            message.masks.append(np.zeros(cv_img0.shape[:2], dtype=np.uint8))
            message.masks.append(np.zeros(cv_img1.shape[:2], dtype=np.uint8))

        # Append to queue (sorted) under lock
        with self.camera_queue_mtx:
            self.camera_queue.append(message)
            if len(self.camera_queue) > 1:
                sorted_list = sorted(self.camera_queue)
                self.camera_queue.clear()
                self.camera_queue.extend(sorted_list)

    # ===================================================================
    #  Visualization — fast-propagated odometry at IMU rate
    # ===================================================================
    def visualize_odometry(self, timestamp):
        """
        Matches C++ ROS1Visualizer::visualize_odometry.
        Uses fast_state_propagate to publish odometry at IMU rate
        without waiting for the full EKF update cycle.
        """
        if not self.vio_manager.initialized():
            return

        # Fast propagate state to current IMU timestamp
        state = self.vio_manager.get_state()
        state_plus = np.zeros(13)
        cov_plus = np.zeros((12, 12))

        if not self.vio_manager.get_propagator().fast_state_propagate(state, timestamp, state_plus, cov_plus):
            return

        # state_plus layout: [qx, qy, qz, qw, px, py, pz, vx_local, vy_local, vz_local, wx, wy, wz]

        if self.pub_odomimu.get_num_connections() == 0:
            return

        odom = Odometry()
        odom.header.stamp = rospy.Time.from_sec(timestamp)
        odom.header.frame_id = "global"
        odom.child_frame_id = "imu"

        # Pose
        odom.pose.pose.orientation.x = state_plus[0]
        odom.pose.pose.orientation.y = state_plus[1]
        odom.pose.pose.orientation.z = state_plus[2]
        odom.pose.pose.orientation.w = state_plus[3]
        odom.pose.pose.position.x = state_plus[4]
        odom.pose.pose.position.y = state_plus[5]
        odom.pose.pose.position.z = state_plus[6]

        # Twist (velocity in local IMU frame, angular velocity)
        odom.twist.twist.linear.x = state_plus[7]
        odom.twist.twist.linear.y = state_plus[8]
        odom.twist.twist.linear.z = state_plus[9]
        odom.twist.twist.angular.x = state_plus[10]
        odom.twist.twist.angular.y = state_plus[11]
        odom.twist.twist.angular.z = state_plus[12]

        # Covariance (ROS order: position then orientation)
        # cov_plus is 12x12: [ori(3), pos(3), vel(3), bias(3)]
        # ROS wants 6x6: [pos(3), ori(3)]
        cov_ros = [0.0] * 36
        for r in range(3):
            for c in range(3):
                # Position covariance → rows/cols 0-2
                cov_ros[6 * r + c] = cov_plus[3 + r, 3 + c]
                # Orientation covariance → rows/cols 3-5
                cov_ros[6 * (r + 3) + (c + 3)] = cov_plus[r, c]
                # Cross terms
                cov_ros[6 * r + (c + 3)] = cov_plus[3 + r, c]
                cov_ros[6 * (r + 3) + c] = cov_plus[r, 3 + c]
        odom.pose.covariance = cov_ros

        self.pub_odomimu.publish(odom)

    def visualize(self):
        """
        Full visualization after an EKF update cycle.
        Matches C++ ROS1Visualizer::visualize (called after feed_measurement_camera).
        """
        if not self.vio_manager.initialized():
            return

        state = self.vio_manager.get_state()
        timestamp = state._timestamp

        # State stores q_GtoI (Global to IMU rotation)
        q_GtoI = state._imu.quat()  # [x, y, z, w]
        p_IinG = state._imu.pos()

        # Convert to ROS convention (T_Body_Global)
        q_ros = np.array([-q_GtoI[0], -q_GtoI[1], -q_GtoI[2], q_GtoI[3]])
        p_ros = p_IinG

        # --- TF ---
        self.tf_broadcaster.sendTransform(
            (p_ros[0], p_ros[1], p_ros[2]),
            (q_ros[0], q_ros[1], q_ros[2], q_ros[3]),
            rospy.Time.from_sec(timestamp),
            "body",
            "global"
        )

        # --- Odometry (post-update, not fast-propagated) ---
        odom = Odometry()
        odom.header.stamp = rospy.Time.from_sec(timestamp)
        odom.header.frame_id = "global"
        odom.child_frame_id = "body"

        odom.pose.pose.position.x = p_ros[0]
        odom.pose.pose.position.y = p_ros[1]
        odom.pose.pose.position.z = p_ros[2]
        odom.pose.pose.orientation.x = q_ros[0]
        odom.pose.pose.orientation.y = q_ros[1]
        odom.pose.pose.orientation.z = q_ros[2]
        odom.pose.pose.orientation.w = q_ros[3]

        for i in range(36):
            odom.pose.covariance[i] = 0.0
        odom.pose.covariance[0] = 1e-3
        odom.pose.covariance[7] = 1e-3
        odom.pose.covariance[14] = 1e-3

        self.pub_odom.publish(odom)

        # --- Path ---
        pose_stamped = PoseStamped()
        pose_stamped.header = odom.header
        pose_stamped.pose = odom.pose.pose

        self.path_msg.header = odom.header
        self.path_msg.poses.append(pose_stamped)

        if len(self.path_msg.poses) > 5000:
            self.path_msg.poses.pop(0)

        self.pub_path.publish(self.path_msg)

        # --- Feature Image ---
        img_viz = self.vio_manager.get_historical_viz_image()
        if img_viz is not None:
            try:
                msg_img = self.bridge.cv2_to_imgmsg(img_viz, encoding="bgr8")
                msg_img.header.stamp = odom.header.stamp
                msg_img.header.frame_id = "body"
                self.pub_feature_img.publish(msg_img)
            except Exception:
                pass


if __name__ == '__main__':
    try:
        node = RosVioManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass