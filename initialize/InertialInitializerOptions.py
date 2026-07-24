import numpy as np
from typing import Any, Dict, Optional, List

class InertialInitializerOptions:
    """
    Struct which stores all options needed for state estimation initialization.
    """
    def __init__(self):
        
        # INITIALIZATION ============================

        # Amount of time we will initialize over (seconds)
        self.init_window_time = 1.0

        # Variance threshold on our acceleration to be classified as moving
        self.init_imu_thresh = 1.0

        # Max disparity we will consider the unit to be stationary
        self.init_max_disparity = 1.0

        # Number of features we should try to track
        self.init_max_features = 50

        # If we should perform dynamic initialization
        self.init_dyn_use = False

        self.init_calib_camImu_dt = 0.0
        self.gravity_mag = 9.81
        # If we should optimize and recover the calibration in our MLE
        self.init_dyn_mle_opt_calib = False

        # Max number of MLE iterations for dynamic initialization
        self.init_dyn_mle_max_iter = 20

        # Max number of MLE threads for dynamic initialization
        self.init_dyn_mle_max_threads = 20

        # Max time for MLE optimization (seconds)
        self.init_dyn_mle_max_time = 5.0

        # Number of poses to use during initialization (max should be cam freq * window)
        self.init_dyn_num_pose = 5

        # Minimum degrees we need to rotate before we try to init (sum of norm)
        self.init_dyn_min_deg = 45.0

        # Magnitude we will inflate initial covariance of orientation
        self.init_dyn_inflation_orientation = 10.0

        # Magnitude we will inflate initial covariance of velocity
        self.init_dyn_inflation_velocity = 10.0

        # Magnitude we will inflate initial covariance of gyroscope bias
        self.init_dyn_inflation_bias_gyro = 100.0

        # Magnitude we will inflate initial covariance of accelerometer bias
        self.init_dyn_inflation_bias_accel = 100.0

        # Minimum reciprocal condition number acceptable for our covariance recovery 
        # (min_sigma / max_sigma < sqrt(min_reciprocal_condition_number))
        self.init_dyn_min_rec_cond = 1e-15

        # Initial IMU gyroscope bias values for dynamic initialization (will be optimized)
        self.init_dyn_bias_g = np.zeros(3)

        # Initial IMU accelerometer bias values for dynamic initialization (will be optimized)
        self.init_dyn_bias_a = np.zeros(3)

    def print_and_load_initializer(self, parser: Optional[Dict[str, Any]] = None):
        """
        This function will load print out all initializer settings loaded.
        :param parser: Dictionary containing configuration parameters.
        """
        print("INITIALIZATION SETTINGS:")
        
        if parser is not None:
            self.init_window_time = parser.get("init_window_time", self.init_window_time)
            self.init_imu_thresh = parser.get("init_imu_thresh", self.init_imu_thresh)
            self.init_max_disparity = parser.get("init_max_disparity", self.init_max_disparity)
            self.init_max_features = parser.get("init_max_features", self.init_max_features)
            self.init_dyn_use = parser.get("init_dyn_use", self.init_dyn_use)
            self.init_dyn_mle_opt_calib = parser.get("init_dyn_mle_opt_calib", self.init_dyn_mle_opt_calib)
            self.init_dyn_mle_max_iter = parser.get("init_dyn_mle_max_iter", self.init_dyn_mle_max_iter)
            self.init_dyn_mle_max_threads = parser.get("init_dyn_mle_max_threads", self.init_dyn_mle_max_threads)
            self.init_dyn_mle_max_time = parser.get("init_dyn_mle_max_time", self.init_dyn_mle_max_time)
            self.init_dyn_num_pose = parser.get("init_dyn_num_pose", self.init_dyn_num_pose)
            self.init_dyn_min_deg = parser.get("init_dyn_min_deg", self.init_dyn_min_deg)
            self.init_dyn_inflation_orientation = parser.get("init_dyn_inflation_ori", self.init_dyn_inflation_orientation)
            self.init_dyn_inflation_velocity = parser.get("init_dyn_inflation_vel", self.init_dyn_inflation_velocity)
            self.init_dyn_inflation_bias_gyro = parser.get("init_dyn_inflation_bg", self.init_dyn_inflation_bias_gyro)
            self.init_dyn_inflation_bias_accel = parser.get("init_dyn_inflation_ba", self.init_dyn_inflation_bias_accel)
            self.init_dyn_min_rec_cond = parser.get("init_dyn_min_rec_cond", self.init_dyn_min_rec_cond)
            
            bias_g_list = parser.get("init_dyn_bias_g", [0.0, 0.0, 0.0])
            self.init_dyn_bias_g = np.array(bias_g_list, dtype=np.float64)
            
            bias_a_list = parser.get("init_dyn_bias_a", [0.0, 0.0, 0.0])
            self.init_dyn_bias_a = np.array(bias_a_list, dtype=np.float64)

        print(f"  - init_window_time: {self.init_window_time:.2f}")
        print(f"  - init_imu_thresh: {self.init_imu_thresh:.2f}")
        print(f"  - init_max_disparity: {self.init_max_disparity:.2f}")
        
        print(f"  - init_max_features: {self.init_max_features:.2f}")

        if self.init_max_features < 15:
            print("[ERROR] number of requested feature tracks to init not enough!!")
            print(f"  init_max_features = {self.init_max_features}")
            exit(1)

        if self.init_imu_thresh <= 0.0 and not self.init_dyn_use:
            print("[ERROR] need to have an IMU threshold for static initialization!")
            print(f"  init_imu_thresh = {self.init_imu_thresh:.3f}")
            print(f"  init_dyn_use = {self.init_dyn_use}")
            exit(1)

        if self.init_max_disparity <= 0.0 and not self.init_dyn_use:
            print("[ERROR] need to have an DISPARITY threshold for static initialization!")
            print(f"  init_max_disparity = {self.init_max_disparity:.3f}")
            print(f"  init_dyn_use = {self.init_dyn_use}")
            exit(1)

        print(f"  - init_dyn_use: {self.init_dyn_use}")
        print(f"  - init_dyn_mle_opt_calib: {self.init_dyn_mle_opt_calib}")
        print(f"  - init_dyn_mle_max_iter: {self.init_dyn_mle_max_iter}")
        print(f"  - init_dyn_mle_max_threads: {self.init_dyn_mle_max_threads}")
        print(f"  - init_dyn_mle_max_time: {self.init_dyn_mle_max_time:.2f}")
        print(f"  - init_dyn_num_pose: {self.init_dyn_num_pose}")
        print(f"  - init_dyn_min_deg: {self.init_dyn_min_deg:.2f}")
        print(f"  - init_dyn_inflation_ori: {self.init_dyn_inflation_orientation:.2e}")
        print(f"  - init_dyn_inflation_vel: {self.init_dyn_inflation_velocity:.2e}")
        print(f"  - init_dyn_inflation_bg: {self.init_dyn_inflation_bias_gyro:.2e}")
        print(f"  - init_dyn_inflation_ba: {self.init_dyn_inflation_bias_accel:.2e}")
        print(f"  - init_dyn_min_rec_cond: {self.init_dyn_min_rec_cond:.2e}")

        if self.init_dyn_num_pose < 4:
            print("[ERROR] number of requested frames to init not enough!!")
            print(f"  init_dyn_num_pose = {self.init_dyn_num_pose} (4 min)")
            exit(1)

        print(f"  - init_dyn_bias_g: {self.init_dyn_bias_g[0]:.2f}, {self.init_dyn_bias_g[1]:.2f}, {self.init_dyn_bias_g[2]:.2f}")
        print(f"  - init_dyn_bias_a: {self.init_dyn_bias_a[0]:.2f}, {self.init_dyn_bias_a[1]:.2f}, {self.init_dyn_bias_a[2]:.2f}")