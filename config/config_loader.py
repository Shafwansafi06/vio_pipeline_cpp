import numpy as np
import os
import sys
import yaml

# --- Project Imports ---
from vio_pipeline_v2.msckf.VioManagerOptions import VioManagerOptions
from vio_pipeline_v2.core.cam_radtan import CamRadtan
from vio_pipeline_v2.core.CamEqui import CamEqui
from vio_pipeline_v2.type.PoseJPL import PoseJPL
from vio_pipeline_v2.type.quat_ops import rot_2_quat

class ConfigLoader:
    """
    Parses OpenVINS/Kalibr configuration files using PyYAML.
    Robustly handles %YAML:1.0 headers and nested list matrices.
    """

    @staticmethod
    def load_all(config_dir: str) -> VioManagerOptions:
        options = VioManagerOptions()
        
        path_est = os.path.join(config_dir, "estimator_config.yaml")
        path_imu = os.path.join(config_dir, "kalibr_imu_chain.yaml")
        path_cam = os.path.join(config_dir, "kalibr_imucam_chain.yaml")

        if not os.path.exists(path_est):
            print(f"\033[91m[ERROR] Config file not found: {path_est}\033[0m")
            sys.exit(1)
        if not os.path.exists(path_imu):
            print(f"\033[91m[ERROR] Config file not found: {path_imu}\033[0m")
            sys.exit(1)
        if not os.path.exists(path_cam):
            print(f"\033[91m[ERROR] Config file not found: {path_cam}\033[0m")
            sys.exit(1)

        print(f"[CONFIG] Loading Estimator: {path_est}")
        ConfigLoader._load_estimator_config(path_est, options)
        
        print(f"[CONFIG] Loading IMU: {path_imu}")
        ConfigLoader._load_imu_config(path_imu, options)
        
        print(f"[CONFIG] Loading Cameras: {path_cam}")
        ConfigLoader._load_camera_config(path_cam, options)

        options.sync_state_options()
        return options

    @staticmethod
    def _read_yaml_safe(path):
        """Helper to read YAML files that might have OpenCV headers."""
        with open(path, 'r') as f:
            content = f.read()
            # Remove OpenCV header if present, otherwise PyYAML fails
            if content.startswith("%YAML:1.0"):
                content = content.replace("%YAML:1.0", "")
            return yaml.safe_load(content)

    @staticmethod
    def _load_estimator_config(path: str, opts: VioManagerOptions):
        data = ConfigLoader._read_yaml_safe(path)
        if not data: return

        # Core filter/state behavior
        opts.state_options.do_fej = bool(data.get("use_fej", opts.state_options.do_fej))
        integration = str(data.get("integration", "rk4")).lower()
        from vio_pipeline_v2.msckf.StateOptions import StateOptions
        if integration == "discrete":
            opts.state_options.integration_method = StateOptions.IntegrationMethod.DISCRETE
        elif integration == "analytical":
            opts.state_options.integration_method = StateOptions.IntegrationMethod.ANALYTICAL
        else:
            opts.state_options.integration_method = StateOptions.IntegrationMethod.RK4

        opts.state_options.do_calib_camera_pose = bool(data.get("calib_cam_extrinsics", opts.state_options.do_calib_camera_pose))
        opts.state_options.do_calib_camera_intrinsics = bool(data.get("calib_cam_intrinsics", opts.state_options.do_calib_camera_intrinsics))
        opts.state_options.do_calib_camera_timeoffset = bool(data.get("calib_cam_timeoffset", opts.state_options.do_calib_camera_timeoffset))
        opts.state_options.do_calib_imu_intrinsics = bool(data.get("calib_imu_intrinsics", opts.state_options.do_calib_imu_intrinsics))
        opts.state_options.do_calib_imu_g_sensitivity = bool(data.get("calib_imu_g_sensitivity", opts.state_options.do_calib_imu_g_sensitivity))

        # General
        opts.num_cameras = int(data.get("max_cameras", 2))
        opts.use_stereo = bool(data.get("use_stereo", True))
        opts.state_options.num_cameras = opts.num_cameras
        opts.use_mask = bool(data.get("use_mask", opts.use_mask))

        # Hovering sensitivity tuning (backward-compatible with older options objects)
        opts.hovering_threshold = float(data.get("hovering_threshold", getattr(opts, "hovering_threshold", 5e-2)))
        opts.hover_smoothing_required = int(data.get("hover_smoothing_required", getattr(opts, "hover_smoothing_required", 2)))
        opts.hover_baseline_threshold = float(data.get("hover_baseline_threshold", getattr(opts, "hover_baseline_threshold", 0.15)))
        opts.hover_epipolar_inlier_threshold = float(
            data.get("hover_epipolar_inlier_threshold", getattr(opts, "hover_epipolar_inlier_threshold", 6e-2))
        )
        opts.hover_bearing_inlier_threshold = float(
            data.get("hover_bearing_inlier_threshold", getattr(opts, "hover_bearing_inlier_threshold", 6e-2))
        )
        opts.hover_min_inliers = int(data.get("hover_min_inliers", getattr(opts, "hover_min_inliers", 6)))
        opts.hover_entry_max_baseline = float(data.get("hover_entry_max_baseline", getattr(opts, "hover_entry_max_baseline", 0.30)))
        opts.hover_entry_max_velocity = float(data.get("hover_entry_max_velocity", getattr(opts, "hover_entry_max_velocity", 0.25)))
        opts.hover_exit_velocity_threshold = float(data.get("hover_exit_velocity_threshold", getattr(opts, "hover_exit_velocity_threshold", 0.5)))
        opts.hover_max_duration_sec = float(data.get("hover_max_duration_sec", getattr(opts, "hover_max_duration_sec", 3.0)))
        # State Size
        opts.state_options.max_clone_size = int(data.get("max_clones", 11))
        opts.state_options.max_slam_features = int(data.get("max_slam", 50))
        opts.max_slam_in_update = int(data.get("max_slam_in_update", 25))
        opts.max_msckf_in_update = int(data.get("max_msckf_in_update", 200))
        
        # Tracking
        opts.use_klt = bool(data.get("use_klt", True))
        opts.num_pts = int(data.get("num_pts", 200))
        opts.fast_threshold = int(data.get("fast_threshold", 20))
        opts.grid_x = int(data.get("grid_x", 5))
        opts.grid_y = int(data.get("grid_y", 3))
        opts.min_px_dist = int(data.get("min_px_dist", 10))
        opts.knn_ratio = float(data.get("knn_ratio", 0.70))
        
        # Initialization
        opts.init_options.init_window_time = float(data.get("init_window_time", 1.0))
        opts.init_options.init_imu_thresh = float(data.get("init_imu_thresh", 1.0))
        opts.init_options.init_max_disparity = float(data.get("init_max_disparity", 1.0))
        opts.init_options.init_max_features = float(data.get("init_max_features", 50.0))
        opts.init_options.init_dyn_use = bool(data.get("init_dyn_use", False))
        opts.init_options.init_dyn_mle_opt_calib = bool(data.get("init_dyn_mle_opt_calib", False))
        opts.init_options.init_dyn_mle_max_iter = int(data.get("init_dyn_mle_max_iter", 50))
        opts.init_options.init_dyn_mle_max_threads = int(data.get("init_dyn_mle_max_threads", 6))
        opts.init_options.init_dyn_mle_max_time = float(data.get("init_dyn_mle_max_time", 0.05))
        opts.init_options.init_dyn_num_pose = int(data.get("init_dyn_num_pose", 6))
        opts.init_options.init_dyn_min_deg = float(data.get("init_dyn_min_deg", 5.0))
        opts.init_options.init_dyn_inflation_orientation = float(data.get("init_dyn_inflation_ori", 10.0))
        opts.init_options.init_dyn_inflation_velocity = float(data.get("init_dyn_inflation_vel", 100.0))
        opts.init_options.init_dyn_inflation_bias_gyro = float(data.get("init_dyn_inflation_bg", 10.0))
        opts.init_options.init_dyn_inflation_bias_accel = float(data.get("init_dyn_inflation_ba", 100.0))
        opts.init_options.init_dyn_min_rec_cond = float(data.get("init_dyn_min_rec_cond", 1e-15))
        bias_g_list = data.get("init_dyn_bias_g", [0.0, 0.0, 0.0])
        opts.init_options.init_dyn_bias_g = np.array(bias_g_list, dtype=np.float64)
        bias_a_list = data.get("init_dyn_bias_a", [0.0, 0.0, 0.0])
        opts.init_options.init_dyn_bias_a = np.array(bias_a_list, dtype=np.float64)
        
        # ZUPT
        opts.try_zupt = bool(data.get("try_zupt", False))
        # Support both historical typo (multipler) and corrected key (multiplier).
        opts.zupt_options.chi2_multipler = float(
            data.get("zupt_chi2_multipler", data.get("zupt_chi2_multiplier", opts.zupt_options.chi2_multipler))
        )
        opts.zupt_max_velocity = float(data.get("zupt_max_velocity", 0.1))
        opts.zupt_noise_multiplier = float(data.get("zupt_noise_multiplier", 10.0))
        opts.zupt_max_disparity = float(data.get("zupt_max_disparity", 1.0))
        opts.zupt_only_at_beginning = bool(data.get("zupt_only_at_beginning", False))
        
        # Gravity
        opts.gravity_mag = float(data.get("gravity_mag", 9.81))
        opts.init_options.gravity_mag = opts.gravity_mag
        
        # Chi2
        opts.up_msckf_sigma_px = float(data.get("up_msckf_sigma_px", 1.0))
        opts.up_msckf_chi2_multipler = float(data.get("up_msckf_chi2_multipler", 1.0))

        # Bind estimator keys into updater option structs.
        opts.msckf_options.sigma_pix = opts.up_msckf_sigma_px
        opts.msckf_options.chi2_multipler = opts.up_msckf_chi2_multipler
        opts.msckf_options.sigma_pix_sq = opts.msckf_options.sigma_pix ** 2

        opts.msckf_options.min_features_for_update = int(
            data.get("msckf_min_features_for_update", opts.msckf_options.min_features_for_update)
        )
        opts.msckf_options.min_update_rows = int(
            data.get("msckf_min_update_rows", opts.msckf_options.min_update_rows)
        )
        opts.msckf_options.max_innovation_condition = float(
            data.get("msckf_max_innovation_condition", opts.msckf_options.max_innovation_condition)
        )
        opts.msckf_options.ekf_innovation_jitter = float(
            data.get("msckf_ekf_innovation_jitter", opts.msckf_options.ekf_innovation_jitter)
        )

    @staticmethod
    def _load_imu_config(path: str, opts: VioManagerOptions):
        data = ConfigLoader._read_yaml_safe(path)
        if not data: return

        # Usually inside 'imu0' key
        imu0 = data.get("imu0", {})

        # IMU model (kalibr/calibrated/rpng)
        from vio_pipeline_v2.msckf.State import StateOptions
        model_str = str(imu0.get("model", "kalibr")).lower()
        if model_str in ["kalibr", "calibrated"]:
            opts.state_options.imu_model = StateOptions.ImuModel.KALIBR
        elif model_str == "rpng":
            opts.state_options.imu_model = StateOptions.ImuModel.RPNG
        
        opts.imu_noises.sigma_a = float(imu0.get("accelerometer_noise_density", 0.01))
        opts.imu_noises.sigma_w = float(imu0.get("gyroscope_noise_density", 0.001))
        opts.imu_noises.sigma_ab = float(imu0.get("accelerometer_random_walk", 1.0e-4))
        opts.imu_noises.sigma_wb = float(imu0.get("gyroscope_random_walk", 1.0e-5))
        
        # Recalculate squared values
        opts.imu_noises.sigma_w_2 = opts.imu_noises.sigma_w**2
        opts.imu_noises.sigma_a_2 = opts.imu_noises.sigma_a**2
        opts.imu_noises.sigma_wb_2 = opts.imu_noises.sigma_wb**2
        opts.imu_noises.sigma_ab_2 = opts.imu_noises.sigma_ab**2

        # IMU intrinsics: Tw, Ta, R_IMUtoGYRO, R_IMUtoACC, Tg
        # C++ loads these and computes Dw = inv(Tw), Da = inv(Ta)
        Tw = np.eye(3)
        Ta = np.eye(3)
        R_IMUtoGYRO = np.eye(3)
        R_IMUtoACC = np.eye(3)
        Tg = np.zeros((3, 3))

        if "Tw" in imu0:
            Tw = np.array(imu0["Tw"], dtype=np.float64)
        if "Ta" in imu0:
            Ta = np.array(imu0["Ta"], dtype=np.float64)
        if "R_IMUtoGYRO" in imu0:
            R_IMUtoGYRO = np.array(imu0["R_IMUtoGYRO"], dtype=np.float64)
        if "R_IMUtoACC" in imu0:
            R_IMUtoACC = np.array(imu0["R_IMUtoACC"], dtype=np.float64)
        if "Tg" in imu0:
            Tg = np.array(imu0["Tg"], dtype=np.float64)

        # Compute Dw = inv(Tw), Da = inv(Ta)
        Dw = np.linalg.solve(Tw, np.eye(3))
        Da = np.linalg.solve(Ta, np.eye(3))
        R_ACCtoIMU = R_IMUtoACC.T
        R_GYROtoIMU = R_IMUtoGYRO.T

        # Extract vectors from matrices (matching C++ logic)
        # KALIBR model: lower-triangular column-wise
        # RPNG model: upper-triangular column-wise
        # For KALIBR: vec = [D(0,0), D(1,0), D(2,0), D(1,1), D(2,1), D(2,2)]
        # For RPNG: vec = [D(0,0), D(0,1), D(1,1), D(0,2), D(1,2), D(2,2)]
        if opts.state_options.imu_model == StateOptions.ImuModel.KALIBR:
            opts.vec_dw = np.array([Dw[0,0], Dw[1,0], Dw[2,0], Dw[1,1], Dw[2,1], Dw[2,2]])
            opts.vec_da = np.array([Da[0,0], Da[1,0], Da[2,0], Da[1,1], Da[2,1], Da[2,2]])
        else:
            opts.vec_dw = np.array([Dw[0,0], Dw[0,1], Dw[1,1], Dw[0,2], Dw[1,2], Dw[2,2]])
            opts.vec_da = np.array([Da[0,0], Da[0,1], Da[1,1], Da[0,2], Da[1,2], Da[2,2]])

        # Tg vector: column-wise flattening
        opts.vec_tg = np.array([Tg[0,0], Tg[1,0], Tg[2,0],
                                Tg[0,1], Tg[1,1], Tg[2,1],
                                Tg[0,2], Tg[1,2], Tg[2,2]])

        # Rotation quaternions
        opts.q_GYROtoIMU = rot_2_quat(R_GYROtoIMU)
        opts.q_ACCtoIMU = rot_2_quat(R_ACCtoIMU)

    @staticmethod
    def _load_camera_config(path: str, opts: VioManagerOptions):
        data = ConfigLoader._read_yaml_safe(path)
        if not data: return

        opts.camera_intrinsics = {}
        opts.camera_extrinsics = {}

        for i in range(opts.num_cameras):
            cam_key = f"cam{i}"
            if cam_key not in data:
                print(f"[WARN] Camera config {cam_key} not found")
                continue
            
            cam_data = data[cam_key]

            # 1. Intrinsics
            intr = cam_data.get("intrinsics", [0,0,0,0])
            dist = cam_data.get("distortion_coeffs", [0,0,0,0])
            res = cam_data.get("resolution", [640, 480])
            w, h = int(res[0]), int(res[1])
            
            # Combine into 8-param vector
            # Ensure dist has 4 params
            while len(dist) < 4: dist.append(0.0)
            params_vec = np.array(intr + dist[:4], dtype=np.float64)
            
            model_type = cam_data.get("distortion_model", "radtan")
            
            if model_type == "radtan":
                cam_obj = CamRadtan(w, h)
            elif model_type == "equidistant":
                cam_obj = CamEqui(w, h)
            else:
                cam_obj = CamRadtan(w, h) # Default
            
            cam_obj.set_value(params_vec)
            opts.camera_intrinsics[i] = cam_obj

            # 2. Extrinsics (T_cam_imu)
            # Standard YAML parses nested lists automatically into Python lists
            T_list = cam_data.get("T_cam_imu") 
            if T_list is None:
                print(f"\033[91m[ERROR] {cam_key} missing T_cam_imu!\033[0m")
                continue

            T_mat = np.array(T_list) # Convert to numpy 4x4
            
            R = T_mat[0:3, 0:3]
            p = T_mat[0:3, 3]
            
            # Convert Rotation Matrix to JPL Quaternion
            q_xyzw = rot_2_quat(R)
            
            pose = PoseJPL()
            pose.set_value(np.concatenate((q_xyzw, p)))
            opts.camera_extrinsics[i] = pose
            
            # 3. Time offset
            if i == 0:
                opts.calib_camimu_dt = float(cam_data.get("timeshift_cam_imu", 0.0))
                opts.init_options.init_calib_camImu_dt = opts.calib_camimu_dt