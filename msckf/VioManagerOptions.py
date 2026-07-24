import numpy as np
import os
import sys
from typing import Dict, Any, List, Optional, Union

# --- Project Imports ---
from vio_pipeline_v2.msckf.StateOptions import StateOptions
from vio_pipeline_v2.msckf.UpdaterOptions import UpdaterOptions
from vio_pipeline_v2.msckf.NoiseManager import NoiseManager
from vio_pipeline_v2.initialize.InertialInitializerOptions import InertialInitializerOptions
from vio_pipeline_v2.core.feat_initializer_options import FeatureInitializerOptions
from vio_pipeline_v2.core.CamEqui import CamEqui
from vio_pipeline_v2.core.cam_radtan import CamRadtan
from vio_pipeline_v2.type.LandmarkRepresentation import LandmarkRepresentation 
from vio_pipeline_v2.type.quat_ops import rot_2_quat, quat_2_Rot

# Colors for terminal output
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

class VioManagerOptions:
    """
    Struct which stores all options needed for state estimation.
    Broken into: Estimator, Trackers, Noise, State, and Simulation.
    """
    def __init__(self):
        
        # ======================================================================
        # ESTIMATOR
        # ======================================================================
        
        # Core state options (Directly instantiated, like in C++)
        self.state_options = StateOptions()
        
        # State initialization options
        self.init_options = InertialInitializerOptions()
        
        # Delay (seconds) before starting SLAM features
        self.dt_slam_delay = 2.0
        
        # Zero Velocity Update (ZUPT) flags
        self.try_zupt = False
        self.zupt_max_velocity = 1.0
        self.zupt_noise_multiplier = 1.0
        self.zupt_max_disparity = 1.0
        self.zupt_only_at_beginning = False
        
        # Timing recording
        self.record_timing_information = False
        self.record_timing_filepath = "ov_msckf_timing.txt"

        # ======================================================================
        # NOISE / CHI2
        # ======================================================================
        
        self.imu_noises = NoiseManager()
        self.msckf_options = UpdaterOptions()
        self.slam_options = UpdaterOptions()
        self.aruco_options = UpdaterOptions()
        self.zupt_options = UpdaterOptions()

        # ======================================================================
        # STATE DEFAULTS
        # ======================================================================
        
        self.gravity_mag = 9.81
        
        # IMU intrinsics (vectors)
        # Default to identity (lower-triangular: [1,0,0,1,0,1])
        self.vec_dw = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        self.vec_da = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        self.vec_tg = np.zeros(9)
        
        # IMU Rotations (Quaternions [x,y,z,w])
        self.q_ACCtoIMU = np.array([0., 0., 0., 1.])
        self.q_GYROtoIMU = np.array([0., 0., 0., 1.])
        
        self.calib_camimu_dt = 0.0
        
        # Camera Intrinsics/Extrinsics containers
        # In Python, we store these here for the Loader to populate, 
        # but they MUST be synced to self.state_options later.
        self.camera_intrinsics = {} # Dict[int, CamBase]
        self.camera_extrinsics = {} # Dict[int, PoseJPL]
        
        self.use_mask = False
        self.masks = {} # Dict[int, cv2.Mat]

        # ======================================================================
        # TRACKERS
        # ======================================================================
        
        self.use_stereo = True
        self.use_klt = True
        self.use_aruco = True
        self.downsize_aruco = True
        self.downsample_cameras = False
        
        self.num_opencv_threads = 4
        self.use_multi_threading_pubs = True
        self.use_multi_threading_subs = False
        
        self.num_pts = 150
        self.fast_threshold = 20
        self.grid_x = 5
        self.grid_y = 5
        self.min_px_dist = 10
        
        # Histogram method: 0=NONE, 1=HISTOGRAM, 2=CLAHE
        self.histogram_method = 1 
        
        self.knn_ratio = 0.85
        self.track_frequency = 20.0
        
        self.featinit_options = FeatureInitializerOptions()

        # ======================================================================
        # HOVERING MODE SWITCH (FIFO <-> LIFO)
        # ======================================================================
        self.sliding_window_size = 12
        # Conservative defaults: reduce hover-mode flapping in dynamic motion.
        self.hovering_threshold = 3e-5
        self.hover_smoothing_required = 4
        self.hover_baseline_threshold = 0.08
        self.hover_epipolar_inlier_threshold = 6e-2
        self.hover_bearing_inlier_threshold = 6e-2
        self.hover_min_inliers = 6
        self.hover_entry_max_baseline = 0.30
        self.hover_entry_max_velocity = 0.25
        self.hover_exit_velocity_threshold = 0.5
        self.hover_max_duration_sec = 3.0
        self.lifo_backward_match_threshold = 0.03
        self.lifo_backward_min_chain_length = 2
        self.lifo_backward_reproj_threshold = 0.06
        self.lifo_backward_mutual_consistency = True
        self.lifo_backward_max_age_steps = 4
        self.lifo_orb_descriptor_fallback = True
        self.lifo_orb_hamming_threshold = 45
        self.lifo_quality_init = 0.5
        self.lifo_quality_inc_match = 0.12
        self.lifo_quality_inc_orb = 0.06
        self.lifo_quality_dec_miss = 0.15
        self.lifo_quality_min_keep = 0.2
        self.lifo_quality_min_output = 0.35
        self.lifo_max_misses = 4

        # ======================================================================
        # SIMULATOR
        # ======================================================================
        
        self.sim_seed_state_init = 0
        self.sim_seed_preturb = 0
        self.sim_seed_measurements = 0
        
        self.sim_do_perturbation = False
        self.sim_traj_path = "src/open_vins/ov_data/sim/udel_gore.txt"
        self.sim_distance_threshold = 1.2
        
        self.sim_freq_cam = 10.0
        self.sim_freq_imu = 400.0
        
        self.sim_min_feature_gen_distance = 5
        self.sim_max_feature_gen_distance = 10

    # ======================================================================
    # HELPER PROPERTIES (To ensure access consistency)
    # ======================================================================
    
    # In C++, state_options is public. In Python, we want to ensure that whenever
    # someone accesses it, it has the latest calibration data loaded into it.
    
    @property
    def num_cameras(self):
        return self.state_options.num_cameras

    @num_cameras.setter
    def num_cameras(self, value):
        self.state_options.num_cameras = value

    def sync_state_options(self):
        """
        Explicitly syncs the loaded calibration data into state_options.
        Must be called after ConfigLoader is done.
        """
        self.state_options.camera_intrinsics = self.camera_intrinsics
        self.state_options.camera_extrinsics = self.camera_extrinsics
        self.state_options.rot_ACCtoIMU = self.q_ACCtoIMU
        self.state_options.rot_GYROtoIMU = self.q_GYROtoIMU
        self.state_options.imu_intrinsics = {
            "dw": self.vec_dw,
            "da": self.vec_da,
            "tg": self.vec_tg
        }
        # In C++, imu_model is set in StateOptions, we default to KALIBR
        self.state_options.imu_model = StateOptions.ImuModel.KALIBR

    def print_and_load(self, parser: Optional[Dict[str, Any]] = None):
        """
        Loads parameters and prints them.
        """
        self.print_and_load_estimator(parser)
        self.print_and_load_trackers(parser)
        self.print_and_load_noise(parser)
        
        # needs to be called last to use num_cameras etc.
        self.print_and_load_state(parser)
        
        # CRITICAL: Sync data after loading everything
        self.sync_state_options()

    def print_and_load_estimator(self, parser: Optional[Dict[str, Any]] = None):
        print("ESTIMATOR PARAMETERS:")
        # In C++: state_options.print()
        # You should implement a print method in StateOptions.py if you want details
        self.init_options.print_and_load_initializer(parser)
        
        if parser is not None:
            self.dt_slam_delay = parser.get("dt_slam_delay", self.dt_slam_delay)
            self.try_zupt = parser.get("try_zupt", self.try_zupt)
            self.zupt_max_velocity = parser.get("zupt_max_velocity", self.zupt_max_velocity)
            self.zupt_noise_multiplier = parser.get("zupt_noise_multiplier", self.zupt_noise_multiplier)
            self.zupt_max_disparity = parser.get("zupt_max_disparity", self.zupt_max_disparity)
            self.zupt_only_at_beginning = parser.get("zupt_only_at_beginning", self.zupt_only_at_beginning)
            self.record_timing_information = parser.get("record_timing_information", self.record_timing_information)
            self.record_timing_filepath = parser.get("record_timing_filepath", self.record_timing_filepath)

        print(f"  - dt_slam_delay: {self.dt_slam_delay:.1f}")
        print(f"  - zero_velocity_update: {int(self.try_zupt)}")
        print(f"  - zupt_max_velocity: {self.zupt_max_velocity:.2f}")
        print(f"  - zupt_noise_multiplier: {self.zupt_noise_multiplier:.2f}")
        print(f"  - zupt_max_disparity: {self.zupt_max_disparity:.4f}")
        print(f"  - zupt_only_at_beginning?: {int(self.zupt_only_at_beginning)}")
        print(f"  - record timing?: {int(self.record_timing_information)}")
        print(f"  - record timing filepath: {self.record_timing_filepath}")

    def print_and_load_noise(self, parser: Optional[Dict[str, Any]] = None):
        print("NOISE PARAMETERS:")
        # Logic is handled by ConfigLoader directly populating imu_noises
        self.imu_noises.print()
        
        print("  Updater MSCKF Feats:")
        self.msckf_options.print()
        print("  Updater SLAM Feats:")
        self.slam_options.print()
        print("  Updater ARUCO Tags:")
        self.aruco_options.print()
        print("  Updater ZUPT:")
        self.zupt_options.print()

    def print_and_load_state(self, parser: Optional[Dict[str, Any]] = None):
        print("STATE PARAMETERS:")
        print(f"  - gravity_mag: {self.gravity_mag:.4f}")
        print(f"  - gravity: 0.0, 0.0, {self.gravity_mag:.3f}")
        print(f"  - use_fej: {self.state_options.do_fej}")
        print(f"  - integration: {self.state_options.integration_method.name}")
        print(f"  - calib_cam_timeoffset: {self.state_options.do_calib_camera_timeoffset}")
        print(f"  - camera masks?: {int(self.use_mask)}")
        
        if len(self.camera_intrinsics) != self.state_options.num_cameras:
            print(f"{RED}[SIM]: camera calib size mismatch{RESET}")
            # Don't exit here if simple print, but warning is good
            # sys.exit(1) 
            pass
            
        print(f"  - calib_camimu_dt: {self.calib_camimu_dt:.4f}")
        print("CAMERA PARAMETERS:")
        for n in range(self.state_options.num_cameras):
            if n not in self.camera_intrinsics: continue
            
            cam = self.camera_intrinsics[n]
            ext = self.camera_extrinsics[n] # PoseJPL object
            
            print(f"cam_{n}_fisheye: {isinstance(cam, CamEqui)}")
            print(f"cam_{n}_wh: {cam.w()} x {cam.h()}")
            print(f"cam_{n}_intrinsic: {cam.get_value().flatten()}")
            
            # Safe access for PoseJPL
            q = ext.quat()
            p = ext.pos()
            print(f"cam_{n}_extrinsic(q): {q}")
            print(f"cam_{n}_extrinsic(p): {p}")
            
            # Print Transformation Matrix
            T = np.eye(4)
            T[0:3, 0:3] = quat_2_Rot(q) # R_ItoC
            T[0:3, 3] = p
            
            # OpenVINS C++ prints T_CtoI (Inverse)
            # R_CtoI = R_ItoC^T
            # p_CinI = -R^T * p
            R_CtoI = T[0:3, 0:3].T
            p_CinI = -R_CtoI @ p
            
            T_display = np.eye(4)
            T_display[0:3, 0:3] = R_CtoI
            T_display[0:3, 3] = p_CinI
            
            print(f"T_C{n}toI:\n{T_display}\n")

        print("IMU PARAMETERS:")
        # Assuming KALIBR model for now
        print(f"imu model: KALIBR")
        print(f"Dw: {self.vec_dw}")
        print(f"Da: {self.vec_da}")
        print(f"Tg: {self.vec_tg}")
        print(f"q_GYROtoI: {self.q_GYROtoIMU}")
        print(f"q_ACCtoI: {self.q_ACCtoIMU}")

    def print_and_load_trackers(self, parser: Optional[Dict[str, Any]] = None):
        print("FEATURE TRACKING PARAMETERS:")
        print(f"  - use_stereo: {int(self.use_stereo)}")
        print(f"  - use_klt: {int(self.use_klt)}")
        print(f"  - use_aruco: {int(self.use_aruco)}")
        print(f"  - downsize aruco: {int(self.downsize_aruco)}")
        print(f"  - downsize cameras: {int(self.downsample_cameras)}")
        print(f"  - num opencv threads: {self.num_opencv_threads}")
        print(f"  - use multi-threading pubs: {int(self.use_multi_threading_pubs)}")
        print(f"  - use multi-threading subs: {int(self.use_multi_threading_subs)}")
        print(f"  - num_pts: {self.num_pts}")
        print(f"  - fast threshold: {self.fast_threshold}")
        print(f"  - grid X by Y: {self.grid_x} by {self.grid_y}")
        print(f"  - min px dist: {self.min_px_dist}")
        print(f"  - hist method: {self.histogram_method}")
        print(f"  - knn ratio: {self.knn_ratio:.3f}")
        print(f"  - track frequency: {self.track_frequency:.1f}")
        print(f"  - sliding_window_size: {self.sliding_window_size}")
        print(f"  - hovering_threshold: {self.hovering_threshold:.6f}")
        print(f"  - hover_smoothing_required: {self.hover_smoothing_required}")
        print(f"  - hover_baseline_threshold: {self.hover_baseline_threshold:.6f}")
        print(f"  - hover_epipolar_inlier_threshold: {self.hover_epipolar_inlier_threshold:.6f}")
        print(f"  - hover_bearing_inlier_threshold: {self.hover_bearing_inlier_threshold:.6f}")
        print(f"  - hover_min_inliers: {self.hover_min_inliers}")
        print(f"  - hover_entry_max_baseline: {self.hover_entry_max_baseline:.3f}")
        print(f"  - hover_entry_max_velocity: {self.hover_entry_max_velocity:.3f}")
        print(f"  - hover_exit_velocity_threshold: {self.hover_exit_velocity_threshold:.3f}")
        print(f"  - hover_max_duration_sec: {self.hover_max_duration_sec:.2f}")
        print(f"  - lifo_backward_match_threshold: {self.lifo_backward_match_threshold:.4f}")
        print(f"  - lifo_backward_min_chain_length: {self.lifo_backward_min_chain_length}")
        print(f"  - lifo_backward_reproj_threshold: {self.lifo_backward_reproj_threshold:.4f}")
        print(f"  - lifo_backward_mutual_consistency: {int(self.lifo_backward_mutual_consistency)}")
        print(f"  - lifo_backward_max_age_steps: {self.lifo_backward_max_age_steps}")
        print(f"  - lifo_orb_descriptor_fallback: {int(self.lifo_orb_descriptor_fallback)}")
        print(f"  - lifo_orb_hamming_threshold: {self.lifo_orb_hamming_threshold}")
        print(f"  - lifo_quality_min_keep: {self.lifo_quality_min_keep:.3f}")
        print(f"  - lifo_quality_min_output: {self.lifo_quality_min_output:.3f}")
        print(f"  - lifo_max_misses: {self.lifo_max_misses}")
        self.featinit_options.print(parser)

    def print_and_load_simulation(self, parser: Optional[Dict[str, Any]] = None):
        print("SIMULATION PARAMETERS:")
        print(f"  - traj path: {self.sim_traj_path}")