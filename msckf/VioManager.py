import numpy as np
import threading
import time
import sys
import os
import cv2
import copy
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
from scipy.optimize import linear_sum_assignment

from sklearn import covariance

# --- Imports from your project structure ---
from vio_pipeline_v2.msckf.VioManagerOptions import VioManagerOptions
from vio_pipeline_v2.msckf.State import State
from vio_pipeline_v2.msckf.state_helper import StateHelper
from vio_pipeline_v2.msckf.Propagator import Propagator
from vio_pipeline_v2.core import FeatureDatabase, feature, TrackKLT, TrackBase, CameraData, ImuData

from vio_pipeline_v2.msckf.UpdaterMSCKF import UpdaterMSCKF
from vio_pipeline_v2.msckf.UpdaterSLAM import UpdaterSLAM
from vio_pipeline_v2.msckf.update_zero_velocity import UpdaterZeroVelocity
from vio_pipeline_v2.type import LandmarkRepresentation, skew_x, rot_2_quat, quat_2_Rot

from vio_pipeline_v2.initialize import InertialInitializer

TrackDescriptor = None
TrackAruco = None
TrackSIM = None
GREEN = "\033[32m"
RESET = "\033[0m"
class VioManager:
    """
    Core class that manages the entire system.
    """
    def __init__(self, params_: VioManagerOptions):
        
        # Nice startup message
        print("=======================================")
        print("OPENVINS ON-MANIFOLD EKF IS STARTING")
        print("=======================================")

        self.params = params_
        self.params.print_and_load_estimator()
        self.params.print_and_load_noise()
        self.params.print_and_load_state()
        self.params.print_and_load_trackers()

        # Globally set thread count
        cv2.setNumThreads(self.params.num_opencv_threads)
        cv2.setRNGSeed(0)

        # Create the state
        self.state = State(self.params.state_options)

        # Set IMU intrinsics
        self.state._calib_imu_dw.set_value(self.params.vec_dw)
        self.state._calib_imu_dw.set_fej(self.params.vec_dw)
        self.state._calib_imu_da.set_value(self.params.vec_da)
        self.state._calib_imu_da.set_fej(self.params.vec_da)
        self.state._calib_imu_tg.set_value(self.params.vec_tg)
        self.state._calib_imu_tg.set_fej(self.params.vec_tg)
        self.state._calib_imu_GYROtoIMU.set_value(self.params.q_GYROtoIMU)
        self.state._calib_imu_GYROtoIMU.set_fej(self.params.q_GYROtoIMU)
        self.state._calib_imu_ACCtoIMU.set_value(self.params.q_ACCtoIMU)
        self.state._calib_imu_ACCtoIMU.set_fej(self.params.q_ACCtoIMU)

        # Timeoffset from camera to IMU
        temp_camimu_dt = np.array([self.params.calib_camimu_dt])
        self.state._calib_dt_CAMtoIMU.set_value(temp_camimu_dt)
        self.state._calib_dt_CAMtoIMU.set_fej(temp_camimu_dt)

        # Loop through and load each of the cameras
        self.state._cam_intrinsics_cameras = self.params.camera_intrinsics
        for i in range(self.state._options.num_cameras):
            self.state._cam_intrinsics[i].set_value(self.params.camera_intrinsics[i].get_value())
            self.state._cam_intrinsics[i].set_fej(self.params.camera_intrinsics[i].get_value())
            self.state._calib_IMUtoCAM[i].set_value(self.params.camera_extrinsics[i].value())
            self.state._calib_IMUtoCAM[i].set_fej(self.params.camera_extrinsics[i].value())

        # Logging / Stats
        self.of_statistics = None
        if self.params.record_timing_information:
            if os.path.exists(self.params.record_timing_filepath):
                os.remove(self.params.record_timing_filepath)
                print(f"[STATS]: found old file {self.params.record_timing_filepath}, deleted...")
            
            os.makedirs(os.path.dirname(self.params.record_timing_filepath), exist_ok=True)
            self.of_statistics = open(self.params.record_timing_filepath, "w")
            header = "# timestamp (sec),tracking,propagation,msckf update,"
            if self.state._options.max_slam_features > 0:
                header += "slam update,slam delayed,"
            header += "re-tri & marg,total\n"
            self.of_statistics.write(header)

        # Feature Extractor
        init_max_features = int(np.floor(self.params.init_options.init_max_features / self.params.state_options.num_cameras))
        
        if self.params.use_klt:
            print("[VIO MANAGER]: Using KLT Feature Tracker")
            self.trackFEATS = TrackKLT(
                self.state._cam_intrinsics_cameras, init_max_features,
                self.state._options.max_aruco_features, self.params.use_stereo, self.params.histogram_method,
                self.params.fast_threshold, self.params.grid_x, self.params.grid_y, self.params.min_px_dist
            )
        else:
            if TrackDescriptor:
                self.trackFEATS = TrackDescriptor(
                    self.state._cam_intrinsics_cameras, init_max_features, self.state._options.max_aruco_features,
                    self.params.use_stereo, self.params.histogram_method, self.params.fast_threshold,
                    self.params.grid_x, self.params.grid_y, self.params.min_px_dist, self.params.knn_ratio
                )
            else:
                raise ImportError("TrackDescriptor class not available")

        # ArUco Tracker
        self.trackARUCO = None
        if self.params.use_aruco:
            if TrackAruco:
                self.trackARUCO = TrackAruco(
                    self.state._cam_intrinsics_cameras, self.state._options.max_aruco_features,
                    self.params.use_stereo, self.params.histogram_method, self.params.downsize_aruco
                )
            else:
                print("[WARN]: TrackAruco class not available")

        # Propagator
        self.propagator = Propagator(self.params.imu_noises, self.params.gravity_mag)

        # Initializer
        self.initializer = InertialInitializer(self.params.init_options, self.trackFEATS.get_feature_database())

        # Updaters
        self.updaterMSCKF = UpdaterMSCKF(self.params.msckf_options, self.params.featinit_options)
        self.updaterSLAM = UpdaterSLAM(self.params.slam_options, self.params.aruco_options, self.params.featinit_options)

        self.updaterZUPT = None
        if self.params.try_zupt:
            self.updaterZUPT = UpdaterZeroVelocity(
                self.params.zupt_options, self.params.imu_noises, self.trackFEATS.get_feature_database(),
                self.propagator, self.params.gravity_mag, self.params.zupt_max_velocity,
                self.params.zupt_noise_multiplier, self.params.zupt_max_disparity
            )

        # Internal State Variables
        self.is_initialized_vio = False
        self.startup_time = -1.0
        self.timelastupdate = -1.0
        self.distance = 0.0
        
        self.thread_init_running = False
        self.thread_init_success = False
        self.camera_queue_init = []
        self.camera_queue_init_mtx = threading.Lock()

        self.did_zupt_update = False
        self.has_moved_since_zupt = False
        self.first_rosbag_time = -1.0
        
        self.good_features_MSCKF = []
        
        # Viz Data
        self.active_tracks_time = -1.0
        self.active_tracks_posinG = {}
        self.active_tracks_uvd = {}
        self.active_image = None
        self.active_feat_linsys_A = {}
        self.active_feat_linsys_b = {}
        self.active_feat_linsys_count = {}

        self.rT1, self.rT2, self.rT3, self.rT4, self.rT5, self.rT6, self.rT7 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Hovering mode control (FIFO by default)
        self.mode = "FIFO"
        self.hover_start_step = None
        self.hover_poses_fixed = {}
        self.covariance_updated = False
        self._hover_feature_measurements = {}
        self._hover_decisions = deque(maxlen=max(1, int(self.params.hover_smoothing_required)))
        self._last_hover_decision = 0
        self._last_camera_timestamp = None
        self._last_camera_imu_rot = None
        self._last_camera_imu_pos = None
        self._last_camera_imu_ts = None

        # LIFO backward-tracking chain store
        self._lifo_backward_chains = {}
        self._lifo_next_chain_id = 1
        self._lifo_orb = cv2.ORB_create(nfeatures=1500)

    def feed_measurement_imu(self, message: ImuData):
        if self.first_rosbag_time < 0:
            self.first_rosbag_time = message.timestamp
            
        oldest_time = self.state.margtimestep()
        if oldest_time > self.state._timestamp:
            oldest_time = -1.0
        
        if not self.is_initialized_vio:
            oldest_time = message.timestamp - self.params.init_options.init_window_time + self.state._calib_dt_CAMtoIMU.value()[0] - 0.10

        self.propagator.feed_imu(message, oldest_time)
        # print("chal rha hai")
        if not self.is_initialized_vio and not self.thread_init_running:
            self.initializer.feed_imu(message, oldest_time)

        if self.is_initialized_vio and self.updaterZUPT and (not self.params.zupt_only_at_beginning or not self.has_moved_since_zupt):
            self.updaterZUPT.feed_imu(message, oldest_time)

    def feed_measurement_camera(self, message: CameraData):
        # print("checkpoint 2")
        # print(f'{message=}')
        self.track_image_and_update(message)

    def feed_measurement_camera_tracks(self, timestamp, tracks, track_count):
        # 1. Update feature database
        db = self.trackFEATS.get_feature_database()
        from vio_pipeline_v2.msckf.state_helper import StateHelper
        for i in range(track_count):
            tr = tracks[i]
            for cam_id in tr.timestamps.keys():
                for idx, t in enumerate(tr.timestamps[cam_id]):
                    db.update_feature(tr.featid, t, cam_id,
                                      tr.uvs[cam_id][idx][0], tr.uvs[cam_id][idx][1],
                                      tr.uvs_norm[cam_id][idx][0], tr.uvs_norm[cam_id][idx][1])
                
        # 2. Try to initialize if not yet initialized
        if not self.is_initialized_vio:
            ts = [0.0]
            cov = []
            order = []
            success = self.initializer.initialize(ts, cov, order, self.state._imu, False)
            if success:
                StateHelper.set_initial_covariance(self.state, cov[0], order)
                self.state._timestamp = ts[0]
                self.startup_time = ts[0]
                db.cleanup_measurements(self.state._timestamp)
                self.is_initialized_vio = True
                print(f"[VioManager]: Initialization succeeded at time {self.state._timestamp}")
            return
            
        # 3. Try Zero Velocity Update (ZUPT)
        zupt_success = False
        if self.updaterZUPT:
            zupt_success = self.updaterZUPT.try_update(self.state, timestamp)
            
        if not zupt_success:
            # Standard MSCKF step: Propagate IMU and augment clone
            self.propagator.propagate_and_clone(self.state, timestamp)
            
            # Find features ready for EKF update (lost tracks or window overflow)
            update_features = []
            db_data = db.get_internal_data()
            for feat_id, feat in list(db_data.items()):
                is_lost = True
                for cam_id, times in feat.timestamps.items():
                    if timestamp in times:
                        is_lost = False
                        break
                if is_lost or len(self.state._clones) >= self.state._options.max_clone_size:
                    if feat.num_measurements() >= 2:
                        update_features.append(feat)
                        
            if len(update_features) > 0:
                self.updaterMSCKF.update(self.state, update_features)
                db.cleanup()
                
            # Marginalize old clones if exceeding sliding window size
            StateHelper.marginalize_old_clone(self.state)
            
        # Clean old IMU readings from cache
        oldest_clone_time = self.state.margtimestep()
        if oldest_clone_time != -1.0:
            self.propagator.clean_old_imu_measurements(oldest_clone_time - 0.5)

    def feed_measurement_simulation(self, timestamp: float, camids: List[int], feats: List[List[Tuple[int, np.ndarray]]]):
        # Start timing
        self.rT1 = time.time()

        # Check if we have simulation tracker
        if TrackSIM is not None and not isinstance(self.trackFEATS, TrackSIM):
            # Replace with simulated tracker
            self.trackFEATS = TrackSIM(self.state._cam_intrinsics_cameras, self.state._options.max_aruco_features)
            # Re-link initializer and zupt
            self.initializer = InertialInitializer(self.params.init_options, self.trackFEATS.get_feature_database())
            if self.updaterZUPT:
                self.updaterZUPT.feature_db = self.trackFEATS.get_feature_database()
            print("[SIM]: Casting our tracker to a TrackSIM object!")

        # Feed simulation tracker
        if hasattr(self.trackFEATS, 'feed_measurement_simulation'):
            self.trackFEATS.feed_measurement_simulation(timestamp, camids, feats)
        
        self.rT2 = time.time()

        # ZUPT Logic
        if self.is_initialized_vio and self.updaterZUPT and (not self.params.zupt_only_at_beginning or not self.has_moved_since_zupt):
            if self.state._timestamp != timestamp:
                self.did_zupt_update = self.updaterZUPT.try_update(self.state, timestamp)
            
            if self.did_zupt_update:
                clean_t = timestamp + self.state._calib_dt_CAMtoIMU.value()[0] - 0.10
                self.propagator.clean_old_imu_measurements(clean_t)
                self.updaterZUPT.clean_old_imu_measurements(clean_t)
                self.propagator.invalidate_cache()
                return

        if not self.is_initialized_vio:
            print("[SIM]: System must be initialized before simulating features!")
            sys.exit(1)

        # Construct dummy camera message for update logic
        message = CameraData()
        message.timestamp = timestamp
        for cid in camids:
            w = self.state._cam_intrinsics_cameras[cid].w()
            h = self.state._cam_intrinsics_cameras[cid].h()
            message.sensor_ids.append(cid)
            message.images.append(np.zeros((h, w), dtype=np.uint8))
            message.masks.append(np.zeros((h, w), dtype=np.uint8))
        
        self.do_feature_propagate_update(message)

    def initialize_with_gt(self, imustate: np.ndarray):
        # Initialize system [time, q, p, v, bg, ba]
        # imustate size 17. Block(1,0,16,1)
        self.state._imu.set_value(imustate[1:17])
        self.state._imu.set_fej(imustate[1:17])

        # Set Covariance
        order = [self.state._imu]
        Cov = (0.02**2) * np.eye(self.state._imu.size())
        Cov[0:3, 0:3] = (0.017**2) * np.eye(3)
        Cov[3:6, 3:6] = (0.05**2) * np.eye(3)
        Cov[6:9, 6:9] = (0.01**2) * np.eye(3)
        StateHelper.set_initial_covariance(self.state, Cov, order)

        # Set time
        self.state._timestamp = imustate[0]
        self.startup_time = imustate[0]
        self.is_initialized_vio = True

        # Cleanup
        self.trackFEATS.get_feature_database().cleanup_measurements(self.state._timestamp)
        if self.trackARUCO:
            self.trackARUCO.get_feature_database().cleanup_measurements(self.state._timestamp)

        print("[INIT]: INITIALIZED FROM GROUNDTRUTH FILE!!!!!")

    def try_to_initialize(self, message: CameraData) -> bool:
        if self.thread_init_running:
            with self.camera_queue_init_mtx:
                self.camera_queue_init.append(message.timestamp)
            return False

        if self.thread_init_success:
            return True

        self.thread_init_running = True
        
        def init_worker():
            init_rt1 = time.time()
            wait_for_jerk = (self.updaterZUPT is None)
            
            # C++ signature: initialize(double &timestamp, MatrixXd &covariance, vector<Type*> &order, shared_ptr<IMU> imu, bool wait_for_jerk)
            # Python initializer returns tuple: (success, timestamp, covariance, order)
            ts = [0.0]
            order = []
            cov = list()

            success = self.initializer.initialize(ts, cov, order, self.state._imu, wait_for_jerk)
            if success:
                bg_norm = float(np.linalg.norm(self.state._imu.bias_g()))
                ba_norm = float(np.linalg.norm(self.state._imu.bias_a()))
                max_bg = 0.35  # rad/s
                max_ba = 3.5   # m/s^2
                if bg_norm > max_bg or ba_norm > max_ba:
                    print(
                        f"[init] rejected: implausible static biases "
                        f"|bg|={bg_norm:.3f} rad/s |ba|={ba_norm:.3f} m/s^2"
                    )
                    imu0 = np.zeros(16)
                    imu0[3] = 1.0
                    self.state._imu.set_value(imu0)
                    self.state._imu.set_fej(imu0)
                    self.thread_init_success = False
                    with self.camera_queue_init_mtx:
                        self.camera_queue_init.clear()
                    self.thread_init_running = False
                    return

                StateHelper.set_initial_covariance(self.state, cov, order)
                self.state._timestamp = ts[0]
                self.startup_time = ts[0]

                self.trackFEATS.get_feature_database().cleanup_measurements(self.state._timestamp)
                new_feats = int(np.floor(self.params.num_pts / self.params.state_options.num_cameras))
                self.trackFEATS.set_num_features(new_feats)
                
                if self.trackARUCO:
                    self.trackARUCO.get_feature_database().cleanup_measurements(self.state._timestamp)

                if np.linalg.norm(self.state._imu.vel()) > self.params.zupt_max_velocity:
                    self.has_moved_since_zupt = True
                    print(f"\n[ZUPT DEBUG] Exited ZUPT at init due to high initial velocity. Time relative to rosbag start: {self.state._timestamp - self.first_rosbag_time:.3f} sec\n")

                init_rT2 = time.time()
                print(f"[init]: successful initialization in {init_rT2 - init_rt1:.4f} seconds (rosbag relative time: {self.state._timestamp - self.first_rosbag_time:.3f} sec)")
                # print(f"{GREEN}[init]: successful initialization in {dt:.4f} seconds{RESET}")

                # 2. Print Orientation (Quaternion)
                q = self.state._imu.quat()
                print(f"{GREEN}[init]: orientation = {q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}{RESET}")

                # 3. Print Gyro Bias
                bg = self.state._imu.bias_g()
                print(f"{GREEN}[init]: bias gyro   = {bg[0]:.4f}, {bg[1]:.4f}, {bg[2]:.4f}{RESET}")

                # 4. Print Velocity
                v = self.state._imu.vel()
                print(f"{GREEN}[init]: velocity    = {v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}{RESET}")

                # 5. Print Accel Bias
                ba = self.state._imu.bias_a()
                print(f"{GREEN}[init]: bias accel  = {ba[0]:.4f}, {ba[1]:.4f}, {ba[2]:.4f}{RESET}")

                # 6. Print Position
                p = self.state._imu.pos()
                print(f"{GREEN}[init]: position    = {p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}{RESET}")

                # Catch up state
                with self.camera_queue_init_mtx:
                    times = [t for t in self.camera_queue_init if t > ts[0]]
                
                clone_rate = int(len(times) / self.params.state_options.max_clone_size) + 1
                for i in range(0, len(times), clone_rate):
                    self.propagator.propagate_and_clone(self.state, times[i])
                    StateHelper.marginalize_old_clone(self.state)
                
                print(f"[init]: moved the state forward {self.state._timestamp - ts[0]:.2f} seconds")
                self.thread_init_success = True
                with self.camera_queue_init_mtx: self.camera_queue_init.clear()
            else:
                self.thread_init_success = False
                with self.camera_queue_init_mtx: self.camera_queue_init.clear()
            
            self.thread_init_running = False

        if not self.params.use_multi_threading_subs:
            init_worker()
        else:
            t = threading.Thread(target=init_worker, daemon=True)
            t.start()

        return False

    def track_image_and_update(self, message_const: CameraData):
        self.rT1 = time.time()

        assert len(message_const.sensor_ids) > 0
        assert len(message_const.sensor_ids) == len(message_const.images)
        for i in range(len(message_const.sensor_ids) - 1):
            assert message_const.sensor_ids[i] != message_const.sensor_ids[i + 1], \
                f"Duplicate sensor_id at index {i}: {message_const.sensor_ids[i]}"
        
        # Deep copy for modification
        message = copy.deepcopy(message_const)
        
        if self.params.downsample_cameras:
            for i in range(len(message.sensor_ids)):
                message.images[i] = cv2.pyrDown(message.images[i])
                message.masks[i] = cv2.pyrDown(message.masks[i])

        self.trackFEATS.feed_new_camera(message)
        # print("checkpoint 6")
        if self.is_initialized_vio and self.trackARUCO:
            self.trackARUCO.feed_new_camera(message)
        
        self.rT2 = time.time()

        if self.is_initialized_vio and self.updaterZUPT and (not self.params.zupt_only_at_beginning or not self.has_moved_since_zupt):
            if self.state._timestamp != message.timestamp:
                self.did_zupt_update = self.updaterZUPT.try_update(self.state, message.timestamp)
            
            if self.did_zupt_update:
                assert self.state._timestamp == message.timestamp, \
                    f"ZUPT succeeded but state timestamp ({self.state._timestamp}) != image timestamp ({message.timestamp})"
                clean_t = message.timestamp + self.state._calib_dt_CAMtoIMU.value()[0] - 0.10
                self.propagator.clean_old_imu_measurements(clean_t)
                self.updaterZUPT.clean_old_imu_measurements(clean_t)
                self.propagator.invalidate_cache()
                return

        if not self.is_initialized_vio:
            self.is_initialized_vio = self.try_to_initialize(message)
            if not self.is_initialized_vio:
                return
            else: 
                self.initializer.Imu_data.clear() 

        self.do_feature_propagate_update(message)

    def do_feature_propagate_update(self, message: CameraData):
        if self.state._timestamp > message.timestamp:
            print(f"[WARN] Image received out of order (dt={message.timestamp - self.state._timestamp})")
            return

        prev_ts = self._last_camera_timestamp

        if self.state._timestamp != message.timestamp:
            if self.mode == "FIFO":
                self.propagator.propagate_and_clone(self.state, message.timestamp)
            else:
                self.propagator.propagate_only(self.state, message.timestamp)

        if self.state._timestamp != message.timestamp:
            print("[WARN] Propagator unable to propagate state forward!")
            return

        features_for_lifo = None

        if prev_ts is not None:
            # Hover decision is computed from raw tracker correspondences,
            # independent from backward-chain quality filtering.
            detect_features = self.trackFEATS.get_feature_database().features_containing(self.state._timestamp, False, True)
            xi_k, d_k, inlier_count, total_count, baseline = self.detect_hovering(detect_features, prev_ts, message.timestamp)
            xi_k = self.apply_temporal_smoothing(xi_k)
            vel_norm = float(np.linalg.norm(self.state._imu.vel()))

            # Entry guard: never switch FIFO->LIFO if estimated motion is already large.
            # This blocks false hover activation during filter divergence.
            entry_blocked = False
            if self.mode == "FIFO":
                if baseline > self.params.hover_entry_max_baseline:
                    entry_blocked = True
                if vel_norm > self.params.hover_entry_max_velocity:
                    entry_blocked = True
                if entry_blocked:
                    xi_k = 0
                    self._hover_decisions.clear()
                    self._hover_decisions.append(0)
                    self._last_hover_decision = 0

            # Hard safety exit: avoid getting stuck in LIFO under clear dynamic motion.
            hard_exit = False
            if self.mode == "LIFO" and vel_norm > self.params.hover_exit_velocity_threshold:
                hard_exit = True

            if self.mode == "LIFO" and self.hover_start_step is not None:
                hover_age = float(message.timestamp - self.hover_start_step)
                if hover_age > self.params.hover_max_duration_sec:
                    hard_exit = True

            if hard_exit:
                xi_k = 0

            print(
                f"[HOVER-DET] t={message.timestamp:.3f} xi={xi_k} d={d_k:.6f} "
                f"inliers={inlier_count}/{total_count} baseline={baseline:.4f} "
                f"vel={vel_norm:.3f} mode={self.mode} entry_blocked={int(entry_blocked)}"
            )

            if self.mode == "FIFO" and xi_k == 1:
                self.switch_to_LIFO()
            elif self.mode == "LIFO" and xi_k == 0:
                self.switch_to_FIFO()

        if self.mode == "LIFO":
            features_for_lifo = self.extract_and_track_BACKWARD(message, prev_ts)

        if self.mode == "FIFO":
            self.fifo_update(message)
        else:
            self.lifo_update(message, features_for_lifo)

        # Cache current IMU pose at camera-frame boundary for next hover decision.
        self._last_camera_imu_rot = self.state._imu.Rot().copy()
        self._last_camera_imu_pos = self.state._imu.pos().copy()
        self._last_camera_imu_ts = message.timestamp
        self._last_camera_timestamp = message.timestamp

    def extract_and_track_FORWARD(self, message: CameraData, _first_image_in_window: float):
        _ = message
        return self.trackFEATS.get_feature_database().features_containing(self.state._timestamp, False, True)

    def extract_and_track_BACKWARD(self, message: CameraData, _last_image_in_window: float):
        prev_ts = _last_image_in_window
        raw_features = self.trackFEATS.get_feature_database().features_containing(self.state._timestamp, False, True)
        observations = self._build_lifo_observations(raw_features, self.state._timestamp, message)
        self._update_backward_feature_chains(observations, self.state._timestamp, prev_ts)
        return self._backward_chains_as_features()

    def _build_lifo_observations(self, features, ts_now: float, message: CameraData):
        observations = []
        image_map = {}
        for i, cam_id in enumerate(message.sensor_ids):
            if i < len(message.images):
                image_map[cam_id] = message.images[i]

        for feat in features:
            for cam_id, ts_list in feat.timestamps.items():
                if cam_id not in feat.uvs or cam_id not in feat.uvs_norm:
                    continue
                for idx, ts in enumerate(ts_list):
                    if abs(ts - ts_now) > 1e-6:
                        continue
                    observations.append({
                        "cam_id": cam_id,
                        "uv": np.array(feat.uvs[cam_id][idx], dtype=float),
                        "uv_norm": np.array(feat.uvs_norm[cam_id][idx], dtype=float),
                        "desc": self._compute_orb_descriptor(image_map.get(cam_id), feat.uvs[cam_id][idx]),
                    })
                    break
        return observations

    def _compute_orb_descriptor(self, image, uv):
        if image is None:
            return None
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        h, w = gray.shape[:2]
        x = float(uv[0])
        y = float(uv[1])
        if x < 2 or y < 2 or x >= w - 2 or y >= h - 2:
            return None
        kp = cv2.KeyPoint(x, y, 31)
        _, desc = self._lifo_orb.compute(gray, [kp])
        if desc is None or len(desc) == 0:
            return None
        return desc[0].copy()

    def _get_chain_last_obs(self, chain, cam_id):
        if cam_id not in chain["timestamps"] or not chain["timestamps"][cam_id]:
            return None
        idx = len(chain["timestamps"][cam_id]) - 1
        return {
            "timestamp": chain["timestamps"][cam_id][idx],
            "uv": chain["uvs"][cam_id][idx],
            "uv_norm": chain["uvs_norm"][cam_id][idx],
            "desc": chain.get("descs", {}).get(cam_id, [None])[idx] if cam_id in chain.get("descs", {}) and len(chain.get("descs", {}).get(cam_id, [])) > idx else None,
        }

    def _append_obs_to_chain(self, chain, ts_now: float, cam_id: int, uv, uv_norm, desc=None):
        if cam_id not in chain["timestamps"]:
            chain["timestamps"][cam_id] = []
            chain["uvs"][cam_id] = []
            chain["uvs_norm"][cam_id] = []
            if "descs" not in chain:
                chain["descs"] = {}
            chain["descs"][cam_id] = []

        if chain["timestamps"][cam_id] and abs(chain["timestamps"][cam_id][-1] - ts_now) <= 1e-6:
            chain["uvs"][cam_id][-1] = np.array(uv, dtype=float)
            chain["uvs_norm"][cam_id][-1] = np.array(uv_norm, dtype=float)
            chain["descs"][cam_id][-1] = None if desc is None else desc.copy()
            return

        chain["timestamps"][cam_id].append(float(ts_now))
        chain["uvs"][cam_id].append(np.array(uv, dtype=float))
        chain["uvs_norm"][cam_id].append(np.array(uv_norm, dtype=float))
        chain["descs"][cam_id].append(None if desc is None else desc.copy())

        while len(chain["timestamps"][cam_id]) > self.params.sliding_window_size:
            chain["timestamps"][cam_id].pop(0)
            chain["uvs"][cam_id].pop(0)
            chain["uvs_norm"][cam_id].pop(0)
            chain["descs"][cam_id].pop(0)

    def _rotation_prior(self, prev_ts: float):
        R_curr = self.state._imu.Rot()
        R_prev = R_curr
        if prev_ts in self.state._clones_IMU:
            R_prev = self.state._clones_IMU[prev_ts].Rot()
        return R_curr @ R_prev.T

    def _predict_uv_norm(self, last_uv_norm, C_q_hat):
        b_prev = np.array([last_uv_norm[0], last_uv_norm[1], 1.0], dtype=float)
        b_prev = b_prev / (np.linalg.norm(b_prev) + 1e-12)
        b_pred = C_q_hat @ b_prev
        if abs(b_pred[2]) < 1e-9:
            return np.array(last_uv_norm, dtype=float)
        return np.array([b_pred[0] / b_pred[2], b_pred[1] / b_pred[2]], dtype=float)

    def _reprojection_residual(self, uv_norm_obs, last_uv_norm, C_q_hat):
        b_prev = np.array([last_uv_norm[0], last_uv_norm[1], 1.0], dtype=float)
        b_curr = np.array([uv_norm_obs[0], uv_norm_obs[1], 1.0], dtype=float)
        b_prev = b_prev / (np.linalg.norm(b_prev) + 1e-12)
        b_curr = b_curr / (np.linalg.norm(b_curr) + 1e-12)
        return float(np.linalg.norm(b_curr - C_q_hat @ b_prev) ** 2)

    def _descriptor_distance(self, d1, d2):
        if d1 is None or d2 is None:
            return None
        return int(cv2.norm(d1, d2, cv2.NORM_HAMMING))

    def _build_cost_and_meta(self, chains, observations, ts_now: float, C_q_hat, frame_dt: float):
        n_old = len(chains)
        n_new = len(observations)
        inf_cost = 1e6
        cost = inf_cost * np.ones((n_old, n_new), dtype=float)
        meta = {}

        for i, (chain_id, chain) in enumerate(chains):
            last = self._get_chain_last_obs(chain, observations[0]["cam_id"] if observations else 0)
            for j, obs in enumerate(observations):
                cam_id = obs["cam_id"]
                last = self._get_chain_last_obs(chain, cam_id)
                if last is None:
                    continue

                age = ts_now - last["timestamp"]
                if age < 0:
                    continue
                age_steps = age / max(1e-6, frame_dt)
                if age_steps > max(1, int(self.params.lifo_backward_max_age_steps)):
                    continue

                pred = self._predict_uv_norm(last["uv_norm"], C_q_hat)
                geom_dist = float(np.linalg.norm(obs["uv_norm"] - pred))
                reproj = self._reprojection_residual(obs["uv_norm"], last["uv_norm"], C_q_hat)
                ddesc = self._descriptor_distance(obs.get("desc"), last.get("desc"))

                allowed = False
                total_cost = inf_cost
                fallback_used = False

                if geom_dist <= self.params.lifo_backward_match_threshold and reproj <= self.params.lifo_backward_reproj_threshold:
                    allowed = True
                    total_cost = geom_dist + 0.5 * reproj
                elif self.params.lifo_orb_descriptor_fallback and ddesc is not None and ddesc <= self.params.lifo_orb_hamming_threshold:
                    # Descriptor-assisted fallback for low-confidence geometry.
                    allowed = True
                    fallback_used = True
                    total_cost = 0.7 * geom_dist + 0.3 * (ddesc / max(1.0, float(self.params.lifo_orb_hamming_threshold)))

                if allowed:
                    cost[i, j] = total_cost
                    meta[(i, j)] = {
                        "fallback": fallback_used,
                        "geom_dist": geom_dist,
                        "reproj": reproj,
                        "ddesc": ddesc,
                    }
        return cost, meta

    def _hungarian_pairs(self, cost):
        if cost.size == 0:
            return set()
        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1e5:
                pairs.add((int(r), int(c)))
        return pairs

    def _update_chain_quality(self, chain, matched: bool, used_fallback: bool = False):
        if matched:
            inc = self.params.lifo_quality_inc_orb if used_fallback else self.params.lifo_quality_inc_match
            chain["quality"] = min(1.0, chain["quality"] + inc)
            chain["misses"] = 0
            chain["hits"] += 1
        else:
            chain["quality"] = max(0.0, chain["quality"] - self.params.lifo_quality_dec_miss)
            chain["misses"] += 1

    def _prune_backward_chains(self):
        remove_ids = []
        for chain_id, chain in self._lifo_backward_chains.items():
            if chain["misses"] > self.params.lifo_max_misses:
                remove_ids.append(chain_id)
                continue
            if chain["quality"] < self.params.lifo_quality_min_keep:
                remove_ids.append(chain_id)

        for cid in remove_ids:
            del self._lifo_backward_chains[cid]

    def _update_backward_feature_chains(self, observations, ts_now: float, prev_ts: float):
        if not observations:
            for chain in self._lifo_backward_chains.values():
                self._update_chain_quality(chain, matched=False)
            self._prune_backward_chains()
            return

        C_q_hat = self._rotation_prior(prev_ts) if prev_ts is not None else np.eye(3)
        frame_dt = max(1e-3, (ts_now - prev_ts)) if prev_ts is not None else 1.0
        obs_by_cam = {}
        for obs in observations:
            obs_by_cam.setdefault(obs["cam_id"], []).append(obs)

        matched_chain_ids = set()

        for cam_id, cam_obs in obs_by_cam.items():
            chains = [(cid, ch) for cid, ch in self._lifo_backward_chains.items() if self._get_chain_last_obs(ch, cam_id) is not None]

            if not chains:
                for obs in cam_obs:
                    chain_id = self._lifo_next_chain_id
                    self._lifo_next_chain_id += 1
                    self._lifo_backward_chains[chain_id] = {
                        "timestamps": {}, "uvs": {}, "uvs_norm": {}, "descs": {},
                        "quality": float(self.params.lifo_quality_init), "misses": 0, "hits": 1,
                    }
                    self._append_obs_to_chain(self._lifo_backward_chains[chain_id], ts_now, cam_id, obs["uv"], obs["uv_norm"], obs.get("desc"))
                    matched_chain_ids.add(chain_id)
                continue

            cost, meta = self._build_cost_and_meta(chains, cam_obs, ts_now, C_q_hat, frame_dt)
            pair_fwd = self._hungarian_pairs(cost)
            pair_bwd = {(i, j) for (j, i) in self._hungarian_pairs(cost.T)}

            if self.params.lifo_backward_mutual_consistency:
                valid_pairs = pair_fwd.intersection(pair_bwd)
            else:
                valid_pairs = pair_fwd

            used_obs = set()
            for i, j in valid_pairs:
                chain_id, chain = chains[i]
                obs = cam_obs[j]
                self._append_obs_to_chain(chain, ts_now, cam_id, obs["uv"], obs["uv_norm"], obs.get("desc"))
                m = meta.get((i, j), {})
                self._update_chain_quality(chain, matched=True, used_fallback=bool(m.get("fallback", False)))
                matched_chain_ids.add(chain_id)
                used_obs.add(j)

            for j, obs in enumerate(cam_obs):
                if j in used_obs:
                    continue
                chain_id = self._lifo_next_chain_id
                self._lifo_next_chain_id += 1
                self._lifo_backward_chains[chain_id] = {
                    "timestamps": {}, "uvs": {}, "uvs_norm": {}, "descs": {},
                    "quality": float(self.params.lifo_quality_init), "misses": 0, "hits": 1,
                }
                self._append_obs_to_chain(self._lifo_backward_chains[chain_id], ts_now, cam_id, obs["uv"], obs["uv_norm"], obs.get("desc"))
                matched_chain_ids.add(chain_id)

        for chain_id, chain in self._lifo_backward_chains.items():
            if chain_id in matched_chain_ids:
                continue
            self._update_chain_quality(chain, matched=False)

        self._prune_backward_chains()

    def _backward_chains_as_features(self, apply_quality_threshold: bool = True):
        out = []
        min_len = max(1, int(self.params.lifo_backward_min_chain_length))
        for chain_id, chain in self._lifo_backward_chains.items():
            total_meas = sum(len(v) for v in chain["timestamps"].values())
            if total_meas < min_len:
                continue
            if apply_quality_threshold and chain.get("quality", 0.0) < float(self.params.lifo_quality_min_output):
                continue
            f = feature.Feature()
            f.featid = chain_id
            f.timestamps = copy.deepcopy(chain["timestamps"])
            f.uvs = copy.deepcopy(chain["uvs"])
            f.uvs_norm = copy.deepcopy(chain["uvs_norm"])
            out.append(f)
        return out

    def detect_hovering(self, features, prev_ts: float, curr_ts: float):
        if prev_ts is None or curr_ts is None:
            return 0, float("inf"), 0, 0, 0.0

        frame_dt = max(1e-6, curr_ts - prev_ts)
        ts_tol = max(1e-6, 0.25 * frame_dt)

        def _find_ts_index(ts_list, target, tol):
            for idx, ts in enumerate(ts_list):
                if abs(ts - target) <= tol:
                    return idx
            return -1

        R_curr = self.state._imu.Rot()
        R_prev = None
        if prev_ts in self.state._clones_IMU:
            R_prev = self.state._clones_IMU[prev_ts].Rot()
        elif self._last_camera_imu_rot is not None:
            R_prev = self._last_camera_imu_rot
        else:
            R_prev = R_curr
        C_q_hat = R_curr @ R_prev.T

        b_prev = []
        b_curr = []
        for feat in features:
            for cam_id, ts_list in feat.timestamps.items():
                if cam_id not in feat.uvs_norm:
                    continue
                idx_prev = _find_ts_index(ts_list, prev_ts, ts_tol)
                idx_curr = _find_ts_index(ts_list, curr_ts, ts_tol)
                if idx_prev < 0 or idx_curr < 0:
                    continue
                z_prev = np.array([feat.uvs_norm[cam_id][idx_prev][0], feat.uvs_norm[cam_id][idx_prev][1], 1.0])
                z_curr = np.array([feat.uvs_norm[cam_id][idx_curr][0], feat.uvs_norm[cam_id][idx_curr][1], 1.0])
                z_prev = z_prev / np.linalg.norm(z_prev)
                z_curr = z_curr / np.linalg.norm(z_curr)
                b_prev.append(z_prev)
                b_curr.append(z_curr)
                break

        if not b_prev:
            return 0, float("inf"), 0, 0, 0.0

        baseline = 0.0
        if prev_ts in self.state._clones_IMU:
            baseline = np.linalg.norm(self.state._imu.pos() - self.state._clones_IMU[prev_ts].pos())
        elif self._last_camera_imu_pos is not None:
            baseline = np.linalg.norm(self.state._imu.pos() - self._last_camera_imu_pos)

        residuals_all = [float(np.linalg.norm(bc - C_q_hat @ bp) ** 2) for bp, bc in zip(b_prev, b_curr)]
        residuals = []
        if baseline > self.params.hover_baseline_threshold:
            A = []
            for bp, bc in zip(b_prev, b_curr):
                A.append((skew_x(C_q_hat @ bp).T @ bc).reshape(-1))
            A = np.array(A)
            if A.shape[0] >= 2:
                _, _, vh = np.linalg.svd(A)
                t_k = vh[-1, :]
                t_k = t_k / (np.linalg.norm(t_k) + 1e-12)
            else:
                t_k = np.zeros(3)

            for bp, bc in zip(b_prev, b_curr):
                epi = float(bc.T @ skew_x(C_q_hat @ bp) @ t_k)
                if abs(epi) < self.params.hover_epipolar_inlier_threshold:
                    residuals.append(float(np.linalg.norm(bc - C_q_hat @ bp) ** 2))
        else:
            for bp, bc in zip(b_prev, b_curr):
                r = float(np.linalg.norm(bc - C_q_hat @ bp) ** 2)
                if r < self.params.hover_bearing_inlier_threshold:
                    residuals.append(r)

        if len(residuals) < self.params.hover_min_inliers:
            # Fallback to robust all-feature statistic when strict inlier test is sparse.
            d_k = float(np.median(residuals_all)) if residuals_all else float("inf")
            decision = 1 if d_k < self.params.hovering_threshold else 0
            return decision, d_k, len(residuals), len(residuals_all), float(baseline)

        d_k = float(np.mean(residuals))
        decision = 1 if d_k < self.params.hovering_threshold else 0
        return decision, d_k, len(residuals), len(residuals_all), float(baseline)

    def apply_temporal_smoothing(self, xi_k: int) -> int:
        self._hover_decisions.append(int(xi_k))
        if len(self._hover_decisions) < self._hover_decisions.maxlen:
            return self._last_hover_decision

        if all(x == self._hover_decisions[0] for x in self._hover_decisions):
            self._last_hover_decision = self._hover_decisions[0]
        return self._last_hover_decision

    def switch_to_LIFO(self):
        self.mode = "LIFO"
        self.hover_start_step = self.state._timestamp
        self.covariance_updated = False
        self.hover_poses_fixed = copy.deepcopy(self.state._clones_IMU)
        self._hover_feature_measurements = {}
        self._lifo_backward_chains = {}
        self._lifo_next_chain_id = 1

        # Seed backward chains from the current database history in the frozen window.
        all_feats = list(self.trackFEATS.get_feature_database().get_internal_data().values())
        for feat_obj in all_feats:
            chain_id = self._lifo_next_chain_id
            self._lifo_next_chain_id += 1
            chain = {
                "timestamps": {},
                "uvs": {},
                "uvs_norm": {},
                "descs": {},
                "quality": float(self.params.lifo_quality_init),
                "misses": 0,
                "hits": 0,
            }
            for cam_id, ts_list in feat_obj.timestamps.items():
                if cam_id not in feat_obj.uvs or cam_id not in feat_obj.uvs_norm:
                    continue
                packed = list(zip(ts_list, feat_obj.uvs[cam_id], feat_obj.uvs_norm[cam_id]))
                packed.sort(key=lambda x: x[0])
                if len(packed) > self.params.sliding_window_size:
                    packed = packed[-self.params.sliding_window_size:]
                chain["timestamps"][cam_id] = [float(p[0]) for p in packed]
                chain["uvs"][cam_id] = [np.array(p[1], dtype=float) for p in packed]
                chain["uvs_norm"][cam_id] = [np.array(p[2], dtype=float) for p in packed]
                chain["descs"][cam_id] = [None for _ in packed]

            chain["hits"] = sum(len(v) for v in chain["timestamps"].values())

            if sum(len(v) for v in chain["timestamps"].values()) > 0:
                self._lifo_backward_chains[chain_id] = chain

        print(f"[HOVER] Switching to LIFO at t={self.hover_start_step:.3f}")

    def switch_to_FIFO(self):
        if not self.covariance_updated:
            self.perform_covariance_update(list(self._hover_feature_measurements.values()))
            self.covariance_updated = True

        self.mode = "FIFO"
        self.hover_start_step = None
        self.hover_poses_fixed = {}
        self.covariance_updated = False
        self._hover_feature_measurements = {}
        self._lifo_backward_chains = {}
        self._lifo_next_chain_id = 1
        print("[HOVER] Switching back to FIFO")

    def msckf_state_only_update(self, features):
        if not features:
            return
        cov_snapshot = self.state._Cov.copy()
        features_local = copy.deepcopy(features)
        self.updaterMSCKF.update(self.state, features_local)
        self.state._Cov = cov_snapshot

    def perform_covariance_update(self, all_hovering_measurements):
        if not all_hovering_measurements:
            return
        features_local = copy.deepcopy(all_hovering_measurements)
        state_snapshot = self._snapshot_state_values()
        self.updaterMSCKF.update(self.state, features_local)
        self._restore_state_values(state_snapshot)

    def _snapshot_state_values(self):
        snapshots = []
        for var in self.state._variables:
            snapshots.append((var, np.copy(var.value()) if var.value() is not None else None))
        return snapshots

    def _restore_state_values(self, snapshots):
        for var, value in snapshots:
            if value is not None:
                var.set_value(value)

    def _collect_hovering_measurements(self, features):
        for feat in features:
            if feat.featid not in self._hover_feature_measurements:
                self._hover_feature_measurements[feat.featid] = copy.deepcopy(feat)
                continue

            merged = self._hover_feature_measurements[feat.featid]
            for cam_id, ts_list in feat.timestamps.items():
                if cam_id not in merged.timestamps:
                    merged.timestamps[cam_id] = list(ts_list)
                    merged.uvs[cam_id] = list(feat.uvs[cam_id])
                    merged.uvs_norm[cam_id] = list(feat.uvs_norm[cam_id])
                    continue

                existing = set(merged.timestamps[cam_id])
                for i, ts in enumerate(ts_list):
                    if ts in existing:
                        continue
                    merged.timestamps[cam_id].append(ts)
                    merged.uvs[cam_id].append(feat.uvs[cam_id][i])
                    merged.uvs_norm[cam_id].append(feat.uvs_norm[cam_id][i])

    def fifo_update(self, message: CameraData):
        self.rT3 = time.time()

        if len(self.state._clones_IMU) < min(self.params.state_options.max_clone_size, 5):
            return

        if not self.has_moved_since_zupt:
            print(f"\n[ZUPT DEBUG] Exited ZUPT normally due to reached clone size. Time relative to rosbag start: {message.timestamp - self.first_rosbag_time:.3f} sec\n")
            self.has_moved_since_zupt = True

        feats_lost = self.trackFEATS.get_feature_database().features_not_containing_newer(self.state._timestamp, False, True)
        feats_marg = []
        feats_slam = []

        if len(self.state._clones_IMU) > self.params.state_options.max_clone_size or len(self.state._clones_IMU) > 5:
            marg_time = self.state.margtimestep()
            feats_marg = self.trackFEATS.get_feature_database().features_containing(marg_time, False, True)
            if self.trackARUCO and message.timestamp - self.startup_time >= self.params.dt_slam_delay:
                feats_slam = self.trackARUCO.get_feature_database().features_containing(marg_time, False, True)

        feats_lost = [f for f in feats_lost if any(cid in message.sensor_ids for cid in f.uvs)]
        feats_lost = [f for f in feats_lost if f not in feats_marg]

        feats_maxtracks = []
        i = len(feats_marg) - 1
        while i >= 0:
            reached_max = any(len(ts) > self.params.state_options.max_clone_size for ts in feats_marg[i].timestamps.values())
            if reached_max:
                feats_maxtracks.append(feats_marg[i])
                del feats_marg[i]
            i -= 1

        curr_aruco = sum(1 for f in self.state._features_SLAM.values() if int(f._featid) <= 4*self.params.state_options.max_aruco_features)

        if self.params.state_options.max_slam_features > 0 and \
           message.timestamp - self.startup_time >= self.params.dt_slam_delay and \
           len(self.state._features_SLAM) < self.params.state_options.max_slam_features + curr_aruco:

            amount = (self.params.state_options.max_slam_features + curr_aruco) - len(self.state._features_SLAM)
            valid = min(amount, len(feats_maxtracks))
            if valid > 0:
                feats_slam.extend(feats_maxtracks[-valid:])
                feats_maxtracks = feats_maxtracks[:-valid]

        for _, landmark in self.state._features_SLAM.items():
            if self.trackARUCO:
                f1 = self.trackARUCO.get_feature_database().get_feature(landmark._featid)
                if f1:
                    feats_slam.append(f1)

            f2 = self.trackFEATS.get_feature_database().get_feature(landmark._featid)
            if f2:
                feats_slam.append(f2)

            current_unique = (landmark._unique_camera_id in message.sensor_ids)
            if f2 is None and current_unique:
                landmark.should_marg = True
            if landmark.update_fail_count > 1:
                landmark.should_marg = True

        StateHelper.marginalize_slam(self.state)

        feats_slam_DELAYED = []
        feats_slam_UPDATE = []
        for f in feats_slam:
            if f.featid in self.state._features_SLAM:
                feats_slam_UPDATE.append(f)
            else:
                feats_slam_DELAYED.append(f)

        featsup_MSCKF = feats_lost + feats_marg + feats_maxtracks
        featsup_MSCKF.sort(key=lambda f: sum(len(ts) for ts in f.timestamps.values()))

        if len(featsup_MSCKF) > self.params.state_options.max_msckf_in_update:
            start_idx = len(featsup_MSCKF) - self.params.state_options.max_msckf_in_update
            featsup_MSCKF = featsup_MSCKF[start_idx:]

        self.updaterMSCKF.update(self.state, featsup_MSCKF)
        self.propagator.invalidate_cache()
        self.rT4 = time.time()

        while feats_slam_UPDATE:
            limit = min(self.params.state_options.max_slam_in_update, len(feats_slam_UPDATE))
            chunk = feats_slam_UPDATE[:limit]
            feats_slam_UPDATE = feats_slam_UPDATE[limit:]

            self.updaterSLAM.update(self.state, chunk)
            self.propagator.invalidate_cache()

        self.rT5 = time.time()
        self.updaterSLAM.delayed_init(self.state, feats_slam_DELAYED)
        self.rT6 = time.time()

        if message.sensor_ids[0] == 0:
            self.retriangulate_active_tracks(message)
            self.good_features_MSCKF.clear()

        for f in featsup_MSCKF:
            if hasattr(f, 'p_FinG'):
                self.good_features_MSCKF.append(f.p_FinG)
            f.to_delete = True

        self.trackFEATS.get_feature_database().cleanup()
        if self.trackARUCO:
            self.trackARUCO.get_feature_database().cleanup()

        self.updaterSLAM.change_anchors(self.state)

        if len(self.state._clones_IMU) > self.params.state_options.max_clone_size:
            marg_time = self.state.margtimestep()
            self.trackFEATS.get_feature_database().cleanup_measurements(marg_time)
            if self.trackARUCO:
                self.trackARUCO.get_feature_database().cleanup_measurements(marg_time)

        StateHelper.marginalize_old_clone(self.state)
        self._print_update_summary(message)

    def lifo_update(self, message: CameraData, features_for_update=None):
        if features_for_update is None:
            features_for_update = []

        # For deferred covariance update we retain broader hovering measurements
        # (without quality threshold) while using strict set for per-step state update.
        features_for_cov = self._backward_chains_as_features(apply_quality_threshold=False)
        self._collect_hovering_measurements(features_for_cov)
        self.msckf_state_only_update(features_for_update)

        if message.sensor_ids and message.sensor_ids[0] == 0:
            self.retriangulate_active_tracks(message)

        self._print_update_summary(message)

    def _print_update_summary(self, message: CameraData):
        rT7 = time.time()
        _p = self.state._imu.pos()
        _v = self.state._imu.vel()
        hover_tag = "[hover]" if self.mode == "LIFO" else "[non-hover]"
        print(f"[VIO:{self.mode}]{hover_tag} t={self.state._timestamp:.3f} pos=[{_p[0]:.2f},{_p[1]:.2f},{_p[2]:.2f}] "
              f"vel=[{_v[0]:.2f},{_v[1]:.2f},{_v[2]:.2f}] clones={len(self.state._clones_IMU)} "
              f"dt={rT7-self.rT1:.3f}s")

        if self.timelastupdate != -1 and self.timelastupdate in self.state._clones_IMU:
            dx = self.state._imu.pos() - self.state._clones_IMU[self.timelastupdate].pos()
            self.distance += np.linalg.norm(dx)
        self.timelastupdate = message.timestamp

    def _pose_for_time(self, ts: float):
        """
        Return a reference pose for visualization at timestamp ts.
        Preference: exact clone -> nearest clone -> current IMU state.
        """
        if ts in self.state._clones_IMU:
            clone = self.state._clones_IMU[ts]
            return clone.Rot(), clone.pos()

        if self.state._clones_IMU:
            nearest_ts = min(self.state._clones_IMU.keys(), key=lambda t: abs(t - ts))
            clone = self.state._clones_IMU[nearest_ts]
            return clone.Rot(), clone.pos()

        return self.state._imu.Rot(), self.state._imu.pos()

    def retriangulate_active_tracks(self, message: CameraData):
        self.active_tracks_time = message.timestamp
        self.active_image = np.zeros((message.images[0].shape[0], message.images[0].shape[1], 3), dtype=np.uint8)
        # Using placeholder for full drawing logic
        # self.trackFEATS.display_active(...)
        self.active_tracks_posinG.clear()
        self.active_tracks_uvd.clear()

        last_obs = self.trackFEATS.get_last_obs()
        last_ids = self.trackFEATS.get_last_ids()

        active_tracks_posinG_new = {}
        active_feat_linsys_A_new = {}
        active_feat_linsys_b_new = {}
        active_feat_linsys_count_new = {}

        feat_uvs_in_cam0 = {}

        R_GtoI_ref, p_IinG_ref = self._pose_for_time(self.active_tracks_time)

        for cam_id in message.sensor_ids:
            R_GtoI = R_GtoI_ref
            p_IinG = p_IinG_ref
            
            R_ItoC = self.state._calib_IMUtoCAM[cam_id].Rot()
            p_IinC = self.state._calib_IMUtoCAM[cam_id].pos()
            
            R_GtoCi = R_ItoC @ R_GtoI
            p_CiinG = p_IinG - R_GtoCi.T @ p_IinC

            if cam_id not in last_obs: continue
            
            for i, kpt in enumerate(last_obs[cam_id]):
                featid = last_ids[cam_id][i]
                pt_d = kpt.pt # (x, y)
                
                if cam_id == 0:
                    feat_uvs_in_cam0[featid] = pt_d
                
                if featid in self.state._features_SLAM: continue

                # Undistort
                # Assuming CamBase has undistort_cv returning Point2f (x,y)
                pt_n = self.state._cam_intrinsics_cameras[cam_id].undistort_cv(pt_d)
                
                b_i = np.array([pt_n[0], pt_n[1], 1.0])
                b_i = R_GtoCi.T @ b_i
                b_i = b_i / np.linalg.norm(b_i)
                Bperp = skew_x(b_i)
                
                Ai = Bperp.T @ Bperp
                bi = Ai @ p_CiinG
                
                if featid not in self.active_feat_linsys_A:
                    active_feat_linsys_A_new[featid] = Ai
                    active_feat_linsys_b_new[featid] = bi
                    active_feat_linsys_count_new[featid] = 1
                else:
                    active_feat_linsys_A_new[featid] = Ai + self.active_feat_linsys_A[featid]
                    active_feat_linsys_b_new[featid] = bi + self.active_feat_linsys_b[featid]
                    active_feat_linsys_count_new[featid] = 1 + self.active_feat_linsys_count[featid]

                if active_feat_linsys_count_new[featid] > 3:
                    A = active_feat_linsys_A_new[featid]
                    b = active_feat_linsys_b_new[featid]
                    
                    try:
                        p_FinG = np.linalg.solve(A, b)
                        p_FinCi = R_GtoCi @ (p_FinG - p_CiinG)
                        
                        # SVD for condition number
                        S = np.linalg.svd(A, compute_uv=False)
                        condA = S[0] / S[-1]
                        
                        if (abs(condA) <= self.params.featinit_options.max_cond_number and 
                            p_FinCi[2] >= self.params.featinit_options.min_dist and 
                            p_FinCi[2] <= self.params.featinit_options.max_dist):
                            active_tracks_posinG_new[featid] = p_FinG
                    except np.linalg.LinAlgError:
                        pass

        # Update
        self.active_feat_linsys_A = active_feat_linsys_A_new
        self.active_feat_linsys_b = active_feat_linsys_b_new
        self.active_feat_linsys_count = active_feat_linsys_count_new
        self.active_tracks_posinG = active_tracks_posinG_new

        # Append SLAM features
        
        for fid, feat in self.state._features_SLAM.items():
            p_FinG = feat.get_xyz(False)
            if LandmarkRepresentation.is_relative_representation(feat._feat_representation):
                if feat._anchor_clone_timestamp not in self.state._clones_IMU:
                    # Anchor clone can be marginalized; skip stale SLAM anchor for viz.
                    continue
                R_ItoC = self.state._calib_IMUtoCAM[feat._anchor_cam_id].Rot()
                p_IinC = self.state._calib_IMUtoCAM[feat._anchor_cam_id].pos()
                R_GtoI = self.state._clones_IMU[feat._anchor_clone_timestamp].Rot()
                p_IinG = self.state._clones_IMU[feat._anchor_clone_timestamp].pos()
                p_FinG = R_GtoI.T @ R_ItoC.T @ (p_FinG - p_IinC) + p_IinG
            self.active_tracks_posinG[feat._featid] = p_FinG

        # Reprojection for cam0
        R_ItoC0 = self.state._calib_IMUtoCAM[0].Rot()
        p_IinC0 = self.state._calib_IMUtoCAM[0].pos()
        R_GtoIi, p_IiinG = self._pose_for_time(self.active_tracks_time)

        for fid, p_FinG in self.active_tracks_posinG.items():
            if fid not in feat_uvs_in_cam0: continue
            
            p_FinIi = R_GtoIi @ (p_FinG - p_IiinG)
            p_FinCi = R_ItoC0 @ p_FinIi + p_IinC0
            depth = p_FinCi[2]
            
            if fid in feat_uvs_in_cam0:
                pt_d = feat_uvs_in_cam0[fid]
                uv_dist = np.array([pt_d[0], pt_d[1]])
            else:
                uv_norm = np.array([p_FinCi[0]/depth, p_FinCi[1]/depth])
                uv_dist = self.state._cam_intrinsics_cameras[0].distort_d(uv_norm)
            
            if depth < 0.1: continue
            
            w = self.state._cam_intrinsics_cameras[0].w()
            h = self.state._cam_intrinsics_cameras[0].h()
            
            if 0 <= uv_dist[0] < w and 0 <= uv_dist[1] < h:
                self.active_tracks_uvd[fid] = np.array([uv_dist[0], uv_dist[1], depth])

    # =======================================================================
    #  Getters / Accessors
    # =======================================================================

    def initialized(self) -> bool:
        """If we are initialized or not."""
        return self.is_initialized_vio and self.timelastupdate != -1

    def initialized_time(self) -> float:
        """Timestamp that the system was initialized at."""
        return self.startup_time

    def get_params(self) -> VioManagerOptions:
        """Accessor for current system parameters."""
        return self.params

    def get_state(self) -> State:
        """Accessor to get the current state."""
        return self.state

    def get_propagator(self) -> Propagator:
        """Accessor to get the current propagator."""
        return self.propagator

    def get_good_features_MSCKF(self) -> List[np.ndarray]:
        """Returns 3d features used in the last update in global frame."""
        return self.good_features_MSCKF

    def get_active_image(self) -> Tuple[float, np.ndarray]:
        """
        Return the image used when projecting the active tracks.
        Returns: (timestamp, image)
        """
        return self.active_tracks_time, self.active_image

    def get_active_tracks(self) -> Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Returns active tracked features in the current frame.
        Returns: (timestamp, feat_posinG, feat_tracks_uvd)
        """
        return self.active_tracks_time, self.active_tracks_posinG, self.active_tracks_uvd
    

    def get_historical_viz_image(self):
        if not self.state or not self.trackFEATS: return None
        highlighted = list(self.state._features_SLAM.keys())
        overlay = "zvupt" if self.did_zupt_update else ("init" if not self.is_initialized_vio else "")
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img = self.trackFEATS.display_history(img, 255, 255, 0, 255, 255, 255, highlighted, overlay)
        if self.trackARUCO:
            img = self.trackARUCO.display_history(img, 0, 255, 255, 255, 255, 255, highlighted, overlay)
        return img

    def get_features_SLAM(self):
        slam_feats = []
        for fid, f in self.state._features_SLAM.items():
            if int(fid) <= 4 * self.params.state_options.max_aruco_features: continue
            
            p_FinG = f.get_xyz(False)
            if LandmarkRepresentation.is_relative_representation(f._feat_representation):
                R_ItoC = self.state._calib_IMUtoCAM[f._anchor_cam_id].Rot()
                p_IinC = self.state._calib_IMUtoCAM[f._anchor_cam_id].pos()
                R_GtoI = self.state._clones_IMU[f._anchor_clone_timestamp].Rot()
                p_IinG = self.state._clones_IMU[f._anchor_clone_timestamp].pos()
                p_FinG = R_GtoI.T @ R_ItoC.T @ (p_FinG - p_IinC) + p_IinG
            
            slam_feats.append(p_FinG)
        return slam_feats

    def get_features_ARUCO(self):
        aruco_feats = []
        for fid, f in self.state._features_SLAM.items():
            if int(fid) > 4 * self.params.state_options.max_aruco_features: continue
            
            p_FinG = f.get_xyz(False)
            if LandmarkRepresentation.is_relative_representation(f._feat_representation):
                R_ItoC = self.state._calib_IMUtoCAM[f._anchor_cam_id].Rot()
                p_IinC = self.state._calib_IMUtoCAM[f._anchor_cam_id].pos()
                R_GtoI = self.state._clones_IMU[f._anchor_clone_timestamp].Rot()
                p_IinG = self.state._clones_IMU[f._anchor_clone_timestamp].pos()
                p_FinG = R_GtoI.T @ R_ItoC.T @ (p_FinG - p_IinC) + p_IinG
            
            aruco_feats.append(p_FinG)
        return aruco_feats