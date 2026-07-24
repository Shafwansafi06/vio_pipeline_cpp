import numpy as np
import scipy.stats
import math
import threading
import copy
from typing import List, Dict, Tuple, Optional

# Imports
from vio_pipeline_v2.msckf.State import State
from vio_pipeline_v2.msckf.Propagator import Propagator
from vio_pipeline_v2.msckf.state_helper import StateHelper
from vio_pipeline_v2.core import FeatureDatabase, FeatureHelper, ImuData
from vio_pipeline_v2.msckf.updater_helper import UpdaterHelper

from vio_pipeline_v2.type import skew_x, log_so3

class UpdaterZeroVelocity:
    def __init__(self, options, noises, db: FeatureDatabase, prop: Propagator, 
                 gravity_mag: float, zupt_max_velocity: float, 
                 zupt_noise_multiplier: float, zupt_max_disparity: float):
        
        self._options = options
        self._noises = noises
        self._db = db
        self._prop = prop
        self._zupt_max_velocity = zupt_max_velocity
        self._zupt_noise_multiplier = zupt_noise_multiplier
        self._zupt_max_disparity = zupt_max_disparity

        # Gravity vector
        self._gravity = np.array([0.0, 0.0, gravity_mag])

        # Precompute squared noises
        self._noises.sigma_w_2 = self._noises.sigma_w ** 2
        self._noises.sigma_a_2 = self._noises.sigma_a ** 2
        self._noises.sigma_wb_2 = self._noises.sigma_wb ** 2
        self._noises.sigma_ab_2 = self._noises.sigma_ab ** 2

        # Chi-Squared Table (Precomputed for speed)
        self.chi_squared_table = {}
        for i in range(1, 1001):
            self.chi_squared_table[i] = scipy.stats.chi2.ppf(0.95, df=i)

        # State tracking variables
        self.last_zupt_state_timestamp = 0.0
        self.last_zupt_count = 0
        self.imu_data = [] 
        self.imu_data_mtx = threading.RLock()
        
        # Time offset logic matches Propagator
        self.last_prop_time_offset = 0.0
        self.have_last_prop_time_offset = False

    def feed_imu(self, message: ImuData, oldest_time: float = -1):
        """
        Thread-safe add IMU message.
        """
        with self.imu_data_mtx:
            self.imu_data.append(message)
            # C++ keeps a 0.10s buffer before oldest_time
            clean_time = oldest_time - 0.10 if oldest_time > 0 else oldest_time
            if clean_time > 0:
                idx = 0
                for i, m in enumerate(self.imu_data):
                    if m.timestamp >= clean_time:
                        idx = i
                        break
                else:
                    # This runs if the loop never breaks (all data is old)
                    idx = len(self.imu_data)

                if idx > 0:
                    del self.imu_data[:idx]

    def clean_old_imu_measurements(self, oldest_time: float):
        """Remove IMU measurements older than oldest_time."""
        with self.imu_data_mtx:
            if not self.imu_data or oldest_time < 0:
                return
            idx = 0
            while idx < len(self.imu_data) and self.imu_data[idx].timestamp < oldest_time:
                idx += 1
            if idx > 0:
                del self.imu_data[:idx]

    def try_update(self, state: State, timestamp: float) -> bool:
        """
        Attempts to perform a Zero Velocity Update (ZUPT).
        Returns True if update was applied (system detected as stationary).
        """
        
        # Checks
        if not self.imu_data:
            self.last_zupt_state_timestamp = 0.0
            return False

        if state._timestamp == timestamp:
            self.last_zupt_state_timestamp = 0.0
            return False

        # Time Offset Init
        if not self.have_last_prop_time_offset:
            self.last_prop_time_offset = state._calib_dt_CAMtoIMU.value()[0]
            self.have_last_prop_time_offset = True

        t_off_new = state._calib_dt_CAMtoIMU.value()[0]
        
        # Time Window
        time0 = state._timestamp + self.last_prop_time_offset
        time1 = timestamp + t_off_new

        # Get IMU data
        with self.imu_data_mtx:
            imu_recent = self._prop.select_imu_readings(self.imu_data, time0, time1)

        self.last_prop_time_offset = t_off_new

        if len(imu_recent) < 2:
            print("[ZUPT]: No IMU data to check zero velocity!")
            self.last_zupt_state_timestamp = 0.0
            return False

        # Configuration Flags
        integrated_accel_constraint = False
        model_time_varying_bias = True
        override_with_disparity_check = True
        explicitly_enforce_zero_motion = False

        # Jacobian Order
        Hx_order = [state._imu.q(), state._imu.bg(), state._imu.ba()]
        if integrated_accel_constraint:
            Hx_order.append(state._imu.v())

        # Dimensions
        h_size = 12 if integrated_accel_constraint else 9
        m_size = 6 * (len(imu_recent) - 1)
        
        H = np.zeros((m_size, h_size))
        res = np.zeros(m_size)

        # Intrinsics
        Dw = State.Dm(state._options.imu_model, state._calib_imu_dw.value())
        Da = State.Dm(state._options.imu_model, state._calib_imu_da.value())
        Tg = State.Tg(state._calib_imu_tg.value())

        dt_summed = 0.0
        valid_count = 0
        
        # 
        # Build System
        for i in range(len(imu_recent) - 1):
            dt = imu_recent[i+1].timestamp - imu_recent[i].timestamp
            if dt <= 0:
                continue

            # Corrected measurements
            a_hat = state._calib_imu_ACCtoIMU.Rot() @ Da @ (imu_recent[i].am - state._imu.bias_a())
            w_hat = state._calib_imu_GYROtoIMU.Rot() @ Dw @ (imu_recent[i].wm - state._imu.bias_g() - Tg @ a_hat)

            # Measurement Noise (Whitening factors)
            w_omega = math.sqrt(dt) / self._noises.sigma_w
            w_accel = math.sqrt(dt) / self._noises.sigma_a
            w_accel_v = 1.0 / (math.sqrt(dt) * self._noises.sigma_a)

            row = 6 * valid_count

            R_GtoI = state._imu.Rot()
            R_jacob = state._imu.Rot_fej() if state._options.do_fej else R_GtoI

            # --- Gyro residual: res = -sqrt(dt)/sigma_w * w_hat ---
            res[row:row+3] = -w_omega * w_hat

            # --- Accel residual ---
            if not integrated_accel_constraint:
                # res_a = -sqrt(dt)/sigma_a * (a_hat - R_GtoI * g)
                res[row+3:row+6] = -w_accel * (a_hat - R_GtoI @ self._gravity)
            else:
                term = state._imu.vel() - self._gravity * dt + R_GtoI.T @ a_hat * dt
                res[row+3:row+6] = -w_accel_v * term

            # --- Gyro Jacobians ---
            # d(res_w)/d(theta) = 0
            # d(res_w)/d(bg)    = -w_omega * I  (C++ line 174)
            # NOTE: C++ omits intrinsic Jacobians (TODO in C++ at line 141)
            H[row:row+3, 0:3] = np.zeros((3, 3))
            H[row:row+3, 3:6] = -w_omega * np.eye(3)

            # --- Accel Jacobians ---
            if not integrated_accel_constraint:
                # d(res_a)/d(theta) = -w_accel * skew_x(R_jacob * g)  (C++ line 175)
                H[row+3:row+6, 0:3] = -w_accel * skew_x(R_jacob @ self._gravity)
                # d(res_a)/d(ba)    = -w_accel * I                     (C++ line 176)
                H[row+3:row+6, 6:9] = -w_accel * np.eye(3)
            else:
                # d(res)/d(theta) = -w_accel_v * R_jacob.T * skew_x(a_hat) * dt  (C++ line 178)
                H[row+3:row+6, 0:3] = -w_accel_v * (R_jacob.T @ skew_x(a_hat)) * dt
                # d(res)/d(ba)    = -w_accel_v * R_jacob.T * dt                   (C++ line 179)
                H[row+3:row+6, 6:9] = -w_accel_v * R_jacob.T * dt
                # d(res)/d(v)     = +w_accel_v * I                                (C++ line 180)
                H[row+3:row+6, 9:12] = w_accel_v * np.eye(3)

            dt_summed += dt
            valid_count += 1

        # Trim H/res to actual used rows
        if valid_count == 0:
            return False
        actual_rows = 6 * valid_count
        H = H[:actual_rows, :]
        res = res[:actual_rows]

        # Compress System
        H, res = UpdaterHelper.measurement_compress_inplace(H, res)
        if H.shape[0] < 1:
            return False

        # Measurement Noise Matrix R
        R = self._zupt_noise_multiplier * np.eye(res.shape[0])

        # Bias Propagation Matrix (Q_bias)
        Q_bias = np.eye(6)
        Q_bias[0:3, 0:3] *= dt_summed * self._noises.sigma_wb_2
        Q_bias[3:6, 3:6] *= dt_summed * self._noises.sigma_ab_2

        # Chi2 Check
        P_marg = StateHelper.get_marginal_covariance(state, Hx_order)
        if model_time_varying_bias:
            # Add bias noise to the covariance block for biases (indices 3:9)
            P_marg[3:9, 3:9] += Q_bias

        S = H @ P_marg @ H.T + R
        
        try:
            chi2 = np.dot(res.T, np.linalg.solve(S, res))
        except np.linalg.LinAlgError:
            chi2 = float('inf')

        # Threshold
        dof = res.shape[0]
        chi2_check = self.chi_squared_table.get(dof, scipy.stats.chi2.ppf(0.95, df=dof))

        # Disparity Check (Visual confirmation of stationarity)
        disparity_passed = False
        if override_with_disparity_check:
            # Check disparity between current state time and target timestamp
            time0_cam = state._timestamp
            time1_cam = timestamp
            
            disp_avg, disp_var, num_features = FeatureHelper.compute_disparity_two_frames(self._db, time0_cam, time1_cam)
            
            if disp_avg != -1 and disp_avg < self._zupt_max_disparity and num_features > 20:
                disparity_passed = True
                print(f"[ZUPT]: Passed disparity ({disp_avg:.3f} < {self._zupt_max_disparity})")

        # Decision
        vel_norm = np.linalg.norm(state._imu.vel())
        if not disparity_passed and (chi2 > self._options.chi2_multipler * chi2_check or vel_norm > self._zupt_max_velocity):
            self.last_zupt_state_timestamp = 0.0
            self.last_zupt_count = 0
            # print(f"[ZUPT]: Rejected |v|={vel_norm:.3f}, chi2={chi2:.3f}")
            return False

        print(f"[ZUPT]: Accepted |v|={vel_norm:.3f}, chi2={chi2:.3f}")

        # Cleanup tracks if consistent ZUPT
        if self.last_zupt_count >= 2:
            self._db.cleanup_measurements_exact(self.last_zupt_state_timestamp)

        # Update
        if not explicitly_enforce_zero_motion:
            
            # Propagate biases first (Random Walk)
            if model_time_varying_bias:
                Phi_bias = np.eye(6)
                Phi_order = [state._imu.bg(), state._imu.ba()]
                StateHelper.EKFPropagation(state, Phi_order, Phi_order, Phi_bias, Q_bias)

            # EKF Update
            StateHelper.EKFUpdate(state, Hx_order, H, res, R)
            state._timestamp = timestamp

        else:
            # Explicit Zero Motion Enforcement
            self._prop.propagate_and_clone(state, timestamp)

            # Hx_order: [clone0(6DOF), clone1(6DOF), imu.v(3DOF)] → 15 cols
            # clone0: cols 0:6  (theta0: 0:3, p0: 3:6)
            # clone1: cols 6:12 (theta1: 6:9, p1: 9:12)
            # vel:    cols 12:15
            H   = np.zeros((9, 15))
            res = np.zeros(9)
            R   = np.eye(9)

            time0_cam = self.last_zupt_state_timestamp
            time1_cam = timestamp

            clone0 = state._clones_IMU[time0_cam]
            clone1 = state._clones_IMU[time1_cam]

            R0 = clone0.Rot_fej() if state._options.do_fej else clone0.Rot()
            R1 = clone1.Rot_fej() if state._options.do_fej else clone1.Rot()

            # Residuals (C++ computes then multiplies all by -1)
            # After -1: res_ori = +log_so3(R0*R1^T), res_pos = -(p1-p0), res_vel = -vel
            res[0:3] = -log_so3(clone0.Rot() @ clone1.Rot().T)
            res[3:6] = clone1.pos() - clone0.pos()
            res[6:9] = state._imu.vel()
            res *= -1

            # Jacobians of log(R0 * R1^T):
            #   d/d(theta0) = +I   (C++ H.block(0, clone0_th_id, 3, 3) =  I)
            #   d/d(theta1) = -R0  (C++ H.block(0, clone1_th_id, 3, 3) = -R0)  BUG4 FIX
            H[0:3, 0:3] = np.eye(3)          # d/d(theta0)
            H[0:3, 6:9] = -R0                 # d/d(theta1) — was wrong in original

            # Jacobians of (p1 - p0):
            #   d/d(p0) = -I,  d/d(p1) = +I
            H[3:6, 3:6]   = -np.eye(3)        # d/d(p0)
            H[3:6, 9:12]  =  np.eye(3)        # d/d(p1)

            # Jacobian of velocity:
            #   d/d(v) = +I
            H[6:9, 12:15] = np.eye(3)

            # Tight noise (C++ values)
            R[0:3, 0:3] *= (1e-2) ** 2
            R[3:6, 3:6] *= (1e-1) ** 2
            R[6:9, 6:9] *= (1e-1) ** 2

            Hx_order = [clone0, clone1, state._imu.v()]

            StateHelper.EKFUpdate(state, Hx_order, H, res, R)

            # Marginalize the NEW clone (C++ marginalizes time1_cam)
            StateHelper.marginalize(state, clone1)
            del state._clones_IMU[time1_cam]

        self.last_zupt_state_timestamp = timestamp
        self.last_zupt_count += 1
        return True