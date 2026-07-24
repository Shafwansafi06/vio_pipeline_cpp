from hmac import new
from re import X
from matplotlib.font_manager import X11FontDirectories
import numpy as np
import threading
import math
from typing import List, Tuple, Optional

from vio_pipeline_v2.msckf.State import State
from vio_pipeline_v2.msckf.state_helper import StateHelper
from vio_pipeline_v2.msckf.StateOptions import StateOptions
from vio_pipeline_v2.msckf.NoiseManager import NoiseManager

from vio_pipeline_v2.core import ImuData

from vio_pipeline_v2.type import skew_x, Omega, rot_2_quat, quat_2_Rot, quat_multiply, quatnorm, exp_so3, Jr_so3, log_so3

class Propagator:
    def __init__(self, imu_noises, gravity_mag):
        self._noises = imu_noises
        
        self._noises.sigma_w_2 = self._noises.sigma_w**2
        self._noises.sigma_a_2 = self._noises.sigma_a**2
        self._noises.sigma_wb_2 = self._noises.sigma_wb**2
        self._noises.sigma_ab_2 = self._noises.sigma_ab**2


        self._gravity = np.array([0, 0, gravity_mag])

        self.imu_data = []
        self.imu_data_mtx = threading.RLock()

        self.last_prop_time_offset =0.0
        self.have_last_prop_time_offset= False

        self.cache_imu_valid = False
        self.cache_state_time = 0.0
        self.cache_state_est = None
        self.cache_state_covariance = None
        self.cache_t_off =0.0

    def feed_imu(self, message, oldest_time = -1):
        # print("being called Propagator.feed_imu")
        with self.imu_data_mtx:
            self.imu_data.append(message)
            if oldest_time >= 0:
                self.clean_old_imu_measurements(oldest_time- 0.10)
    
    def clean_old_imu_measurements(self, oldest_time):
        if not self.imu_data:
            return
        if (oldest_time < 0):
            return
        idx =0
        while idx < len(self.imu_data) and self.imu_data[idx].timestamp < oldest_time:
            idx +=1
        if idx >0:
            del self.imu_data[:idx]     
        # print(f"clean old imu measurements \n\n{self.imu_data=}")
    
    def invalidate_cache(self):
        self.cache_imu_valid = False



    def propagate_and_clone(self, state, timestamp):
        if state._timestamp == timestamp:
            print("[propagator]: propagate_and_clone called with same timestamp")
            exit(1)

        if state._timestamp > timestamp:
            print("[propagator]: propagate_and_clone called with backward timestamp")
            exit(1)

        if not self.have_last_prop_time_offset:
            self.last_prop_time_offset = state._calib_dt_CAMtoIMU.value()[0]
            self.have_last_prop_time_offset = True

        t_off_new = state._calib_dt_CAMtoIMU.value()[0]

        time0 = state._timestamp + self.last_prop_time_offset
        time1 = timestamp + t_off_new

        with self.imu_data_mtx:
            # print(f"propage and clone \n\n {self.imu_data=}")
            prop_data = self.select_imu_readings(self.imu_data, time0, time1,True)

        dim = state.imu_intrinsic_size() +15

        Phi_summed = np.eye(dim)
        Qd_summed = np.zeros((dim,dim))
        dt_summed =0.0
        if len(prop_data) > 1:
            for i in range(len(prop_data) - 1):
                
                # Compute F and Qdi for this interval
                F, Qdi = self.predict_and_compute(state, prop_data[i], prop_data[i+1])

                # Summation
                # Phi_summed = F * Phi_summed
                # Qd_summed = F * Qd_summed * F.T + Qdi
                Phi_summed = F @ Phi_summed
                Qd_summed = F @ Qd_summed @ F.T + Qdi
                
                # Ensure symmetry
                Qd_summed = 0.5 * (Qd_summed + Qd_summed.T)
                
                dt_summed += (prop_data[i+1].timestamp - prop_data[i].timestamp)

        # Verify time consistency
        # assert abs((time1 - time0) - dt_summed) < 1e-4

        # Last angular velocity for cloning (corrected)
        last_a = np.zeros(3)
        last_w = np.zeros(3)
        
        if prop_data:
            last_msg = prop_data[-1]
            Dw = State.Dm(state._options.imu_model, state._calib_imu_dw.value())
            Da = State.Dm(state._options.imu_model, state._calib_imu_da.value())
            Tg = State.Tg(state._calib_imu_tg.value())
            
            # Correct measurements
            # last_a = R_ACCtoIMU * Da * (am - ba)
            last_a = state._calib_imu_ACCtoIMU.Rot() @ Da @ (last_msg.am - state._imu.bias_a())
            
            # last_w = R_GYROtoIMU * Dw * (wm - bg - Tg*last_a)
            term = last_msg.wm - state._imu.bias_g() - Tg @ last_a
            last_w = state._calib_imu_GYROtoIMU.Rot() @ Dw @ term

        # Update Covariance using summed Phi/Qd
        Phi_order = []
        Phi_order.append(state._imu)
        
        if state._options.do_calib_imu_intrinsics:
            Phi_order.append(state._calib_imu_dw)
            Phi_order.append(state._calib_imu_da)
            if state._options.do_calib_imu_g_sensitivity:
                Phi_order.append(state._calib_imu_tg)
            
            if state._options.imu_model == StateOptions.ImuModel.KALIBR:
                Phi_order.append(state._calib_imu_GYROtoIMU)
            else:
                Phi_order.append(state._calib_imu_ACCtoIMU)

        StateHelper.EKFPropagation(state, Phi_order, Phi_order, Phi_summed, Qd_summed)

        # Set timestamp
        state._timestamp = timestamp
        self.last_prop_time_offset = t_off_new

        # Stochastic Cloning
        StateHelper.augment_clone(state, last_w)

    def propagate_only(self, state, timestamp):
        """
        Propagate IMU state and covariance to timestamp without cloning.
        Used by LIFO hovering mode where clone window is frozen.
        """
        if state._timestamp == timestamp:
            return

        if state._timestamp > timestamp:
            print("[propagator]: propagate_only called with backward timestamp")
            return

        if not self.have_last_prop_time_offset:
            self.last_prop_time_offset = state._calib_dt_CAMtoIMU.value()[0]
            self.have_last_prop_time_offset = True

        t_off_new = state._calib_dt_CAMtoIMU.value()[0]
        time0 = state._timestamp + self.last_prop_time_offset
        time1 = timestamp + t_off_new

        with self.imu_data_mtx:
            prop_data = self.select_imu_readings(self.imu_data, time0, time1, True)

        if len(prop_data) < 2:
            return

        dim = state.imu_intrinsic_size() + 15
        Phi_summed = np.eye(dim)
        Qd_summed = np.zeros((dim, dim))

        for i in range(len(prop_data) - 1):
            F, Qdi = self.predict_and_compute(state, prop_data[i], prop_data[i + 1])
            Phi_summed = F @ Phi_summed
            Qd_summed = F @ Qd_summed @ F.T + Qdi
            Qd_summed = 0.5 * (Qd_summed + Qd_summed.T)

        Phi_order = [state._imu]
        if state._options.do_calib_imu_intrinsics:
            Phi_order.append(state._calib_imu_dw)
            Phi_order.append(state._calib_imu_da)
            if state._options.do_calib_imu_g_sensitivity:
                Phi_order.append(state._calib_imu_tg)
            if state._options.imu_model == StateOptions.ImuModel.KALIBR:
                Phi_order.append(state._calib_imu_GYROtoIMU)
            else:
                Phi_order.append(state._calib_imu_ACCtoIMU)

        StateHelper.EKFPropagation(state, Phi_order, Phi_order, Phi_summed, Qd_summed)

        state._timestamp = timestamp
        self.last_prop_time_offset = t_off_new


    # @staticmethod
    def select_imu_readings(self, imu_data: List[ImuData], time0: float, time1: float, warn: bool = True) -> List[ImuData]:
        # print(f"{imu_data=}, {time0=},{time1=}")
        prop_data = []

        if not imu_data:
            if warn: print("[WARN]: No IMU measurements.")
            return prop_data
        # print(f"select imu reading \n\n {imu_data=}")
        for i in range(len(imu_data) - 1):
            curr_t = imu_data[i].timestamp
            next_t = imu_data[i+1].timestamp

            # CASE 1: Start of integration period (Split current)
            if next_t > time0 and curr_t < time0:
                data = self.interpolate_data(imu_data[i], imu_data[i+1], time0)
                prop_data.append(data)
                continue

            # CASE 2: Middle of integration period
            if curr_t >= time0 and next_t <= time1:
                prop_data.append(imu_data[i])
                continue

            # CASE 3: End of integration period
            if next_t > time1:
                # Handle edge case where we started late (dropped data)
                if curr_t > time1 and i == 0:
                    break
                
                # Add current point (either interpolated or raw)
                if curr_t > time1:
                    # Previous was < time1 (implied by loop structure or CASE 2 failure)
                    # Actually C++ logic handles i-1 here
                    data = self.interpolate_data(imu_data[i-1], imu_data[i], time1)
                    prop_data.append(data)
                else:
                    prop_data.append(imu_data[i])

                # If last added point doesn't match end time exactly, add interpolated end point
                if prop_data[-1].timestamp != time1:
                    data = self.interpolate_data(imu_data[i], imu_data[i+1], time1)
                    prop_data.append(data)
                
                break

        if not prop_data:
            if warn: print(f"[WARN]: No IMU measurements selected.")
            return prop_data

        # Stretch last measurement if we ran out of data
        if prop_data[-1].timestamp != time1:
            # if warn: print(f"[WARN]: Missing inertial measurements ({time1 - imu_data[-1].timestamp}s)")
            data = self.interpolate_data(imu_data[-2], imu_data[-1], time1)
            prop_data.append(data)

        # Remove Zero DT
        final_data = []
        if len(prop_data) > 0:
            final_data.append(prop_data[0])
            for i in range(1, len(prop_data)):
                if abs(prop_data[i].timestamp - prop_data[i-1].timestamp) > 1e-12:
                    final_data.append(prop_data[i])
        if not final_data:
            if warn: print(f"[WARN]: No IMU measurements selected.")
            return final_data
        
        return final_data


    def fast_state_propagate(self, state: State, timestamp: float, 
                             state_plus: np.ndarray, covariance: np.ndarray) -> bool:
        """
        Gets what the state and its covariance will be at a given timestamp.
        Propagates a clone of the current IMU state without updating the main filter state.
        
        Args:
            state: Pointer to state
            timestamp: Time to propagate to (IMU clock frame)
            state_plus: Output propagated state (13x1: q, p, v, w_local)
            covariance: Output propagated covariance (12x12: th, p, v, w)
            
        Returns:
            True if successful
        """
        
        # 1. Store/Check Cache
        if not self.cache_imu_valid:
            self.cache_state_time = state._timestamp
            self.cache_state_est = state._imu.value().copy() # Deep copy essential
            # Marginal covariance of just the IMU state (15x15)
            self.cache_state_covariance = StateHelper.get_marginal_covariance(state, [state._imu])
            self.cache_t_off = state._calib_dt_CAMtoIMU.value()[0]
            self.cache_imu_valid = True

        # 2. Select IMU readings
        # Calculate IMU time range
        time0 = self.cache_state_time + self.cache_t_off
        time1 = timestamp + self.cache_t_off
        
        with self.imu_data_mtx:
            prop_data = self.select_imu_readings(self.imu_data, time0, time1, warn=False)
            
        if len(prop_data) < 2:
            return False

        # Biases from cached state
        bias_g = self.cache_state_est[10:13]
        bias_a = self.cache_state_est[13:16]

        # Calibration matrices
        Dw = State.Dm(state._options.imu_model, state._calib_imu_dw.value())
        Da = State.Dm(state._options.imu_model, state._calib_imu_da.value())
        Tg = State.Tg(state._calib_imu_tg.value())
        R_ACCtoIMU = state._calib_imu_ACCtoIMU.Rot()
        R_GYROtoIMU = state._calib_imu_GYROtoIMU.Rot()

        # 3. Iterate through IMU measurements
        for i in range(len(prop_data) - 1):
            data_minus = prop_data[i]
            data_plus = prop_data[i+1]
            dt = data_plus.timestamp - data_minus.timestamp
            
            if dt <= 0: continue

            # Corrected Accel
            a_hat1 = R_ACCtoIMU @ Da @ (data_minus.am - bias_a)
            a_hat2 = R_ACCtoIMU @ Da @ (data_plus.am - bias_a)
            a_hat = 0.5 * (a_hat1 + a_hat2)

            # Corrected Gyro
            w_hat1 = R_GYROtoIMU @ Dw @ (data_minus.wm - bias_g - Tg @ a_hat1)
            w_hat2 = R_GYROtoIMU @ Dw @ (data_plus.wm - bias_g - Tg @ a_hat2)
            w_hat = 0.5 * (w_hat1 + w_hat2)

            # Current State Estimates from Cache
            q_curr = self.cache_state_est[0:4]
            R_Gtoi = quat_2_Rot(q_curr)
            p_iinG = self.cache_state_est[4:7]
            v_iinG = self.cache_state_est[7:10]

            # F Matrix (15x15) construction
            F = np.zeros((15, 15))
            
            # Theta block
            exp_w = exp_so3(-w_hat * dt)
            F[0:3, 0:3] = exp_w
            F[0:3, 9:12] = -exp_w @ Jr_so3(-w_hat * dt) * dt
            F[9:12, 9:12] = np.eye(3)
            
            # Velocity block
            F[6:9, 0:3] = -R_Gtoi.T @ skew_x(a_hat * dt)
            F[6:9, 6:9] = np.eye(3)
            F[6:9, 12:15] = -R_Gtoi.T * dt
            F[12:15, 12:15] = np.eye(3)
            
            # Position block
            F[3:6, 0:3] = -0.5 * R_Gtoi.T @ skew_x(a_hat * dt * dt)
            F[3:6, 3:6] = np.eye(3)
            F[3:6, 6:9] = np.eye(3) * dt
            F[3:6, 12:15] = -0.5 * R_Gtoi.T * dt * dt
            F[3:6, 3:6] = np.eye(3) # Redundant but safe

            # G Matrix (15x12)
            G = np.zeros((15, 12))
            G[0:3, 0:3] = -exp_w @ Jr_so3(-w_hat * dt) * dt
            G[6:9, 3:6] = -R_Gtoi.T * dt
            G[3:6, 3:6] = -0.5 * R_Gtoi.T * dt * dt
            G[9:12, 6:9] = np.eye(3)
            G[12:15, 9:12] = np.eye(3)

            # Noise Covariance
            Qc = np.zeros((12, 12))
            Qc[0:3, 0:3] = (self._noises.sigma_w_2 / dt) * np.eye(3)
            Qc[3:6, 3:6] = (self._noises.sigma_a_2 / dt) * np.eye(3)
            Qc[6:9, 6:9] = (self._noises.sigma_wb_2 / dt) * np.eye(3)
            Qc[9:12, 9:12] = (self._noises.sigma_ab_2 / dt) * np.eye(3)

            Qd = G @ Qc @ G.T
            Qd = 0.5 * (Qd + Qd.T)

            # Update Covariance
            self.cache_state_covariance = F @ self.cache_state_covariance @ F.T + Qd

            # Update Mean
            # Orientation
            dq = rot_2_quat(exp_w @ R_Gtoi)
            self.cache_state_est[0:4] = dq
            
            # Position
            self.cache_state_est[4:7] = (p_iinG + v_iinG * dt + 
                                         0.5 * R_Gtoi.T @ a_hat * dt * dt - 
                                         0.5 * self._gravity * dt * dt)
                                         
            # Velocity
            self.cache_state_est[7:10] = (v_iinG + R_Gtoi.T @ a_hat * dt - 
                                          self._gravity * dt)

        # 4. Finalize Cache
        self.cache_state_time = time1
        self.cache_t_off = 0.0

        # 5. Populate Output State
        q_Gtoi = self.cache_state_est[0:4]
        v_iinG = self.cache_state_est[7:10]
        p_iinG = self.cache_state_est[4:7]
        
        state_plus[:] = 0
        state_plus[0:4] = q_Gtoi
        state_plus[4:7] = p_iinG
        
        # Output Velocity in Local Frame (v_iinI)
        # v_local = R_GtoI * v_global
        state_plus[7:10] = quat_2_Rot(q_Gtoi) @ v_iinG
        
        # Last angular velocity
        last_m = prop_data[-1]
        last_a = R_ACCtoIMU @ Da @ (last_m.am - bias_a)
        last_w = R_GYROtoIMU @ Dw @ (last_m.wm - bias_g - Tg @ last_a)
        state_plus[10:13] = last_w

        # 6. Populate Output Covariance
        covariance[:] = 0
        
        # Transform velocity covariance to local frame
        # Phi transform: [I 0 0; 0 I 0; 0 0 R] for [th, p, v]
        Phi = np.eye(15)
        Phi[6:9, 6:9] = quat_2_Rot(q_Gtoi)
        
        cov_tmp = Phi @ self.cache_state_covariance @ Phi.T
        
        # Copy relevant blocks [th, p, v] -> 9x9
        covariance[0:9, 0:9] = cov_tmp[0:9, 0:9]
        
        # Angular velocity noise (Approx)
        dt_last = prop_data[-1].timestamp - prop_data[-2].timestamp
        covariance[9:12, 9:12] = (self._noises.sigma_w_2 / dt_last) * np.eye(3)
        
        return True

    def predict_and_compute(self, state: State, data_minus: ImuData, data_plus: ImuData) -> Tuple[np.ndarray, np.ndarray]:
        
        dt = data_plus.timestamp - data_minus.timestamp

        # 1. IMU intrinsic calibration estimates
        Dw = State.Dm(state._options.imu_model, state._calib_imu_dw.value())
        Da = State.Dm(state._options.imu_model, state._calib_imu_da.value())
        Tg = State.Tg(state._calib_imu_tg.value())

        # 2. Corrected imu acc measurements with our current biases
        a_hat1 = data_minus.am - state._imu.bias_a()
        a_hat2 = data_plus.am - state._imu.bias_a()
        a_hat_avg = 0.5 * (a_hat1 + a_hat2)

        # 3. Convert "raw" imu to its corrected frame using the IMU intrinsics
        a_uncorrected = a_hat_avg.copy()
        R_ACCtoIMU = state._calib_imu_ACCtoIMU.Rot()

        # Apply Corrections
        a_hat1 = R_ACCtoIMU @ Da @ a_hat1
        a_hat2 = R_ACCtoIMU @ Da @ a_hat2
        a_hat_avg = R_ACCtoIMU @ Da @ a_hat_avg

        # 4. Corrected imu gyro measurements with our current biases and gravity sensitivity
        w_hat1 = data_minus.wm - state._imu.bias_g() - Tg @ a_hat1
        w_hat2 = data_plus.wm - state._imu.bias_g() - Tg @ a_hat2
        w_hat_avg = 0.5 * (w_hat1 + w_hat2)

        # 5. Convert "raw" imu to its corrected frame using the IMU intrinsics
        w_uncorrected = w_hat_avg
        R_GYROtoIMU = state._calib_imu_GYROtoIMU.Rot()

        # Apply Corrections
        w_hat1 = R_GYROtoIMU @ Dw @ w_hat1
        w_hat2 = R_GYROtoIMU @ Dw @ w_hat2
        w_hat_avg = R_GYROtoIMU @ Dw @ w_hat_avg

        # Predict Mean
        Xi_sum = np.zeros((3, 18))

        if state._options.integration_method in [StateOptions.IntegrationMethod.RK4, StateOptions.IntegrationMethod.ANALYTICAL]:
            self.compute_Xi_sum(state, dt, w_hat_avg, a_hat_avg, Xi_sum)
        new_q = np.zeros(4); new_v = np.zeros(3); new_p = np.zeros(3)
        
        if state._options.integration_method == StateOptions.IntegrationMethod.ANALYTICAL:
            self.predict_mean_analytic(state, dt, w_hat_avg, a_hat_avg, new_q, new_v, new_p, Xi_sum)
        elif state._options.integration_method == StateOptions.IntegrationMethod.RK4:
            self.predict_mean_rk4(state, dt, w_hat1, a_hat1, w_hat2, a_hat2, new_q, new_v, new_p)
        else:
            self.predict_mean_discrete(state, dt, w_hat_avg, a_hat_avg, new_q, new_v, new_p)

        # Compute Jacobians F and G
        dim = state.imu_intrinsic_size() + 15
        F = np.zeros((dim, dim))
        G = np.zeros((dim, 12))

        if state._options.integration_method == StateOptions.IntegrationMethod.ANALYTICAL or state._options.integration_method == StateOptions.IntegrationMethod.RK4:
            self.compute_F_and_G_analytic(state, dt, w_hat_avg,a_hat_avg, w_uncorrected, a_uncorrected,new_q, new_v, new_p, Xi_sum, F, G)
        else:
            self.compute_F_and_G_discrete(state, dt, w_hat_avg, a_hat_avg, new_q, new_v, new_p, F, G)

        # Discrete Noise Covariance (Qd)
        # Qc (Continuous) -> Qd (Discrete)
        # Equations (129) and (130) of Trawny tech report
        Qc = np.zeros((12, 12))
        Qc[0:3, 0:3] = (self._noises.sigma_w**2 / dt) * np.eye(3)
        Qc[3:6, 3:6] = (self._noises.sigma_a**2 / dt) * np.eye(3)
        Qc[6:9, 6:9] = (self._noises.sigma_wb**2 / dt) * np.eye(3)
        Qc[9:12, 9:12] = (self._noises.sigma_ab**2 / dt) * np.eye(3)

        Qd = G @ Qc @ G.T
        Qd = 0.5 * (Qd + Qd.T)

        # Update State Mean In-Place
        imu_x = state._imu.value().copy()
        imu_x[0:4] = new_q
        imu_x[4:7] = new_p
        imu_x[7:10] = new_v # C++ uses block(7,0,3,1) for velocity

        
        state._imu.set_value(imu_x)
        state._imu.set_fej(imu_x)
        return F, Qd

       

    def predict_mean_discrete(self, state, dt, w_hat, a_hat, new_q, new_v, new_p):
        w_norm = np.linalg.norm(w_hat)
        R_Gtoi = state._imu.Rot()

        # Orientation
        if w_norm > 1e-12:
            half_th = 0.5 * w_norm * dt
            bigO = math.cos(half_th) * np.eye(4) + (1/w_norm)*math.sin(half_th) * Omega(w_hat)
        else:
            bigO = np.eye(4) + 0.5 * dt * Omega(w_hat)
        
        new_q[:] = quatnorm(bigO @ state._imu.quat())

        # Velocity: v + R^T * a * dt - g * dt
        # Note: OpenVINS defines global gravity usually as [0,0,9.81]. 
        # Equation: v_dot = R_ItoG^T * a - g. 
        # C++ Code: R_Gtoi.transpose() * a_hat * dt - _gravity * dt
        new_v[:] = state._imu.vel() + R_Gtoi.T @ a_hat * dt - self._gravity * dt

        # Position
        new_p[:] = state._imu.pos() + state._imu.vel() * dt + \
                   0.5 * R_Gtoi.T @ a_hat * dt * dt - \
                   0.5 * self._gravity * dt * dt

    def predict_mean_rk4(self, state, dt, w_hat1, a_hat1, w_hat2, a_hat2, new_q, new_v, new_p):
        w_hat = w_hat1.copy()
        a_hat = a_hat1.copy()
        # Linear interpolation for RK4 steps
        w_alpha = (w_hat2 - w_hat1) / dt
        a_jerk = (a_hat2 - a_hat1) / dt

        # State at 0
        q_0 = state._imu.quat()
        p_0 = state._imu.pos()
        v_0 = state._imu.vel()

        dq_0 = np.array([0, 0, 0, 1.0])

        # Helper for derivative
        def compute_deriv(q, v, w, a):
            # q_dot = 0.5 * Omega(w) * q
            q_dot = 0.5 * Omega(w) @ q
            p_dot = v
            R = quat_2_Rot(quat_multiply(q, q_0)) # Global to Current
            v_dot = R.T @ a - self._gravity
            return q_dot, p_dot, v_dot

        # K1 (Start)
        k1_q_dot, k1_p_dot, k1_v_dot = compute_deriv(dq_0, v_0, w_hat1, a_hat1)
        k1_q = k1_q_dot * dt; k1_p = k1_p_dot * dt; k1_v = k1_v_dot * dt

        # K2 (Mid)
        w_hat+= 0.5 * w_alpha * dt
        a_hat += 0.5 * a_jerk * dt
        
        dq_1 = quatnorm(dq_0 + 0.5 * k1_q)
        v_1 = v_0 + 0.5 * k1_v
        
        k2_q_dot, k2_p_dot, k2_v_dot = compute_deriv(dq_1, v_1, w_hat, a_hat)
        k2_q = k2_q_dot * dt; k2_p = k2_p_dot * dt; k2_v = k2_v_dot * dt

        # K3 (Mid)
        dq_2 = quatnorm(dq_0 + 0.5 * k2_q)
        v_2 = v_0 + 0.5 * k2_v
        
        k3_q_dot, k3_p_dot, k3_v_dot = compute_deriv(dq_2, v_2, w_hat, a_hat)
        k3_q = k3_q_dot * dt; k3_p = k3_p_dot * dt; k3_v = k3_v_dot * dt

        # K4 (End)
        dq_3 = quatnorm(dq_0 + k3_q)
        v_3 = v_0 + k3_v
        
        k4_q_dot, k4_p_dot, k4_v_dot = compute_deriv(dq_3, v_3, w_hat2, a_hat2)
        k4_q = k4_q_dot * dt; k4_p = k4_p_dot * dt; k4_v = k4_v_dot * dt

        # Integrate
        dq = quatnorm(dq_0 + (1/6)*k1_q + (1/3)*k2_q + (1/3)*k3_q + (1/6)*k4_q)
        new_q[:] = quat_multiply(dq, q_0)
        new_p[:] = p_0 + (1/6)*k1_p + (1/3)*k2_p + (1/3)*k3_p + (1/6)*k4_p
        new_v[:] = v_0 + (1/6)*k1_v + (1/3)*k2_v + (1/3)*k3_v + (1/6)*k4_v


    def compute_Xi_sum(self, state, dt, w_hat, a_hat, Xi_sum):
        
        w_norm = np.linalg.norm(w_hat)
        d_th = w_norm*dt 
        k_hat = np.zeros(3)
        if w_norm>1e-12:
            k_hat = w_hat / w_norm

        I3 = np.eye(3)
        d_t2 = dt**2
        d_t3 = dt**3
        w_norm2 = w_norm**2
        w_norm3 = w_norm**3
        cos_dth = math.cos(d_th)
        sin_dth = math.sin(d_th)
        d_th2 = d_th**2
        d_th3 = d_th**3
        sK = skew_x(k_hat)
        sK2 = sK @ sK
        sA = skew_x(a_hat)

        R_ktok1 = exp_so3(-w_hat*dt)
        Jr_ktok1 = Jr_so3(-w_hat*dt)

        Xi_1 = np.zeros((3,3))
        Xi_2 = np.zeros((3,3))
        Xi_3 = np.zeros((3,3))
        Xi_4 = np.zeros((3,3))

        small_w = (w_norm< (1.0/ 180*np.pi/2))

        if not small_w:
            Xi_1 = I3*dt + (1.0 -cos_dth)/ w_norm*sK + (dt- sin_dth/w_norm)*sK2
            Xi_2 = 0.5 * d_t2 * I3 + (d_th - sin_dth) / w_norm2 * sK + (0.5 * d_t2 - (1.0 - cos_dth) / w_norm2) * sK2

            # first order integration with constant omega and constant acc
            term1 = (sin_dth - d_th) / w_norm2 * (sA @ sK)
            term2 = (sin_dth - d_th * cos_dth) / w_norm2 * (sK @ sA)
            term3 = (0.5 * d_t2 - (1.0 - cos_dth) / w_norm2) * (sA @ sK2)
            
            # Complex cross terms
            # (sK2 * sA + k_hat.dot(a_hat) * sK)
            cross_term_1 = (sK2 @ sA) + np.dot(k_hat, a_hat) * sK
            coeff_1 = (0.5 * d_t2 + (1.0 - cos_dth - d_th * sin_dth) / w_norm2)
            
            # k_hat.dot(a_hat) * sK2
            cross_term_2 = np.dot(k_hat, a_hat) * sK2
            coeff_2 = (3 * sin_dth - 2 * d_th - d_th * cos_dth) / w_norm2
            
            Xi_3 = 0.5 * d_t2 * sA + term1 + term2 + term3 + coeff_1 * cross_term_1 - coeff_2 * cross_term_2

            # second order integration with constant omega and constant acc
            # Terms for Xi_4
            # (sA @ sK)
            xi4_term1 = (2 * (1.0 - cos_dth) - d_th2) / (2 * w_norm3) * (sA @ sK)
            
            # (sK @ sA)
            xi4_term2 = ((2 * (1.0 - cos_dth) - d_th * sin_dth) / w_norm3) * (sK @ sA)
            
            # (sA @ sK2)
            xi4_term3 = ((sin_dth - d_th) / w_norm3 + d_t3 / 6) * (sA @ sK2)
            
            # (sK2 @ sA)
            xi4_coeff4 = (d_th - 2 * sin_dth + d_th3 / 6 + d_th * cos_dth) / w_norm3
            xi4_term4 = xi4_coeff4 * (sK2 @ sA) 
            
            # k_hat.dot(a_hat) * sK
            xi4_term5 = xi4_coeff4 * (np.dot(k_hat, a_hat) * sK)
            
            # k_hat.dot(a_hat) * sK2
            xi4_coeff6 = (4 * cos_dth - 4 + d_th2 + d_th * sin_dth) / w_norm3
            xi4_term6 = xi4_coeff6 * (np.dot(k_hat, a_hat) * sK2)
            
            Xi_4 = (1.0/6.0) * d_t3 * sA + xi4_term1 + xi4_term2 + xi4_term3 + xi4_term4 + xi4_term5 + xi4_term6

        else:
            # Small angle approximation
            
            # first order rotation integration with constant omega
            Xi_1 = dt * (I3 + sin_dth * sK + (1.0 - cos_dth) * sK2)

            # second order rotation integration with constant omega
            Xi_2 = 0.5 * dt * Xi_1

            # first order integration with constant omega and constant acc
            # term = sA + sin_dth * (-sA * sK + sK * sA + k_hat.dot(a_hat) * sK2) + ...
            term_small = (sA + 
                          sin_dth * (-sA @ sK + sK @ sA + np.dot(k_hat, a_hat) * sK2) + 
                          (1.0 - cos_dth) * (sA @ sK2 + sK2 @ sA + np.dot(k_hat, a_hat) * sK))
            
            Xi_3 = 0.5 * d_t2 * term_small

            # second order integration with constant omega and constant acc
            Xi_4 = (1.0/3.0) * dt * Xi_3

        # Store the integrated parameters
        # Xi_sum is passed by reference (numpy array slice or object)
        # Xi_sum structure: [R_ktok1 | Xi_1 | Xi_2 | Jr_ktok1 | Xi_3 | Xi_4]
        Xi_sum.fill(0)
        Xi_sum[0:3, 0:3] = R_ktok1
        Xi_sum[0:3, 3:6] = Xi_1
        Xi_sum[0:3, 6:9] = Xi_2
        Xi_sum[0:3, 9:12] = Jr_ktok1
        Xi_sum[0:3, 12:15] = Xi_3
        Xi_sum[0:3, 15:18] = Xi_4
    
    def predict_mean_analytic(self, state, dt, w_hat, a_hat, new_q, new_v, new_p, Xi_sum):

        R_Gtok =state._imu.Rot()

        R_ktok1_mat = Xi_sum[0:3, 0:3]
        q_ktok1 = rot_2_quat(R_ktok1_mat)

        Xi_1 = Xi_sum[0:3, 3:6]
        Xi_2 = Xi_sum[0:3, 6:9]

        new_q[:] = quat_multiply(q_ktok1, state._imu.quat())
        term_v = R_Gtok.T @ Xi_1 @ a_hat
        new_v[:] = state._imu.vel() + term_v - self._gravity * dt

        term_p = R_Gtok.T @ Xi_2 @ a_hat
        new_p[:] = state._imu.pos() + state._imu.vel() * dt + term_p - 0.5 * self._gravity * dt * dt

    def compute_F_and_G_analytic(self, state: State, dt: float, w_hat: np.ndarray,
                                 a_hat: np.ndarray, w_uncorrected: np.ndarray,
                                 a_uncorrected: np.ndarray, new_q: np.ndarray, new_v: np.ndarray,
                                 new_p: np.ndarray, Xi_sum: np.ndarray, F: np.ndarray,
                                 G: np.ndarray):
        """Analytically compute state transition matrix F and noise Jacobian G."""
        
        # Get the locations of each entry of the imu state
        local_size = 0
        th_id = local_size; local_size += state._imu.q().size()
        p_id = local_size; local_size += state._imu.p().size()
        v_id = local_size; local_size += state._imu.v().size()
        bg_id = local_size; local_size += state._imu.bg().size()
        ba_id = local_size; local_size += state._imu.ba().size()

        # Calibration IDs
        Dw_id = -1
        Da_id = -1
        Tg_id = -1
        th_atoI_id = -1
        th_wtoI_id = -1
        
        if state._options.do_calib_imu_intrinsics:
            Dw_id = local_size; local_size += state._calib_imu_dw.size()
            Da_id = local_size; local_size += state._calib_imu_da.size()
            if state._options.do_calib_imu_g_sensitivity:
                Tg_id = local_size; local_size += state._calib_imu_tg.size()
            
            if state._options.imu_model == StateOptions.ImuModel.KALIBR:
                th_wtoI_id = local_size; local_size += state._calib_imu_GYROtoIMU.size()
            else:
                th_atoI_id = local_size; local_size += state._calib_imu_ACCtoIMU.size()

        # Previous State
        if state._options.do_fej:
            R_k = state._imu.Rot_fej()
            v_k = state._imu.vel_fej()
            p_k = state._imu.pos_fej()
        else:
            R_k = state._imu.Rot()
            v_k = state._imu.vel()
            p_k = state._imu.pos()

        dR_ktok1 = quat_2_Rot(new_q) @ R_k.T

        # Calibration Matrices
        Dw = State.Dm(state._options.imu_model, state._calib_imu_dw.value())
        Da = State.Dm(state._options.imu_model, state._calib_imu_da.value())
        Tg = State.Tg(state._calib_imu_tg.value())
        R_atoI = state._calib_imu_ACCtoIMU.Rot()
        R_wtoI = state._calib_imu_GYROtoIMU.Rot()
        
        a_k = R_atoI @ Da @ a_uncorrected
        w_k = R_wtoI @ Dw @ w_uncorrected

        # Extract Integration Components
        Xi_1 = Xi_sum[0:3, 3:6]
        Xi_2 = Xi_sum[0:3, 6:9]
        Jr_ktok1 = Xi_sum[0:3, 9:12]
        Xi_3 = Xi_sum[0:3, 12:15]
        Xi_4 = Xi_sum[0:3, 15:18]

        # F Matrix Filling
        F[th_id:th_id+3, th_id:th_id+3] = dR_ktok1
        term_p_skew = skew_x(new_p - p_k - v_k * dt + 0.5 * self._gravity * dt * dt)
        F[p_id:p_id+3, th_id:th_id+3] = -term_p_skew @ R_k.T
        term_v_skew = skew_x(new_v - v_k + self._gravity * dt)
        F[v_id:v_id+3, th_id:th_id+3] = -term_v_skew @ R_k.T

        F[p_id:p_id+3, p_id:p_id+3] = np.eye(3)
        F[p_id:p_id+3, v_id:v_id+3] = np.eye(3) * dt
        F[v_id:v_id+3, v_id:v_id+3] = np.eye(3)

        F[th_id:th_id+3, bg_id:bg_id+3] = -dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ Dw
        F[p_id:p_id+3, bg_id:bg_id+3] = R_k.T @ Xi_4 @ R_wtoI @ Dw
        F[v_id:v_id+3, bg_id:bg_id+3] = R_k.T @ Xi_3 @ R_wtoI @ Dw
        F[bg_id:bg_id+3, bg_id:bg_id+3] = np.eye(3)

        F[th_id:th_id+3, ba_id:ba_id+3] = dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ Dw @ Tg @ R_atoI @ Da
        F[p_id:p_id+3, ba_id:ba_id+3] = -R_k.T @ (Xi_2 + Xi_4 @ R_wtoI @ Dw @ Tg) @ R_atoI @ Da
        F[v_id:v_id+3, ba_id:ba_id+3] = -R_k.T @ (Xi_1 + Xi_3 @ R_wtoI @ Dw @ Tg) @ R_atoI @ Da
        F[ba_id:ba_id+3, ba_id:ba_id+3] = np.eye(3)

        # 1. Omega Intrinsics Dw
        if Dw_id != -1:
            H_Dw = self.compute_H_Dw(state, w_uncorrected)
            dim = state._calib_imu_dw.size()
            F[th_id:th_id+3, Dw_id:Dw_id+dim] = dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ H_Dw
            F[p_id:p_id+3, Dw_id:Dw_id+dim] = -R_k.T @ Xi_4 @ R_wtoI @ H_Dw
            F[v_id:v_id+3, Dw_id:Dw_id+dim] = -R_k.T @ Xi_3 @ R_wtoI @ H_Dw
            F[Dw_id:Dw_id+dim, Dw_id:Dw_id+dim] = np.eye(dim)

        # 2. Accel Intrinsics Da
        if Da_id != -1:
            H_Da = self.compute_H_Da(state, a_uncorrected)
            dim = state._calib_imu_da.size()
            F[th_id:th_id+3, Da_id:Da_id+dim] = -dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ Dw @ Tg @ R_atoI @ H_Da
            F[p_id:p_id+3, Da_id:Da_id+dim] = R_k.T @ (Xi_2 + Xi_4 @ R_wtoI @ Dw @ Tg) @ R_atoI @ H_Da
            F[v_id:v_id+3, Da_id:Da_id+dim] = R_k.T @ (Xi_1 + Xi_3 @ R_wtoI @ Dw @ Tg) @ R_atoI @ H_Da
            F[Da_id:Da_id+dim, Da_id:Da_id+dim] = np.eye(dim)

        # 3. Gravity Sensitivity Tg
        if Tg_id != -1:
            H_Tg = self.compute_H_Tg(state, a_k)
            dim = state._calib_imu_tg.size()
            F[th_id:th_id+3, Tg_id:Tg_id+dim] = -dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ Dw @ H_Tg
            F[p_id:p_id+3, Tg_id:Tg_id+dim] = R_k.T @ Xi_4 @ R_wtoI @ Dw @ H_Tg
            F[v_id:v_id+3, Tg_id:Tg_id+dim] = R_k.T @ Xi_3 @ R_wtoI @ Dw @ H_Tg
            F[Tg_id:Tg_id+dim, Tg_id:Tg_id+dim] = np.eye(dim)

        # 4. R_ACCtoIMU
        if th_atoI_id != -1:
            term_skew = skew_x(a_k)
            F[th_id:th_id+3, th_atoI_id:th_atoI_id+3] = -dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ Dw @ Tg @ term_skew
            F[p_id:p_id+3, th_atoI_id:th_atoI_id+3] = R_k.T @ (Xi_2 + Xi_4 @ R_wtoI @ Dw @ Tg) @ term_skew
            F[v_id:v_id+3, th_atoI_id:th_atoI_id+3] = R_k.T @ (Xi_1 + Xi_3 @ R_wtoI @ Dw @ Tg) @ term_skew
            F[th_atoI_id:th_atoI_id+3, th_atoI_id:th_atoI_id+3] = np.eye(3)

        # 5. R_GYROtoIMU
        if th_wtoI_id != -1:
            term_skew = skew_x(w_k)
            F[th_id:th_id+3, th_wtoI_id:th_wtoI_id+3] = dR_ktok1 @ Jr_ktok1 * dt @ term_skew
            F[p_id:p_id+3, th_wtoI_id:th_wtoI_id+3] = -R_k.T @ Xi_4 @ term_skew
            F[v_id:v_id+3, th_wtoI_id:th_wtoI_id+3] = -R_k.T @ Xi_3 @ term_skew
            F[th_wtoI_id:th_wtoI_id+3, th_wtoI_id:th_wtoI_id+3] = np.eye(3)

        # G Matrix
        G[th_id:th_id+3, 0:3] = -dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ Dw
        G[p_id:p_id+3, 0:3] = R_k.T @ Xi_4 @ R_wtoI @ Dw
        G[v_id:v_id+3, 0:3] = R_k.T @ Xi_3 @ R_wtoI @ Dw
        G[th_id:th_id+3, 3:6] = dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ Dw @ Tg @ R_atoI @ Da
        G[p_id:p_id+3, 3:6] = -R_k.T @ (Xi_2 + Xi_4 @ R_wtoI @ Dw @ Tg) @ R_atoI @ Da
        G[v_id:v_id+3, 3:6] = -R_k.T @ (Xi_1 + Xi_3 @ R_wtoI @ Dw @ Tg) @ R_atoI @ Da
        G[bg_id:bg_id+3, 6:9] = dt * np.eye(3)
        G[ba_id:ba_id+3, 9:12] = dt * np.eye(3)


    def compute_F_and_G_discrete(self, state, dt, w_hat, a_hat, w_uncorrected, a_uncorrected, 
                                 new_q, new_v, new_p, F, G):
        
        # IDs for block matrix
        local_size = 0
        th_id = local_size; local_size += 3
        p_id = local_size; local_size += 3
        v_id = local_size; local_size += 3
        bg_id = local_size; local_size += 3
        ba_id = local_size; local_size += 3

        # Optional IDs
        Dw_id = Da_id = Tg_id = th_atoI_id = th_wtoI_id = -1
        
        if state._options.do_calib_imu_intrinsics:
            Dw_id = local_size; local_size += 6
            Da_id = local_size; local_size += 6
            if state._options.do_calib_imu_g_sensitivity:
                Tg_id = local_size; local_size += 9
            
            if state._options.imu_model == StateOptions.ImuModel.KALIBR:
                th_wtoI_id = local_size; local_size += 3
            else:
                th_atoI_id = local_size; local_size += 3

        # Current State (k)
        R_k = state._imu.Rot()
        v_k = state._imu.vel()
        p_k = state._imu.pos()
        if state._options.do_fej:
            R_k = state._imu.Rot_fej()
            v_k = state._imu.vel_fej()
            p_k = state._imu.pos_fej()

        # Delta Rotation
        dR_ktok1 = quat_2_Rot(new_q) @ R_k.T

        # Matrices
        Dw = State.Dm(state._options.imu_model, state._calib_imu_dw.value())
        Da = State.Dm(state._options.imu_model, state._calib_imu_da.value())
        Tg = State.Tg(state._calib_imu_tg.value())
        R_atoI = state._calib_imu_ACCtoIMU.Rot()
        R_wtoI = state._calib_imu_GYROtoIMU.Rot()
        
        # Corrected raw inputs
        a_k = R_atoI @ Da @ a_uncorrected
        w_k = R_wtoI @ Dw @ w_uncorrected
        
        # Jacobian of SO3
        # Jr_ktok1 = Jr_so3(log_so3(dR_ktok1))
        # Note: log_so3 returns rotation vector. Jr takes rotation vector.
        Jr_ktok1 = Jr_so3(log_so3(dR_ktok1))

        # --- F Construction ---
        
        # Theta
        F[th_id:th_id+3, th_id:th_id+3] = dR_ktok1
        F[th_id:th_id+3, bg_id:bg_id+3] = -dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ Dw
        F[th_id:th_id+3, ba_id:ba_id+3] = dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ Dw @ Tg @ R_atoI @ Da

        # Position
        term_p = new_p - p_k - v_k * dt + 0.5 * self._gravity * dt * dt
        F[p_id:p_id+3, th_id:th_id+3] = -skew_x(term_p) @ R_k.T
        F[p_id:p_id+3, p_id:p_id+3] = np.eye(3)
        F[p_id:p_id+3, v_id:v_id+3] = np.eye(3) * dt
        F[p_id:p_id+3, ba_id:ba_id+3] = -0.5 * R_k.T * dt * dt @ R_atoI @ Da

        # Velocity
        term_v = new_v - v_k + self._gravity * dt
        F[v_id:v_id+3, th_id:th_id+3] = -skew_x(term_v) @ R_k.T
        F[v_id:v_id+3, v_id:v_id+3] = np.eye(3)
        F[v_id:v_id+3, ba_id:ba_id+3] = -R_k.T * dt @ R_atoI @ Da

        # Biases (Identity)
        F[bg_id:bg_id+3, bg_id:bg_id+3] = np.eye(3)
        F[ba_id:ba_id+3, ba_id:ba_id+3] = np.eye(3)

        # Calibration Blocks
        if Dw_id != -1:
            H_Dw = self.compute_H_Dw(state, w_uncorrected)
            F[th_id:th_id+3, Dw_id:Dw_id+6] = dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ H_Dw
            F[Dw_id:Dw_id+6, Dw_id:Dw_id+6] = np.eye(6)

        if Da_id != -1:
            H_Da = self.compute_H_Da(state, a_uncorrected)
            F[th_id:th_id+3, Da_id:Da_id+6] = -dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ Dw @ Tg @ R_atoI @ H_Da
            F[p_id:p_id+3, Da_id:Da_id+6]   = 0.5 * R_k.T * dt * dt @ R_atoI @ H_Da
            F[v_id:v_id+3, Da_id:Da_id+6]   = R_k.T * dt @ R_atoI @ H_Da
            F[Da_id:Da_id+6, Da_id:Da_id+6] = np.eye(6)

        if Tg_id != -1:
            H_Tg = self.compute_H_Tg(state, a_k)
            F[th_id:th_id+3, Tg_id:Tg_id+9] = -dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ Dw @ H_Tg
            F[Tg_id:Tg_id+9, Tg_id:Tg_id+9] = np.eye(9)

        if th_atoI_id != -1: # Acc Alignment
            F[th_id:th_id+3, th_atoI_id:th_atoI_id+3] = -dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ Dw @ Tg @ skew_x(a_k)
            F[p_id:p_id+3, th_atoI_id:th_atoI_id+3] = 0.5 * R_k.T * dt * dt @ skew_x(a_k)
            F[v_id:v_id+3, th_atoI_id:th_atoI_id+3] = R_k.T * dt @ skew_x(a_k)
            F[th_atoI_id:th_atoI_id+3, th_atoI_id:th_atoI_id+3] = np.eye(3)

        if th_wtoI_id != -1: # Gyro Alignment
            F[th_id:th_id+3, th_wtoI_id:th_wtoI_id+3] = dR_ktok1 @ Jr_ktok1 * dt @ skew_x(w_k)
            F[th_wtoI_id:th_wtoI_id+3, th_wtoI_id:th_wtoI_id+3] = np.eye(3)

        # --- G Construction ---
        G[th_id:th_id+3, 0:3] = -dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ Dw
        G[th_id:th_id+3, 3:6] = dR_ktok1 @ Jr_ktok1 * dt @ R_wtoI @ Dw @ Tg @ R_atoI @ Da
        G[v_id:v_id+3, 3:6]   = -R_k.T * dt @ R_atoI @ Da
        G[p_id:p_id+3, 3:6]   = -0.5 * R_k.T * dt * dt @ R_atoI @ Da
        G[bg_id:bg_id+3, 6:9] = dt * np.eye(3)
        G[ba_id:ba_id+3, 9:12] = dt * np.eye(3)

    def compute_H_Dw(self, state, w_uncorrected):
        w = w_uncorrected
        I = np.eye(3)
        H = np.zeros((3, 6))
        if state._options.imu_model == StateOptions.ImuModel.KALIBR:
            H[:, 0:3] = w[0] * I
            H[:, 3] = w[1] * I[:, 1]
            H[:, 4] = w[1] * I[:, 2]
            H[:, 5] = w[2] * I[:, 2]
        else: # RPNG
            H[:, 0] = w[0] * I[:, 0]
            H[:, 1] = w[1] * I[:, 0]
            H[:, 2] = w[1] * I[:, 1]
            H[:, 3:6] = w[2] * I
        return H

    def compute_H_Da(self, state, a_uncorrected):
        a = a_uncorrected
        I = np.eye(3)
        H = np.zeros((3, 6))
        if state._options.imu_model == StateOptions.ImuModel.KALIBR:
            H[:, 0:3] = a[0] * I
            H[:, 3] = a[1] * I[:, 1]
            H[:, 4] = a[1] * I[:, 2]
            H[:, 5] = a[2] * I[:, 2]
        else:
            H[:, 0] = a[0] * I[:, 0]
            H[:, 1] = a[1] * I[:, 0]
            H[:, 2] = a[1] * I[:, 1]
            H[:, 3:6] = a[2] * I
        return H

    def compute_H_Tg(self, state, a_inI):
        a = a_inI
        I = np.eye(3)
        H = np.zeros((3, 9))
        H[:, 0:3] = a[0] * I
        H[:, 3:6] = a[1] * I
        H[:, 6:9] = a[2] * I
        return H

    @staticmethod
    def interpolate_data(data0: ImuData, data1: ImuData, time: float) -> ImuData:
        lambda_val = (time - data0.timestamp) / (data1.timestamp - data0.timestamp)
        new_wm = (1 - lambda_val) * data0.wm + lambda_val * data1.wm
        new_am = (1 - lambda_val) * data0.am + lambda_val * data1.am
        return ImuData(timestamp=time, wm=new_wm, am=new_am)