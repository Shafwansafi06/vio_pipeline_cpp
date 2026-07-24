
import numpy as np
import math


from vio_pipeline_v2.type import quat_2_Rot, rot_2_quat, IMU
from vio_pipeline_v2.initialize.InitailizerHelper import InitializerHelper


class StaticInitializer:
    def __init__(self, config, database, Imu_data):
        self.config = config
        self.database = database
        self.Imu_data = Imu_data

    def initialize(self,timestamp, covariance, order, t_imu, wait_for_jerk=True):
        if len(self.Imu_data) < 2:
            return False
        
        newest_time = sorted([msg.timestamp for msg in self.Imu_data])[-1]
        oldest_time = self.Imu_data[0].timestamp

        # Check if we have enough IMU data
        if (newest_time - oldest_time) < self.config.init_window_time:
            return False
        
        # Split IMU data into two windows
        window_1to0 = []
        window_2to1 = []

        t_mid = newest_time - 0.5 * self.config.init_window_time
        t_start = newest_time - self.config.init_window_time

        for data in self.Imu_data:
            if data.timestamp >= t_start and data.timestamp < t_mid:
                window_2to1.append(data)
            elif data.timestamp >= t_mid and data.timestamp <= newest_time:
                window_1to0.append(data)

        if len(window_1to0) < 2 or len(window_2to1) < 2:
            return False
        
        #variance for window 1to0
        a_val_1to0 = np.array([d.am for d in window_1to0])
        a_avg_1to0 = np.mean(a_val_1to0, axis=0)

        diffs_1to0 = a_val_1to0 - a_avg_1to0
        a_var_1to0 = np.sum(np.sum(diffs_1to0**2 , axis=1))
        a_var_1to0 = math.sqrt(a_var_1to0 / (len(window_1to0)-1))
        
        #variance for window 2to1
        a_val_2to1 = np.array([d.am for d in window_2to1])
        w_val_2to1 = np.array([d.wm for d in window_2to1])

        a_avg_2to1 = np.mean(a_val_2to1,axis=0)
        w_avg_2to1 = np.mean(w_val_2to1, axis=0)

        diffs_2to1 = a_val_2to1 - a_avg_2to1
        a_var_2to1 = np.sum(np.sum(diffs_2to1**2 , axis=1))
        a_var_2to1 = math.sqrt(a_var_2to1/(len(window_2to1)-1))

        thresh = self.config.init_imu_thresh

        if a_var_1to0<thresh and wait_for_jerk:
            return False
        
        if a_var_2to1 > thresh and wait_for_jerk:
            return False
        
        if (a_var_1to0> thresh or a_var_2to1>thresh) and not wait_for_jerk:
            return False
        
        z_axis = a_avg_2to1 / np.linalg.norm(a_avg_2to1)
        Ro = np.eye(3)
        InitializerHelper.gram_schmidt(z_axis, Ro)
        q_GtoI= rot_2_quat(Ro)

        gravity_inG = np.array([0.0, 0.0, self.config.gravity_mag])
        bg = w_avg_2to1

        rot_mat= quat_2_Rot(q_GtoI)

        ba = a_avg_2to1 - np.dot(rot_mat, gravity_inG)

        imu_state= np.zeros(16)
        imu_state[0:4] = q_GtoI
        imu_state[10:13] = bg
        imu_state[13:16] = ba

        if t_imu is not None:
            t_imu.set_value(imu_state)
            t_imu.set_fej(imu_state)

        cov_dim = t_imu.size()
        temp_cov = (0.02**2) * np.eye(cov_dim)

        temp_cov[0:3, 0:3] = (0.02**2)*np.eye(3)
        temp_cov[3:6, 3:6] = (0.02**2)*np.eye(3)
        temp_cov[6:9, 6:9] = (0.02**2)*np.eye(3)

        # covariance = (0.02**2)* np.eye(cov_dim)

        # covariance[0:3,0:3] = (0.02**2)*np.eye(3)
        # covariance[0:3,0:3] = (0.05**2)*np.eye(3)
        # covariance[0:3,0:3] = (0.01**2)*np.eye(3)
        if isinstance(covariance, list):
            if len(covariance)>0:
                covariance[0] = temp_cov
            else:
                covariance.append(temp_cov)
        else:
            print("inplace update failed")
            
        order.clear()
        order.append(t_imu)

        if isinstance(timestamp, list):
            timestamp[0]= window_2to1[-1].timestamp
        else:
            timestamp = [window_2to1[-1].timestamp]

        return True