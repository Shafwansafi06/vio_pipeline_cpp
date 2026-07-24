# imports
from mimetypes import init

from numpy import true_divide
from vio_pipeline_v2.initialize.static import StaticInitializer
# from vio_pipeline_v2.initialize.dynamic import DynamicInitializer

from vio_pipeline_v2.core.feature_helper import FeatureHelper
DynamicInitializer = None



# dynamic initialization of modules



class InertialInitializer:
    def __init__(self, config, database):
        self.config = config
        self.database = database
        self.Imu_data = []

        self.init_static = StaticInitializer(self.config,self.database, self.Imu_data)
        self.init_dynamic = None
        if self.config.init_dyn_use:
            try:
                self.init_dynamic = DynamicInitializer(self.config,self.database, self.Imu_data)
            except Exception as e:
                print("Dynamic initializer failed to initialize:", e)
                self.init_dynamic = None
        



    def feed_imu(self, imu_msg, oldest_time=-1):
        self.Imu_data.append(imu_msg)

        # Remove old IMU data
        if oldest_time is not None:
            self.Imu_data[:] = [msg for msg in self.Imu_data if msg.timestamp >= oldest_time]


    def initialize(self,timestamp, covariance ,order, t_imu , wait_for_jerk=True):
        
        newest_time = -1
        for feat_id , feat in self.database.get_internal_data().items():
            for cam_id, camtimepair in feat.timestamps.items():
                for time in camtimepair:
                    newest_time = max(newest_time, time)
                
        oldest_time = newest_time - self.config.init_window_time - 0.10

        if newest_time < 0 or oldest_time < 0:
            return False
        
        self.database.cleanup_measurements(oldest_time)
        self.Imu_data[:] = [msg for msg in self.Imu_data if msg.timestamp >= oldest_time + self.config.init_calib_camImu_dt]

        disparity_detected_moving_1to0 = False
        disparity_detected_moving_2to1 = False

        if self.config.init_max_disparity > 0:
            newest_time_allowed = newest_time -0.5*self.config.init_window_time
            num_features0 = 0
            num_features1 = 0
            avg_disp0 = 0.0
            avg_disp1 = 0.0
            avg_var0 = 0.0
            avg_var1 = 0.0

            avg_disp0, avg_var0, num_features0 = FeatureHelper.compute_disparity(self.database, newest_time_allowed)
            avg_disp1, avg_var1, num_features1 = FeatureHelper.compute_disparity(self.database, newest_time, newest_time_allowed)

            feature_threshold = 15

            if num_features0 < feature_threshold or num_features1 < feature_threshold:
                return False  
            
            disparity_detected_moving_1to0 = (avg_disp0 > self.config.init_max_disparity)
            disparity_detected_moving_2to1 = (avg_disp1 > self.config.init_max_disparity)

        has_jerk = (not disparity_detected_moving_1to0) and ( disparity_detected_moving_2to1)
        is_still = (not disparity_detected_moving_1to0) and ( not disparity_detected_moving_2to1)

        if not hasattr(self, '_init_attempt'):
            self._init_attempt = 0
        self._init_attempt += 1
        if self._init_attempt % 50 == 1:
            print(f"[init] attempt={self._init_attempt} disp0={avg_disp0:.2f} disp1={avg_disp1:.2f} "
                  f"thresh={self.config.init_max_disparity:.1f} still={is_still} jerk={has_jerk} wfj={wait_for_jerk}")

        if (((has_jerk and wait_for_jerk ) or (is_still and not wait_for_jerk)) and self.config.init_imu_thresh >0.0):
            print(f"[init] Entering static init at attempt {self._init_attempt}")
            return self.init_static.initialize(timestamp, covariance, order, t_imu, wait_for_jerk)
        
        elif self.config.init_dyn_use and (not is_still):
            print("Using dynamic initializer.")
            if self.init_dynamic:
                clone_imu ={}
                features_slam ={}
                success, ts, cov, order,_,_ = self.init_dynamic.initialize(timestamp, covariance, order, t_imu, clone_imu, features_slam)
                return success, ts, cov, order
            else:
                print("Dynamic initializer not available.")
                return False
        else:
            pass  # Init conditions not met, will retry

        return False
