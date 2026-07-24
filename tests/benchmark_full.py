# /// script
# dependencies = [
#   "numpy",
# ]
# ///
import sys
import os
import time
import math
from unittest.mock import MagicMock

# Mock dependencies to run without ROS / OpenCV / Scipy
sys.modules['cv2'] = MagicMock()
sys.modules['rospkg'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.covariance'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.font_manager'] = MagicMock()

scipy_mock = MagicMock()
stats_mock = MagicMock()
chi2_mock = MagicMock()

def mock_ppf(q, df):
    if df <= 0: return 0.0
    if df == 1: return 3.841
    if df == 2: return 5.991
    if df == 3: return 7.815
    if df == 4: return 9.488
    if df == 5: return 11.070
    k = float(df)
    term = 1.0 - 2.0 / (9.0 * k)
    sd = math.sqrt(2.0 / (9.0 * k))
    z = 1.64485362695
    return k * ((term + z * sd) ** 3.0)

chi2_mock.ppf.side_effect = mock_ppf
stats_mock.chi2 = chi2_mock
scipy_mock.stats = stats_mock
sys.modules['scipy'] = scipy_mock
sys.modules['scipy.stats'] = stats_mock

optimize_mock = MagicMock()
def mock_lsa(cost_matrix):
    import numpy as np
    rows, cols = cost_matrix.shape
    size = min(rows, cols)
    return np.arange(size), np.arange(size)

optimize_mock.linear_sum_assignment.side_effect = mock_lsa
sys.modules['scipy.optimize'] = optimize_mock

# Add local path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import numpy as np
from vio_pipeline_v2.msckf.VioManager import VioManager
from vio_pipeline_v2.msckf.VioManagerOptions import VioManagerOptions
from vio_pipeline_v2.core.sensor_data import ImuData
from vio_pipeline_v2.core.feature import Feature
from vio_pipeline_v2.core.cam_radtan import CamRadtan

def main():
    print("Running Python VIO Full Pipeline mathematical benchmark...")
    
    # 1. Setup options
    params = VioManagerOptions()
    params.init_options.init_window_time = 0.5
    params.init_options.init_imu_thresh = 2.0
    params.init_options.init_max_disparity = 2.0
    params.init_options.gravity_mag = 9.81
    params.zupt_max_velocity = -1.0 # disable ZUPT
    params.state_options.max_clone_size = 11
    params.state_options.do_fej = True
    params.state_options.num_cameras = 1
    
    # 2. Re-create synthetic IMU readings (60 readings at 100 Hz)
    imu_readings = []
    g_vec = np.array([0, 0, 9.81])
    for i in range(61):
        ts = i * 0.01
        am = g_vec + np.array([0.05 * math.sin(10 * ts), 0.02 * math.cos(10 * ts), 0.01 * math.sin(5 * ts)])
        wm = np.array([0.01 * math.sin(5 * ts), -0.01 * math.cos(5 * ts), 0.02 * math.sin(10 * ts)])
        imu_readings.append(ImuData(ts, wm, am))
        
    # 3. Re-create camera frames (6 frames at 10 Hz from t=0.0 to t=0.60)
    pts_3d = []
    for i in range(30):
        pts_3d.append(np.array([
            -1.0 + 2.0 * (i % 5) / 4.0,
            -1.0 + 2.0 * (i // 5) / 5.0,
            2.0 + 2.0 * i / 30.0
        ]))
    from vio_pipeline_v2.type.PoseJPL import PoseJPL
    cam = CamRadtan(640, 480)
    cam.camera_values = np.array([460.0, 460.0, 320.0, 240.0, 0.01, -0.02, 0.001, -0.002])
    params.camera_intrinsics[0] = cam
    params.camera_extrinsics[0] = PoseJPL()
    
    frames_tracks = []
    for i in range(7):
        ts = i * 0.10
        frame_obs = []
        for pid, pt in enumerate(pts_3d):
            u_norm = np.array([pt[0]/pt[2], pt[1]/pt[2]])
            u_dist = cam.distort_f(u_norm)
            # Create a mock Feature and FeatureMeasurement
            f = Feature()
            f.featid = pid
            # Populate dict matching Feature class structure
            f.timestamps[0] = [ts]
            f.uvs[0] = [u_dist]
            f.uvs_norm[0] = [u_norm]
            frame_obs.append(f)
        frames_tracks.append(frame_obs)

    # 4. Benchmark loop
    NUM_ITERS = 100
    t0 = time.perf_counter()
    for run in range(NUM_ITERS):
        # Instantiate fresh VioManager
        vio = VioManager(params)
        
        # Feed data chronologically
        next_imu_idx = 0
        next_frame_idx = 0
        next_frame_time = 0.0
        
        for step in range(61):
            current_time = step * 0.01
            
            # Feed IMU measurements up to this time
            while next_imu_idx < len(imu_readings) and imu_readings[next_imu_idx].timestamp <= current_time:
                vio.feed_measurement_imu(imu_readings[next_imu_idx])
                next_imu_idx += 1
                
            # Feed camera frames if time matches multiple of 0.10
            if abs(current_time - next_frame_time) < 1e-5 and next_frame_idx < len(frames_tracks):
                active_tracks = []
                for f_global in frames_tracks[next_frame_idx]:
                    # Mock active track (only past measurements)
                    f_active = Feature()
                    f_active.featid = f_global.featid
                    # collect all measurements up to current_time
                    for j in range(next_frame_idx + 1):
                        meas_t = j * 0.10
                        f_active.timestamps[0] = f_active.timestamps.get(0, []) + [meas_t]
                        # get corresponding coordinates
                        f_active.uvs[0] = f_active.uvs.get(0, []) + [frames_tracks[j][f_global.featid].uvs[0][0]]
                        f_active.uvs_norm[0] = f_active.uvs_norm.get(0, []) + [frames_tracks[j][f_global.featid].uvs_norm[0][0]]
                    active_tracks.append(f_active)
                    
                vio.feed_measurement_camera_tracks(current_time, active_tracks, len(active_tracks))
                next_frame_idx += 1
                next_frame_time += 0.10
                
    t1 = time.perf_counter()
    total_time = t1 - t0
    avg_time_ms = (total_time / NUM_ITERS) * 1000.0
    
    print(f"Total time for {NUM_ITERS} full VIO runs: {total_time:.4f} seconds")
    print(f"Average time per full VIO run (0.60s window): {avg_time_ms:.4f} ms")

if __name__ == '__main__':
    main()
