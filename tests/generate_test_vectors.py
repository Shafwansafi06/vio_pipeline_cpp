import sys
import os
import math
import json
from unittest.mock import MagicMock

# 1. Mock external dependencies
sys.modules['cv2'] = MagicMock()
sys.modules['rospkg'] = MagicMock()

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

sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.covariance'] = MagicMock()

# 2. Add local workspace to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import numpy as np
from vio_pipeline_v2.msckf.State import State
from vio_pipeline_v2.msckf.StateOptions import StateOptions
from vio_pipeline_v2.msckf.NoiseManager import NoiseManager
from vio_pipeline_v2.msckf.Propagator import Propagator
from vio_pipeline_v2.msckf.VioManagerOptions import VioManagerOptions
from vio_pipeline_v2.core import ImuData, FeatureDatabase
from vio_pipeline_v2.core.cam_radtan import CamRadtan
from vio_pipeline_v2.msckf.state_helper import StateHelper
from vio_pipeline_v2.msckf.UpdaterMSCKF import UpdaterMSCKF
from vio_pipeline_v2.msckf.UpdaterOptions import UpdaterOptions
from vio_pipeline_v2.msckf.update_zero_velocity import UpdaterZeroVelocity
from vio_pipeline_v2.initialize.static import StaticInitializer
from vio_pipeline_v2.initialize.InertialInitializerOptions import InertialInitializerOptions

def generate():
    np.random.seed(42)
    
    # Options setup
    options = VioManagerOptions()
    options.init_window_time = 0.5
    options.init_imu_thresh = 2.0
    options.init_max_disparity = 2.0
    options.init_max_features = 30
    options.gravity_mag = 9.81
    
    # State options
    state_opt = StateOptions()
    state_opt.do_fej = True
    state_opt.num_cameras = 1
    state_opt.max_clone_size = 11
    
    state = State(state_opt)
    
    # Noise manager
    noises = NoiseManager()
    noises.sigma_w = 0.005
    noises.sigma_a = 0.01
    noises.sigma_wb = 0.001
    noises.sigma_ab = 0.002
    
    # Initializer
    init_opt = InertialInitializerOptions()
    init_opt.init_window_time = 0.5
    init_opt.init_imu_thresh = 2.0
    init_opt.init_max_disparity = 2.0
    init_opt.gravity_mag = 9.81
    
    # Generate synthetic IMU readings
    # 60 readings at 100 Hz (from t=0.0 to t=0.6)
    imu_readings = []
    g_vec = np.array([0, 0, 9.81])
    
    for i in range(61):
        ts = i * 0.01
        # stationary with small oscillations
        am = g_vec + np.array([0.05 * math.sin(10 * ts), 0.02 * math.cos(10 * ts), 0.01 * math.sin(5 * ts)])
        wm = np.array([0.01 * math.sin(5 * ts), -0.01 * math.cos(5 * ts), 0.02 * math.sin(10 * ts)])
        imu_readings.append({
            'timestamp': ts,
            'am': am.tolist(),
            'wm': wm.tolist()
        })
        
    # Generate camera frames (at 10 Hz from t=0.0 to t=0.6)
    # 3D points in space
    pts_3d = []
    for i in range(30):
        # x, y in [-1, 1], z in [2, 4]
        pts_3d.append(np.array([
            -1.0 + 2.0 * (i % 5) / 4.0,
            -1.0 + 2.0 * (i // 5) / 5.0,
            2.0 + 2.0 * i / 30.0
        ]))
        
    # Pinhole RadTan model
    cam = CamRadtan(640, 480)
    # Set camera intrinsic values: fx, fy, cx, cy, k1, k2, p1, p2
    cam.camera_values = np.array([460.0, 460.0, 320.0, 240.0, 0.01, -0.02, 0.001, -0.002])
    
    frames = []
    for i in range(7):
        ts = i * 0.10
        # Project points to camera
        frame_obs = []
        for pid, pt in enumerate(pts_3d):
            # Camera frame is identity (stationary camera)
            u_norm = np.array([pt[0]/pt[2], pt[1]/pt[2]])
            u_dist = cam.distort_f(u_norm)
            
            frame_obs.append({
                'featid': pid,
                'uv': u_dist.tolist(),
                'uv_norm': u_norm.tolist()
            })
        frames.append({
            'timestamp': ts,
            'features': frame_obs
        })
        
    # Run simulation to get expected results
    # 1. Initialize
    imu_objs = [ImuData(r['timestamp'], r['wm'], r['am']) for r in imu_readings]
    static_init = StaticInitializer(init_opt, None, imu_objs[:51])
    
    init_ts_list = [0.0]
    init_cov_list = []
    init_order_list = []
    
    # Run static initialization on the window
    init_success = static_init.initialize(init_ts_list, init_cov_list, init_order_list, state._imu, wait_for_jerk=False)
    assert init_success
    
    init_ts = init_ts_list[0]
    init_cov = init_cov_list[0]
    
    # Setup initial state covariance
    state._timestamp = init_ts
    state._Cov.fill(0.0)
    StateHelper.set_initial_covariance(state, init_cov, [state._imu])
    
    # Save values at initialization
    init_q = state._imu.quat().tolist()
    init_p = state._imu.pos().tolist()
    init_v = state._imu.vel().tolist()
    init_bg = state._imu.bias_g().tolist()
    init_ba = state._imu.bias_a().tolist()
    init_P = state._Cov.diagonal().tolist()
    
    # 2. Propagate to t=0.60
    propagator = Propagator(noises, 9.81)
    for r in imu_objs:
        propagator.feed_imu(r)
        
    # We propagate from init_ts to 0.60
    propagator.propagate_only(state, 0.60)
    
    prop_q = state._imu.quat().tolist()
    prop_p = state._imu.pos().tolist()
    prop_v = state._imu.vel().tolist()
    prop_bg = state._imu.bias_g().tolist()
    prop_ba = state._imu.bias_a().tolist()
    prop_P = state._Cov.diagonal().tolist()
    
    # Dump test vector JSON
    test_vectors = {
        'init_options': {
            'init_window_time': init_opt.init_window_time,
            'init_imu_thresh': init_opt.init_imu_thresh,
            'init_max_disparity': init_opt.init_max_disparity,
            'gravity_mag': init_opt.gravity_mag
        },
        'noises': {
            'sigma_w': noises.sigma_w,
            'sigma_a': noises.sigma_a,
            'sigma_wb': noises.sigma_wb,
            'sigma_ab': noises.sigma_ab
        },
        'camera': {
            'fx': 460.0, 'fy': 460.0,
            'cx': 320.0, 'cy': 240.0,
            'd': [0.01, -0.02, 0.001, -0.002]
        },
        'imu_readings': imu_readings,
        'frames': frames
    }
    
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    with open(os.path.dirname(__file__) + '/test_vectors.json', 'w') as f:
        json.dump(test_vectors, f, indent=2)
        
    # Generate C++ Header
    with open(os.path.dirname(__file__) + '/test_data.hpp', 'w') as f:
        f.write("#pragma once\n")
        f.write("#include <Eigen/Core>\n")
        f.write("#include \"../core/sensor_data.hpp\"\n")
        f.write("#include \"../core/feature.hpp\"\n\n")
        
        f.write("namespace tests {\n\n")
        
        f.write(f"constexpr int NUM_IMU_READINGS = {len(imu_readings)};\n")
        f.write("inline core::ImuData imu_readings[NUM_IMU_READINGS] = {\n")
        for read in imu_readings:
            am = read['am']
            wm = read['wm']
            f.write(f"    {{ {read['timestamp']}, Eigen::Vector3d({wm[0]}, {wm[1]}, {wm[2]}), Eigen::Vector3d({am[0]}, {am[1]}, {am[2]}) }},\n")
        f.write("};\n\n")
        
        f.write(f"constexpr int NUM_FRAMES = {len(frames)};\n")
        f.write(f"constexpr int NUM_TRACKS = {len(pts_3d)};\n")
        
        f.write("inline core::Feature features[NUM_TRACKS] = {\n")
        for tr_id in range(len(pts_3d)):
            f.write("    {\n")
            f.write(f"        {tr_id}, // featid\n")
            f.write("        false, // to_delete\n")
            f.write("        {\n")
            
            num_m = 0
            for frame in frames:
                for obs in frame['features']:
                    if obs['featid'] == tr_id:
                        uv = obs['uv']
                        uv_n = obs['uv_norm']
                        f.write(f"            {{ 0, {frame['timestamp']}, Eigen::Vector2d({uv[0]}, {uv[1]}), Eigen::Vector2d({uv_n[0]}, {uv_n[1]}) }},\n")
                        num_m += 1
                        break
            f.write("        },\n")
            f.write(f"        {num_m}, // num_measurements\n")
            f.write("        Eigen::Vector3d::Zero(),\n")
            f.write("        -1,\n")
            f.write("        0.0,\n")
            f.write("        Eigen::Vector3d::Zero()\n")
            f.write("    },\n")
        f.write("};\n\n")
        
        # Expected Init Values
        f.write(f"constexpr double INIT_TS = {init_ts};\n")
        f.write(f"inline const Eigen::Vector4d EXPECTED_INIT_Q = Eigen::Vector4d({init_q[0]}, {init_q[1]}, {init_q[2]}, {init_q[3]});\n")
        f.write(f"inline const Eigen::Vector3d EXPECTED_INIT_P = Eigen::Vector3d({init_p[0]}, {init_p[1]}, {init_p[2]});\n")
        f.write(f"inline const Eigen::Vector3d EXPECTED_INIT_V = Eigen::Vector3d({init_v[0]}, {init_v[1]}, {init_v[2]});\n")
        f.write(f"inline const Eigen::Vector3d EXPECTED_INIT_BG = Eigen::Vector3d({init_bg[0]}, {init_bg[1]}, {init_bg[2]});\n")
        f.write(f"inline const Eigen::Vector3d EXPECTED_INIT_BA = Eigen::Vector3d({init_ba[0]}, {init_ba[1]}, {init_ba[2]});\n")
        
        # Expected Prop Values at t=0.60
        f.write(f"inline const Eigen::Vector4d EXPECTED_PROP_Q = Eigen::Vector4d({prop_q[0]}, {prop_q[1]}, {prop_q[2]}, {prop_q[3]});\n")
        f.write(f"inline const Eigen::Vector3d EXPECTED_PROP_P = Eigen::Vector3d({prop_p[0]}, {prop_p[1]}, {prop_p[2]});\n")
        f.write(f"inline const Eigen::Vector3d EXPECTED_PROP_V = Eigen::Vector3d({prop_v[0]}, {prop_v[1]}, {prop_v[2]});\n")
        f.write(f"inline const Eigen::Vector3d EXPECTED_PROP_BG = Eigen::Vector3d({prop_bg[0]}, {prop_bg[1]}, {prop_bg[2]});\n")
        f.write(f"inline const Eigen::Vector3d EXPECTED_PROP_BA = Eigen::Vector3d({prop_ba[0]}, {prop_ba[1]}, {prop_ba[2]});\n")
        
        f.write("} // namespace tests\n")
        
    print("Test vectors generated successfully in tests/test_vectors.json and tests/test_data.hpp")

if __name__ == '__main__':
    generate()
