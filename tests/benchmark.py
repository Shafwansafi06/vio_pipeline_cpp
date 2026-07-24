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
from vio_pipeline_v2.msckf.State import State
from vio_pipeline_v2.msckf.StateOptions import StateOptions
from vio_pipeline_v2.msckf.NoiseManager import NoiseManager
from vio_pipeline_v2.msckf.Propagator import Propagator
from vio_pipeline_v2.msckf.state_helper import StateHelper
from vio_pipeline_v2.core import ImuData

def main():
    print("Running Python VIO Pipeline mathematical benchmark...")
    
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
    
    # Propagator
    propagator = Propagator(noises, 9.81)
    
    # Generate 100 dummy IMU readings
    imu_readings = []
    for i in range(101):
        ts = i * 0.01
        imu_readings.append(ImuData(ts, np.array([0.01, -0.02, 0.03]), np.array([0.0, 0.0, 9.81])))
        
    for r in imu_readings:
        propagator.feed_imu(r)
        
    # Benchmarking propagation
    NUM_ITERS = 1000
    
    t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        # Reset state timestamp
        state._timestamp = 0.0
        # Propagate from 0.0 to 1.0
        propagator.propagate_only(state, 1.0)
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    avg_time_ms = (total_time / NUM_ITERS) * 1000.0
    
    print(f"Total time for {NUM_ITERS} propagations: {total_time:.4f} seconds")
    print(f"Average time per propagation (1.0s window): {avg_time_ms:.4f} ms")

if __name__ == '__main__':
    main()
