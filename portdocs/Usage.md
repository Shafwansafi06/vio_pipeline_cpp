# VIO C++ Pipeline Usage Guide

This document describes how to compile the new C++ VIO library, execute the mathematical equivalence tests, run the performance benchmarks, and integrate the pipeline into your applications.

## Compilation

The C++ pipeline requires a C++17 compliant compiler (GCC 7+ or Clang 5+) and CMake 3.5+. The Eigen library is vendorized in `vendor/eigen` and configured automatically.

### 1. Build optimized Release configuration
To compile the library and the executable targets:
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

This builds the following targets inside the `build` directory:
* `libvio_pipeline.a`: Static library of the ported VIO pipeline.
* `verify_math`: Verification executable matching state values against Python.
* `benchmark`: Execution profiler for IMU propagation.
* `compare_openvins`: Benchmark comparison script pitting DOD against Open_VINS.

---

## Running Verification & Benchmarks

### 1. Mathematical Equivalence Verification
Asserts that C++ EKF states match the Python pipeline simulation to within $10^{-7}$:
```bash
./build/verify_math
```
*Expected Output:*
```
Starting mathematical equivalence tests...
[VioManager]: Initialization succeeded at time 0.24
[Test]: Checking state initialization equivalence...
[SUCCESS]: Initialization EKF state matches Python exactly.
[Test]: Checking EKF state propagation equivalence at t=0.60...
[SUCCESS]: EKF state propagation matches Python exactly.
All mathematical equivalence tests passed successfully!
```

### 2. Run C++ Propagation Benchmark
```bash
./build/benchmark
```

### 3. Run Python Benchmark (using uv)
```bash
uv run tests/benchmark.py
```

### 4. Run Open_VINS vs DOD Benchmark
```bash
./build/compare_openvins
```

---

## API Integration Example

To integrate the DOD VIO pipeline into a custom front-end wrapper:

```cpp
#include "msckf/vio_manager.hpp"
#include "core/sensor_data.hpp"

int main() {
    // 1. Initialize Pipeline Config Options
    msckf::VioManagerOptions params;
    params.init_opt.init_window_time = 0.5;
    params.init_opt.init_imu_thresh = 2.0;
    
    // Configure Camera Calibration
    params.num_cameras = 1;
    double cam_vals[8] = { 460.0, 460.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0 };
    core::init_camera(params.cam_models[0], core::CameraModelType::RADTAN, 640, 480, cam_vals);
    
    // 2. Instantiate VioManager
    msckf::VioManager vio(params);
    
    // 3. Feed IMU Readings
    core::ImuData imu_msg;
    imu_msg.timestamp = 0.01;
    imu_msg.wm = Eigen::Vector3d(0.01, -0.02, 0.03); // Gyro
    imu_msg.am = Eigen::Vector3d(0.0, 0.0, 9.81);    // Acc
    vio.feed_measurement_imu(imu_msg);
    
    // 4. Feed Visual Feature Tracks
    double current_time = 0.10;
    core::Feature frame_tracks[100];
    int track_count = 50; // populate with frontend tracker outputs
    
    vio.feed_measurement_camera_tracks(current_time, frame_tracks, track_count);
    
    // 5. Query EKF State
    if (vio.is_initialized()) {
        const msckf::State& state = vio.get_state();
        Eigen::Vector3d pos = Eigen::Map<const Eigen::Vector3d>(state.imu.value + 4);
        std::cout << "Current IMU position: " << pos.transpose() << std::endl;
    }
    
    return 0;
}
```
*Note: Link your target application against `libvio_pipeline.a` and include the path to `vendor/eigen`.*
