#include <iostream>
#include <cmath>
#include <cassert>
#include "test_data.hpp"
#include "../msckf/vio_manager.hpp"
#include "../arena.hpp"

// Check function for vector tolerance
bool check_close(const Eigen::VectorXd& val, const Eigen::VectorXd& exp, double tol = 1e-7) {
    if (val.size() != exp.size()) return false;
    double diff = (val - exp).norm();
    if (diff > tol) {
        std::cout << "Tolerance exceeded: diff = " << diff << "\n"
                  << "Value   : " << val.transpose() << "\n"
                  << "Expected: " << exp.transpose() << "\n";
        return false;
    }
    return true;
}

// Special check for quaternions (handles sign ambiguity)
bool check_close_quat(const Eigen::Vector4d& val, const Eigen::Vector4d& exp, double tol = 1e-7) {
    double diff1 = (val - exp).norm();
    double diff2 = (val + exp).norm();
    double diff = std::min(diff1, diff2);
    if (diff > tol) {
        std::cout << "Quaternion Tolerance exceeded: diff = " << diff << "\n"
                  << "Value   : " << val.transpose() << "\n"
                  << "Expected: " << exp.transpose() << "\n";
        return false;
    }
    return true;
}

int main() {
    std::cout << "Starting mathematical equivalence tests...\n";
    
    // 1. Setup options matching Python simulation settings
    msckf::VioManagerOptions params;
    
    // Initializer Options
    params.init_opt.init_window_time = 0.5;
    params.init_opt.init_imu_thresh = 2.0;
    params.init_opt.init_max_disparity = 2.0;
    params.init_opt.init_max_features = 30;
    params.init_opt.gravity_mag = 9.81;
    
    // Noises
    params.noises.sigma_w = 0.005;
    params.noises.sigma_a = 0.01;
    params.noises.sigma_wb = 0.001;
    params.noises.sigma_ab = 0.002;
    params.noises.sigma_w_2 = 0.005 * 0.005;
    params.noises.sigma_a_2 = 0.01 * 0.01;
    params.noises.sigma_wb_2 = 0.001 * 0.001;
    params.noises.sigma_ab_2 = 0.002 * 0.002;
    
    // Camera intrinsic calibration model
    params.num_cameras = 1;
    // fx, fy, cx, cy, k1, k2, p1, p2
    double cam_vals[8] = { 460.0, 460.0, 320.0, 240.0, 0.01, -0.02, 0.001, -0.002 };
    core::init_camera(params.cam_models[0], core::CameraModelType::RADTAN, 640, 480, cam_vals);
    
    // ZUPT disabled for static propagation test or matched to options
    params.zupt_max_velocity = -1.0;
    params.zupt_noise_multiplier = 1.0;
    params.zupt_max_disparity = 1.0;
    
    // 2. Instantiate pipeline manager
    ArenaAllocator global_arena(MiB(16));
    msckf::VioManagerData* vio = global_arena.allocate<msckf::VioManagerData>();
    assert(vio != nullptr);
    msckf::init_vio_manager(*vio, params);
    
    // Chronologically feed sensor data
    int next_imu_idx = 0;
    int next_frame_idx = 0;
    
    double next_frame_time = 0.0;
    
    bool init_verified = false;
    
    // Feed data step-by-step
    for (int step = 0; step <= 60; ++step) {
        double current_time = step * 0.01;
        
        // Feed IMU measurements up to this time
        while (next_imu_idx < tests::NUM_IMU_READINGS && tests::imu_readings[next_imu_idx].timestamp <= current_time) {
            msckf::feed_measurement_imu(*vio, tests::imu_readings[next_imu_idx]);
            next_imu_idx++;
        }
        
        // Feed camera frames if time matches multiple of 0.10
        if (std::abs(current_time - next_frame_time) < 1e-5 && next_frame_idx < tests::NUM_FRAMES) {
            
            // Build the active frame tracks array
            core::Feature frame_tracks[tests::NUM_TRACKS];
            int track_count = 0;
            
            for (int t = 0; t < tests::NUM_TRACKS; ++t) {
                const core::Feature& global_feat = tests::features[t];
                
                // Copy only measurements up to current_time to simulate active tracking
                core::Feature active_feat = global_feat;
                active_feat.num_measurements = 0;
                for (int m = 0; m < global_feat.num_measurements; ++m) {
                    if (global_feat.measurements[m].timestamp <= current_time) {
                        active_feat.measurements[active_feat.num_measurements++] = global_feat.measurements[m];
                    }
                }
                
                if (active_feat.num_measurements > 0) {
                    frame_tracks[track_count++] = active_feat;
                }
            }
            
            msckf::feed_measurement_camera_tracks(*vio, current_time, frame_tracks, track_count);
            next_frame_idx++;
            next_frame_time += 0.10;
        }
        
        // 3. Verify initialization at EXPECTED_INIT_TS
        if (msckf::is_initialized(*vio) && !init_verified) {
            std::cout << "[Test]: Checking state initialization equivalence...\n";
            const msckf::State& state = msckf::get_state(*vio);
            
            Eigen::Vector4d q = Eigen::Map<const Eigen::Vector4d>(state.imu.value);
            Eigen::Vector3d p = Eigen::Map<const Eigen::Vector3d>(state.imu.value + 4);
            Eigen::Vector3d v = Eigen::Map<const Eigen::Vector3d>(state.imu.value + 7);
            Eigen::Vector3d bg = Eigen::Map<const Eigen::Vector3d>(state.imu.value + 10);
            Eigen::Vector3d ba = Eigen::Map<const Eigen::Vector3d>(state.imu.value + 13);
            
            assert(check_close_quat(q, tests::EXPECTED_INIT_Q));
            assert(check_close(p, tests::EXPECTED_INIT_P));
            assert(check_close(v, tests::EXPECTED_INIT_V));
            assert(check_close(bg, tests::EXPECTED_INIT_BG));
            assert(check_close(ba, tests::EXPECTED_INIT_BA));
            
            std::cout << "[SUCCESS]: Initialization EKF state matches Python exactly.\n";
            init_verified = true;
        }
    }
    
    // 4. Verify propagation state at t=0.60
    std::cout << "[Test]: Checking EKF state propagation equivalence at t=0.60...\n";
    const msckf::State& state = msckf::get_state(*vio);
    
    Eigen::Vector4d q = Eigen::Map<const Eigen::Vector4d>(state.imu.value);
    Eigen::Vector3d p = Eigen::Map<const Eigen::Vector3d>(state.imu.value + 4);
    Eigen::Vector3d v = Eigen::Map<const Eigen::Vector3d>(state.imu.value + 7);
    Eigen::Vector3d bg = Eigen::Map<const Eigen::Vector3d>(state.imu.value + 10);
    Eigen::Vector3d ba = Eigen::Map<const Eigen::Vector3d>(state.imu.value + 13);
    
    assert(check_close_quat(q, tests::EXPECTED_PROP_Q));
    assert(check_close(p, tests::EXPECTED_PROP_P));
    assert(check_close(v, tests::EXPECTED_PROP_V));
    assert(check_close(bg, tests::EXPECTED_PROP_BG));
    assert(check_close(ba, tests::EXPECTED_PROP_BA));
    
    std::cout << "[SUCCESS]: EKF state propagation matches Python exactly.\n";
    
    std::cout << "All mathematical equivalence tests passed successfully!\n";
    return 0;
}
