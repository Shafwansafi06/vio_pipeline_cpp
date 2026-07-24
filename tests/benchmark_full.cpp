#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include "../msckf/vio_manager.hpp"
#include "../arena.hpp"
#include "test_data.hpp"

int main() {
    std::cout << "Running C++ VIO Full Pipeline mathematical benchmark...\n";
    
    // 1. Setup options
    msckf::VioManagerOptions params;
    params.init_opt.init_window_time = 0.5;
    params.init_opt.init_imu_thresh = 2.0;
    params.init_opt.init_max_disparity = 2.0;
    params.init_opt.gravity_mag = 9.81;
    params.zupt_max_velocity = -1.0; // disable ZUPT
    params.num_cameras = 1;
    
    // Camera intrinsics
    double cam_vals[8] = { 460.0, 460.0, 320.0, 240.0, 0.01, -0.02, 0.001, -0.002 };
    core::init_camera(params.cam_models[0], core::CameraModelType::RADTAN, 640, 480, cam_vals);
    
    constexpr int NUM_ITERS = 10000;
    
    ArenaAllocator global_arena(MiB(16));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < NUM_ITERS; ++run) {
        global_arena.reset();
        msckf::VioManagerData* vio = global_arena.allocate<msckf::VioManagerData>();
        if (vio == nullptr) return 1;
        msckf::init_vio_manager(*vio, params);
        
        int next_imu_idx = 0;
        int next_frame_idx = 0;
        double next_frame_time = 0.0;
        
        for (int step = 0; step <= 60; ++step) {
            double current_time = step * 0.01;
            
            // Feed IMU measurements up to this time
            while (next_imu_idx < tests::NUM_IMU_READINGS && tests::imu_readings[next_imu_idx].timestamp <= current_time) {
                msckf::feed_measurement_imu(*vio, tests::imu_readings[next_imu_idx]);
                next_imu_idx++;
            }
            
            // Feed camera frames if time matches multiple of 0.10
            if (std::abs(current_time - next_frame_time) < 1e-5 && next_frame_idx < tests::NUM_FRAMES) {
                core::Feature frame_tracks[tests::NUM_TRACKS];
                int track_count = 0;
                
                for (int t = 0; t < tests::NUM_TRACKS; ++t) {
                    const core::Feature& global_feat = tests::features[t];
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
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = t1 - t0;
    double total_time = diff.count();
    double avg_time_ms = (total_time / NUM_ITERS) * 1000.0;
    
    std::cout << "Total time for " << NUM_ITERS << " full VIO runs: " << total_time << " seconds\n";
    std::cout << "Average time per full VIO run (0.60s window): " << avg_time_ms << " ms\n";
    
    return 0;
}
