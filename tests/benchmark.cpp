#include <iostream>
#include <chrono>
#include <vector>
#include "../msckf/state.hpp"
#include "../msckf/propagator.hpp"
#include "../core/sensor_data.hpp"

int main() {
    std::cout << "Running C++ VIO Pipeline mathematical benchmark...\n";
    
    // Setup state
    msckf::StateOptions opt;
    opt.do_fej = true;
    opt.num_cameras = 1;
    opt.max_clone_size = 11;
    
    msckf::State state;
    msckf::init_state(state, opt);
    
    // Setup noises
    msckf::PropagatorNoises noises;
    noises.sigma_w = 0.005;
    noises.sigma_a = 0.01;
    noises.sigma_wb = 0.001;
    noises.sigma_ab = 0.002;
    noises.sigma_w_2 = 0.005 * 0.005;
    noises.sigma_a_2 = 0.01 * 0.01;
    noises.sigma_wb_2 = 0.001 * 0.001;
    noises.sigma_ab_2 = 0.002 * 0.002;
    
    // Instantiate propagator
    msckf::PropagatorData propagator;
    msckf::init_propagator(propagator, noises, 9.81);
    
    // Generate 100 dummy IMU readings
    std::vector<core::ImuData> imu_readings;
    for (int i = 0; i <= 100; ++i) {
        double ts = i * 0.01;
        core::ImuData data;
        data.timestamp = ts;
        data.wm = Eigen::Vector3d(0.01, -0.02, 0.03);
        data.am = Eigen::Vector3d(0.0, 0.0, 9.81);
        imu_readings.push_back(data);
    }
    
    constexpr int NUM_ITERS = 1000;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERS; ++i) {
        state.timestamp = 0.0;
        msckf::propagate_only(propagator, state, 1.0, imu_readings.data(), imu_readings.size());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = t1 - t0;
    double total_time = diff.count();
    double avg_time_ms = (total_time / NUM_ITERS) * 1000.0;
    
    std::cout << "Total time for " << NUM_ITERS << " propagations: " << total_time << " seconds\n";
    std::cout << "Average time per propagation (1.0s window): " << avg_time_ms << " ms\n";
    
    return 0;
}
