#pragma once
#include <Eigen/Core>
#include "../core/sensor_data.hpp"
#include "../type/type.hpp"
#include "../core/feature.hpp"

namespace initialize {

struct InitializerOptions {
    double init_window_time = 1.0;
    double init_imu_thresh = 1.0;
    double init_max_disparity = 1.0;
    int init_max_features = 50;
    bool init_dyn_use = false;
    double init_calib_camImu_dt = 0.0;
    double gravity_mag = 9.81;
    // Dynamic (linear MLE) initializer parameters (ov_init defaults).
    int init_dyn_num_pose = 6;
    double init_dyn_min_deg = 10.0;
    Eigen::Vector3d init_dyn_bias_g = Eigen::Vector3d::Zero();
    Eigen::Vector3d init_dyn_bias_a = Eigen::Vector3d::Zero();
};

bool static_initialize(const InitializerOptions& config,
                           const core::ImuData* imu_data, int imu_count,
                           type::Variable& t_imu,
                           Eigen::Matrix<double, 15, 15>& covariance,
                           double& timestamp,
                           bool wait_for_jerk = true);

bool dynamic_initialize(const InitializerOptions& config,
                        core::FeatureDatabase& db,
                        const core::ImuData* imu_data, int imu_count,
                        const double camera_extrinsics[2][7], int num_cameras,
                        type::Variable& t_imu,
                        Eigen::Matrix<double, 15, 15>& covariance,
                        double& timestamp);

bool run_initialization(const InitializerOptions& config,
                        core::FeatureDatabase& db,
                        const core::ImuData* imu_data, int imu_count,
                        type::Variable& t_imu,
                        Eigen::Matrix<double, 15, 15>& covariance,
                        double& timestamp,
                        int& init_attempt,
                        bool wait_for_jerk = true);

} // namespace initialize
