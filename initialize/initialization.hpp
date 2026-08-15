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
    // Reject a dynamic-init solution whose linear system it does not actually
    // explain: ||A x - b|| / ||b||. DOD's dynamic initializer is the linear
    // stage only -- official follows it with a Ceres MLE refinement -- so an
    // ill-posed window can produce a solution that satisfies the |g|
    // constraint and is still garbage. Default is off (accept anything).
    double init_dyn_max_residual = 1e9;
    // Reject a dynamic-init solution whose recovered gravity points somewhere
    // the accelerometer does not agree with. Default is off.
    double init_dyn_max_gravity_deg = 1e9;
    // Official inflates the covariance recovered by its MLE before handing it
    // to the filter (init_dyn_inflation_* in its EuRoC config). DOD has no MLE
    // stage, so its linear solution is *less* trustworthy than official's, not
    // more -- but it was shipping a fixed, confident prior with no inflation
    // at all.
    double init_dyn_min_rec_cond = 0.0; // off by default
    // The dynamic initializer's LINEAR stage cannot observe velocity reliably:
    // over a short window the least-squares fit trades velocity against feature
    // depth almost freely, so the recovered v0 is wrong even when the residual,
    // the conditioning and the recovered gravity all look healthy. Measured on
    // EuRoC at the init instant (truth from Leica):
    //
    //   MH_01  recovered 0.445 m/s, true 0.048 -> run survives (0.1132 m)
    //   MH_02  recovered 0.162 m/s, true 0.029 -> run DIVERGES (6.5e5 m)
    //
    // Zeroing it instead: MH_01 0.1131 m, MH_02 0.1744 m. Official does not hit
    // this because it refines the linear solution with a Ceres MLE over the IMU
    // residuals; port that and this can be revisited. Until then the velocity is
    // discarded and its covariance left inflated.
    // Default OFF: on a genuinely moving start the linear velocity does carry
    // information (tests/verify_dynamic_init recovers 0.92 m/s truth and fails
    // if it is discarded). Turned ON in tools/euroc_options.hpp, where the
    // measurements above were taken.
    bool init_dyn_zero_velocity = false;
    // Diagnostic only: accelerometer excitation (m/s^2 std) over the window.
    double init_dyn_min_excitation = 0.0;
    double init_dyn_inflation_ori = 10.0;
    double init_dyn_inflation_vel = 100.0;
    double init_dyn_inflation_bg = 10.0;
    double init_dyn_inflation_ba = 100.0;
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
