#pragma once
#include <Eigen/Core>
#include "../type/type.hpp"

namespace msckf {

// Worst case the code can actually reach: IMU 15 + dt 1 + 2 camera extrinsics
// (6 each) + 2 intrinsics (8 each) + IMU intrinsics 27 + 20 clones (6 each) +
// 50 SLAM landmarks (3 each) = 341. 512 was headroom nobody needed, and it is
// not free: state.Cov is a fixed-size matrix, so every block operation strides
// by the CAPACITY, not by the live state size. At 240 live dims that wasted
// over half of every cache line fetched, on a 2 MB matrix. init_state()
// validates the configured maximum against this at runtime.
constexpr int STATE_COV_CAPACITY = 384;
constexpr int IMU_ERROR_STATE_CAPACITY = 39;

enum class IntegrationMethod {
    DISCRETE = 1,
    RK4 = 2,
    ANALYTICAL = 3
};

enum class ImuModel {
    KALIBR = 1,
    RPNG = 2
};

struct StateOptions {
    bool do_fej = true;
    IntegrationMethod integration_method = IntegrationMethod::RK4;
    bool do_calib_camera_pose = false;
    bool do_calib_camera_intrinsics = false;
    bool do_calib_camera_timeoffset = false;
    bool do_calib_imu_intrinsics = false;
    bool do_calib_imu_g_sensitivity = false;
    ImuModel imu_model = ImuModel::KALIBR;
    
    // MUST stay <= 19. clones_IMU below holds 20 and state_helper.cpp stops
    // inserting at `num_clones < 20`, but marginalisation only fires on
    // `num_clones > max_clone_size` -- so at 20 the window never slides and
    // the filter silently starves (measured: 40_4 goes from ATE 12.81 at 15 to
    // 94149 at 20, with meas_overflow=45428). No runtime guard rejects it.
    int max_clone_size = 11;
    int max_slam_features = 25;
    int max_slam_in_update = 1000;
    int max_msckf_in_update = 1000;
    int max_aruco_features = 1024;
    int num_cameras = 1;
    
    type::LandmarkRepresentation feat_rep_msckf = type::LandmarkRepresentation::GLOBAL_3D;
    type::LandmarkRepresentation feat_rep_slam = type::LandmarkRepresentation::GLOBAL_3D;
    type::LandmarkRepresentation feat_rep_aruco = type::LandmarkRepresentation::GLOBAL_3D;
};

struct State {
    // Config Options
    StateOptions options;
    
    // Base persistent states
    type::Variable imu;
    type::Variable calib_imu_dw;
    type::Variable calib_imu_da;
    type::Variable calib_imu_tg;
    type::Variable calib_imu_GYROtoIMU;
    type::Variable calib_imu_ACCtoIMU;
    type::Variable calib_dt_CAMtoIMU;
    
    // Camera calibrations (max 2 cameras)
    type::Variable calib_IMUtoCAM[2];
    type::Variable cam_intrinsics[2];
    
    // Dynamic pose clone variables window (max 20)
    type::Variable clones_IMU[20];
    int num_clones = 0;
    
    // Persistent SLAM features (max 50)
    type::Variable features_SLAM[50];
    int num_slam_features = 0;
    
    // Ordered pointers to currently active variables in the covariance matrix
    type::Variable* variables[100];
    int num_variables = 0;
    
    // Dense covariance matrix
    Eigen::Matrix<double, STATE_COV_CAPACITY, STATE_COV_CAPACITY> Cov;
    int cov_size = 0;
    
    double timestamp = 0.0;
};

// Initializer
void init_state(State& state, const StateOptions& options);

// Intrinsic matrix builders
Eigen::Matrix3d Dm(ImuModel model, const double* vec);
Eigen::Matrix3d Tg(const double* vec);

// Helper size calculator
int imu_intrinsic_size(const State& state);

// Oldest clone timestamp finder
double margtimestep(const State& state);

} // namespace msckf
