#include "state.hpp"

#include <cstdio>
#include <cstdlib>
#include <cassert>

namespace msckf {

void init_state(State& state, const StateOptions& options) {
    state.options = options;
    // features_SLAM and clones_IMU are fixed-capacity. A config asking for more
    // than fits used to run straight off the end of the array (max_slam = 100
    // segfaults mid-sequence); clamp instead so an over-large config costs
    // landmarks, not memory corruption.
    // The covariance is fixed-capacity; a config that could outgrow it must fail
    // loudly here rather than corrupt memory later (the asserts below compile
    // out under NDEBUG, which Release defines).
    {
        const int worst_case = 15 + 1 + 2 * 6 + 2 * 8 + 27 +
                               6 * int(sizeof(state.clones_IMU) / sizeof(state.clones_IMU[0])) +
                               3 * int(sizeof(state.features_SLAM) / sizeof(state.features_SLAM[0]));
        if (worst_case > STATE_COV_CAPACITY) {
            std::fprintf(stderr,
                         "init_state: STATE_COV_CAPACITY=%d cannot hold worst-case state %d\n",
                         STATE_COV_CAPACITY, worst_case);
            std::exit(1);
        }
    }
    const int slam_capacity = int(sizeof(state.features_SLAM) / sizeof(state.features_SLAM[0]));
    const int clone_capacity = int(sizeof(state.clones_IMU) / sizeof(state.clones_IMU[0]));
    if (state.options.max_slam_features > slam_capacity) state.options.max_slam_features = slam_capacity;
    if (state.options.max_clone_size > clone_capacity) state.options.max_clone_size = clone_capacity;
    state.num_variables = 0;
    state.num_clones = 0;
    state.num_slam_features = 0;
    state.timestamp = 0.0;
    state.Cov.setZero();
    
    int current_id = 0;
    
    // 1. Initialize IMU State
    type::init_imu(state.imu);
    state.imu.id = current_id;
    state.variables[state.num_variables++] = &state.imu;
    current_id += state.imu.size;
    
    // 2. Initialize IMU Intrinsics (dw, da, tg)
    type::init_vec(state.calib_imu_dw, 6);
    type::init_vec(state.calib_imu_da, 6);
    type::init_vec(state.calib_imu_tg, 9);
    type::init_jplquat(state.calib_imu_GYROtoIMU);
    type::init_jplquat(state.calib_imu_ACCtoIMU);
    
    // Set default values (Identity)
    Eigen::Matrix<double, 6, 1> imu_default = Eigen::Matrix<double, 6, 1>::Zero();
    if (options.imu_model == ImuModel::KALIBR) {
        imu_default[0] = 1.0;
        imu_default[3] = 1.0;
        imu_default[5] = 1.0;
    } else {
        imu_default[0] = 1.0;
        imu_default[2] = 1.0;
        imu_default[5] = 1.0;
    }
    
    type::set_variable_value(state.calib_imu_dw, imu_default);
    type::set_variable_fej(state.calib_imu_dw, imu_default);
    type::set_variable_value(state.calib_imu_da, imu_default);
    type::set_variable_fej(state.calib_imu_da, imu_default);
    
    if (options.do_calib_imu_intrinsics) {
        state.calib_imu_dw.id = current_id;
        state.variables[state.num_variables++] = &state.calib_imu_dw;
        current_id += state.calib_imu_dw.size;
        
        state.calib_imu_da.id = current_id;
        state.variables[state.num_variables++] = &state.calib_imu_da;
        current_id += state.calib_imu_da.size;
        
        if (options.do_calib_imu_g_sensitivity) {
            state.calib_imu_tg.id = current_id;
            state.variables[state.num_variables++] = &state.calib_imu_tg;
            current_id += state.calib_imu_tg.size;
        }
        
        if (options.imu_model == ImuModel::KALIBR) {
            state.calib_imu_GYROtoIMU.id = current_id;
            state.variables[state.num_variables++] = &state.calib_imu_GYROtoIMU;
            current_id += state.calib_imu_GYROtoIMU.size;
        } else {
            state.calib_imu_ACCtoIMU.id = current_id;
            state.variables[state.num_variables++] = &state.calib_imu_ACCtoIMU;
            current_id += state.calib_imu_ACCtoIMU.size;
        }
    }
    
    // 3. Time Offset
    type::init_vec(state.calib_dt_CAMtoIMU, 1);
    if (options.do_calib_camera_timeoffset) {
        state.calib_dt_CAMtoIMU.id = current_id;
        state.variables[state.num_variables++] = &state.calib_dt_CAMtoIMU;
        current_id += state.calib_dt_CAMtoIMU.size;
    }
    
    // 4. Cameras Pose and Intrinsics
    for (int i = 0; i < options.num_cameras; ++i) {
        type::init_posejpl(state.calib_IMUtoCAM[i]);
        type::init_vec(state.cam_intrinsics[i], 8);
        
        if (options.do_calib_camera_pose) {
            state.calib_IMUtoCAM[i].id = current_id;
            state.variables[state.num_variables++] = &state.calib_IMUtoCAM[i];
            current_id += state.calib_IMUtoCAM[i].size;
        }
        if (options.do_calib_camera_intrinsics) {
            state.cam_intrinsics[i].id = current_id;
            state.variables[state.num_variables++] = &state.cam_intrinsics[i];
            current_id += state.cam_intrinsics[i].size;
        }
    }
    
    state.cov_size = current_id;
    
    // 5. Initialize Covariance
    state.Cov.block(0, 0, current_id, current_id).setIdentity();
    state.Cov.block(0, 0, current_id, current_id) *= (1e-3 * 1e-3);
    
    auto set_diag = [&](const type::Variable& var, double noise_std) {
        int idx = var.id;
        int dim = var.size;
        if (idx != -1) {
            state.Cov.block(idx, idx, dim, dim).setIdentity();
            state.Cov.block(idx, idx, dim, dim) *= (noise_std * noise_std);
        }
    };
    
    if (options.do_calib_imu_intrinsics) {
        set_diag(state.calib_imu_dw, 0.005);
        set_diag(state.calib_imu_da, 0.008);
        
        if (options.do_calib_imu_g_sensitivity) {
            set_diag(state.calib_imu_tg, 0.005);
        }
        
        if (options.imu_model == ImuModel::KALIBR) {
            set_diag(state.calib_imu_GYROtoIMU, 0.005);
        } else {
            set_diag(state.calib_imu_ACCtoIMU, 0.005);
        }
    }
    
    if (options.do_calib_camera_timeoffset) {
        set_diag(state.calib_dt_CAMtoIMU, 0.01);
    }
    
    if (options.do_calib_camera_pose) {
        for (int i = 0; i < options.num_cameras; ++i) {
            int idx = state.calib_IMUtoCAM[i].id;
            if (idx != -1) {
                state.Cov.block(idx, idx, 3, 3) = (0.005 * 0.005) * Eigen::Matrix3d::Identity();
                state.Cov.block(idx + 3, idx + 3, 3, 3) = (0.015 * 0.015) * Eigen::Matrix3d::Identity();
            }
        }
    }
    
    if (options.do_calib_camera_intrinsics) {
        for (int i = 0; i < options.num_cameras; ++i) {
            int idx = state.cam_intrinsics[i].id;
            if (idx != -1) {
                state.Cov.block(idx, idx, 4, 4) = (1.0 * 1.0) * Eigen::Matrix4d::Identity();
                state.Cov.block(idx + 4, idx + 4, 4, 4) = (0.005 * 0.005) * Eigen::Matrix4d::Identity();
            }
        }
    }
}

Eigen::Matrix3d Dm(ImuModel model, const double* vec) {
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    if (model == ImuModel::KALIBR) {
        D(0, 0) = vec[0];
        D(1, 0) = vec[1];
        D(2, 0) = vec[2];
        D(1, 1) = vec[3];
        D(2, 1) = vec[4];
        D(2, 2) = vec[5];
    } else {
        D(0, 0) = vec[0];
        D(0, 1) = vec[1];
        D(1, 1) = vec[2];
        D(0, 2) = vec[3];
        D(1, 2) = vec[4];
        D(2, 2) = vec[5];
    }
    return D;
}

Eigen::Matrix3d Tg(const double* vec) {
    Eigen::Matrix3d Tg = Eigen::Matrix3d::Zero();
    Tg.col(0) = Eigen::Map<const Eigen::Vector3d>(vec);
    Tg.col(1) = Eigen::Map<const Eigen::Vector3d>(vec + 3);
    Tg.col(2) = Eigen::Map<const Eigen::Vector3d>(vec + 6);
    return Tg;
}

int imu_intrinsic_size(const State& state) {
    int sz = 0;
    if (state.options.do_calib_imu_intrinsics) {
        sz += 15;
        if (state.options.do_calib_imu_g_sensitivity) {
            sz += 9;
        }
    }
    return sz;
}

double margtimestep(const State& state) {
    if (state.num_clones == 0) {
        return -1.0;
    }
    double min_ts = state.clones_IMU[0].timestamp;
    for (int i = 1; i < state.num_clones; ++i) {
        if (state.clones_IMU[i].timestamp < min_ts) {
            min_ts = state.clones_IMU[i].timestamp;
        }
    }
    return min_ts;
}

} // namespace msckf
