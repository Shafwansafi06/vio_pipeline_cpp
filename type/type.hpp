#pragma once
#include <Eigen/Core>

namespace type {

enum class VariableType {
    IMU,
    VEC,
    JPLQUAT,
    POSEJPL,
    LANDMARK
};

enum class LandmarkRepresentation {
    GLOBAL_3D = 1,
    GLOBAL_FULL_INVERSE_DEPTH = 2,
    ANCHORED_3D = 3,
    ANCHORED_FULL_INVERSE_DEPTH = 4,
    ANCHORED_MSCKF_INVERSE_DEPTH = 5,
    ANCHORED_INVERSE_DEPTH_SINGLE = 6,
    UNKNOWN = 7
};

bool is_relative_representation(LandmarkRepresentation rep);

struct Variable {
    VariableType type;
    int id = -1;        // starting offset in EKF covariance (-1 if not active)
    int size = 0;       // error state size
    int storage_size = 0; // nominal value storage size
    
    // Memory footprint is fixed, no dynamic allocation
    double value[16] = {0.0};
    double fej[16] = {0.0};
    
    // Landmark Bookkeeping
    int feat_id = -1;
    int anchor_cam_id = -1;
    LandmarkRepresentation representation = LandmarkRepresentation::GLOBAL_3D;
    bool should_marg = false;
    int update_fail_count = 0;
    double uv_norm_zero[3] = {0.0};
    double uv_norm_zero_fej[3] = {0.0};
    
    // Clone Bookkeeping
    double timestamp = -1.0;
    
    // Member getters
    Eigen::Matrix3d Rot() const;
    Eigen::Matrix3d Rot_fej() const;
    Eigen::Vector3d pos() const;
    Eigen::Vector3d pos_fej() const;
    Eigen::Vector3d vel() const;
    Eigen::Vector3d vel_fej() const;
    Eigen::Vector3d bias_g() const;
    Eigen::Vector3d bias_a() const;
};

// Manipulators
void init_imu(Variable& var);
void init_vec(Variable& var, int dim);
void init_jplquat(Variable& var);
void init_posejpl(Variable& var);
void init_landmark(Variable& var, int size, LandmarkRepresentation rep);

void update_variable(Variable& var, const Eigen::Ref<const Eigen::VectorXd>& dx);
void set_variable_value(Variable& var, const Eigen::Ref<const Eigen::VectorXd>& val);
void set_variable_fej(Variable& var, const Eigen::Ref<const Eigen::VectorXd>& val);


// Special Landmark Helpers
Eigen::Vector3d get_landmark_xyz(const Variable& var, bool get_fej);
void set_landmark_from_xyz(Variable& var, const Eigen::Vector3d& xyz, bool is_fej);

// Special PoseJPL Helpers
Eigen::Matrix3d get_pose_rot(const Variable& var, bool get_fej = false);
Eigen::Vector3d get_pose_pos(const Variable& var, bool get_fej = false);

// Special Clone function
Variable clone_variable(const Variable& var);

} // namespace type
