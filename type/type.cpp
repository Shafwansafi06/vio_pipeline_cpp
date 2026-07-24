#include "type.hpp"
#include "quat_ops.hpp"
#include <cassert>
#include <cstring>
#include <cmath>

namespace type {

bool is_relative_representation(LandmarkRepresentation rep) {
    return (rep == LandmarkRepresentation::ANCHORED_3D ||
            rep == LandmarkRepresentation::ANCHORED_FULL_INVERSE_DEPTH ||
            rep == LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH ||
            rep == LandmarkRepresentation::ANCHORED_INVERSE_DEPTH_SINGLE);
}

void init_imu(Variable& var) {
    var.type = VariableType::IMU;
    var.size = 15;
    var.storage_size = 16;
    std::memset(var.value, 0, sizeof(var.value));
    std::memset(var.fej, 0, sizeof(var.fej));
    var.value[3] = 1.0;
    var.fej[3] = 1.0;
    var.id = -1;
}

void init_vec(Variable& var, int dim) {
    assert(dim <= 16);
    var.type = VariableType::VEC;
    var.size = dim;
    var.storage_size = dim;
    std::memset(var.value, 0, sizeof(var.value));
    std::memset(var.fej, 0, sizeof(var.fej));
    var.id = -1;
}

void init_jplquat(Variable& var) {
    var.type = VariableType::JPLQUAT;
    var.size = 3;
    var.storage_size = 4;
    std::memset(var.value, 0, sizeof(var.value));
    std::memset(var.fej, 0, sizeof(var.fej));
    var.value[3] = 1.0;
    var.fej[3] = 1.0;
    var.id = -1;
}

void init_posejpl(Variable& var) {
    var.type = VariableType::POSEJPL;
    var.size = 6;
    var.storage_size = 7;
    std::memset(var.value, 0, sizeof(var.value));
    std::memset(var.fej, 0, sizeof(var.fej));
    var.value[3] = 1.0;
    var.fej[3] = 1.0;
    var.id = -1;
}

void init_landmark(Variable& var, int size, LandmarkRepresentation rep) {
    var.type = VariableType::LANDMARK;
    var.size = size;
    var.storage_size = size;
    var.representation = rep;
    std::memset(var.value, 0, sizeof(var.value));
    std::memset(var.fej, 0, sizeof(var.fej));
    var.feat_id = -1;
    var.anchor_cam_id = -1;
    var.should_marg = false;
    var.update_fail_count = 0;
    std::memset(var.uv_norm_zero, 0, sizeof(var.uv_norm_zero));
    std::memset(var.uv_norm_zero_fej, 0, sizeof(var.uv_norm_zero_fej));
    var.id = -1;
}

void update_variable(Variable& var, const Eigen::Ref<const Eigen::VectorXd>& dx) {
    assert(dx.size() == var.size);
    if (var.type == VariableType::IMU) {
        Eigen::Vector4d q;
        q << var.value[0], var.value[1], var.value[2], var.value[3];
        
        Eigen::Vector4d dq = Eigen::Vector4d::Zero();
        dq.head<3>() = 0.5 * dx.head<3>();
        dq[3] = 1.0;
        dq = quatnorm(dq);
        
        Eigen::Vector4d new_q = quat_multiply(dq, q);
        var.value[0] = new_q[0];
        var.value[1] = new_q[1];
        var.value[2] = new_q[2];
        var.value[3] = new_q[3];
        
        var.value[4] += dx[3];
        var.value[5] += dx[4];
        var.value[6] += dx[5];
        
        var.value[7] += dx[6];
        var.value[8] += dx[7];
        var.value[9] += dx[8];
        
        var.value[10] += dx[9];
        var.value[11] += dx[10];
        var.value[12] += dx[11];
        
        var.value[13] += dx[12];
        var.value[14] += dx[13];
        var.value[15] += dx[14];
    }
    else if (var.type == VariableType::VEC) {
        for (int i = 0; i < var.size; ++i) {
            var.value[i] += dx[i];
        }
    }
    else if (var.type == VariableType::JPLQUAT) {
        Eigen::Vector4d q;
        q << var.value[0], var.value[1], var.value[2], var.value[3];
        
        Eigen::Vector4d dq = Eigen::Vector4d::Zero();
        dq.head<3>() = 0.5 * dx;
        dq[3] = 1.0;
        dq = quatnorm(dq);
        
        Eigen::Vector4d new_q = quat_multiply(dq, q);
        var.value[0] = new_q[0];
        var.value[1] = new_q[1];
        var.value[2] = new_q[2];
        var.value[3] = new_q[3];
    }
    else if (var.type == VariableType::POSEJPL) {
        Eigen::Vector4d q;
        q << var.value[0], var.value[1], var.value[2], var.value[3];
        
        Eigen::Vector4d dq = Eigen::Vector4d::Zero();
        dq.head<3>() = 0.5 * dx.head<3>();
        dq[3] = 1.0;
        dq = quatnorm(dq);
        
        Eigen::Vector4d new_q = quat_multiply(dq, q);
        var.value[0] = new_q[0];
        var.value[1] = new_q[1];
        var.value[2] = new_q[2];
        var.value[3] = new_q[3];
        
        var.value[4] += dx[3];
        var.value[5] += dx[4];
        var.value[6] += dx[5];
    }
    else if (var.type == VariableType::LANDMARK) {
        for (int i = 0; i < var.size; ++i) {
            var.value[i] += dx[i];
        }
    }
}

void set_variable_value(Variable& var, const Eigen::Ref<const Eigen::VectorXd>& val) {
    assert(val.size() == var.storage_size);
    for (int i = 0; i < var.storage_size; ++i) {
        var.value[i] = val[i];
    }
}

void set_variable_fej(Variable& var, const Eigen::Ref<const Eigen::VectorXd>& val) {
    assert(val.size() == var.storage_size);
    for (int i = 0; i < var.storage_size; ++i) {
        var.fej[i] = val[i];
    }
}

Eigen::Vector3d get_landmark_xyz(const Variable& var, bool get_fej) {
    const double* current_val = get_fej ? var.fej : var.value;
    
    if (var.representation == LandmarkRepresentation::GLOBAL_3D ||
        var.representation == LandmarkRepresentation::ANCHORED_3D) {
        return Eigen::Vector3d(current_val[0], current_val[1], current_val[2]);
    }
    
    if (var.representation == LandmarkRepresentation::GLOBAL_FULL_INVERSE_DEPTH ||
        var.representation == LandmarkRepresentation::ANCHORED_FULL_INVERSE_DEPTH) {
        double theta = current_val[0];
        double phi = current_val[1];
        double rho = current_val[2];
        double inv_rho = (rho != 0.0) ? (1.0 / rho) : 0.0;
        return Eigen::Vector3d(inv_rho * std::cos(theta) * std::sin(phi),
                               inv_rho * std::sin(theta) * std::sin(phi),
                               inv_rho * std::cos(phi));
    }
    
    if (var.representation == LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH) {
        double inv_z = current_val[2];
        double z = (inv_z != 0.0) ? (1.0 / inv_z) : 0.0;
        return Eigen::Vector3d(z * current_val[0], z * current_val[1], z);
    }
    
    if (var.representation == LandmarkRepresentation::ANCHORED_INVERSE_DEPTH_SINGLE) {
        double rho = current_val[0];
        double inv_rho = (rho != 0.0) ? (1.0 / rho) : 0.0;
        const double* bearing = get_fej ? var.uv_norm_zero_fej : var.uv_norm_zero;
        return Eigen::Vector3d(inv_rho * bearing[0], inv_rho * bearing[1], inv_rho * bearing[2]);
    }
    
    return Eigen::Vector3d::Zero();
}

void set_landmark_from_xyz(Variable& var, const Eigen::Vector3d& xyz, bool is_fej) {
    double* target_val = is_fej ? var.fej : var.value;
    
    if (var.representation == LandmarkRepresentation::GLOBAL_3D ||
        var.representation == LandmarkRepresentation::ANCHORED_3D) {
        target_val[0] = xyz[0];
        target_val[1] = xyz[1];
        target_val[2] = xyz[2];
        return;
    }
    
    if (var.representation == LandmarkRepresentation::GLOBAL_FULL_INVERSE_DEPTH ||
        var.representation == LandmarkRepresentation::ANCHORED_FULL_INVERSE_DEPTH) {
        double norm = xyz.norm();
        double rho = (norm != 0.0) ? (1.0 / norm) : 0.0;
        double phi = (std::abs(rho * xyz[2]) <= 1.0) ? std::acos(rho * xyz[2]) : 0.0;
        double theta = std::atan2(xyz[1], xyz[0]);
        target_val[0] = theta;
        target_val[1] = phi;
        target_val[2] = rho;
        return;
    }
    
    if (var.representation == LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH) {
        double z = xyz[2];
        double inv_z = (z != 0.0) ? (1.0 / z) : 0.0;
        target_val[0] = xyz[0] * inv_z;
        target_val[1] = xyz[1] * inv_z;
        target_val[2] = inv_z;
        return;
    }
    
    if (var.representation == LandmarkRepresentation::ANCHORED_INVERSE_DEPTH_SINGLE) {
        double z = xyz[2];
        double inv_z = (z != 0.0) ? (1.0 / z) : 0.0;
        target_val[0] = inv_z;
        
        Eigen::Vector3d bearing = Eigen::Vector3d::Zero();
        if (z != 0.0) {
            bearing = inv_z * xyz;
        }
        if (!is_fej) {
            var.uv_norm_zero[0] = bearing[0];
            var.uv_norm_zero[1] = bearing[1];
            var.uv_norm_zero[2] = bearing[2];
        } else {
            var.uv_norm_zero_fej[0] = bearing[0];
            var.uv_norm_zero_fej[1] = bearing[1];
            var.uv_norm_zero_fej[2] = bearing[2];
        }
        return;
    }
}

Eigen::Matrix3d get_pose_rot(const Variable& var, bool get_fej) {
    Eigen::Vector4d q;
    const double* src = get_fej ? var.fej : var.value;
    q << src[0], src[1], src[2], src[3];
    return quat_2_Rot(q);
}

Eigen::Vector3d get_pose_pos(const Variable& var, bool get_fej) {
    const double* src = get_fej ? var.fej : var.value;
    return Eigen::Vector3d(src[4], src[5], src[6]);
}

Variable clone_variable(const Variable& var) {
    return var;
}

Eigen::Matrix3d Variable::Rot() const {
    Eigen::Vector4d q;
    q << value[0], value[1], value[2], value[3];
    return quat_2_Rot(q);
}

Eigen::Matrix3d Variable::Rot_fej() const {
    Eigen::Vector4d q;
    q << fej[0], fej[1], fej[2], fej[3];
    return quat_2_Rot(q);
}

Eigen::Vector3d Variable::pos() const {
    return Eigen::Vector3d(value[4], value[5], value[6]);
}

Eigen::Vector3d Variable::pos_fej() const {
    return Eigen::Vector3d(fej[4], fej[5], fej[6]);
}

Eigen::Vector3d Variable::vel() const {
    return Eigen::Vector3d(value[7], value[8], value[9]);
}

Eigen::Vector3d Variable::vel_fej() const {
    return Eigen::Vector3d(fej[7], fej[8], fej[9]);
}

Eigen::Vector3d Variable::bias_g() const {
    return Eigen::Vector3d(value[10], value[11], value[12]);
}

Eigen::Vector3d Variable::bias_a() const {
    return Eigen::Vector3d(value[13], value[14], value[15]);
}

} // namespace type
