#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace type {

Eigen::Vector4d rot_2_quat(const Eigen::Matrix3d& rot);
Eigen::Matrix3d skew_x(const Eigen::Vector3d& w);
Eigen::Matrix3d quat_2_Rot(const Eigen::Vector4d& q);
Eigen::Vector4d quat_multiply(const Eigen::Vector4d& q, const Eigen::Vector4d& p);
Eigen::Vector3d vee(const Eigen::Matrix3d& w_x);
Eigen::Matrix3d exp_so3(const Eigen::Vector3d& w);
Eigen::Vector3d log_so3(const Eigen::Matrix3d& R);
Eigen::Matrix4d exp_se3(const Eigen::Matrix<double, 6, 1>& vec);
Eigen::Matrix<double, 6, 1> log_se3(const Eigen::Matrix4d& mat);
Eigen::Matrix4d hat_se3(const Eigen::Matrix<double, 6, 1>& vec);
Eigen::Matrix4d Inv_se3(const Eigen::Matrix4d& T);
Eigen::Vector4d Inv(const Eigen::Vector4d& q);
Eigen::Matrix4d Omega(const Eigen::Vector3d& w);
Eigen::Vector4d quatnorm(const Eigen::Vector4d& q);
Eigen::Matrix3d Jl_so3(const Eigen::Vector3d& w);
Eigen::Matrix3d Jr_so3(const Eigen::Vector3d& w);
Eigen::Vector3d rot2rpy(const Eigen::Matrix3d& rot);
Eigen::Matrix3d rot_x(double t);
Eigen::Matrix3d rot_y(double t);
Eigen::Matrix3d rot_z(double t);

} // namespace type
