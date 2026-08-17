#include "initialization.hpp"
#include "../type/quat_ops.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <map>
#include <cstdint>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace initialize {

static void gram_schmidt(const Eigen::Vector3d& gravity_inI, Eigen::Matrix3d& R_GtoI) {
    double norm_g = gravity_inI.norm();
    if (norm_g < 1e-6) {
        R_GtoI.setIdentity();
        return;
    }
    
    Eigen::Vector3d z_axis = gravity_inI / norm_g;
    Eigen::Vector3d e_1(1.0, 0.0, 0.0);
    Eigen::Vector3d e_2(0.0, 1.0, 0.0);
    
    double inner1 = e_1.dot(z_axis);
    double inner2 = e_2.dot(z_axis);
    
    Eigen::Vector3d x_axis;
    if (std::abs(inner1) < std::abs(inner2)) {
        x_axis = z_axis.cross(e_1);
    } else {
        x_axis = z_axis.cross(e_2);
    }
    
    double norm_x = x_axis.norm();
    if (norm_x < 1e-6) {
        x_axis = Eigen::Vector3d(1.0, 0.0, 0.0);
    } else {
        x_axis /= norm_x;
    }
    
    Eigen::Vector3d y_axis = z_axis.cross(x_axis);
    y_axis.normalize();
    
    R_GtoI.col(0) = x_axis;
    R_GtoI.col(1) = y_axis;
    R_GtoI.col(2) = z_axis;
}

bool static_initialize(const InitializerOptions& config,
                                   const core::ImuData* imu_data, int imu_count,
                                   type::Variable& t_imu,
                                   Eigen::Matrix<double, 15, 15>& covariance,
                                   double& timestamp,
                                   bool wait_for_jerk) {
    if (imu_count < 2) return false;
    
    double newest_time = imu_data[0].timestamp;
    double oldest_time = imu_data[0].timestamp;
    for (int i = 0; i < imu_count; ++i) {
        if (imu_data[i].timestamp > newest_time) newest_time = imu_data[i].timestamp;
        if (imu_data[i].timestamp < oldest_time) oldest_time = imu_data[i].timestamp;
    }
    
    if ((newest_time - oldest_time) < config.init_window_time) return false;
    
    double t_mid = newest_time - 0.5 * config.init_window_time;
    double t_start = newest_time - config.init_window_time;
    
    // Allocate pointers on stack to avoid memory allocations
    const core::ImuData* window_1to0[2000];
    int count_1to0 = 0;
    const core::ImuData* window_2to1[2000];
    int count_2to1 = 0;
    
    for (int i = 0; i < imu_count; ++i) {
        double t = imu_data[i].timestamp;
        if (t >= t_start && t < t_mid) {
            if (count_2to1 < 2000) window_2to1[count_2to1++] = &imu_data[i];
        } else if (t >= t_mid && t <= newest_time) {
            if (count_1to0 < 2000) window_1to0[count_1to0++] = &imu_data[i];
        }
    }
    
    if (count_1to0 < 2 || count_2to1 < 2) return false;
    
    Eigen::Vector3d a_avg_1to0 = Eigen::Vector3d::Zero();
    for (int i = 0; i < count_1to0; ++i) a_avg_1to0 += window_1to0[i]->am;
    a_avg_1to0 /= count_1to0;
    
    double a_var_1to0 = 0.0;
    for (int i = 0; i < count_1to0; ++i) {
        Eigen::Vector3d diff = window_1to0[i]->am - a_avg_1to0;
        a_var_1to0 += diff.squaredNorm();
    }
    a_var_1to0 = std::sqrt(a_var_1to0 / (count_1to0 - 1));
    
    Eigen::Vector3d a_avg_2to1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d w_avg_2to1 = Eigen::Vector3d::Zero();
    for (int i = 0; i < count_2to1; ++i) {
        a_avg_2to1 += window_2to1[i]->am;
        w_avg_2to1 += window_2to1[i]->wm;
    }
    a_avg_2to1 /= count_2to1;
    w_avg_2to1 /= count_2to1;
    
    double a_var_2to1 = 0.0;
    for (int i = 0; i < count_2to1; ++i) {
        Eigen::Vector3d diff = window_2to1[i]->am - a_avg_2to1;
        a_var_2to1 += diff.squaredNorm();
    }
    a_var_2to1 = std::sqrt(a_var_2to1 / (count_2to1 - 1));
    
    double thresh = config.init_imu_thresh;
    
    if (wait_for_jerk) {
        if (a_var_1to0 < thresh) return false;
        if (a_var_2to1 > thresh) return false;
    } else {
        if (a_var_1to0 > thresh || a_var_2to1 > thresh) return false;
    }
    
    Eigen::Vector3d z_axis = a_avg_2to1.normalized();
    Eigen::Matrix3d Ro = Eigen::Matrix3d::Identity();
    gram_schmidt(z_axis, Ro);
    Eigen::Vector4d q_GtoI = type::rot_2_quat(Ro);
    
    Eigen::Vector3d gravity_inG(0.0, 0.0, config.gravity_mag);
    Eigen::Vector3d bg = w_avg_2to1;
    Eigen::Matrix3d rot_mat = type::quat_2_Rot(q_GtoI);
    Eigen::Vector3d ba = a_avg_2to1 - rot_mat * gravity_inG;
    
    Eigen::Matrix<double, 16, 1> imu_state = Eigen::Matrix<double, 16, 1>::Zero();
    imu_state.head<4>() = q_GtoI;
    imu_state.segment<3>(10) = bg;
    imu_state.tail<3>() = ba;
    
    int original_id = t_imu.id;
    type::init_imu(t_imu);
    t_imu.id = original_id;
    type::set_variable_value(t_imu, imu_state);
    type::set_variable_fej(t_imu, imu_state);
    
    // Matches official StaticInitializer.cpp exactly: theta/bg/ba stay at
    // 0.02^2, but position and velocity get their own overrides (0.05^2 and
    // 0.01^2) rather than sharing the uniform 0.02^2 this port previously
    // used for all 15 error-state dims.
    covariance = (0.02 * 0.02) * Eigen::Matrix<double, 15, 15>::Identity();
    covariance.block<3, 3>(3, 3) = (0.05 * 0.05) * Eigen::Matrix3d::Identity(); // position
    covariance.block<3, 3>(6, 6) = (0.01 * 0.01) * Eigen::Matrix3d::Identity(); // velocity
    
    double latest_ts_2to1 = window_2to1[0]->timestamp;
    for (int i = 0; i < count_2to1; ++i) {
        if (window_2to1[i]->timestamp > latest_ts_2to1) {
            latest_ts_2to1 = window_2to1[i]->timestamp;
        }
    }
    timestamp = latest_ts_2to1;
    return true;
}

// ============================================================================
// Dynamic (linear MLE) initializer -- ported from OpenVINS ov_init
// DynamicInitializer (linear stage only, no Ceres refinement). Recovers
// gravity direction, velocity, and feature depths from a window of camera
// poses connected by IMU continuous-preintegration (CPI-v1 means), solved
// under the |g| = gravity_mag constraint via the Dongsi closed-form
// (companion-matrix eigenvalue) method. Used for sequences with NO clean
// stationary start (e.g. EuRoC MH_*), where static_initialize cannot fire.
// ============================================================================
namespace {

core::ImuData interp_imu(const core::ImuData& a, const core::ImuData& b, double t) {
    double lam = (t - a.timestamp) / (b.timestamp - a.timestamp);
    core::ImuData d;
    d.timestamp = t;
    d.am = (1.0 - lam) * a.am + lam * b.am;
    d.wm = (1.0 - lam) * a.wm + lam * b.wm;
    return d;
}

// Select IMU readings in [t0,t1], interpolating the endpoints. Mirrors
// InitializerHelper::select_imu_readings. Returns count written to out.
int select_imu(const core::ImuData* imu, int n, double t0, double t1,
               core::ImuData* out, int cap) {
    int c = 0;
    if (n == 0) return 0;
    for (int i = 0; i < n - 1; ++i) {
        if (imu[i + 1].timestamp > t0 && imu[i].timestamp < t0) {
            if (c < cap) out[c++] = interp_imu(imu[i], imu[i + 1], t0);
            continue;
        }
        if (imu[i].timestamp >= t0 && imu[i + 1].timestamp <= t1) {
            if (c < cap) out[c++] = imu[i];
            continue;
        }
        if (imu[i + 1].timestamp > t1) {
            if (imu[i].timestamp > t1 && i == 0) {
                break;
            } else if (imu[i].timestamp > t1) {
                if (c < cap) out[c++] = interp_imu(imu[i - 1], imu[i], t1);
            } else {
                if (c < cap) out[c++] = imu[i];
            }
            if (c > 0 && out[c - 1].timestamp != t1) {
                if (c < cap) out[c++] = interp_imu(imu[i], imu[i + 1], t1);
            }
            break;
        }
    }
    if (c == 0) return 0;
    for (int i = 0; i < c - 1; ++i) {
        if (std::abs(out[i + 1].timestamp - out[i].timestamp) < 1e-12) {
            for (int j = i; j < c - 1; ++j) out[j] = out[j + 1];
            --c; --i;
        }
    }
    return c;
}

// CPI-v1 means only (DT, R_k2tau, alpha_tau, beta_tau), imu_avg=true.
// Drops the Jacobian/covariance machinery of ov_core::CpiV1 -- the linear
// initializer uses only the mean preintegration.
struct CpiMeans {
    double DT = 0.0;
    Eigen::Matrix3d R_k2tau = Eigen::Matrix3d::Identity();
    Eigen::Vector3d alpha_tau = Eigen::Vector3d::Zero();
    Eigen::Vector3d beta_tau = Eigen::Vector3d::Zero();
    Eigen::Vector3d b_w = Eigen::Vector3d::Zero();
    Eigen::Vector3d b_a = Eigen::Vector3d::Zero();

    void feed(double t0, double t1, const Eigen::Vector3d& w0, const Eigen::Vector3d& a0,
              const Eigen::Vector3d& w1, const Eigen::Vector3d& a1) {
        double dt = t1 - t0;
        DT += dt;
        if (dt == 0.0) return;
        Eigen::Vector3d w_hat = 0.5 * ((w0 - b_w) + (w1 - b_w));
        Eigen::Vector3d a_hat = 0.5 * ((a0 - b_a) + (a1 - b_a));
        double mag_w = w_hat.norm();
        double w_dt = mag_w * dt;
        bool small_w = (mag_w < 0.008726646);
        double dt_2 = dt * dt;
        double cos_wt = std::cos(w_dt);
        double sin_wt = std::sin(w_dt);
        const Eigen::Matrix3d eye3 = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d w_x = type::skew_x(w_hat);
        Eigen::Matrix3d w_x_2 = w_x * w_x;
        Eigen::Matrix3d R_tau2tau1 = small_w
            ? (eye3 - dt * w_x + (dt_2 / 2.0) * w_x_2)
            : (eye3 - (sin_wt / mag_w) * w_x + ((1.0 - cos_wt) / (mag_w * mag_w)) * w_x_2);
        Eigen::Matrix3d R_k2tau1 = R_tau2tau1 * R_k2tau;
        Eigen::Matrix3d R_tau12k = R_k2tau1.transpose();
        double f_1, f_2, f_3, f_4;
        if (small_w) {
            f_1 = -(dt * dt * dt / 3.0);
            f_2 = (dt * dt * dt * dt / 8.0);
            f_3 = -(dt_2 / 2.0);
            f_4 = (dt * dt * dt / 6.0);
        } else {
            f_1 = (w_dt * cos_wt - sin_wt) / (mag_w * mag_w * mag_w);
            f_2 = (w_dt * w_dt - 2.0 * cos_wt - 2.0 * w_dt * sin_wt + 2.0) / (2.0 * std::pow(mag_w, 4));
            f_3 = -(1.0 - cos_wt) / (mag_w * mag_w);
            f_4 = (w_dt - sin_wt) / (mag_w * mag_w * mag_w);
        }
        Eigen::Matrix3d alpha_arg = (dt_2 / 2.0) * eye3 + f_1 * w_x + f_2 * w_x_2;
        Eigen::Matrix3d Beta_arg = dt * eye3 + f_3 * w_x + f_4 * w_x_2;
        Eigen::Matrix3d H_al = R_tau12k * alpha_arg;
        Eigen::Matrix3d H_be = R_tau12k * Beta_arg;
        alpha_tau += beta_tau * dt + H_al * a_hat;
        beta_tau += H_be * a_hat;
        R_k2tau = R_k2tau1;
    }
};

// Compute coefficients for the constrained |g| optimization polynomial.
// Copied verbatim from ov_init InitializerHelper::compute_dongsi_coeff
// (auto-generated MATLAB expansion). Returns coeffs highest->constant.
Eigen::Matrix<double, 7, 1> compute_dongsi_coeff(const Eigen::MatrixXd& D, const Eigen::MatrixXd& d, double gravity_mag) {
    double D1_1 = D(0, 0), D1_2 = D(0, 1), D1_3 = D(0, 2);
    double D2_1 = D(1, 0), D2_2 = D(1, 1), D2_3 = D(1, 2);
    double D3_1 = D(2, 0), D3_2 = D(2, 1), D3_3 = D(2, 2);
    double d1 = d(0, 0), d2 = d(1, 0), d3 = d(2, 0);
    double g = gravity_mag;
    double D1_1_sq = D1_1 * D1_1, D1_2_sq = D1_2 * D1_2, D1_3_sq = D1_3 * D1_3;
    double D2_1_sq = D2_1 * D2_1, D2_2_sq = D2_2 * D2_2, D2_3_sq = D2_3 * D2_3;
    double D3_1_sq = D3_1 * D3_1, D3_2_sq = D3_2 * D3_2, D3_3_sq = D3_3 * D3_3;
    double d1_sq = d1 * d1, d2_sq = d2 * d2, d3_sq = d3 * d3;
    double g_sq = g * g;
    Eigen::Matrix<double, 7, 1> coeff = Eigen::Matrix<double, 7, 1>::Zero();
    coeff(6) =
        -(-D1_1_sq * D2_2_sq * D3_3_sq * g_sq + D1_1_sq * D2_2_sq * d3_sq + 2 * D1_1_sq * D2_2 * D2_3 * D3_2 * D3_3 * g_sq -
          D1_1_sq * D2_2 * D2_3 * d2 * d3 - D1_1_sq * D2_2 * D3_2 * d2 * d3 - D1_1_sq * D2_3_sq * D3_2_sq * g_sq +
          D1_1_sq * D2_3 * D3_2 * d2_sq + D1_1_sq * D2_3 * D3_2 * d3_sq - D1_1_sq * D2_3 * D3_3 * d2 * d3 -
          D1_1_sq * D3_2 * D3_3 * d2 * d3 + D1_1_sq * D3_3_sq * d2_sq + 2 * D1_1 * D1_2 * D2_1 * D2_2 * D3_3_sq * g_sq -
          2 * D1_1 * D1_2 * D2_1 * D2_2 * d3_sq - 2 * D1_1 * D1_2 * D2_1 * D2_3 * D3_2 * D3_3 * g_sq + D1_1 * D1_2 * D2_1 * D2_3 * d2 * d3 +
          D1_1 * D1_2 * D2_1 * D3_2 * d2 * d3 - 2 * D1_1 * D1_2 * D2_2 * D2_3 * D3_1 * D3_3 * g_sq + D1_1 * D1_2 * D2_2 * D2_3 * d1 * d3 +
          D1_1 * D1_2 * D2_2 * D3_1 * d2 * d3 + 2 * D1_1 * D1_2 * D2_3_sq * D3_1 * D3_2 * g_sq - D1_1 * D1_2 * D2_3 * D3_1 * d2_sq -
          D1_1 * D1_2 * D2_3 * D3_1 * d3_sq - D1_1 * D1_2 * D2_3 * D3_2 * d1 * d2 + D1_1 * D1_2 * D2_3 * D3_3 * d1 * d3 +
          D1_1 * D1_2 * D3_1 * D3_3 * d2 * d3 - D1_1 * D1_2 * D3_3_sq * d1 * d2 - 2 * D1_1 * D1_3 * D2_1 * D2_2 * D3_2 * D3_3 * g_sq +
          D1_1 * D1_3 * D2_1 * D2_2 * d2 * d3 + 2 * D1_1 * D1_3 * D2_1 * D2_3 * D3_2_sq * g_sq - D1_1 * D1_3 * D2_1 * D3_2 * d2_sq -
          D1_1 * D1_3 * D2_1 * D3_2 * d3_sq + D1_1 * D1_3 * D2_1 * D3_3 * d2 * d3 + 2 * D1_1 * D1_3 * D2_2_sq * D3_1 * D3_3 * g_sq -
          D1_1 * D1_3 * D2_2_sq * d1 * d3 - 2 * D1_1 * D1_3 * D2_2 * D2_3 * D3_1 * D3_2 * g_sq + D1_1 * D1_3 * D2_2 * D3_2 * d1 * d2 +
          D1_1 * D1_3 * D2_3 * D3_1 * d2 * d3 - D1_1 * D1_3 * D2_3 * D3_2 * d1 * d3 + D1_1 * D1_3 * D3_1 * D3_2 * d2 * d3 -
          2 * D1_1 * D1_3 * D3_1 * D3_3 * d2_sq + D1_1 * D1_3 * D3_2 * D3_3 * d1 * d2 + D1_1 * D2_1 * D2_2 * D3_2 * d1 * d3 -
          D1_1 * D2_1 * D2_3 * D3_2 * d1 * d2 + D1_1 * D2_1 * D3_2 * D3_3 * d1 * d3 - D1_1 * D2_1 * D3_3_sq * d1 * d2 -
          D1_1 * D2_2_sq * D3_1 * d1 * d3 + D1_1 * D2_2 * D2_3 * D3_1 * d1 * d2 - D1_1 * D2_3 * D3_1 * D3_2 * d1 * d3 +
          D1_1 * D2_3 * D3_1 * D3_3 * d1 * d2 - D1_2_sq * D2_1_sq * D3_3_sq * g_sq + D1_2_sq * D2_1_sq * d3_sq +
          2 * D1_2_sq * D2_1 * D2_3 * D3_1 * D3_3 * g_sq - D1_2_sq * D2_1 * D2_3 * d1 * d3 - D1_2_sq * D2_1 * D3_1 * d2 * d3 -
          D1_2_sq * D2_3_sq * D3_1_sq * g_sq + D1_2_sq * D2_3 * D3_1 * d1 * d2 + 2 * D1_2 * D1_3 * D2_1_sq * D3_2 * D3_3 * g_sq -
          D1_2 * D1_3 * D2_1_sq * d2 * d3 - 2 * D1_2 * D1_3 * D2_1 * D2_2 * D3_1 * D3_3 * g_sq + D1_2 * D1_3 * D2_1 * D2_2 * d1 * d3 -
          2 * D1_2 * D1_3 * D2_1 * D2_3 * D3_1 * D3_2 * g_sq + D1_2 * D1_3 * D2_1 * D3_1 * d2_sq + D1_2 * D1_3 * D2_1 * D3_1 * d3_sq -
          D1_2 * D1_3 * D2_1 * D3_3 * d1 * d3 + 2 * D1_2 * D1_3 * D2_2 * D2_3 * D3_1_sq * g_sq - D1_2 * D1_3 * D2_2 * D3_1 * d1 * d2 -
          D1_2 * D1_3 * D3_1_sq * d2 * d3 + D1_2 * D1_3 * D3_1 * D3_3 * d1 * d2 - D1_2 * D2_1_sq * D3_2 * d1 * d3 +
          D1_2 * D2_1 * D2_2 * D3_1 * d1 * d3 + D1_2 * D2_1 * D2_3 * D3_2 * d1_sq + D1_2 * D2_1 * D2_3 * D3_2 * d3_sq -
          D1_2 * D2_1 * D2_3 * D3_3 * d2 * d3 - D1_2 * D2_1 * D3_1 * D3_3 * d1 * d3 - D1_2 * D2_1 * D3_2 * D3_3 * d2 * d3 +
          D1_2 * D2_1 * D3_3_sq * d1_sq + D1_2 * D2_1 * D3_3_sq * d2_sq - D1_2 * D2_2 * D2_3 * D3_1 * d1_sq -
          D1_2 * D2_2 * D2_3 * D3_1 * d3_sq + D1_2 * D2_2 * D2_3 * D3_3 * d1 * d3 + D1_2 * D2_2 * D3_1 * D3_3 * d2 * d3 -
          D1_2 * D2_2 * D3_3_sq * d1 * d2 + D1_2 * D2_3_sq * D3_1 * d2 * d3 - D1_2 * D2_3_sq * D3_2 * d1 * d3 +
          D1_2 * D2_3 * D3_1_sq * d1 * d3 - D1_2 * D2_3 * D3_1 * D3_3 * d1_sq - D1_2 * D2_3 * D3_1 * D3_3 * d2_sq +
          D1_2 * D2_3 * D3_2 * D3_3 * d1 * d2 - D1_3_sq * D2_1_sq * D3_2_sq * g_sq + 2 * D1_3_sq * D2_1 * D2_2 * D3_1 * D3_2 * g_sq -
          D1_3_sq * D2_1 * D3_1 * d2 * d3 + D1_3_sq * D2_1 * D3_2 * d1 * d3 - D1_3_sq * D2_2_sq * D3_1_sq * g_sq +
          D1_3_sq * D3_1_sq * d2_sq - D1_3_sq * D3_1 * D3_2 * d1 * d2 + D1_3 * D2_1_sq * D3_2 * d1 * d2 -
          D1_3 * D2_1 * D2_2 * D3_1 * d1 * d2 - D1_3 * D2_1 * D2_2 * D3_2 * d1_sq - D1_3 * D2_1 * D2_2 * D3_2 * d3_sq +
          D1_3 * D2_1 * D2_2 * D3_3 * d2 * d3 + D1_3 * D2_1 * D3_1 * D3_3 * d1 * d2 + D1_3 * D2_1 * D3_2_sq * d2 * d3 -
          D1_3 * D2_1 * D3_2 * D3_3 * d1_sq - D1_3 * D2_1 * D3_2 * D3_3 * d2_sq + D1_3 * D2_2_sq * D3_1 * d1_sq +
          D1_3 * D2_2_sq * D3_1 * d3_sq - D1_3 * D2_2_sq * D3_3 * d1 * d3 - D1_3 * D2_2 * D2_3 * D3_1 * d2 * d3 +
          D1_3 * D2_2 * D2_3 * D3_2 * d1 * d3 - D1_3 * D2_2 * D3_1 * D3_2 * d2 * d3 + D1_3 * D2_2 * D3_2 * D3_3 * d1 * d2 -
          D1_3 * D2_3 * D3_1_sq * d1 * d2 + D1_3 * D2_3 * D3_1 * D3_2 * d1_sq + D1_3 * D2_3 * D3_1 * D3_2 * d2_sq -
          D1_3 * D2_3 * D3_2_sq * d1 * d2 + D2_1 * D2_2 * D3_2 * D3_3 * d1 * d3 - D2_1 * D2_2 * D3_3_sq * d1 * d2 -
          D2_1 * D2_3 * D3_2_sq * d1 * d3 + D2_1 * D2_3 * D3_2 * D3_3 * d1 * d2 - D2_2_sq * D3_1 * D3_3 * d1 * d3 +
          D2_2_sq * D3_3_sq * d1_sq + D2_2 * D2_3 * D3_1 * D3_2 * d1 * d3 + D2_2 * D2_3 * D3_1 * D3_3 * d1 * d2 -
          2 * D2_2 * D2_3 * D3_2 * D3_3 * d1_sq - D2_3_sq * D3_1 * D3_2 * d1 * d2 + D2_3_sq * D3_2_sq * d1_sq) /
        g_sq;
    coeff(5) =
        (-(2 * D1_1_sq * D2_2_sq * D3_3 * g_sq - 2 * D1_1_sq * D2_2 * D2_3 * D3_2 * g_sq + 2 * D1_1_sq * D2_2 * D3_3_sq * g_sq -
           2 * D1_1_sq * D2_2 * d3_sq - 2 * D1_1_sq * D2_3 * D3_2 * D3_3 * g_sq + 2 * D1_1_sq * D2_3 * d2 * d3 +
           2 * D1_1_sq * D3_2 * d2 * d3 - 2 * D1_1_sq * D3_3 * d2_sq - 4 * D1_1 * D1_2 * D2_1 * D2_2 * D3_3 * g_sq +
           2 * D1_1 * D1_2 * D2_1 * D2_3 * D3_2 * g_sq - 2 * D1_1 * D1_2 * D2_1 * D3_3_sq * g_sq + 2 * D1_1 * D1_2 * D2_1 * d3_sq +
           2 * D1_1 * D1_2 * D2_2 * D2_3 * D3_1 * g_sq + 2 * D1_1 * D1_2 * D2_3 * D3_1 * D3_3 * g_sq - 2 * D1_1 * D1_2 * D2_3 * d1 * d3 -
           2 * D1_1 * D1_2 * D3_1 * d2 * d3 + 2 * D1_1 * D1_2 * D3_3 * d1 * d2 + 2 * D1_1 * D1_3 * D2_1 * D2_2 * D3_2 * g_sq +
           2 * D1_1 * D1_3 * D2_1 * D3_2 * D3_3 * g_sq - 2 * D1_1 * D1_3 * D2_1 * d2 * d3 - 2 * D1_1 * D1_3 * D2_2_sq * D3_1 * g_sq -
           4 * D1_1 * D1_3 * D2_2 * D3_1 * D3_3 * g_sq + 2 * D1_1 * D1_3 * D2_2 * d1 * d3 + 2 * D1_1 * D1_3 * D2_3 * D3_1 * D3_2 * g_sq +
           2 * D1_1 * D1_3 * D3_1 * d2_sq - 2 * D1_1 * D1_3 * D3_2 * d1 * d2 - 2 * D1_1 * D2_1 * D3_2 * d1 * d3 +
           2 * D1_1 * D2_1 * D3_3 * d1 * d2 + 2 * D1_1 * D2_2_sq * D3_3_sq * g_sq - 2 * D1_1 * D2_2_sq * d3_sq -
           4 * D1_1 * D2_2 * D2_3 * D3_2 * D3_3 * g_sq + 2 * D1_1 * D2_2 * D2_3 * d2 * d3 + 2 * D1_1 * D2_2 * D3_1 * d1 * d3 +
           2 * D1_1 * D2_2 * D3_2 * d2 * d3 + 2 * D1_1 * D2_3_sq * D3_2_sq * g_sq - 2 * D1_1 * D2_3 * D3_1 * d1 * d2 -
           2 * D1_1 * D2_3 * D3_2 * d2_sq - 2 * D1_1 * D2_3 * D3_2 * d3_sq + 2 * D1_1 * D2_3 * D3_3 * d2 * d3 +
           2 * D1_1 * D3_2 * D3_3 * d2 * d3 - 2 * D1_1 * D3_3_sq * d2_sq + 2 * D1_2_sq * D2_1_sq * D3_3 * g_sq -
           2 * D1_2_sq * D2_1 * D2_3 * D3_1 * g_sq - 2 * D1_2 * D1_3 * D2_1_sq * D3_2 * g_sq + 2 * D1_2 * D1_3 * D2_1 * D2_2 * D3_1 * g_sq +
           2 * D1_2 * D1_3 * D2_1 * D3_1 * D3_3 * g_sq - 2 * D1_2 * D1_3 * D2_3 * D3_1_sq * g_sq - 2 * D1_2 * D2_1 * D2_2 * D3_3_sq * g_sq +
           2 * D1_2 * D2_1 * D2_2 * d3_sq + 2 * D1_2 * D2_1 * D2_3 * D3_2 * D3_3 * g_sq - 2 * D1_2 * D2_1 * D3_3 * d1_sq -
           2 * D1_2 * D2_1 * D3_3 * d2_sq + 2 * D1_2 * D2_2 * D2_3 * D3_1 * D3_3 * g_sq - 2 * D1_2 * D2_2 * D2_3 * d1 * d3 -
           2 * D1_2 * D2_2 * D3_1 * d2 * d3 + 2 * D1_2 * D2_2 * D3_3 * d1 * d2 - 2 * D1_2 * D2_3_sq * D3_1 * D3_2 * g_sq +
           2 * D1_2 * D2_3 * D3_1 * d1_sq + 2 * D1_2 * D2_3 * D3_1 * d2_sq + 2 * D1_2 * D2_3 * D3_1 * d3_sq -
           2 * D1_2 * D2_3 * D3_3 * d1 * d3 - 2 * D1_2 * D3_1 * D3_3 * d2 * d3 + 2 * D1_2 * D3_3_sq * d1 * d2 -
           2 * D1_3_sq * D2_1 * D3_1 * D3_2 * g_sq + 2 * D1_3_sq * D2_2 * D3_1_sq * g_sq + 2 * D1_3 * D2_1 * D2_2 * D3_2 * D3_3 * g_sq -
           2 * D1_3 * D2_1 * D2_2 * d2 * d3 - 2 * D1_3 * D2_1 * D2_3 * D3_2_sq * g_sq + 2 * D1_3 * D2_1 * D3_2 * d1_sq +
           2 * D1_3 * D2_1 * D3_2 * d2_sq + 2 * D1_3 * D2_1 * D3_2 * d3_sq - 2 * D1_3 * D2_1 * D3_3 * d2 * d3 -
           2 * D1_3 * D2_2_sq * D3_1 * D3_3 * g_sq + 2 * D1_3 * D2_2_sq * d1 * d3 + 2 * D1_3 * D2_2 * D2_3 * D3_1 * D3_2 * g_sq -
           2 * D1_3 * D2_2 * D3_1 * d1_sq - 2 * D1_3 * D2_2 * D3_1 * d3_sq - 2 * D1_3 * D2_2 * D3_2 * d1 * d2 +
           2 * D1_3 * D2_2 * D3_3 * d1 * d3 + 2 * D1_3 * D3_1 * D3_3 * d2_sq - 2 * D1_3 * D3_2 * D3_3 * d1 * d2 -
           2 * D2_1 * D2_2 * D3_2 * d1 * d3 + 2 * D2_1 * D2_2 * D3_3 * d1 * d2 - 2 * D2_1 * D3_2 * D3_3 * d1 * d3 +
           2 * D2_1 * D3_3_sq * d1 * d2 + 2 * D2_2_sq * D3_1 * d1 * d3 - 2 * D2_2_sq * D3_3 * d1_sq - 2 * D2_2 * D2_3 * D3_1 * d1 * d2 +
           2 * D2_2 * D2_3 * D3_2 * d1_sq + 2 * D2_2 * D3_1 * D3_3 * d1 * d3 - 2 * D2_2 * D3_3_sq * d1_sq -
           2 * D2_3 * D3_1 * D3_3 * d1 * d2 + 2 * D2_3 * D3_2 * D3_3 * d1_sq) /
         g_sq);
    coeff(4) =
        ((D1_1_sq * D2_2_sq * g_sq + 4 * D1_1_sq * D2_2 * D3_3 * g_sq - 2 * D1_1_sq * D2_3 * D3_2 * g_sq + D1_1_sq * D3_3_sq * g_sq -
          D1_1_sq * d2_sq - D1_1_sq * d3_sq - 2 * D1_1 * D1_2 * D2_1 * D2_2 * g_sq - 4 * D1_1 * D1_2 * D2_1 * D3_3 * g_sq +
          2 * D1_1 * D1_2 * D2_3 * D3_1 * g_sq + D1_1 * D1_2 * d1 * d2 + 2 * D1_1 * D1_3 * D2_1 * D3_2 * g_sq -
          4 * D1_1 * D1_3 * D2_2 * D3_1 * g_sq - 2 * D1_1 * D1_3 * D3_1 * D3_3 * g_sq + D1_1 * D1_3 * d1 * d3 + D1_1 * D2_1 * d1 * d2 +
          4 * D1_1 * D2_2_sq * D3_3 * g_sq - 4 * D1_1 * D2_2 * D2_3 * D3_2 * g_sq + 4 * D1_1 * D2_2 * D3_3_sq * g_sq -
          4 * D1_1 * D2_2 * d3_sq - 4 * D1_1 * D2_3 * D3_2 * D3_3 * g_sq + 4 * D1_1 * D2_3 * d2 * d3 + D1_1 * D3_1 * d1 * d3 +
          4 * D1_1 * D3_2 * d2 * d3 - 4 * D1_1 * D3_3 * d2_sq + D1_2_sq * D2_1_sq * g_sq + 2 * D1_2 * D1_3 * D2_1 * D3_1 * g_sq -
          4 * D1_2 * D2_1 * D2_2 * D3_3 * g_sq + 2 * D1_2 * D2_1 * D2_3 * D3_2 * g_sq - 2 * D1_2 * D2_1 * D3_3_sq * g_sq -
          D1_2 * D2_1 * d1_sq - D1_2 * D2_1 * d2_sq + 2 * D1_2 * D2_1 * d3_sq + 2 * D1_2 * D2_2 * D2_3 * D3_1 * g_sq +
          D1_2 * D2_2 * d1 * d2 + 2 * D1_2 * D2_3 * D3_1 * D3_3 * g_sq - 3 * D1_2 * D2_3 * d1 * d3 - 3 * D1_2 * D3_1 * d2 * d3 +
          4 * D1_2 * D3_3 * d1 * d2 + D1_3_sq * D3_1_sq * g_sq + 2 * D1_3 * D2_1 * D2_2 * D3_2 * g_sq +
          2 * D1_3 * D2_1 * D3_2 * D3_3 * g_sq - 3 * D1_3 * D2_1 * d2 * d3 - 2 * D1_3 * D2_2_sq * D3_1 * g_sq -
          4 * D1_3 * D2_2 * D3_1 * D3_3 * g_sq + 4 * D1_3 * D2_2 * d1 * d3 + 2 * D1_3 * D2_3 * D3_1 * D3_2 * g_sq - D1_3 * D3_1 * d1_sq +
          2 * D1_3 * D3_1 * d2_sq - D1_3 * D3_1 * d3_sq - 3 * D1_3 * D3_2 * d1 * d2 + D1_3 * D3_3 * d1 * d3 + D2_1 * D2_2 * d1 * d2 -
          3 * D2_1 * D3_2 * d1 * d3 + 4 * D2_1 * D3_3 * d1 * d2 + D2_2_sq * D3_3_sq * g_sq - D2_2_sq * d1_sq - D2_2_sq * d3_sq -
          2 * D2_2 * D2_3 * D3_2 * D3_3 * g_sq + D2_2 * D2_3 * d2 * d3 + 4 * D2_2 * D3_1 * d1 * d3 + D2_2 * D3_2 * d2 * d3 -
          4 * D2_2 * D3_3 * d1_sq + D2_3_sq * D3_2_sq * g_sq - 3 * D2_3 * D3_1 * d1 * d2 + 2 * D2_3 * D3_2 * d1_sq - D2_3 * D3_2 * d2_sq -
          D2_3 * D3_2 * d3_sq + D2_3 * D3_3 * d2 * d3 + D3_1 * D3_3 * d1 * d3 + D3_2 * D3_3 * d2 * d3 - D3_3_sq * d1_sq - D3_3_sq * d2_sq) /
         g_sq);
    coeff(3) =
        ((2 * D1_1 * d2_sq + 2 * D1_1 * d3_sq + 2 * D2_2 * d1_sq + 2 * D2_2 * d3_sq + 2 * D3_3 * d1_sq + 2 * D3_3 * d2_sq -
          2 * D1_1 * D2_2_sq * g_sq - 2 * D1_1_sq * D2_2 * g_sq - 2 * D1_1 * D3_3_sq * g_sq - 2 * D1_1_sq * D3_3 * g_sq -
          2 * D2_2 * D3_3_sq * g_sq - 2 * D2_2_sq * D3_3 * g_sq - 2 * D1_2 * d1 * d2 - 2 * D1_3 * d1 * d3 - 2 * D2_1 * d1 * d2 -
          2 * D2_3 * d2 * d3 - 2 * D3_1 * d1 * d3 - 2 * D3_2 * d2 * d3 + 2 * D1_1 * D1_2 * D2_1 * g_sq + 2 * D1_1 * D1_3 * D3_1 * g_sq +
          2 * D1_2 * D2_1 * D2_2 * g_sq - 8 * D1_1 * D2_2 * D3_3 * g_sq + 4 * D1_1 * D2_3 * D3_2 * g_sq + 4 * D1_2 * D2_1 * D3_3 * g_sq -
          2 * D1_2 * D2_3 * D3_1 * g_sq - 2 * D1_3 * D2_1 * D3_2 * g_sq + 4 * D1_3 * D2_2 * D3_1 * g_sq + 2 * D1_3 * D3_1 * D3_3 * g_sq +
          2 * D2_2 * D2_3 * D3_2 * g_sq + 2 * D2_3 * D3_2 * D3_3 * g_sq) /
         g_sq);
    coeff(2) =
        (-(d1_sq + d2_sq + d3_sq - D1_1_sq * g_sq - D2_2_sq * g_sq - D3_3_sq * g_sq - 4 * D1_1 * D2_2 * g_sq + 2 * D1_2 * D2_1 * g_sq -
           4 * D1_1 * D3_3 * g_sq + 2 * D1_3 * D3_1 * g_sq - 4 * D2_2 * D3_3 * g_sq + 2 * D2_3 * D3_2 * g_sq) /
         g_sq);
    coeff(1) = (-(2 * D1_1 * g_sq + 2 * D2_2 * g_sq + 2 * D3_3 * g_sq) / g_sq);
    coeff(0) = 1;
    return coeff;
}

} // namespace

// ---------------------------------------------------------------------------
// Feature-less dynamic initialization (sqrtVINS, Peng et al., T-RO 2025, Sec. V-A).
//
// The existing dynamic initializer solves for [features, velocity, gravity]
// jointly. Over a short, low-parallax window that system is nearly rank
// deficient in the feature block, and the velocity it returns is unreliable --
// measured against Leica truth, MH_02 recovered 0.162 m/s where the truth was
// 0.029 and the run diverged, while every quality metric (residual,
// conditioning, recovered gravity direction) looked healthy. The mitigation
// shipped for that was to discard the velocity entirely, which is a workaround,
// not a fix.
//
// This recovers velocity and gravity WITHOUT ever estimating a 3D point:
//
//   Prop. 1  For a feature seen from two frames, the two bearings rotated into
//            {I0} span an epipolar plane whose normal n is perpendicular to the
//            relative translation t. Stacking M = sum(n n^T) over features, t is
//            the eigenvector of M for the smallest eigenvalue -- the up-to-scale
//            translation direction, from bearings alone.
//
//   Prop. 2  Preintegration gives the same relative translation with scale:
//                s t = A x + c,   x = [v_I0; g],
//                A = [ (dT_j - dT_i) I,  (dT_j^2 - dT_i^2)/2 I ]
//                c = (alpha_j - alpha_i) + (R_j^T - R_i^T) p_CinI
//            Projecting onto the two directions orthogonal to t eliminates the
//            unknown scale s, leaving 2 equations per frame pair in 6 unknowns.
//
// Cost is O(M) in features rather than O(M^3), and low parallax degrades the
// direction estimate gracefully instead of making the system singular.
bool featureless_initialize(const InitializerOptions& config,
                            core::FeatureDatabase& db,
                            const core::ImuData* imu_data, int imu_count,
                            const double camera_extrinsics[2][7], int num_cameras,
                            type::Variable& t_imu,
                            Eigen::Matrix<double, 15, 15>& covariance,
                            double& timestamp);


bool featureless_initialize(const InitializerOptions& config,
                            core::FeatureDatabase& db,
                            const core::ImuData* imu_data, int imu_count,
                            const double camera_extrinsics[2][7], int num_cameras,
                            type::Variable& t_imu,
                            Eigen::Matrix<double, 15, 15>& covariance,
                            double& timestamp) {
    (void)num_cameras;
    double newest_cam_time = -1.0;
    for (int f = 0; f < (int)db.count; ++f) {
        const core::Feature& feat = db.features[f];
        for (int m = 0; m < feat.num_measurements; ++m) {
            newest_cam_time = std::max(newest_cam_time, feat.measurements[m].timestamp);
        }
    }
    if (newest_cam_time < 0.0) return false;

    // Camera times spanning the window, evenly spaced.
    std::vector<double> times;
    {
        std::map<double, bool> unique_times;
        const double oldest_allowed = newest_cam_time - config.init_window_time;
        for (int f = 0; f < (int)db.count; ++f) {
            const core::Feature& feat = db.features[f];
            for (int m = 0; m < feat.num_measurements; ++m) {
                const double t = feat.measurements[m].timestamp;
                if (t >= oldest_allowed) unique_times[t] = true;
            }
        }
        if ((int)unique_times.size() < 2) return false;
        // Take up to kMaxTimes samples spread evenly over whatever the window
        // actually holds, rather than a fixed spacing. Fixed spacing gave only 3
        // times here -- 3 pairs, 6 equations for 6 unknowns -- and the resulting
        // gravity was 3.2 deg off on MH_01, which is a 0.5 m/s^2 horizontal
        // acceleration leak and diverges. The pair count is O(k^2), so k is
        // capped rather than unbounded.
        constexpr int kMaxTimes = 8;
        std::vector<double> all;
        all.reserve(unique_times.size());
        for (auto const& tp : unique_times) all.push_back(tp.first);
        const int take = std::min<int>(kMaxTimes, int(all.size()));
        for (int i = 0; i < take; ++i) {
            const int idx = (take == 1) ? 0 : int((std::int64_t)i * (all.size() - 1) / (take - 1));
            if (times.empty() || all[idx] != times.back()) times.push_back(all[idx]);
        }
    }
    if (times.size() < 3) return false;

    const double t0 = times.front();
    const Eigen::Vector3d gyroscope_bias = config.init_dyn_bias_g;
    const Eigen::Vector3d accelerometer_bias = config.init_dyn_bias_a;

    // Preintegrate I0 -> Ik for each selected time.
    std::map<double, CpiMeans> cpi;
    static core::ImuData seg[4096];
    for (double tk : times) {
        CpiMeans c;
        c.b_w = gyroscope_bias;
        c.b_a = accelerometer_bias;
        if (tk != t0) {
            const double a = t0 + config.init_calib_camImu_dt;
            const double b = tk + config.init_calib_camImu_dt;
            const int ns = select_imu(imu_data, imu_count, a, b, seg, 4096);
            if (ns < 2) return false;
            if (std::abs((seg[ns - 1].timestamp - seg[0].timestamp) - (b - a)) > 0.01) return false;
            for (int k = 0; k < ns - 1; ++k) {
                c.feed(seg[k].timestamp, seg[k + 1].timestamp, seg[k].wm, seg[k].am,
                       seg[k + 1].wm, seg[k + 1].am);
            }
        }
        cpi[tk] = c;
    }

    // Camera-IMU extrinsics: state stores T_cam_imu = (R_ItoC, p_IinC).
    const Eigen::Vector4d q_ItoC(camera_extrinsics[0][0], camera_extrinsics[0][1],
                                 camera_extrinsics[0][2], camera_extrinsics[0][3]);
    const Eigen::Matrix3d R_ItoC = type::quat_2_Rot(q_ItoC);
    const Eigen::Vector3d p_IinC(camera_extrinsics[0][4], camera_extrinsics[0][5],
                                 camera_extrinsics[0][6]);
    const Eigen::Vector3d p_CinI = -R_ItoC.transpose() * p_IinC;

    // Two equations per frame pair, six unknowns [v_I0; g].
    std::vector<Eigen::Matrix<double, 1, 6>> rows;
    std::vector<double> rhs;
    int used_pairs = 0;

    for (std::size_t a = 0; a < times.size(); ++a) {
        for (std::size_t b = a + 1; b < times.size(); ++b) {
            const double ti = times[a];
            const double tj = times[b];
            const Eigen::Matrix3d R_i = cpi[ti].R_k2tau;  // R_{I0 -> Ii}
            const Eigen::Matrix3d R_j = cpi[tj].R_k2tau;

            // Prop. 1: epipolar-plane normals from bearings rotated into {I0}.
            Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
            int shared = 0;
            for (int f = 0; f < (int)db.count; ++f) {
                const core::Feature& feat = db.features[f];
                const Eigen::Vector2d* uv_i = nullptr;
                const Eigen::Vector2d* uv_j = nullptr;
                for (int m = 0; m < feat.num_measurements; ++m) {
                    if (feat.measurements[m].cam_id != 0) continue;
                    if (feat.measurements[m].timestamp == ti) uv_i = &feat.measurements[m].uv_norm;
                    if (feat.measurements[m].timestamp == tj) uv_j = &feat.measurements[m].uv_norm;
                }
                if (uv_i == nullptr || uv_j == nullptr) continue;
                Eigen::Vector3d bi(uv_i->x(), uv_i->y(), 1.0);
                Eigen::Vector3d bj(uv_j->x(), uv_j->y(), 1.0);
                bi.normalize();
                bj.normalize();
                // camera -> IMU -> {I0}
                const Eigen::Vector3d b_i0 = R_i.transpose() * (R_ItoC.transpose() * bi);
                const Eigen::Vector3d b_j0 = R_j.transpose() * (R_ItoC.transpose() * bj);
                const Eigen::Vector3d n = b_i0.cross(b_j0);
                const double n_norm = n.norm();
                if (n_norm < 1e-9) continue;   // parallel bearings carry no direction
                const Eigen::Vector3d n_unit = n / n_norm;
                M.noalias() += n_unit * n_unit.transpose();
                ++shared;
            }
            if (shared < 8) continue;

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(M);
            if (eig.info() != Eigen::Success) continue;
            // Ascending eigenvalues: t is the null direction, e spans its complement.
            const Eigen::Vector3d t_dir = eig.eigenvectors().col(0);
            const Eigen::Vector3d e1 = eig.eigenvectors().col(1);
            const Eigen::Vector3d e2 = eig.eigenvectors().col(2);

            // Prop. 2: s t = A x + c, projected onto e to drop the scale s.
            const double dTi = cpi[ti].DT;
            const double dTj = cpi[tj].DT;
            Eigen::Matrix<double, 3, 6> A;
            A.leftCols<3>() = (dTj - dTi) * Eigen::Matrix3d::Identity();
            // NEGATIVE: this pipeline's state assembly (and the feature-based
            // initializer it shares it with) uses
            //     p_Ik = v dT - 1/2 g dT^2 + alpha
            // i.e. the recovered `gravity_inI0` is the UPWARD vector, not the
            // physical downward one the paper writes with a +1/2 g dT^2. The
            // solved quantity has to match the convention it is consumed in.
            A.rightCols<3>() = -0.5 * (dTj * dTj - dTi * dTi) * Eigen::Matrix3d::Identity();
            const Eigen::Vector3d c =
                (cpi[tj].alpha_tau - cpi[ti].alpha_tau) +
                (R_j.transpose() - R_i.transpose()) * p_CinI;

            rows.push_back(e1.transpose() * A);
            rhs.push_back(-e1.dot(c));
            rows.push_back(e2.transpose() * A);
            rhs.push_back(-e2.dot(c));
            ++used_pairs;
        }
    }

    if (used_pairs < 3 || (int)rows.size() < 6) {
        std::cout << "[init-f]: only " << used_pairs << " usable frame pairs\n";
        return false;
    }

    Eigen::MatrixXd Amat(rows.size(), 6);
    Eigen::VectorXd bvec(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        Amat.row(i) = rows[i];
        bvec(i) = rhs[i];
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Amat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const double rec_cond = svd.singularValues()(5) / svd.singularValues()(0);
    if (!(rec_cond > config.init_dyn_min_rec_cond)) {
        std::cout << "[init-f]: rec_cond " << rec_cond << " too small\n";
        return false;
    }
    Eigen::Matrix<double, 6, 1> x = svd.solve(bvec);

    Eigen::Vector3d v_I0inI0 = x.head<3>();
    Eigen::Vector3d gravity_inI0 = x.tail<3>();
    if (gravity_inI0.norm() < 1e-6) return false;

    // |g| is known; enforce it, then re-solve the velocity with gravity fixed.
    gravity_inI0 = gravity_inI0.normalized() * config.gravity_mag;
    {
        Eigen::MatrixXd Av = Amat.leftCols<3>();
        Eigen::VectorXd bv = bvec - Amat.rightCols<3>() * gravity_inI0;
        v_I0inI0 = Av.colPivHouseholderQr().solve(bv);
    }

    // Cross-check the recovered gravity against the accelerometer: over a short
    // window the mean specific force is dominated by gravity.
    double gravity_angle_deg = -1.0;
    {
        Eigen::Vector3d mean_am = Eigen::Vector3d::Zero();
        int n = 0;
        for (int k = 0; k < imu_count; ++k) {
            if (imu_data[k].timestamp < t0) continue;
            if (imu_data[k].timestamp > times.back()) break;
            mean_am += imu_data[k].am;
            ++n;
        }
        if (n > 0) {
            mean_am /= double(n);
            const double c = mean_am.normalized().dot(gravity_inI0.normalized());
            gravity_angle_deg = 180.0 / M_PI * std::acos(std::max(-1.0, std::min(1.0, c)));
        }
    }
    std::cout << "[init-f]: featureless init OK  |g|=" << gravity_inI0.norm()
              << " |v0|=" << v_I0inI0.norm() << " pairs=" << used_pairs
              << " times=" << times.size()
              << " rec_cond=" << rec_cond
              << " gravity-vs-accel=" << gravity_angle_deg << " deg\n";

    // Same state assembly as the dynamic initializer: gravity-align, then
    // propagate I0 -> newest.
    Eigen::Matrix3d R_GtoI0 = Eigen::Matrix3d::Identity();
    gram_schmidt(gravity_inI0, R_GtoI0);
    const Eigen::Vector4d q_GtoI0 = type::rot_2_quat(R_GtoI0);

    const CpiMeans& cpiN = cpi[times.back()];
    const double DTn = cpiN.DT;
    const Eigen::Vector3d p_IkinI0 =
        v_I0inI0 * DTn - 0.5 * gravity_inI0 * DTn * DTn + cpiN.alpha_tau;
    const Eigen::Vector3d v_IkinI0 = v_I0inI0 - gravity_inI0 * DTn + cpiN.beta_tau;
    const Eigen::Vector4d q_I0toIk = type::rot_2_quat(cpiN.R_k2tau);
    const Eigen::Vector4d q_GtoIk = type::quat_multiply(q_I0toIk, q_GtoI0);
    const Eigen::Vector3d p_IkinG = R_GtoI0.transpose() * p_IkinI0;
    const Eigen::Vector3d v_IkinG = R_GtoI0.transpose() * v_IkinI0;

    Eigen::Matrix<double, 16, 1> imu_state = Eigen::Matrix<double, 16, 1>::Zero();
    imu_state.head<4>() = q_GtoIk;
    imu_state.segment<3>(4) = p_IkinG;
    imu_state.segment<3>(7) = v_IkinG;
    imu_state.segment<3>(10) = gyroscope_bias;
    imu_state.tail<3>() = accelerometer_bias;

    const int original_id = t_imu.id;
    type::init_imu(t_imu);
    t_imu.id = original_id;
    type::set_variable_value(t_imu, imu_state);
    type::set_variable_fej(t_imu, imu_state);

    covariance = (0.02 * 0.02) * Eigen::Matrix<double, 15, 15>::Identity();
    covariance.block<3, 3>(3, 3) = (0.05 * 0.05) * Eigen::Matrix3d::Identity();
    covariance.block<3, 3>(6, 6) = (0.05 * 0.05) * Eigen::Matrix3d::Identity();
    covariance.block<3, 3>(12, 12) = (0.05 * 0.05) * Eigen::Matrix3d::Identity();
    covariance.block<3, 3>(0, 0) *= config.init_dyn_inflation_ori;
    covariance.block<3, 3>(6, 6) *= config.init_dyn_inflation_vel;
    covariance.block<3, 3>(9, 9) *= config.init_dyn_inflation_bg;
    covariance.block<3, 3>(12, 12) *= config.init_dyn_inflation_ba;

    timestamp = times.back();
    return true;
}

bool dynamic_initialize(const InitializerOptions& config,
                        core::FeatureDatabase& db,
                        const core::ImuData* imu_data, int imu_count,
                        const double camera_extrinsics[2][7], int num_cameras,
                        type::Variable& t_imu,
                        Eigen::Matrix<double, 15, 15>& covariance,
                        double& timestamp) {
    // Newest / oldest camera time across all features in the DB.
    double newest_cam_time = -1.0;
    for (int f = 0; f < (int)db.count; ++f) {
        const core::Feature& feat = db.features[f];
        for (int m = 0; m < feat.num_measurements; ++m) {
            newest_cam_time = std::max(newest_cam_time, feat.measurements[m].timestamp);
        }
    }
    double oldest_time = newest_cam_time - config.init_window_time;
    if (newest_cam_time < 0.0 || oldest_time < 0.0) return false;

    core::cleanup_db_measurements(db, oldest_time);

    // Need old IMU readings spanning the window (so I0 is covered).
    double oldest_imu = (imu_count > 0) ? imu_data[0].timestamp : newest_cam_time;
    bool have_old_imu = (oldest_imu < oldest_time + config.init_calib_camImu_dt) && imu_count >= 2;
    if (!have_old_imu) return false;

    // Require enough features in the window.
    int valid_db = 0;
    for (int f = 0; f < (int)db.count; ++f)
        if (db.features[f].num_measurements > 0) ++valid_db;
    if (valid_db < 0.75 * config.init_max_features) return false;

    const int min_num_meas_to_optimize = (int)config.init_window_time;
    const int min_valid_features = 8;

    // ---- Select the set of camera pose-times at ~init_dyn_num_pose spacing ----
    std::map<double, bool> map_camera_times;
    map_camera_times[newest_cam_time] = true;
    std::map<int, int> feat_num_meas; // feat index -> #times selected
    int count_valid_features = 0;
    double oldest_camera_time = INFINITY;
    double pose_dt_avg = config.init_window_time / (double)(config.init_dyn_num_pose + 1);
    for (int f = 0; f < (int)db.count; ++f) {
        // Bound the linear system. state = [3 per feature, vel, gravity], so
        // system_size grows as 3N and A is (2 * observations) x system_size --
        // letting N follow however many features the tracker happens to supply
        // makes this allocate and multiply unboundedly. Official keeps its
        // initializer on a separate, bounded feature set sized by
        // init_max_features; this is the same bound applied directly.
        if (count_valid_features >= config.init_max_features) break;
        const core::Feature& feat = db.features[f];
        std::vector<double> times;
        for (int m = 0; m < feat.num_measurements; ++m) {
            double time = feat.measurements[m].timestamp;
            double time_dt = INFINITY;
            for (auto const& tmp : map_camera_times) time_dt = std::min(time_dt, std::abs(time - tmp.first));
            for (double tt : times) time_dt = std::min(time_dt, std::abs(time - tt));
            if (time_dt >= pose_dt_avg || time_dt == 0.0) times.push_back(time);
        }
        feat_num_meas[f] = (int)times.size();
        if ((int)times.size() < min_num_meas_to_optimize) continue;
        for (double tt : times) {
            map_camera_times[tt] = true;
            oldest_camera_time = std::min(oldest_camera_time, tt);
        }
        ++count_valid_features;
    }
    if ((int)map_camera_times.size() < config.init_dyn_num_pose) return false;
    if (count_valid_features < min_valid_features) return false;

    Eigen::Vector3d gyroscope_bias = config.init_dyn_bias_g;
    Eigen::Vector3d accelerometer_bias = config.init_dyn_bias_a;

    // ---- Require enough orientation change (avoid degenerate pure-translation) ----
    double time0_in_imu = oldest_camera_time + config.init_calib_camImu_dt;
    double time1_in_imu = newest_cam_time + config.init_calib_camImu_dt;
    static core::ImuData readings[4096];
    int nread = select_imu(imu_data, imu_count, time0_in_imu, time1_in_imu, readings, 4096);
    if (nread <= 2) return false;
    double theta_inI_norm = 0.0;
    for (int k = 0; k < nread - 1; ++k) {
        double dt = readings[k + 1].timestamp - readings[k].timestamp;
        Eigen::Vector3d wm = 0.5 * (readings[k].wm + readings[k + 1].wm) - gyroscope_bias;
        theta_inI_norm += (-wm * dt).norm();
    }
    if (180.0 / M_PI * theta_inI_norm < config.init_dyn_min_deg) {
        std::cout << "[init-d]: only " << 180.0 / M_PI * theta_inI_norm << " deg change ("
                  << config.init_dyn_min_deg << " thresh)\n";
        return false;
    }

    // ---- Preintegrate I0 -> Ik for every pose-time (means only) ----
    const int size_feature = 3;
    int num_features = count_valid_features;
    int system_size = size_feature * num_features + 3 + 3;

    std::map<double, CpiMeans> cpi_I0toIi;
    static core::ImuData seg[4096];
    for (auto const& tp : map_camera_times) {
        double cur = tp.first;
        if (cur == oldest_camera_time) { cpi_I0toIi[cur] = CpiMeans{}; continue; }
        double t0 = oldest_camera_time + config.init_calib_camImu_dt;
        double t1 = cur + config.init_calib_camImu_dt;
        int ns = select_imu(imu_data, imu_count, t0, t1, seg, 4096);
        if (ns < 2) return false;
        double dt_imu = seg[ns - 1].timestamp - seg[0].timestamp;
        if (std::abs(dt_imu - (t1 - t0)) > 0.01) return false;
        CpiMeans cpi;
        cpi.b_w = gyroscope_bias;
        cpi.b_a = accelerometer_bias;
        for (int k = 0; k < ns - 1; ++k)
            cpi.feed(seg[k].timestamp, seg[k + 1].timestamp, seg[k].wm, seg[k].am, seg[k + 1].wm, seg[k + 1].am);
        cpi_I0toIi[cur] = cpi;
    }

    // ---- Build the linear system A x = b, state = [features, vel, grav] ----
    // First count the actual rows we will append (2 per used observation).
    std::map<int, int> A_index_features;
    int idx_feat = 0;
    int num_rows = 0;
    for (int f = 0; f < (int)db.count; ++f) {
        if (feat_num_meas[f] < min_num_meas_to_optimize) continue;
        A_index_features[f] = idx_feat++;
        const core::Feature& feat = db.features[f];
        for (int m = 0; m < feat.num_measurements; ++m)
            if (map_camera_times.count(feat.measurements[m].timestamp)) num_rows += 2;
    }
    if (num_rows < system_size) return false;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(num_rows, system_size);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(num_rows);
    int index_meas = 0;
    for (int f = 0; f < (int)db.count; ++f) {
        if (feat_num_meas[f] < min_num_meas_to_optimize) continue;
        const core::Feature& feat = db.features[f];
        for (int m = 0; m < feat.num_measurements; ++m) {
            const core::FeatureMeasurement& obs = feat.measurements[m];
            double time = obs.timestamp;
            if (!map_camera_times.count(time)) continue;
            int cam_id = obs.cam_id;
            Eigen::Vector4d q_ItoC(camera_extrinsics[cam_id][0], camera_extrinsics[cam_id][1],
                                   camera_extrinsics[cam_id][2], camera_extrinsics[cam_id][3]);
            Eigen::Vector3d p_IinC(camera_extrinsics[cam_id][4], camera_extrinsics[cam_id][5],
                                   camera_extrinsics[cam_id][6]);
            Eigen::Matrix3d R_ItoC = type::quat_2_Rot(q_ItoC);
            Eigen::Vector2d uv_norm = obs.uv_norm;

            double DT = 0.0;
            Eigen::Matrix3d R_I0toIk = Eigen::Matrix3d::Identity();
            Eigen::Vector3d alpha_I0toIk = Eigen::Vector3d::Zero();
            auto it = cpi_I0toIi.find(time);
            if (it != cpi_I0toIi.end()) { DT = it->second.DT; R_I0toIk = it->second.R_k2tau; alpha_I0toIk = it->second.alpha_tau; }

            Eigen::Matrix<double, 2, 3> H_proj;
            H_proj << 1, 0, -uv_norm(0), 0, 1, -uv_norm(1);
            Eigen::Matrix<double, 2, 3> Y = H_proj * R_ItoC * R_I0toIk;
            Eigen::Matrix<double, 2, 1> b_i = Y * alpha_I0toIk - H_proj * p_IinC;
            A.block(index_meas, size_feature * A_index_features[f], 2, 3) = Y;       // feat
            A.block(index_meas, size_feature * num_features + 0, 2, 3) = -DT * Y;    // vel
            A.block(index_meas, size_feature * num_features + 3, 2, 3) = 0.5 * DT * DT * Y; // grav
            b.block(index_meas, 0, 2, 1) = b_i;
            index_meas += 2;
        }
    }

    // ---- Constrained solve with |g| = gravity_mag (Dongsi closed-form) ----
    Eigen::MatrixXd A1 = A.block(0, 0, A.rows(), A.cols() - 3);
    Eigen::MatrixXd A1A1_inv = (A1.transpose() * A1).llt().solve(Eigen::MatrixXd::Identity(A1.cols(), A1.cols()));
    Eigen::MatrixXd A2 = A.block(0, A.cols() - 3, A.rows(), 3);
    Eigen::MatrixXd Temp = A2.transpose() * (Eigen::MatrixXd::Identity(A1.rows(), A1.rows()) - A1 * A1A1_inv * A1.transpose());
    Eigen::MatrixXd D = Temp * A2;
    Eigen::MatrixXd dd = Temp * b;
    Eigen::Matrix<double, 7, 1> coeff = compute_dongsi_coeff(D, dd, config.gravity_mag);

    Eigen::Matrix<double, 6, 6> companion = Eigen::Matrix<double, 6, 6>::Zero();
    companion.diagonal(-1).setOnes();
    companion.col(companion.cols() - 1) = -coeff.reverse().head(6);
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd0(companion);
    if ((int)svd0.rank() != (int)companion.rows()) return false;
    Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> solver(companion, false);
    if (solver.info() != Eigen::Success) return false;

    bool lambda_found = false;
    double lambda_min = -1.0, cost_min = INFINITY;
    Eigen::MatrixXd I_dd = Eigen::MatrixXd::Identity(D.rows(), D.rows());
    for (int i = 0; i < solver.eigenvalues().size(); ++i) {
        auto val = solver.eigenvalues()(i);
        if (val.imag() == 0.0) {
            double lambda = val.real();
            Eigen::VectorXd sg = (D - lambda * I_dd).llt().solve(I_dd) * dd;
            double cost = std::abs(sg.norm() - config.gravity_mag);
            if (!lambda_found || cost < cost_min) { lambda_found = true; lambda_min = lambda; cost_min = cost; }
        }
    }
    if (!lambda_found) return false;

    // Excitation of the window. Velocity is only observable from vision +
    // preintegration if the platform actually accelerated: with a near-static
    // window the least-squares solution trades velocity against feature depth
    // almost freely, and the recovered v0 is meaningless even though the fit,
    // the conditioning and the recovered gravity all look healthy. (Measured:
    // MH_02 recovers 0.162 m/s against a true 0.029, and the run diverges;
    // MH_01 recovers 0.445 against a true 0.048 and merely survives. Official
    // does not hit this because it refines the linear solution with a Ceres
    // MLE over the IMU residuals, which DOD does not implement.)
    double accel_excitation = 0.0;
    {
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        int n = 0;
        for (int k = 0; k < imu_count; ++k) {
            if (imu_data[k].timestamp < oldest_camera_time) continue;
            if (imu_data[k].timestamp > newest_cam_time) break;
            mean += imu_data[k].am;
            ++n;
        }
        if (n > 1) {
            mean /= double(n);
            double sum = 0.0;
            for (int k = 0; k < imu_count; ++k) {
                if (imu_data[k].timestamp < oldest_camera_time) continue;
                if (imu_data[k].timestamp > newest_cam_time) break;
                sum += (imu_data[k].am - mean).squaredNorm();
            }
            accel_excitation = std::sqrt(sum / double(n));
        }
    }

    Eigen::VectorXd state_grav = (D - lambda_min * I_dd).llt().solve(I_dd) * dd;
    Eigen::VectorXd state_feat_vel = -A1A1_inv * A1.transpose() * A2 * state_grav + A1A1_inv * A1.transpose() * b;
    Eigen::Vector3d v_I0inI0 = state_feat_vel.segment<3>(size_feature * num_features + 0);

    Eigen::Vector3d gravity_inI0 = state_grav;

    // How well does the recovered state actually explain the measurements?
    {
        Eigen::VectorXd solution(A.cols());
        solution.head(A1.cols()) = state_feat_vel;
        solution.tail(3) = state_grav;
        const double b_norm = b.norm();
        const double relative_residual = b_norm > 1e-12 ? (A * solution - b).norm() / b_norm : 0.0;
        // Physical cross-check: over a short window with modest acceleration the
        // mean specific force is dominated by gravity, so the recovered gravity
        // direction should sit close to -mean(am). A linear solve that latched
        // onto the wrong root gets this badly wrong even when it satisfies the
        // |g| constraint and explains the pixels.
        Eigen::Vector3d mean_am = Eigen::Vector3d::Zero();
        int used = 0;
        for (int k = 0; k < imu_count; ++k) {
            if (imu_data[k].timestamp < oldest_camera_time) continue;
            if (imu_data[k].timestamp > newest_cam_time) break;
            mean_am += imu_data[k].am;
            ++used;
        }
        double gravity_angle_deg = 0.0;
        if (used > 0 && state_grav.norm() > 1e-9) {
            mean_am /= double(used);
            const double cosine = mean_am.normalized().dot((-state_grav).normalized());
            gravity_angle_deg = 180.0 / M_PI * std::acos(std::max(-1.0, std::min(1.0, cosine)));
        }
        // Conditioning of the linear system. An under-constrained window (too
        // little parallax) still fits its own measurements -- the residual
        // stays small because the solution is not unique, not because it is
        // right. Official guards this with init_dyn_min_rec_cond; DOD checked
        // nothing but the companion matrix's rank.
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_A(A);
        const double smax = svd_A.singularValues()(0);
        const double smin = svd_A.singularValues()(svd_A.singularValues().size() - 1);
        const double rec_cond = smax > 0.0 ? smin / smax : 0.0;
        std::cout << "[init-d]: rec_cond " << rec_cond << " excitation " << accel_excitation;
        if (rec_cond < config.init_dyn_min_rec_cond) {
            std::cout << " -- rejected (below " << config.init_dyn_min_rec_cond << ")\n";
            return false;
        }
        std::cout << "[init-d]: linear residual " << relative_residual
                  << " gravity-vs-accel " << gravity_angle_deg << " deg"
                  << " |v0| " << state_feat_vel.segment<3>(size_feature * num_features + 0).norm() << "\n";
        if (gravity_angle_deg > config.init_dyn_max_gravity_deg) {
            std::cout << "[init-d]: rejected, gravity " << gravity_angle_deg
                      << " deg from measured specific force\n";
            return false;
        }
        if (relative_residual > config.init_dyn_max_residual) {
            std::cout << "[init-d]: rejected, residual " << relative_residual
                      << " > " << config.init_dyn_max_residual << "\n";
            return false;
        }
    }

    if (std::abs(gravity_inI0.norm() - config.gravity_mag) > 1e-3) {
        std::cout << "[init-d]: gravity did not converge (|g|=" << gravity_inI0.norm() << ")\n";
        return false;
    }

    // ---- Recover the newest IMU pose, gravity-align, seed the state ----
    Eigen::Matrix3d R_GtoI0 = Eigen::Matrix3d::Identity();
    gram_schmidt(gravity_inI0, R_GtoI0);
    Eigen::Vector4d q_GtoI0 = type::rot_2_quat(R_GtoI0);

    const CpiMeans& cpiN = cpi_I0toIi[newest_cam_time];
    double DTn = cpiN.DT;
    Eigen::Vector3d p_IkinI0 = v_I0inI0 * DTn - 0.5 * gravity_inI0 * DTn * DTn + cpiN.alpha_tau;
    Eigen::Vector3d v_IkinI0 = v_I0inI0 - gravity_inI0 * DTn + cpiN.beta_tau;
    Eigen::Vector4d q_I0toIk = type::rot_2_quat(cpiN.R_k2tau);
    Eigen::Vector4d q_GtoIk = type::quat_multiply(q_I0toIk, q_GtoI0);
    Eigen::Vector3d p_IkinG = R_GtoI0.transpose() * p_IkinI0;
    Eigen::Vector3d v_IkinG = R_GtoI0.transpose() * v_IkinI0;
    if (config.init_dyn_zero_velocity || accel_excitation < config.init_dyn_min_excitation) {
        // Discard the velocity the linear stage could not observe, keeping the
        // gravity alignment and pose it did recover. This zeroes the velocity
        // actually handed to the filter -- zeroing v_I0inI0 instead does NOT
        // work, because the CPI chain rebuilds v_Ik from it and the window's
        // own delta-v survives (measured: MH_02 still diverged at 4.1e4).
        std::cout << "[init-d]: discarding unobservable velocity (was "
                  << v_IkinG.norm() << " m/s, window excitation " << accel_excitation << ")\n";
        v_IkinG.setZero();
    }

    std::cout << "[init-d]: dynamic init OK  |g|=" << gravity_inI0.norm()
              << " |v0|=" << v_I0inI0.norm() << " feats=" << count_valid_features
              << " poses=" << map_camera_times.size() << "\n";

    Eigen::Matrix<double, 16, 1> imu_state = Eigen::Matrix<double, 16, 1>::Zero();
    imu_state.head<4>() = q_GtoIk;
    imu_state.segment<3>(4) = p_IkinG;
    imu_state.segment<3>(7) = v_IkinG;
    imu_state.segment<3>(10) = gyroscope_bias;
    imu_state.tail<3>() = accelerometer_bias;

    int original_id = t_imu.id;
    type::init_imu(t_imu);
    t_imu.id = original_id;
    type::set_variable_value(t_imu, imu_state);
    // FEJ deliberately keeps the SOLVED velocity even when the state velocity
    // above was discarded. Zeroing both diverges on MH_02 (3.4e4 m) while
    // zeroing only the value gives 0.1744 m -- the Jacobians are evidently
    // better conditioned around the solve's own linearisation point than around
    // a velocity the solve never produced. This asymmetry is empirical and is
    // the sort of thing official's MLE refinement would make unnecessary.
    Eigen::Matrix<double, 16, 1> fej_state = imu_state;
    fej_state.segment<3>(7) = R_GtoI0.transpose() * v_IkinI0;
    type::set_variable_fej(t_imu, fej_state);

    // Prior covariance: dynamic init is less certain than a still start, so
    // velocity gets a looser prior. (Official inflates the MLE information
    // matrix; without the Ceres refinement we use a fixed conservative prior.)
    covariance = (0.02 * 0.02) * Eigen::Matrix<double, 15, 15>::Identity();
    covariance.block<3, 3>(3, 3) = (0.05 * 0.05) * Eigen::Matrix3d::Identity();  // position
    covariance.block<3, 3>(6, 6) = (0.05 * 0.05) * Eigen::Matrix3d::Identity();  // velocity
    covariance.block<3, 3>(12, 12) = (0.05 * 0.05) * Eigen::Matrix3d::Identity(); // accel bias
    // Official's inflation factors, applied for the same reason it applies
    // them: the recovered state is a starting guess, and a prior that claims
    // more confidence than the solve deserves leaves the filter unable to
    // correct the error it started with.
    covariance.block<3, 3>(0, 0) *= config.init_dyn_inflation_ori;
    covariance.block<3, 3>(6, 6) *= config.init_dyn_inflation_vel;
    covariance.block<3, 3>(9, 9) *= config.init_dyn_inflation_bg;
    covariance.block<3, 3>(12, 12) *= config.init_dyn_inflation_ba;

    timestamp = newest_cam_time;
    return true;
}

bool run_initialization(const InitializerOptions& config,
                        core::FeatureDatabase& db,
                        const core::ImuData* imu_data, int imu_count,
                        type::Variable& t_imu,
                        Eigen::Matrix<double, 15, 15>& covariance,
                        double& timestamp,
                        int& init_attempt,
                        bool wait_for_jerk) {
    double newest_time = -1.0;
    for (int i = 0; i < (int)db.count; ++i) {
        const core::Feature& feat = db.features[i];
        for (int m = 0; m < feat.num_measurements; ++m) {
            if (feat.measurements[m].timestamp > newest_time) {
                newest_time = feat.measurements[m].timestamp;
            }
        }
    }
    
    double oldest_time = newest_time - config.init_window_time - 0.10;
    if (newest_time < 0.0 || oldest_time < 0.0) return false;
    
    core::cleanup_db_measurements(db, oldest_time);
    
    double t_imu_filter = oldest_time + config.init_calib_camImu_dt;
    
    // Stack-allocated subset buffer to avoid dynamic allocation
    core::ImuData filtered_imu[2000];
    int filtered_count = 0;
    for (int i = 0; i < imu_count; ++i) {
        if (imu_data[i].timestamp >= t_imu_filter) {
            if (filtered_count < 2000) {
                filtered_imu[filtered_count++] = imu_data[i];
            }
        }
    }
    
    bool disparity_detected_moving_1to0 = false;
    bool disparity_detected_moving_2to1 = false;
    
    double avg_disp0 = 0.0;
    double avg_disp1 = 0.0;
    
    if (config.init_max_disparity > 0.0) {
        double newest_time_allowed = newest_time - 0.5 * config.init_window_time;
        double var0 = 0.0, var1 = 0.0;
        int num_features0 = 0, num_features1 = 0;
        
        core::compute_disparity(db, newest_time_allowed, -1.0, avg_disp0, var0, num_features0);
        core::compute_disparity(db, newest_time, newest_time_allowed, avg_disp1, var1, num_features1);
        
        int feature_threshold = 15;
        if (num_features0 < feature_threshold || num_features1 < feature_threshold) {
            return false;
        }
        
        disparity_detected_moving_1to0 = (avg_disp0 > config.init_max_disparity);
        disparity_detected_moving_2to1 = (avg_disp1 > config.init_max_disparity);
    }
    
    bool has_jerk = (!disparity_detected_moving_1to0) && disparity_detected_moving_2to1;
    bool is_still = (!disparity_detected_moving_1to0) && (!disparity_detected_moving_2to1);
    
    init_attempt++;
    if (init_attempt % 50 == 1) {
        std::cout << "[init] attempt=" << init_attempt
                  << " disp0=" << avg_disp0 << " disp1=" << avg_disp1
                  << " thresh=" << config.init_max_disparity
                  << " still=" << is_still << " jerk=" << has_jerk
                  << " wfj=" << wait_for_jerk << "\n";
    }
    
    if (((has_jerk && wait_for_jerk) || (is_still && !wait_for_jerk)) && config.init_imu_thresh > 0.0) {
        std::cout << "[init] Entering static init at attempt " << init_attempt << "\n";
        return static_initialize(config, filtered_imu, filtered_count, t_imu, covariance, timestamp, wait_for_jerk);
    }
    
    return false;
}

} // namespace initialize
