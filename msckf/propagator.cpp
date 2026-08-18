#include "propagator.hpp"
#include "state_helper.hpp"
#include "../type/quat_ops.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace msckf {

static core::ImuData interpolate_data(const core::ImuData& imu_1, const core::ImuData& imu_2, double timestamp) {
    double lambda_ = 0.0;
    if (imu_2.timestamp != imu_1.timestamp) {
        lambda_ = (timestamp - imu_1.timestamp) / (imu_2.timestamp - imu_1.timestamp);
    }
    core::ImuData data;
    data.timestamp = timestamp;
    data.am = (1.0 - lambda_) * imu_1.am + lambda_ * imu_2.am;
    data.wm = (1.0 - lambda_) * imu_1.wm + lambda_ * imu_2.wm;
    return data;
}

// Times an IMU reading that belonged in a propagation window was discarded
// because the caller's scratch array was full. Every one of these means the
// window handed to the integrator is shorter than the interval it claims to
// cover, so the callers below refuse a saturated window outright rather than
// integrating part of it and reporting success.
long imu_window_truncations = 0;

// One propagation window is bounded by the interval between two camera frames,
// which is 50 ms at EuRoC's 20 Hz and never more than a few hundred ms on any
// shipped config; at 200 Hz IMU that is 10 samples, and 2000 is 10 s of IMU.
// The bound exists to keep the scratch array on the stack, not because the
// window is expected to approach it. A saturated window is therefore a fault,
// not a working limit -- see the refusals at each call site.
constexpr int PROP_WINDOW_CAPACITY = 2000;

int select_imu_readings(const core::ImuData* imu_data, int imu_count,
                               double time0, double time1,
                               core::ImuData* selected_out, int max_out_capacity) {
    int count = 0;
    if (imu_count == 0) return 0;
    
    for (int i = 0; i < imu_count - 1; ++i) {
        const core::ImuData& curr = imu_data[i];
        const core::ImuData& next_m = imu_data[i+1];
        
        if (next_m.timestamp > time0 && curr.timestamp < time0) {
            if (count < max_out_capacity) {
                selected_out[count++] = interpolate_data(curr, next_m, time0);
            } else { ++imu_window_truncations; }
            continue;
        }
        
        if (curr.timestamp >= time0 && next_m.timestamp <= time1) {
            if (count < max_out_capacity) {
                selected_out[count++] = curr;
            } else { ++imu_window_truncations; }
            continue;
        }
        
        if (next_m.timestamp > time1) {
            if (curr.timestamp > time1 && i == 0) {
                break;
            } else if (curr.timestamp > time1) {
                if (i > 0 && count < max_out_capacity) {
                    selected_out[count++] = interpolate_data(imu_data[i-1], curr, time1);
                } else if (i > 0) { ++imu_window_truncations; }
            } else {
                if (count < max_out_capacity) {
                    selected_out[count++] = curr;
                } else { ++imu_window_truncations; }
            }
            
            if (count == 0 || selected_out[count - 1].timestamp != time1) {
                if (count < max_out_capacity) {
                    selected_out[count++] = interpolate_data(curr, next_m, time1);
                } else { ++imu_window_truncations; }
            }
            break;
        }
    }
    
    if (count == 0) return 0;

    // Official compares the trailing timestamp to time1 with an exact !=, not a
    // tolerance. A 1e-12 window here suppressed the final interpolated sample on
    // windows whose end lands within a picosecond of a reading.
    if (selected_out[count - 1].timestamp != time1) {
        if (count < max_out_capacity) {
            selected_out[count++] = interpolate_data(imu_data[imu_count - 2], imu_data[imu_count - 1], time1);
        } else { ++imu_window_truncations; }
    }

    // Zero-dt removal. Official erases the EARLIER reading of the pair and
    // re-examines the same index; keeping the earlier one instead (as this did)
    // hands the propagator a different wm/am at that step.
    for (int i = 0; i + 1 < count; ++i) {
        if (std::abs(selected_out[i + 1].timestamp - selected_out[i].timestamp) < 1e-12) {
            for (int j = i; j + 1 < count; ++j) selected_out[j] = selected_out[j + 1];
            --count;
            --i;
        }
    }
    return count;
}

void init_propagator(PropagatorData& prop, const PropagatorNoises& noises, double gravity_mag) {
    prop.noises = noises;
    prop.gravity << 0.0, 0.0, gravity_mag;
    prop.last_prop_time_offset = 0.0;
    prop.have_last_prop_time_offset = false;
}

bool propagate_and_clone(PropagatorData& prop, State& state, double timestamp, const core::ImuData* imu_data, int imu_count) {
    // Refuse any frame that does not advance the state, not just one that
    // repeats it exactly. A stamp that goes backwards makes time1 < time0, and
    // select_imu_readings() below then returns a window running the wrong way;
    // the integration that follows is meaningless rather than merely short.
    // The caller counts the refusal (frame_unpropagated_drops).
    if (timestamp <= state.timestamp) return false;
    
    if (!prop.have_last_prop_time_offset) {
        prop.last_prop_time_offset = state.calib_dt_CAMtoIMU.value[0];
        prop.have_last_prop_time_offset = true;
    }
    
    double t_off_new = state.calib_dt_CAMtoIMU.value[0];
    double time0 = state.timestamp + prop.last_prop_time_offset;
    double time1 = timestamp + t_off_new;
    
    core::ImuData prop_data[PROP_WINDOW_CAPACITY];
    int prop_count = select_imu_readings(imu_data, imu_count, time0, time1, prop_data,
                                         PROP_WINDOW_CAPACITY);
    // A window that filled the scratch array was truncated: the readings past
    // the bound were dropped, so integrating what is left would advance the
    // state by less than the interval it is being asked to cover, while
    // reporting success.
    if (prop_count >= PROP_WINDOW_CAPACITY) return false;
    
    int dim = imu_intrinsic_size(state) + 15;
    assert(dim <= MAX_IMU_ERROR_STATE_SIZE);
    ImuTransitionMatrix Phi_summed = ImuTransitionMatrix::Zero();
    ImuTransitionMatrix Qd_summed = ImuTransitionMatrix::Zero();
    Phi_summed.topLeftCorner(dim, dim).setIdentity();
    
    if (prop_count > 1) {
        for (int i = 0; i < prop_count - 1; ++i) {
            ImuTransitionMatrix F;
            ImuTransitionMatrix Qdi;
            ImuTransitionMatrix tmp;
            predict_and_compute(prop, state, prop_data[i], prop_data[i+1], F, Qdi);

            tmp.topLeftCorner(dim, dim).noalias() =
                F.topLeftCorner(dim, dim) * Phi_summed.topLeftCorner(dim, dim);
            Phi_summed.topLeftCorner(dim, dim) = tmp.topLeftCorner(dim, dim);
            tmp.topLeftCorner(dim, dim).noalias() =
                F.topLeftCorner(dim, dim) * Qd_summed.topLeftCorner(dim, dim);
            Qd_summed.topLeftCorner(dim, dim).noalias() =
                tmp.topLeftCorner(dim, dim) * F.topLeftCorner(dim, dim).transpose();
            Qd_summed.topLeftCorner(dim, dim) += Qdi.topLeftCorner(dim, dim);
            Qd_summed.topLeftCorner(dim, dim) = 0.5 *
                (Qd_summed.topLeftCorner(dim, dim) +
                 Qd_summed.topLeftCorner(dim, dim).transpose());
        }
    }
    
    Eigen::Vector3d last_a = Eigen::Vector3d::Zero();
    Eigen::Vector3d last_w = Eigen::Vector3d::Zero();
    
    if (prop_count > 0) {
        const core::ImuData& last_msg = prop_data[prop_count - 1];
        Eigen::Matrix3d Dw = Dm(state.options.imu_model, state.calib_imu_dw.value);
        Eigen::Matrix3d Da = Dm(state.options.imu_model, state.calib_imu_da.value);
        Eigen::Matrix3d Tg = msckf::Tg(state.calib_imu_tg.value);
        
        last_a = state.calib_imu_ACCtoIMU.Rot() * Da * (last_msg.am - state.imu.bias_a());
        Eigen::Vector3d term = last_msg.wm - state.imu.bias_g() - Tg * last_a;
        last_w = state.calib_imu_GYROtoIMU.Rot() * Dw * term;
    }
    
    type::Variable* Phi_order[10];
    int num_Phi = 0;
    Phi_order[num_Phi++] = &state.imu;
    
    if (state.options.do_calib_imu_intrinsics) {
        Phi_order[num_Phi++] = &state.calib_imu_dw;
        Phi_order[num_Phi++] = &state.calib_imu_da;
        if (state.options.do_calib_imu_g_sensitivity) {
            Phi_order[num_Phi++] = &state.calib_imu_tg;
        }
        if (state.options.imu_model == ImuModel::KALIBR) {
            Phi_order[num_Phi++] = &state.calib_imu_GYROtoIMU;
        } else {
            Phi_order[num_Phi++] = &state.calib_imu_ACCtoIMU;
        }
    }
    
    EKFPropagation(
        state, Phi_order, num_Phi, Phi_order, num_Phi,
        Phi_summed.topLeftCorner(dim, dim), Qd_summed.topLeftCorner(dim, dim));
    
    state.timestamp = timestamp;
    prop.last_prop_time_offset = t_off_new;
    
    augment_clone(state, last_w);
    return true;
}

bool propagate_only(PropagatorData& prop, State& state, double timestamp, const core::ImuData* imu_data, int imu_count) {
    if (state.timestamp == timestamp) return false;
    
    if (!prop.have_last_prop_time_offset) {
        prop.last_prop_time_offset = state.calib_dt_CAMtoIMU.value[0];
        prop.have_last_prop_time_offset = true;
    }
    
    double t_off_new = state.calib_dt_CAMtoIMU.value[0];
    double time0 = state.timestamp + prop.last_prop_time_offset;
    double time1 = timestamp + t_off_new;
    
    core::ImuData prop_data[PROP_WINDOW_CAPACITY];
    int prop_count = select_imu_readings(imu_data, imu_count, time0, time1, prop_data,
                                         PROP_WINDOW_CAPACITY);
    // A window that filled the scratch array was truncated: the readings past
    // the bound were dropped, so integrating what is left would advance the
    // state by less than the interval it is being asked to cover, while
    // reporting success.
    if (prop_count >= PROP_WINDOW_CAPACITY) return false;
    
    if (prop_count < 2) return false;
    
    int dim = imu_intrinsic_size(state) + 15;
    assert(dim <= MAX_IMU_ERROR_STATE_SIZE);
    ImuTransitionMatrix Phi_summed = ImuTransitionMatrix::Zero();
    ImuTransitionMatrix Qd_summed = ImuTransitionMatrix::Zero();
    Phi_summed.topLeftCorner(dim, dim).setIdentity();
    
    for (int i = 0; i < prop_count - 1; ++i) {
        ImuTransitionMatrix F;
        ImuTransitionMatrix Qdi;
        ImuTransitionMatrix tmp;
        predict_and_compute(prop, state, prop_data[i], prop_data[i+1], F, Qdi);

        tmp.topLeftCorner(dim, dim).noalias() =
            F.topLeftCorner(dim, dim) * Phi_summed.topLeftCorner(dim, dim);
        Phi_summed.topLeftCorner(dim, dim) = tmp.topLeftCorner(dim, dim);
        tmp.topLeftCorner(dim, dim).noalias() =
            F.topLeftCorner(dim, dim) * Qd_summed.topLeftCorner(dim, dim);
        Qd_summed.topLeftCorner(dim, dim).noalias() =
            tmp.topLeftCorner(dim, dim) * F.topLeftCorner(dim, dim).transpose();
        Qd_summed.topLeftCorner(dim, dim) += Qdi.topLeftCorner(dim, dim);
        Qd_summed.topLeftCorner(dim, dim) = 0.5 *
            (Qd_summed.topLeftCorner(dim, dim) +
             Qd_summed.topLeftCorner(dim, dim).transpose());
    }
    
    type::Variable* Phi_order[10];
    int num_Phi = 0;
    Phi_order[num_Phi++] = &state.imu;
    
    if (state.options.do_calib_imu_intrinsics) {
        Phi_order[num_Phi++] = &state.calib_imu_dw;
        Phi_order[num_Phi++] = &state.calib_imu_da;
        if (state.options.do_calib_imu_g_sensitivity) {
            Phi_order[num_Phi++] = &state.calib_imu_tg;
        }
        if (state.options.imu_model == ImuModel::KALIBR) {
            Phi_order[num_Phi++] = &state.calib_imu_GYROtoIMU;
        } else {
            Phi_order[num_Phi++] = &state.calib_imu_ACCtoIMU;
        }
    }
    
    EKFPropagation(
        state, Phi_order, num_Phi, Phi_order, num_Phi,
        Phi_summed.topLeftCorner(dim, dim), Qd_summed.topLeftCorner(dim, dim));
    
    state.timestamp = timestamp;
    prop.last_prop_time_offset = t_off_new;
    return true;
}

void predict_and_compute(const PropagatorData& prop, const State& state, const core::ImuData& data_minus, const core::ImuData& data_plus,
                                    ImuTransitionMatrix& F, ImuTransitionMatrix& Qd) {
    double dt = data_plus.timestamp - data_minus.timestamp;
    
    Eigen::Matrix3d Dw = Dm(state.options.imu_model, state.calib_imu_dw.value);
    Eigen::Matrix3d Da = Dm(state.options.imu_model, state.calib_imu_da.value);
    Eigen::Matrix3d Tg = msckf::Tg(state.calib_imu_tg.value);
    
    Eigen::Vector3d a_hat1 = data_minus.am - state.imu.bias_a();
    Eigen::Vector3d a_hat2 = data_plus.am - state.imu.bias_a();
    Eigen::Vector3d a_hat_avg = 0.5 * (a_hat1 + a_hat2);
    
    Eigen::Vector3d a_uncorrected = a_hat_avg;
    Eigen::Matrix3d R_ACCtoIMU = state.calib_imu_ACCtoIMU.Rot();
    
    a_hat1 = R_ACCtoIMU * Da * a_hat1;
    a_hat2 = R_ACCtoIMU * Da * a_hat2;
    a_hat_avg = R_ACCtoIMU * Da * a_hat_avg;
    
    Eigen::Vector3d w_hat1 = data_minus.wm - state.imu.bias_g() - Tg * a_hat1;
    Eigen::Vector3d w_hat2 = data_plus.wm - state.imu.bias_g() - Tg * a_hat2;
    Eigen::Vector3d w_hat_avg = 0.5 * (w_hat1 + w_hat2);
    
    Eigen::Vector3d w_uncorrected = w_hat_avg;
    Eigen::Matrix3d R_GYROtoIMU = state.calib_imu_GYROtoIMU.Rot();
    
    w_hat1 = R_GYROtoIMU * Dw * w_hat1;
    w_hat2 = R_GYROtoIMU * Dw * w_hat2;
    w_hat_avg = R_GYROtoIMU * Dw * w_hat_avg;
    
    Eigen::Matrix<double, 3, 18> Xi_sum = Eigen::Matrix<double, 3, 18>::Zero();
    if (state.options.integration_method == IntegrationMethod::RK4 || state.options.integration_method == IntegrationMethod::ANALYTICAL) {
        compute_Xi_sum(state, dt, w_hat_avg, a_hat_avg, Xi_sum);
    }
    
    Eigen::Vector4d new_q;
    Eigen::Vector3d new_v, new_p;
    
    if (state.options.integration_method == IntegrationMethod::ANALYTICAL) {
        predict_mean_analytic(prop, state, dt, w_hat_avg, a_hat_avg, new_q, new_v, new_p, Xi_sum);
    } else if (state.options.integration_method == IntegrationMethod::RK4) {
        predict_mean_rk4(prop, state, dt, w_hat1, a_hat1, w_hat2, a_hat2, new_q, new_v, new_p);
    } else {
        predict_mean_discrete(prop, state, dt, w_hat_avg, a_hat_avg, new_q, new_v, new_p);
    }
    
    int dim = imu_intrinsic_size(state) + 15;
    F.setZero();
    ImuNoiseJacobian G = ImuNoiseJacobian::Zero();
    
    if (state.options.integration_method == IntegrationMethod::ANALYTICAL || state.options.integration_method == IntegrationMethod::RK4) {
        compute_F_and_G_analytic(prop, state, dt, w_hat_avg, a_hat_avg, w_uncorrected, a_uncorrected, new_q, new_v, new_p, Xi_sum, F, G);
    } else {
        compute_F_and_G_discrete(prop, state, dt, w_hat_avg, a_hat_avg, w_uncorrected, a_uncorrected, new_q, new_v, new_p, F, G);
    }
    
    Eigen::Matrix<double, 12, 12> Qc = Eigen::Matrix<double, 12, 12>::Zero();
    Qc.block<3, 3>(0, 0) = (prop.noises.sigma_w * prop.noises.sigma_w / dt) * Eigen::Matrix3d::Identity();
    Qc.block<3, 3>(3, 3) = (prop.noises.sigma_a * prop.noises.sigma_a / dt) * Eigen::Matrix3d::Identity();
    Qc.block<3, 3>(6, 6) = (prop.noises.sigma_wb * prop.noises.sigma_wb / dt) * Eigen::Matrix3d::Identity();
    Qc.block<3, 3>(9, 9) = (prop.noises.sigma_ab * prop.noises.sigma_ab / dt) * Eigen::Matrix3d::Identity();
    
    Qd.setZero();
    ImuNoiseJacobian GQc = ImuNoiseJacobian::Zero();
    GQc.topRows(dim).noalias() = G.topRows(dim) * Qc;
    Qd.topLeftCorner(dim, dim).noalias() =
        GQc.topRows(dim) * G.topRows(dim).transpose();
    Qd.topLeftCorner(dim, dim) = 0.5 *
        (Qd.topLeftCorner(dim, dim) +
         Qd.topLeftCorner(dim, dim).transpose());
    
    Eigen::Matrix<double, 16, 1> imu_x = Eigen::Map<const Eigen::Matrix<double, 16, 1>>(state.imu.value);
    imu_x.head<4>() = new_q;
    imu_x.segment<3>(4) = new_p;
    imu_x.segment<3>(7) = new_v;
    
    type::set_variable_value(*const_cast<type::Variable*>(&state.imu), imu_x);
    type::set_variable_fej(*const_cast<type::Variable*>(&state.imu), imu_x);
}

void compute_Xi_sum(const State& state, double dt, const Eigen::Vector3d& w_hat, const Eigen::Vector3d& a_hat,
                                Eigen::Matrix<double, 3, 18>& Xi_sum) {
    // Transcribed from official Propagator::compute_Xi_sum term-for-term.
    // Three things here are NOT free to tidy up, each verified by bitdiff_prop:
    //   - std::pow(dt, 3) is not bit-equal to dt*dt*dt, likewise w_norm3/d_th3.
    //   - `1.0/6 * d_th3` is not bit-equal to `d_th3 / 6.0` (1/6 is inexact).
    //   - `coef * sA * sK` groups as `(coef * sA) * sK` -- scaling the matrix
    //     first, then multiplying. Writing `coef * (sA * sK)` reassociates and
    //     changes the last bits.
    double w_norm = w_hat.norm();
    double d_th = w_norm * dt;
    Eigen::Vector3d k_hat = Eigen::Vector3d::Zero();
    if (w_norm > 1e-12) {
        k_hat = w_hat / w_norm;
    }

    Eigen::Matrix3d I_3x3 = Eigen::Matrix3d::Identity();
    double d_t2 = std::pow(dt, 2);
    double d_t3 = std::pow(dt, 3);
    double w_norm2 = std::pow(w_norm, 2);
    double w_norm3 = std::pow(w_norm, 3);
    double cos_dth = std::cos(d_th);
    double sin_dth = std::sin(d_th);
    double d_th2 = std::pow(d_th, 2);
    double d_th3 = std::pow(d_th, 3);
    Eigen::Matrix3d sK = type::skew_x(k_hat);
    Eigen::Matrix3d sK2 = sK * sK;
    Eigen::Matrix3d sA = type::skew_x(a_hat);

    Eigen::Matrix3d R_ktok1, Xi_1, Xi_2, Jr_ktok1, Xi_3, Xi_4;
    R_ktok1 = type::exp_so3(-w_hat * dt);
    Jr_ktok1 = type::Jr_so3(-w_hat * dt);

    bool small_w = (w_norm < 1.0 / 180 * M_PI / 2);
    if (!small_w) {

        Xi_1 = I_3x3 * dt + (1.0 - cos_dth) / w_norm * sK + (dt - sin_dth / w_norm) * sK2;

        Xi_2 = 1.0 / 2 * d_t2 * I_3x3 + (d_th - sin_dth) / w_norm2 * sK + (1.0 / 2 * d_t2 - (1.0 - cos_dth) / w_norm2) * sK2;

        Xi_3 = 1.0 / 2 * d_t2 * sA + (sin_dth - d_th) / w_norm2 * sA * sK + (sin_dth - d_th * cos_dth) / w_norm2 * sK * sA +
               (1.0 / 2 * d_t2 - (1.0 - cos_dth) / w_norm2) * sA * sK2 +
               (1.0 / 2 * d_t2 + (1.0 - cos_dth - d_th * sin_dth) / w_norm2) * (sK2 * sA + k_hat.dot(a_hat) * sK) -
               (3 * sin_dth - 2 * d_th - d_th * cos_dth) / w_norm2 * k_hat.dot(a_hat) * sK2;

        Xi_4 = 1.0 / 6 * d_t3 * sA + (2 * (1.0 - cos_dth) - d_th2) / (2 * w_norm3) * sA * sK +
               ((2 * (1.0 - cos_dth) - d_th * sin_dth) / w_norm3) * sK * sA + ((sin_dth - d_th) / w_norm3 + d_t3 / 6) * sA * sK2 +
               ((d_th - 2 * sin_dth + 1.0 / 6 * d_th3 + d_th * cos_dth) / w_norm3) * (sK2 * sA + k_hat.dot(a_hat) * sK) +
               (4 * cos_dth - 4 + d_th2 + d_th * sin_dth) / w_norm3 * k_hat.dot(a_hat) * sK2;

    } else {

        Xi_1 = dt * (I_3x3 + sin_dth * sK + (1.0 - cos_dth) * sK2);

        Xi_2 = 1.0 / 2 * dt * Xi_1;

        Xi_3 = 1.0 / 2 * d_t2 *
               (sA + sin_dth * (-sA * sK + sK * sA + k_hat.dot(a_hat) * sK2) + (1.0 - cos_dth) * (sA * sK2 + sK2 * sA + k_hat.dot(a_hat) * sK));

        Xi_4 = 1.0 / 3 * dt * Xi_3;
    }

    Xi_sum.setZero();
    Xi_sum.block(0, 0, 3, 3) = R_ktok1;
    Xi_sum.block(0, 3, 3, 3) = Xi_1;
    Xi_sum.block(0, 6, 3, 3) = Xi_2;
    Xi_sum.block(0, 9, 3, 3) = Jr_ktok1;
    Xi_sum.block(0, 12, 3, 3) = Xi_3;
    Xi_sum.block(0, 15, 3, 3) = Xi_4;
}

void predict_mean_discrete(const PropagatorData& prop, const State& state, double dt, const Eigen::Vector3d& w_hat, const Eigen::Vector3d& a_hat,
                                      Eigen::Vector4d& new_q, Eigen::Vector3d& new_v, Eigen::Vector3d& new_p) {
    double w_norm = w_hat.norm();
    Eigen::Matrix3d R_Gtoi = state.imu.Rot();
    Eigen::Matrix4d bigO;
    
    if (w_norm > 1e-12) {
        double half_th = 0.5 * w_norm * dt;
        bigO = std::cos(half_th) * Eigen::Matrix4d::Identity() + (1.0 / w_norm) * std::sin(half_th) * type::Omega(w_hat);
    } else {
        bigO = Eigen::Matrix4d::Identity() + 0.5 * dt * type::Omega(w_hat);
    }
    
    Eigen::Vector4d q_val(state.imu.value[0], state.imu.value[1], state.imu.value[2], state.imu.value[3]);
    new_q = (bigO * q_val).normalized();
    
    Eigen::Vector3d vel(state.imu.value[7], state.imu.value[8], state.imu.value[9]);
    Eigen::Vector3d pos(state.imu.value[4], state.imu.value[5], state.imu.value[6]);
    
    new_v = vel + R_Gtoi.transpose() * a_hat * dt - prop.gravity * dt;
    new_p = pos + vel * dt + 0.5 * R_Gtoi.transpose() * a_hat * dt * dt - 0.5 * prop.gravity * dt * dt;
}

void predict_mean_rk4(const PropagatorData& prop, const State& state, double dt,
                                  const Eigen::Vector3d& w_hat1, const Eigen::Vector3d& a_hat1,
                                  const Eigen::Vector3d& w_hat2, const Eigen::Vector3d& a_hat2,
                                  Eigen::Vector4d& new_q, Eigen::Vector3d& new_v, Eigen::Vector3d& new_p) {
    // Transcribed from official Propagator::predict_mean_rk4. Two details that
    // look redundant and are not (both caught by bitdiff_prop):
    //   - The k4 stage uses w_hat/a_hat reached by two accumulated
    //     `+= 0.5 * w_alpha * dt` steps, NOT w_hat2 directly. The two are equal
    //     in exact arithmetic and differ in the last bits.
    //   - Renormalisation goes through quatnorm(), which flips the sign when
    //     q(3) < 0. Eigen's .normalized() does not, so it can return the
    //     antipodal quaternion.
    Eigen::Vector3d w_hat = w_hat1;
    Eigen::Vector3d a_hat = a_hat1;
    Eigen::Vector3d w_alpha = (w_hat2 - w_hat1) / dt;
    Eigen::Vector3d a_jerk = (a_hat2 - a_hat1) / dt;

    Eigen::Vector4d q_0(state.imu.value[0], state.imu.value[1], state.imu.value[2], state.imu.value[3]);
    Eigen::Vector3d p_0(state.imu.value[4], state.imu.value[5], state.imu.value[6]);
    Eigen::Vector3d v_0(state.imu.value[7], state.imu.value[8], state.imu.value[9]);

    // k1 ---------------------------------------------------------------------
    Eigen::Vector4d dq_0(0.0, 0.0, 0.0, 1.0);
    Eigen::Vector4d q0_dot = 0.5 * type::Omega(w_hat) * dq_0;
    Eigen::Vector3d p0_dot = v_0;
    Eigen::Matrix3d R_Gto0 = type::quat_2_Rot(type::quat_multiply(dq_0, q_0));
    Eigen::Vector3d v0_dot = R_Gto0.transpose() * a_hat - prop.gravity;
    Eigen::Vector4d k1_q = q0_dot * dt;
    Eigen::Vector3d k1_p = p0_dot * dt;
    Eigen::Vector3d k1_v = v0_dot * dt;

    w_hat += 0.5 * w_alpha * dt;
    a_hat += 0.5 * a_jerk * dt;

    // k2 ---------------------------------------------------------------------
    Eigen::Vector4d dq_1 = type::quatnorm(dq_0 + 0.5 * k1_q);
    Eigen::Vector3d v_1 = v_0 + 0.5 * k1_v;
    Eigen::Vector4d q1_dot = 0.5 * type::Omega(w_hat) * dq_1;
    Eigen::Vector3d p1_dot = v_1;
    Eigen::Matrix3d R_Gto1 = type::quat_2_Rot(type::quat_multiply(dq_1, q_0));
    Eigen::Vector3d v1_dot = R_Gto1.transpose() * a_hat - prop.gravity;
    Eigen::Vector4d k2_q = q1_dot * dt;
    Eigen::Vector3d k2_p = p1_dot * dt;
    Eigen::Vector3d k2_v = v1_dot * dt;

    // k3 ---------------------------------------------------------------------
    Eigen::Vector4d dq_2 = type::quatnorm(dq_0 + 0.5 * k2_q);
    Eigen::Vector3d v_2 = v_0 + 0.5 * k2_v;
    Eigen::Vector4d q2_dot = 0.5 * type::Omega(w_hat) * dq_2;
    Eigen::Vector3d p2_dot = v_2;
    Eigen::Matrix3d R_Gto2 = type::quat_2_Rot(type::quat_multiply(dq_2, q_0));
    Eigen::Vector3d v2_dot = R_Gto2.transpose() * a_hat - prop.gravity;
    Eigen::Vector4d k3_q = q2_dot * dt;
    Eigen::Vector3d k3_p = p2_dot * dt;
    Eigen::Vector3d k3_v = v2_dot * dt;

    w_hat += 0.5 * w_alpha * dt;
    a_hat += 0.5 * a_jerk * dt;

    // k4 ---------------------------------------------------------------------
    Eigen::Vector4d dq_3 = type::quatnorm(dq_0 + k3_q);
    Eigen::Vector3d v_3 = v_0 + k3_v;
    Eigen::Vector4d q3_dot = 0.5 * type::Omega(w_hat) * dq_3;
    Eigen::Vector3d p3_dot = v_3;
    Eigen::Matrix3d R_Gto3 = type::quat_2_Rot(type::quat_multiply(dq_3, q_0));
    Eigen::Vector3d v3_dot = R_Gto3.transpose() * a_hat - prop.gravity;
    Eigen::Vector4d k4_q = q3_dot * dt;
    Eigen::Vector3d k4_p = p3_dot * dt;
    Eigen::Vector3d k4_v = v3_dot * dt;

    Eigen::Vector4d dq = type::quatnorm(dq_0 + (1.0 / 6.0) * k1_q + (1.0 / 3.0) * k2_q + (1.0 / 3.0) * k3_q + (1.0 / 6.0) * k4_q);
    new_q = type::quat_multiply(dq, q_0);
    new_p = p_0 + (1.0 / 6.0) * k1_p + (1.0 / 3.0) * k2_p + (1.0 / 3.0) * k3_p + (1.0 / 6.0) * k4_p;
    new_v = v_0 + (1.0 / 6.0) * k1_v + (1.0 / 3.0) * k2_v + (1.0 / 3.0) * k3_v + (1.0 / 6.0) * k4_v;
}

void predict_mean_analytic(const PropagatorData& prop, const State& state, double dt, const Eigen::Vector3d& w_hat, const Eigen::Vector3d& a_hat,
                                      Eigen::Vector4d& new_q, Eigen::Vector3d& new_v, Eigen::Vector3d& new_p,
                                      const Eigen::Matrix<double, 3, 18>& Xi_sum) {
    Eigen::Matrix3d R_Gtok = state.imu.Rot();
    Eigen::Matrix3d R_ktok1_mat = Xi_sum.block<3, 3>(0, 0);
    Eigen::Vector4d q_ktok1 = type::rot_2_quat(R_ktok1_mat);
    
    Eigen::Matrix3d Xi_1 = Xi_sum.block<3, 3>(0, 3);
    Eigen::Matrix3d Xi_2 = Xi_sum.block<3, 3>(0, 6);
    
    Eigen::Vector4d q_curr(state.imu.value[0], state.imu.value[1], state.imu.value[2], state.imu.value[3]);
    new_q = type::quat_multiply(q_ktok1, q_curr);
    
    Eigen::Vector3d vel(state.imu.value[7], state.imu.value[8], state.imu.value[9]);
    Eigen::Vector3d pos(state.imu.value[4], state.imu.value[5], state.imu.value[6]);
    
    new_v = vel + R_Gtok.transpose() * Xi_1 * a_hat - prop.gravity * dt;
    new_p = pos + vel * dt + R_Gtok.transpose() * Xi_2 * a_hat - 0.5 * prop.gravity * dt * dt;
}

void compute_F_and_G_analytic(const PropagatorData& prop, const State& state, double dt, const Eigen::Vector3d& w_hat, const Eigen::Vector3d& a_hat,
                                          const Eigen::Vector3d& w_uncorrected, const Eigen::Vector3d& a_uncorrected,
                                          const Eigen::Vector4d& new_q, const Eigen::Vector3d& new_v, const Eigen::Vector3d& new_p,
                                          const Eigen::Matrix<double, 3, 18>& Xi_sum, ImuTransitionMatrix& F, ImuNoiseJacobian& G) {
    int th_id = 0;
    int p_id = 3;
    int v_id = 6;
    int bg_id = 9;
    int ba_id = 12;
    
    int local_size = 15;
    int Dw_id = -1, Da_id = -1, Tg_id = -1, th_atoI_id = -1, th_wtoI_id = -1;
    
    if (state.options.do_calib_imu_intrinsics) {
        Dw_id = local_size; local_size += 6;
        Da_id = local_size; local_size += 6;
        if (state.options.do_calib_imu_g_sensitivity) {
            Tg_id = local_size; local_size += 9;
        }
        if (state.options.imu_model == ImuModel::KALIBR) {
            th_wtoI_id = local_size; local_size += 3;
        } else {
            th_atoI_id = local_size; local_size += 3;
        }
    }
    
    Eigen::Matrix3d R_k = state.options.do_fej ? state.imu.Rot_fej() : state.imu.Rot();
    Eigen::Vector3d v_k = state.options.do_fej ? state.imu.vel_fej() : state.imu.vel();
    Eigen::Vector3d p_k = state.options.do_fej ? state.imu.pos_fej() : state.imu.pos();
    
    Eigen::Matrix3d dR_ktok1 = type::quat_2_Rot(new_q) * R_k.transpose();
    
    Eigen::Matrix3d Dw = Dm(state.options.imu_model, state.calib_imu_dw.value);
    Eigen::Matrix3d Da = Dm(state.options.imu_model, state.calib_imu_da.value);
    Eigen::Matrix3d Tg = msckf::Tg(state.calib_imu_tg.value);
    Eigen::Matrix3d R_atoI = state.calib_imu_ACCtoIMU.Rot();
    Eigen::Matrix3d R_wtoI = state.calib_imu_GYROtoIMU.Rot();
    
    Eigen::Vector3d a_k = R_atoI * Da * a_uncorrected;
    Eigen::Vector3d w_k = R_wtoI * Dw * w_uncorrected;
    
    Eigen::Matrix3d Xi_1 = Xi_sum.block<3, 3>(0, 3);
    Eigen::Matrix3d Xi_2 = Xi_sum.block<3, 3>(0, 6);
    Eigen::Matrix3d Jr_ktok1 = Xi_sum.block<3, 3>(0, 9);
    Eigen::Matrix3d Xi_3 = Xi_sum.block<3, 3>(0, 12);
    Eigen::Matrix3d Xi_4 = Xi_sum.block<3, 3>(0, 15);
    
    F.block<3, 3>(th_id, th_id) = dR_ktok1;
    Eigen::Matrix3d term_p_skew = type::skew_x(new_p - p_k - v_k * dt + 0.5 * prop.gravity * dt * dt);
    F.block<3, 3>(p_id, th_id) = -term_p_skew * R_k.transpose();
    Eigen::Matrix3d term_v_skew = type::skew_x(new_v - v_k + prop.gravity * dt);
    F.block<3, 3>(v_id, th_id) = -term_v_skew * R_k.transpose();
    
    F.block<3, 3>(p_id, p_id) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(p_id, v_id) = Eigen::Matrix3d::Identity() * dt;
    F.block<3, 3>(v_id, v_id) = Eigen::Matrix3d::Identity();
    
    F.block<3, 3>(th_id, bg_id) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw;
    F.block<3, 3>(p_id, bg_id) = R_k.transpose() * Xi_4 * R_wtoI * Dw;
    F.block<3, 3>(v_id, bg_id) = R_k.transpose() * Xi_3 * R_wtoI * Dw;
    F.block<3, 3>(bg_id, bg_id) = Eigen::Matrix3d::Identity();
    
    F.block<3, 3>(th_id, ba_id) = dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * R_atoI * Da;
    F.block<3, 3>(p_id, ba_id) = -R_k.transpose() * (Xi_2 + Xi_4 * R_wtoI * Dw * Tg) * R_atoI * Da;
    F.block<3, 3>(v_id, ba_id) = -R_k.transpose() * (Xi_1 + Xi_3 * R_wtoI * Dw * Tg) * R_atoI * Da;
    F.block<3, 3>(ba_id, ba_id) = Eigen::Matrix3d::Identity();
    
    if (Dw_id != -1) {
        Eigen::Matrix<double, 3, 6> H_Dw = compute_H_Dw(state, w_uncorrected);
        F.block<3, 6>(th_id, Dw_id) = dR_ktok1 * Jr_ktok1 * dt * R_wtoI * H_Dw;
        F.block<3, 6>(p_id, Dw_id) = -R_k.transpose() * Xi_4 * R_wtoI * H_Dw;
        F.block<3, 6>(v_id, Dw_id) = -R_k.transpose() * Xi_3 * R_wtoI * H_Dw;
        F.block<6, 6>(Dw_id, Dw_id) = Eigen::Matrix<double, 6, 6>::Identity();
    }
    
    if (Da_id != -1) {
        Eigen::Matrix<double, 3, 6> H_Da = compute_H_Da(state, a_uncorrected);
        F.block<3, 6>(th_id, Da_id) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * R_atoI * H_Da;
        F.block<3, 6>(p_id, Da_id) = R_k.transpose() * (Xi_2 + Xi_4 * R_wtoI * Dw * Tg) * R_atoI * H_Da;
        F.block<3, 6>(v_id, Da_id) = R_k.transpose() * (Xi_1 + Xi_3 * R_wtoI * Dw * Tg) * R_atoI * H_Da;
        F.block<6, 6>(Da_id, Da_id) = Eigen::Matrix<double, 6, 6>::Identity();
    }
    
    if (Tg_id != -1) {
        Eigen::Matrix<double, 3, 9> H_Tg = compute_H_Tg(state, a_k);
        F.block<3, 9>(th_id, Tg_id) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * H_Tg;
        F.block<3, 9>(p_id, Tg_id) = R_k.transpose() * Xi_4 * R_wtoI * Dw * H_Tg;
        F.block<3, 9>(v_id, Tg_id) = R_k.transpose() * Xi_3 * R_wtoI * Dw * H_Tg;
        F.block<9, 9>(Tg_id, Tg_id) = Eigen::Matrix<double, 9, 9>::Identity();
    }
    
    if (th_atoI_id != -1) {
        Eigen::Matrix3d term_skew = type::skew_x(a_k);
        F.block<3, 3>(th_id, th_atoI_id) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * term_skew;
        F.block<3, 3>(p_id, th_atoI_id) = R_k.transpose() * (Xi_2 + Xi_4 * R_wtoI * Dw * Tg) * term_skew;
        F.block<3, 3>(v_id, th_atoI_id) = R_k.transpose() * (Xi_1 + Xi_3 * R_wtoI * Dw * Tg) * term_skew;
        F.block<3, 3>(th_atoI_id, th_atoI_id) = Eigen::Matrix3d::Identity();
    }
    
    if (th_wtoI_id != -1) {
        Eigen::Matrix3d term_skew = type::skew_x(w_k);
        F.block<3, 3>(th_id, th_wtoI_id) = dR_ktok1 * Jr_ktok1 * dt * term_skew;
        F.block<3, 3>(p_id, th_wtoI_id) = -R_k.transpose() * Xi_4 * term_skew;
        F.block<3, 3>(v_id, th_wtoI_id) = -R_k.transpose() * Xi_3 * term_skew;
        F.block<3, 3>(th_wtoI_id, th_wtoI_id) = Eigen::Matrix3d::Identity();
    }
    
    G.block<3, 3>(th_id, 0) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw;
    G.block<3, 3>(p_id, 0) = R_k.transpose() * Xi_4 * R_wtoI * Dw;
    G.block<3, 3>(v_id, 0) = R_k.transpose() * Xi_3 * R_wtoI * Dw;
    
    G.block<3, 3>(th_id, 3) = dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * R_atoI * Da;
    G.block<3, 3>(p_id, 3) = -R_k.transpose() * (Xi_2 + Xi_4 * R_wtoI * Dw * Tg) * R_atoI * Da;
    G.block<3, 3>(v_id, 3) = -R_k.transpose() * (Xi_1 + Xi_3 * R_wtoI * Dw * Tg) * R_atoI * Da;
    
    G.block<3, 3>(bg_id, 6) = dt * Eigen::Matrix3d::Identity();
    G.block<3, 3>(ba_id, 9) = dt * Eigen::Matrix3d::Identity();
}

void compute_F_and_G_discrete(const PropagatorData& prop, const State& state, double dt, const Eigen::Vector3d& w_hat, const Eigen::Vector3d& a_hat,
                                          const Eigen::Vector3d& w_uncorrected, const Eigen::Vector3d& a_uncorrected,
                                          const Eigen::Vector4d& new_q, const Eigen::Vector3d& new_v, const Eigen::Vector3d& new_p,
                                          ImuTransitionMatrix& F, ImuNoiseJacobian& G) {
    int th_id = 0;
    int p_id = 3;
    int v_id = 6;
    int bg_id = 9;
    int ba_id = 12;
    
    int local_size = 15;
    int Dw_id = -1, Da_id = -1, Tg_id = -1, th_atoI_id = -1, th_wtoI_id = -1;
    
    if (state.options.do_calib_imu_intrinsics) {
        Dw_id = local_size; local_size += 6;
        Da_id = local_size; local_size += 6;
        if (state.options.do_calib_imu_g_sensitivity) {
            Tg_id = local_size; local_size += 9;
        }
        if (state.options.imu_model == ImuModel::KALIBR) {
            th_wtoI_id = local_size; local_size += 3;
        } else {
            th_atoI_id = local_size; local_size += 3;
        }
    }
    
    Eigen::Matrix3d R_k = state.options.do_fej ? state.imu.Rot_fej() : state.imu.Rot();
    Eigen::Vector3d v_k = state.options.do_fej ? state.imu.vel_fej() : state.imu.vel();
    Eigen::Vector3d p_k = state.options.do_fej ? state.imu.pos_fej() : state.imu.pos();
    
    Eigen::Matrix3d dR_ktok1 = type::quat_2_Rot(new_q) * R_k.transpose();
    
    Eigen::Matrix3d Dw = Dm(state.options.imu_model, state.calib_imu_dw.value);
    Eigen::Matrix3d Da = Dm(state.options.imu_model, state.calib_imu_da.value);
    Eigen::Matrix3d Tg = msckf::Tg(state.calib_imu_tg.value);
    Eigen::Matrix3d R_atoI = state.calib_imu_ACCtoIMU.Rot();
    Eigen::Matrix3d R_wtoI = state.calib_imu_GYROtoIMU.Rot();
    
    Eigen::Vector3d a_k = R_atoI * Da * a_uncorrected;
    Eigen::Vector3d w_k = R_wtoI * Dw * w_uncorrected;
    
    Eigen::Matrix3d Jr_ktok1 = type::Jr_so3(type::log_so3(dR_ktok1));
    
    F.block<3, 3>(th_id, th_id) = dR_ktok1;
    F.block<3, 3>(th_id, bg_id) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw;
    F.block<3, 3>(th_id, ba_id) = dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * R_atoI * Da;
    
    Eigen::Vector3d term_p = new_p - p_k - v_k * dt + 0.5 * prop.gravity * dt * dt;
    F.block<3, 3>(p_id, th_id) = -type::skew_x(term_p) * R_k.transpose();
    F.block<3, 3>(p_id, p_id) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(p_id, v_id) = Eigen::Matrix3d::Identity() * dt;
    F.block<3, 3>(p_id, ba_id) = -0.5 * R_k.transpose() * dt * dt * R_atoI * Da;
    
    Eigen::Vector3d term_v = new_v - v_k + prop.gravity * dt;
    F.block<3, 3>(v_id, th_id) = -type::skew_x(term_v) * R_k.transpose();
    F.block<3, 3>(v_id, v_id) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(v_id, ba_id) = -R_k.transpose() * dt * R_atoI * Da;
    
    F.block<3, 3>(bg_id, bg_id) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(ba_id, ba_id) = Eigen::Matrix3d::Identity();
    
    if (Dw_id != -1) {
        Eigen::Matrix<double, 3, 6> H_Dw = compute_H_Dw(state, w_uncorrected);
        F.block<3, 6>(th_id, Dw_id) = dR_ktok1 * Jr_ktok1 * dt * R_wtoI * H_Dw;
        F.block<6, 6>(Dw_id, Dw_id) = Eigen::Matrix<double, 6, 6>::Identity();
    }
    
    if (Da_id != -1) {
        Eigen::Matrix<double, 3, 6> H_Da = compute_H_Da(state, a_uncorrected);
        F.block<3, 6>(th_id, Da_id) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * R_atoI * H_Da;
        F.block<3, 6>(p_id, Da_id) = 0.5 * R_k.transpose() * dt * dt * R_atoI * H_Da;
        F.block<3, 6>(v_id, Da_id) = R_k.transpose() * dt * R_atoI * H_Da;
        F.block<6, 6>(Da_id, Da_id) = Eigen::Matrix<double, 6, 6>::Identity();
    }
    
    if (Tg_id != -1) {
        Eigen::Matrix<double, 3, 9> H_Tg = compute_H_Tg(state, a_k);
        F.block<3, 9>(th_id, Tg_id) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * H_Tg;
        F.block<9, 9>(Tg_id, Tg_id) = Eigen::Matrix<double, 9, 9>::Identity();
    }
    
    if (th_atoI_id != -1) {
        F.block<3, 3>(th_id, th_atoI_id) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * type::skew_x(a_k);
        F.block<3, 3>(p_id, th_atoI_id) = 0.5 * R_k.transpose() * dt * dt * type::skew_x(a_k);
        F.block<3, 3>(v_id, th_atoI_id) = R_k.transpose() * dt * type::skew_x(a_k);
        F.block<3, 3>(th_atoI_id, th_atoI_id) = Eigen::Matrix3d::Identity();
    }
    
    if (th_wtoI_id != -1) {
        F.block<3, 3>(th_id, th_wtoI_id) = dR_ktok1 * Jr_ktok1 * dt * type::skew_x(w_k);
        F.block<3, 3>(th_wtoI_id, th_wtoI_id) = Eigen::Matrix3d::Identity();
    }
    
    G.block<3, 3>(th_id, 0) = -dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw;
    G.block<3, 3>(th_id, 3) = dR_ktok1 * Jr_ktok1 * dt * R_wtoI * Dw * Tg * R_atoI * Da;
    G.block<3, 3>(v_id, 3) = -R_k.transpose() * dt * R_atoI * Da;
    G.block<3, 3>(p_id, 3) = -0.5 * R_k.transpose() * dt * dt * R_atoI * Da;
    G.block<3, 3>(bg_id, 6) = dt * Eigen::Matrix3d::Identity();
    G.block<3, 3>(ba_id, 9) = dt * Eigen::Matrix3d::Identity();
}

Eigen::Matrix<double, 3, 6> compute_H_Dw(const State& state, const Eigen::Vector3d& w_uncorrected) {
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 3, 6> H = Eigen::Matrix<double, 3, 6>::Zero();
    if (state.options.imu_model == ImuModel::KALIBR) {
        H.block<3, 3>(0, 0) = w_uncorrected[0] * I;
        H.col(3) = w_uncorrected[1] * I.col(1);
        H.col(4) = w_uncorrected[1] * I.col(2);
        H.col(5) = w_uncorrected[2] * I.col(2);
    } else {
        H.col(0) = w_uncorrected[0] * I.col(0);
        H.col(1) = w_uncorrected[1] * I.col(0);
        H.col(2) = w_uncorrected[1] * I.col(1);
        H.block<3, 3>(0, 3) = w_uncorrected[2] * I;
    }
    return H;
}

Eigen::Matrix<double, 3, 6> compute_H_Da(const State& state, const Eigen::Vector3d& a_uncorrected) {
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 3, 6> H = Eigen::Matrix<double, 3, 6>::Zero();
    if (state.options.imu_model == ImuModel::KALIBR) {
        H.block<3, 3>(0, 0) = a_uncorrected[0] * I;
        H.col(3) = a_uncorrected[1] * I.col(1);
        H.col(4) = a_uncorrected[1] * I.col(2);
        H.col(5) = a_uncorrected[2] * I.col(2);
    } else {
        H.col(0) = a_uncorrected[0] * I.col(0);
        H.col(1) = a_uncorrected[1] * I.col(0);
        H.col(2) = a_uncorrected[1] * I.col(1);
        H.block<3, 3>(0, 3) = a_uncorrected[2] * I;
    }
    return H;
}

Eigen::Matrix<double, 3, 9> compute_H_Tg(const State& state, const Eigen::Vector3d& a_inI) {
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 3, 9> H = Eigen::Matrix<double, 3, 9>::Zero();
    H.block<3, 3>(0, 0) = a_inI[0] * I;
    H.block<3, 3>(0, 3) = a_inI[1] * I;
    H.block<3, 3>(0, 6) = a_inI[2] * I;
    return H;
}

bool initialize_covariance(PropagatorData& prop, State& state, double timestamp, const core::ImuData* imu_data, int imu_count,
                                      Eigen::Matrix<double, 15, 15>& covariance_out, double& state_time_out,
                                      Eigen::Matrix<double, 16, 1>& state_est_out) {
    double time0 = state.timestamp + prop.last_prop_time_offset;
    double time1 = timestamp + state.calib_dt_CAMtoIMU.value[0];
    
    core::ImuData prop_data[PROP_WINDOW_CAPACITY];
    int prop_count = select_imu_readings(imu_data, imu_count, time0, time1, prop_data,
                                         PROP_WINDOW_CAPACITY);
    // A window that filled the scratch array was truncated: the readings past
    // the bound were dropped, so integrating what is left would advance the
    // state by less than the interval it is being asked to cover, while
    // reporting success.
    if (prop_count >= PROP_WINDOW_CAPACITY) return false;
    
    if (prop_count < 2) return false;
    
    Eigen::Vector3d bias_g = state.imu.bias_g();
    Eigen::Vector3d bias_a = state.imu.bias_a();
    
    Eigen::Matrix3d Dw = Dm(state.options.imu_model, state.calib_imu_dw.value);
    Eigen::Matrix3d Da = Dm(state.options.imu_model, state.calib_imu_da.value);
    Eigen::Matrix3d Tg = msckf::Tg(state.calib_imu_tg.value);
    Eigen::Matrix3d R_ACCtoIMU = state.calib_imu_ACCtoIMU.Rot();
    Eigen::Matrix3d R_GYROtoIMU = state.calib_imu_GYROtoIMU.Rot();
    
    Eigen::Matrix<double, 16, 1> cache_state_est = Eigen::Map<const Eigen::Matrix<double, 16, 1>>(state.imu.value);
    Eigen::Matrix<double, 15, 15> cache_state_covariance = Eigen::Matrix<double, 15, 15>::Zero();
    // Default initial covariance for IMU elements
    cache_state_covariance = (0.02 * 0.02) * Eigen::Matrix<double, 15, 15>::Identity();
    
    for (int i = 0; i < prop_count - 1; ++i) {
        const core::ImuData& data_minus = prop_data[i];
        const core::ImuData& data_plus = prop_data[i+1];
        double dt = data_plus.timestamp - data_minus.timestamp;
        
        if (dt <= 0.0) continue;
        
        Eigen::Vector3d a_hat1 = R_ACCtoIMU * Da * (data_minus.am - bias_a);
        Eigen::Vector3d a_hat2 = R_ACCtoIMU * Da * (data_plus.am - bias_a);
        Eigen::Vector3d a_hat = 0.5 * (a_hat1 + a_hat2);
        
        Eigen::Vector3d w_hat1 = R_GYROtoIMU * Dw * (data_minus.wm - bias_g - Tg * a_hat1);
        Eigen::Vector3d w_hat2 = R_GYROtoIMU * Dw * (data_plus.wm - bias_g - Tg * a_hat2);
        Eigen::Vector3d w_hat = 0.5 * (w_hat1 + w_hat2);
        
        Eigen::Vector4d q_curr = cache_state_est.head<4>();
        Eigen::Matrix3d R_Gtoi = type::quat_2_Rot(q_curr);
        Eigen::Vector3d p_iinG = cache_state_est.segment<3>(4);
        Eigen::Vector3d v_iinG = cache_state_est.segment<3>(7);
        
        Eigen::Matrix<double, 15, 15> F_block = Eigen::Matrix<double, 15, 15>::Zero();
        Eigen::Matrix3d exp_w = type::exp_so3(-w_hat * dt);
        F_block.block<3, 3>(0, 0) = exp_w;
        F_block.block<3, 3>(0, 9) = -exp_w * type::Jr_so3(-w_hat * dt) * dt;
        F_block.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
        
        F_block.block<3, 3>(6, 0) = -R_Gtoi.transpose() * type::skew_x(a_hat * dt);
        F_block.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
        F_block.block<3, 3>(6, 12) = -R_Gtoi.transpose() * dt;
        F_block.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();
        
        F_block.block<3, 3>(3, 0) = -0.5 * R_Gtoi.transpose() * type::skew_x(a_hat * dt * dt);
        F_block.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
        F_block.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;
        F_block.block<3, 3>(3, 12) = -0.5 * R_Gtoi.transpose() * dt * dt;
        
        Eigen::Matrix<double, 15, 12> G_block = Eigen::Matrix<double, 15, 12>::Zero();
        G_block.block<3, 3>(0, 0) = -exp_w * type::Jr_so3(-w_hat * dt) * dt;
        G_block.block<3, 3>(6, 3) = -R_Gtoi.transpose() * dt;
        G_block.block<3, 3>(3, 3) = -0.5 * R_Gtoi.transpose() * dt * dt;
        G_block.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
        G_block.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();
        
        Eigen::Matrix<double, 12, 12> Qc = Eigen::Matrix<double, 12, 12>::Zero();
        Qc.block<3, 3>(0, 0) = (prop.noises.sigma_w_2 / dt) * Eigen::Matrix3d::Identity();
        Qc.block<3, 3>(3, 3) = (prop.noises.sigma_a_2 / dt) * Eigen::Matrix3d::Identity();
        Qc.block<3, 3>(6, 6) = (prop.noises.sigma_wb_2 / dt) * Eigen::Matrix3d::Identity();
        Qc.block<3, 3>(9, 9) = (prop.noises.sigma_ab_2 / dt) * Eigen::Matrix3d::Identity();
        
        Eigen::Matrix<double, 15, 15> Qdi = G_block * Qc * G_block.transpose();
        Qdi = 0.5 * (Qdi + Qdi.transpose());
        
        cache_state_covariance = F_block * cache_state_covariance * F_block.transpose() + Qdi;
        
        Eigen::Vector4d dq = type::rot_2_quat(exp_w * R_Gtoi);
        cache_state_est.head<4>() = dq;
        cache_state_est.segment<3>(4) = p_iinG + v_iinG * dt + 0.5 * R_Gtoi.transpose() * a_hat * dt * dt - 0.5 * prop.gravity * dt * dt;
        cache_state_est.segment<3>(7) = v_iinG + R_Gtoi.transpose() * a_hat * dt - prop.gravity * dt;
    }
    
    state_time_out = time1;
    Eigen::Vector4d q_Gtoi = cache_state_est.head<4>();
    Eigen::Vector3d v_iinG = cache_state_est.segment<3>(7);
    Eigen::Vector3d p_iinG = cache_state_est.segment<3>(4);
    
    state_est_out.head<4>() = q_Gtoi;
    state_est_out.segment<3>(4) = p_iinG;
    state_est_out.segment<3>(7) = type::quat_2_Rot(q_Gtoi) * v_iinG;
    
    const core::ImuData& last_msg = prop_data[prop_count - 1];
    Eigen::Vector3d last_a = state.calib_imu_ACCtoIMU.Rot() * Da * (last_msg.am - bias_a);
    Eigen::Vector3d term = last_msg.wm - bias_g - Tg * last_a;
    Eigen::Vector3d last_w = state.calib_imu_GYROtoIMU.Rot() * Dw * term;
    state_est_out.segment<3>(10) = last_w;
    
    covariance_out.setZero();
    Eigen::Matrix<double, 15, 15> Phi = Eigen::Matrix<double, 15, 15>::Identity();
    Phi.block<3, 3>(6, 6) = type::quat_2_Rot(q_Gtoi);
    
    Eigen::Matrix<double, 15, 15> cov_tmp = Phi * cache_state_covariance * Phi.transpose();
    covariance_out.block<9, 9>(0, 0) = cov_tmp.block<9, 9>(0, 0);
    
    double dt_last = prop_data[prop_count-1].timestamp - prop_data[prop_count-2].timestamp;
    covariance_out.block<3, 3>(9, 9) = (prop.noises.sigma_w_2 / dt_last) * Eigen::Matrix3d::Identity();
    return true;
}

} // namespace msckf
