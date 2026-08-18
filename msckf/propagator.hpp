#pragma once
#include <Eigen/Core>
#include "../core/sensor_data.hpp"
#include "state.hpp"

namespace msckf {

constexpr int MAX_IMU_ERROR_STATE_SIZE = IMU_ERROR_STATE_CAPACITY;

// Times a propagation window lost readings to the scratch-array bound. Defined
// in propagator.cpp; nonzero means an integration interval was truncated and
// the frame refused.
extern long imu_window_truncations;
using ImuTransitionMatrix = Eigen::Matrix<double, MAX_IMU_ERROR_STATE_SIZE,
                                          MAX_IMU_ERROR_STATE_SIZE>;
using ImuNoiseJacobian = Eigen::Matrix<double, MAX_IMU_ERROR_STATE_SIZE, 12>;

struct PropagatorNoises {
    double sigma_w = 0.005;
    double sigma_a = 0.01;
    double sigma_wb = 0.001;
    double sigma_ab = 0.002;
    double sigma_w_2 = 0.005 * 0.005;
    double sigma_a_2 = 0.01 * 0.01;
    double sigma_wb_2 = 0.001 * 0.001;
    double sigma_ab_2 = 0.002 * 0.002;
};

struct PropagatorData {
    PropagatorNoises noises;
    Eigen::Vector3d gravity;
    double last_prop_time_offset = 0.0;
    bool have_last_prop_time_offset = false;
};

void init_propagator(PropagatorData& prop, const PropagatorNoises& noises, double gravity_mag);
bool propagate_and_clone(PropagatorData& prop, State& state, double timestamp, const core::ImuData* imu_data, int imu_count);
bool propagate_only(PropagatorData& prop, State& state, double timestamp, const core::ImuData* imu_data, int imu_count);
void predict_and_compute(const PropagatorData& prop, const State& state, const core::ImuData& data_minus, const core::ImuData& data_plus, ImuTransitionMatrix& F, ImuTransitionMatrix& Qd);
void compute_Xi_sum(const State& state, double dt, const Eigen::Vector3d& w_hat, const Eigen::Vector3d& a_hat, Eigen::Matrix<double, 3, 18>& Xi_sum);
void predict_mean_discrete(const PropagatorData& prop, const State& state, double dt, const Eigen::Vector3d& w_hat, const Eigen::Vector3d& a_hat, Eigen::Vector4d& new_q, Eigen::Vector3d& new_v, Eigen::Vector3d& new_p);
void predict_mean_rk4(const PropagatorData& prop, const State& state, double dt, const Eigen::Vector3d& w_hat1, const Eigen::Vector3d& a_hat1, const Eigen::Vector3d& w_hat2, const Eigen::Vector3d& a_hat2, Eigen::Vector4d& new_q, Eigen::Vector3d& new_v, Eigen::Vector3d& new_p);
void predict_mean_analytic(const PropagatorData& prop, const State& state, double dt, const Eigen::Vector3d& w_hat, const Eigen::Vector3d& a_hat, Eigen::Vector4d& new_q, Eigen::Vector3d& new_v, Eigen::Vector3d& new_p, const Eigen::Matrix<double, 3, 18>& Xi_sum);
void compute_F_and_G_analytic(const PropagatorData& prop, const State& state, double dt, const Eigen::Vector3d& w_hat, const Eigen::Vector3d& a_hat, const Eigen::Vector3d& w_uncorrected, const Eigen::Vector3d& a_uncorrected, const Eigen::Vector4d& new_q, const Eigen::Vector3d& new_v, const Eigen::Vector3d& new_p, const Eigen::Matrix<double, 3, 18>& Xi_sum, ImuTransitionMatrix& F, ImuNoiseJacobian& G);
void compute_F_and_G_discrete(const PropagatorData& prop, const State& state, double dt, const Eigen::Vector3d& w_hat, const Eigen::Vector3d& a_hat, const Eigen::Vector3d& w_uncorrected, const Eigen::Vector3d& a_uncorrected, const Eigen::Vector4d& new_q, const Eigen::Vector3d& new_v, const Eigen::Vector3d& new_p, ImuTransitionMatrix& F, ImuNoiseJacobian& G);
Eigen::Matrix<double, 3, 6> compute_H_Dw(const State& state, const Eigen::Vector3d& w_uncorrected);
Eigen::Matrix<double, 3, 6> compute_H_Da(const State& state, const Eigen::Vector3d& a_uncorrected);
Eigen::Matrix<double, 3, 9> compute_H_Tg(const State& state, const Eigen::Vector3d& a_inI);
bool initialize_covariance(PropagatorData& prop, State& state, double timestamp, const core::ImuData* imu_data, int imu_count, Eigen::Matrix<double, 15, 15>& covariance_out, double& state_time_out, Eigen::Matrix<double, 16, 1>& state_est_out);

int select_imu_readings(const core::ImuData* imu_data, int imu_count,
                       double time0, double time1, core::ImuData* output_data, int max_output);

} // namespace msckf
