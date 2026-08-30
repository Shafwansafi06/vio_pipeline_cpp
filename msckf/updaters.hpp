#pragma once
#include <Eigen/Core>
#include "state.hpp"
#include "../core/feature.hpp"
#include "../core/cam.hpp"
#include "../core/sensor_data.hpp"
#include "propagator.hpp"

namespace msckf {

// Sub-stage timing inside update_msckf (ms, accumulated), defined in updaters.cpp.
extern long msckf_calls, msckf_rows_pre, msckf_cols, msckf_rows_post, msckf_cov;
extern double msckf_ms_tri, msckf_ms_jac, msckf_ms_chi2, msckf_ms_assemble,
              msckf_ms_alloc, msckf_ms_compress, msckf_ms_ekf;


// Anchor-change accounting, defined in updaters.cpp. Each failure path kills a
// SLAM landmark, so they are counted rather than silent.
extern long anchor_marg_no_old, anchor_marg_depth, anchor_marg_transform, anchor_change_ok;
extern long dinit_seen, dinit_skip_cap, dinit_skip_meas, dinit_skip_tri, dinit_ok;

struct UpdaterOptions {
    double sigma_pix = 1.0;
    double sigma_pix_sq = 1.0;
    double chi2_multipler = 1.0;
    int min_update_rows = 1;
    double max_innovation_condition = 1e12;
    int min_features_for_update = 1;

    // Parallax-limited measurement noise (core::parallax_noise_scale). 0.0 is
    // off and is bit-identical to no model at all; the mid-altitude runner is
    // the only configuration that sets it. Not tuned on EuRoC -- EuRoC never
    // reaches the depth-to-parallax ratios this exists for.
    double parallax_noise_lambda = 0.0;
    double parallax_noise_max = 100.0;   // hard cap on the variance multiplier
};

struct UpdaterMSCKFData {
    UpdaterOptions options;
    core::FeatureInitializerOptions feat_init_options;
};

void init_updater_msckf(UpdaterMSCKFData& updater, const UpdaterOptions& options,
                        const core::FeatureInitializerOptions& feat_init_options);
void update_msckf(UpdaterMSCKFData& updater, State& state, core::Feature** feature_vec, int feature_count, const core::CameraModel* cam_models);

// SchurVINS-style alternative (msckf/updater_schur.cpp): the landmark block is
// removed with a Schur complement on the normal equations instead of a
// left-nullspace projection plus QR compression.
void update_msckf_schur(UpdaterMSCKFData& updater, State& state, core::Feature** feature_vec,
                        int feature_count, const core::CameraModel* cam_models);
extern double schur_ms_tri, schur_ms_jac, schur_ms_chi2, schur_ms_accum,
              schur_ms_reduce, schur_ms_ekf, schur_ms_backsub;
extern long schur_calls, schur_dim;
extern long schur_slam_calls, schur_slam_llt_fail, schur_slam_empty, schur_slam_landmarks;
extern long schur_gated;

core::ClonesCamera build_clones_camera(const State& state);
double chi2_ppf_95(int dof);

struct UpdaterSLAMData {
    UpdaterOptions options_slam;
    UpdaterOptions options_aruco;
    core::FeatureInitializerOptions feat_init_options;
};

void init_updater_slam(UpdaterSLAMData& updater, const UpdaterOptions& options_slam, const UpdaterOptions& options_aruco, const core::FeatureInitializerOptions& feat_init_options);
void delayed_init_slam(UpdaterSLAMData& updater, State& state, core::Feature** feature_vec, int feature_count, const core::CameraModel* cam_models);
void update_slam(UpdaterSLAMData& updater, State& state, core::Feature** feature_vec, int feature_count, const core::CameraModel* cam_models);

// Re-anchors any SLAM landmark whose anchor clone is about to be
// marginalized onto the newest clone, transferring covariance via a
// deterministic (Q=0) EKF propagation. Must run after clone augmentation and
// before marginalize_old_clone() each frame.
void change_anchors(UpdaterSLAMData& updater, State& state);
void perform_anchor_change(State& state, type::Variable& landmark, double new_anchor_timestamp, int new_cam_id);

struct UpdaterZeroVelocityData {
    PropagatorNoises noises;
    Eigen::Vector3d gravity;
    double zupt_max_velocity = 0.0;
    double zupt_noise_multiplier = 1.0;
    double zupt_max_disparity = 0.0;
    double zupt_chi2_multiplier = 1.0;
    double last_prop_time_offset = 0.0;
    bool have_last_prop_time_offset = false;
    double last_zupt_state_timestamp = 0.0;
    int last_zupt_count = 0;
};

void init_updater_zupt(UpdaterZeroVelocityData& updater, const PropagatorNoises& noises, double gravity_mag, double zupt_max_velocity, double zupt_noise_multiplier, double zupt_max_disparity, double zupt_chi2_multiplier = 1.0);
bool try_update_zupt(UpdaterZeroVelocityData& updater, State& state, double timestamp, const core::ImuData* imu_data, int imu_count, const core::FeatureDatabase& db, bool wait_for_jerk = true);

// Global measurement helper methods
void get_feature_jacobian_full(const State& state, const core::Feature& feature,
                               const core::CameraModel& cam_model, int cam_id,
                               Eigen::MatrixXd& H_f, Eigen::MatrixXd& H_x, Eigen::VectorXd& res,
                               type::Variable** Hx_order, int& num_Hx);

// H_f is modified in place (Givens applied to it as well), matching official.
void nullspace_project_inplace(Eigen::MatrixXd& H_f, Eigen::MatrixXd& H_x, Eigen::VectorXd& res);

void measurement_compress_inplace(Eigen::MatrixXd& H_x, Eigen::VectorXd& res);

} // namespace msckf
