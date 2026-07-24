#pragma once
#include <Eigen/Core>
#include "state.hpp"
#include "../core/feature.hpp"
#include "../core/cam.hpp"

namespace msckf {

// Jacobian of the feature's global position (p_FinG) w.r.t. its anchored
// inverse-depth parameters (lambda = [alpha, beta, rho]) and w.r.t. the
// anchor clone's IMU pose. `landmark.anchor_cam_id` selects the anchor
// camera and `landmark.timestamp` is repurposed to store the anchor clone's
// timestamp (type::Variable has no dedicated field for this since only
// clones use `timestamp` upstream). Only ANCHORED_MSCKF_INVERSE_DEPTH is
// supported (the only representation this pipeline configures); anchor
// camera-extrinsics calibration Jacobians are not computed since
// do_calib_camera_pose is off in every current config.
//
// Returns false if the anchor clone is no longer in the sliding window
// (needs re-anchoring via change_anchors()/perform_anchor_change() first).
bool get_feature_jacobian_representation(const State& state, const type::Variable& landmark,
                                          Eigen::Matrix3d& H_f,
                                          Eigen::Matrix<double, 3, 6>& H_anchor_clone,
                                          type::Variable*& anchor_clone_out);

// Full residual + Jacobian chain for one already-anchored SLAM landmark
// against one tracked feature's (possibly multi-camera) observations this
// frame. `Hx_order`/`num_Hx` do NOT include the landmark itself -- callers
// append it alongside H_f, mirroring the H_x/H_f split used by
// delayed_init_slam()/update_slam() before calling state_helper::initialize()
// or EKFUpdate().
bool get_feature_jacobian_slam(const State& state, const type::Variable& landmark,
                                const core::Feature& feature, const core::CameraModel* cam_models,
                                Eigen::MatrixXd& H_f, Eigen::MatrixXd& H_x, Eigen::VectorXd& res,
                                type::Variable** Hx_order, int& num_Hx);

} // namespace msckf
