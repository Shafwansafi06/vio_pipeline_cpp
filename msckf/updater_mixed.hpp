#pragma once

#include "state.hpp"
#include "../core/cam.hpp"
#include "../core/feature.hpp"

#include <Eigen/Core>

namespace msckf {

// `representation` selects what H_f is expressed in, exactly as official's
// UpdaterHelper::get_feature_jacobian_representation does: GLOBAL_3D leaves the
// feature in global XYZ (no anchor, no anchor-clone coupling), anything else
// uses the anchored inverse-depth chain. Official's EuRoC config runs
// feat_rep_msckf GLOBAL_3D and feat_rep_slam ANCHORED_MSCKF_INVERSE_DEPTH, so
// the two updaters pass different values here. Defaulted to the anchored form
// to keep the parity tests' call sites unchanged.
void get_feature_jacobian_mixed(const State& state, const core::Feature& feature,
                                const core::CameraModel* camera_models,
                                Eigen::MatrixXd& H_f, Eigen::MatrixXd& H_x,
                                Eigen::VectorXd& residual,
                                type::Variable** Hx_order, int& num_Hx,
                                type::LandmarkRepresentation representation =
                                    type::LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH);

} // namespace msckf
