#pragma once

#include "state.hpp"
#include "../core/cam.hpp"
#include "../core/feature.hpp"

#include <Eigen/Core>

namespace msckf {

void get_feature_jacobian_mixed(const State& state, const core::Feature& feature,
                                const core::CameraModel* camera_models,
                                Eigen::MatrixXd& H_f, Eigen::MatrixXd& H_x,
                                Eigen::VectorXd& residual,
                                type::Variable** Hx_order, int& num_Hx);

} // namespace msckf
