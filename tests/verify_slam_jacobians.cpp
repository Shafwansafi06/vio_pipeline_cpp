// Finite-difference validation of the anchored-inverse-depth representation
// Jacobian (msckf::get_feature_jacobian_representation), which is the only
// genuinely new math introduced by the SLAM updater. Perturbs the landmark's
// lambda=[alpha,beta,rho] parameters and the anchor clone's pose in their
// true tangent spaces (matching type::update_variable's conventions) and
// compares the resulting change in p_FinG against the analytic H_f / H_anchor.
#include <algorithm>
#include <iostream>

#include "../core/cam.hpp"
#include "../msckf/state.hpp"
#include "../msckf/updater_slam_helper.hpp"
#include "../type/quat_ops.hpp"

namespace {

Eigen::Vector3d compute_p_FinG(const type::Variable& landmark, const type::Variable& clone,
                                const type::Variable& calib) {
    const Eigen::Vector3d p_FinA = type::get_landmark_xyz(landmark, false);
    return clone.Rot().transpose() * calib.Rot().transpose() * (p_FinA - calib.pos()) + clone.pos();
}

// Predicted (distorted) pixel for one observing clone -- mirrors the forward
// model inside get_feature_jacobian_slam exactly, so finite-differencing
// this function exercises the same math get_feature_jacobian_slam's H_x/H_f
// are supposed to equal (up to sign: H = d(predicted)/d(x), not d(res)/d(x)).
Eigen::Vector2d predict_pixel(const type::Variable& landmark, const type::Variable& anchor_clone,
                               const type::Variable& anchor_calib, const type::Variable& obs_clone,
                               const type::Variable& obs_calib, const core::CameraModel& cam) {
    const Eigen::Vector3d p_FinG = compute_p_FinG(landmark, anchor_clone, anchor_calib);
    const Eigen::Vector3d p_FinI = obs_clone.Rot() * (p_FinG - obs_clone.pos());
    const Eigen::Vector3d p_FinC = obs_calib.Rot() * p_FinI + obs_calib.pos();
    const Eigen::Vector2d normalized = p_FinC.head<2>() / p_FinC.z();
    return core::distort(cam, normalized);
}

} // namespace

int main() {
    msckf::StateOptions options;
    options.do_fej = false; // isolate the base formula first
    options.num_cameras = 1;
    msckf::State state;
    msckf::init_state(state, options);

    type::init_posejpl(state.clones_IMU[0]);
    Eigen::VectorXd dx0(6);
    dx0 << 0.2, -0.1, 0.05, 1.0, 2.0, 0.3;
    type::update_variable(state.clones_IMU[0], dx0);
    for (int i = 0; i < 7; ++i) state.clones_IMU[0].fej[i] = state.clones_IMU[0].value[i];
    state.clones_IMU[0].timestamp = 10.0;
    state.num_clones = 1;

    type::init_posejpl(state.calib_IMUtoCAM[0]);
    Eigen::VectorXd dxc(6);
    dxc << 0.05, 0.02, -0.03, 0.01, -0.02, 0.05;
    type::update_variable(state.calib_IMUtoCAM[0], dxc);
    for (int i = 0; i < 7; ++i) state.calib_IMUtoCAM[0].fej[i] = state.calib_IMUtoCAM[0].value[i];

    type::Variable landmark{};
    landmark.type = type::VariableType::LANDMARK;
    landmark.representation = type::LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
    landmark.size = 3;
    landmark.storage_size = 3;
    landmark.anchor_cam_id = 0;
    landmark.timestamp = 10.0;
    const Eigen::Vector3d p_FinA_nominal(0.3, -0.2, 2.5);
    type::set_landmark_from_xyz(landmark, p_FinA_nominal, false);
    for (int i = 0; i < 3; ++i) landmark.fej[i] = landmark.value[i];

    Eigen::Matrix3d H_f;
    Eigen::Matrix<double, 3, 6> H_anchor;
    type::Variable* anchor_out = nullptr;
    if (!msckf::get_feature_jacobian_representation(state, landmark, H_f, H_anchor, anchor_out)) {
        std::cerr << "get_feature_jacobian_representation() failed\n";
        return 1;
    }

    const Eigen::Vector3d p0 = compute_p_FinG(landmark, state.clones_IMU[0], state.calib_IMUtoCAM[0]);
    const double eps = 1e-6;
    double max_err_f = 0.0;
    for (int col = 0; col < 3; ++col) {
        type::Variable lm2 = landmark;
        lm2.value[col] += eps;
        const Eigen::Vector3d p1 = compute_p_FinG(lm2, state.clones_IMU[0], state.calib_IMUtoCAM[0]);
        const Eigen::Vector3d numeric = (p1 - p0) / eps;
        const Eigen::Vector3d analytic = H_f.col(col);
        const double err = (numeric - analytic).norm();
        max_err_f = std::max(max_err_f, err);
        std::cerr << "H_f col " << col << " numeric=[" << numeric.transpose() << "] analytic=[" << analytic.transpose()
                   << "] err=" << err << "\n";
    }

    double max_err_anchor = 0.0;
    for (int col = 0; col < 6; ++col) {
        Eigen::VectorXd dxp = Eigen::VectorXd::Zero(6);
        dxp[col] = eps;
        type::Variable clone2 = state.clones_IMU[0];
        type::update_variable(clone2, dxp);
        const Eigen::Vector3d p1 = compute_p_FinG(landmark, clone2, state.calib_IMUtoCAM[0]);
        const Eigen::Vector3d numeric = (p1 - p0) / eps;
        const Eigen::Vector3d analytic = H_anchor.col(col);
        const double err = (numeric - analytic).norm();
        max_err_anchor = std::max(max_err_anchor, err);
        std::cerr << "H_anchor col " << col << " numeric=[" << numeric.transpose() << "] analytic=["
                   << analytic.transpose() << "] err=" << err << "\n";
    }

    std::cout << "max H_f error: " << max_err_f << ", max H_anchor error: " << max_err_anchor << "\n";
    if (max_err_f > 1e-4 || max_err_anchor > 1e-4) {
        std::cerr << "JACOBIAN MISMATCH DETECTED\n";
        return 1;
    }
    std::cout << "representation Jacobians verified against finite differences\n";

    // --- Part 2: full get_feature_jacobian_slam() chain, two observing
    // clones (the anchor itself, plus a second clone at a later timestamp),
    // single camera. Exercises the anchor-accumulation logic directly.
    type::init_posejpl(state.clones_IMU[1]);
    Eigen::VectorXd dx1(6);
    dx1 << -0.1, 0.15, -0.05, 1.2, 1.8, 0.4;
    type::update_variable(state.clones_IMU[1], dx1);
    for (int i = 0; i < 7; ++i) state.clones_IMU[1].fej[i] = state.clones_IMU[1].value[i];
    state.clones_IMU[1].timestamp = 10.1;
    state.num_clones = 2;

    core::CameraModel cam{};
    const double cam_values[8] = {300.0, 300.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0};
    core::init_camera(cam, core::CameraModelType::RADTAN, 640, 480, cam_values);
    core::CameraModel cam_models[1] = {cam};

    core::Feature feature{};
    feature.featid = 42;
    feature.num_measurements = 2;
    feature.measurements[0].cam_id = 0;
    feature.measurements[0].timestamp = 10.0; // anchor clone itself
    feature.measurements[1].cam_id = 0;
    feature.measurements[1].timestamp = 10.1; // second clone
    const Eigen::Vector2d uv0 =
        predict_pixel(landmark, state.clones_IMU[0], state.calib_IMUtoCAM[0], state.clones_IMU[0],
                      state.calib_IMUtoCAM[0], cam);
    const Eigen::Vector2d uv1 =
        predict_pixel(landmark, state.clones_IMU[0], state.calib_IMUtoCAM[0], state.clones_IMU[1],
                      state.calib_IMUtoCAM[0], cam);
    feature.measurements[0].uv = uv0;
    feature.measurements[1].uv = uv1;

    Eigen::MatrixXd H_f_full, H_x_full;
    Eigen::VectorXd res_full;
    type::Variable* Hx_order[100];
    int num_Hx = 0;
    if (!msckf::get_feature_jacobian_slam(state, landmark, feature, cam_models, H_f_full, H_x_full, res_full,
                                          Hx_order, num_Hx)) {
        std::cerr << "get_feature_jacobian_slam() failed\n";
        return 1;
    }

    int col_offsets[100];
    int total_cols = 0;
    for (int i = 0; i < num_Hx; ++i) {
        col_offsets[i] = total_cols;
        total_cols += Hx_order[i]->size;
    }

    double max_err_full = 0.0;
    for (int m = 0; m < 2; ++m) {
        const type::Variable& obs_clone = (m == 0) ? state.clones_IMU[0] : state.clones_IMU[1];
        const Eigen::Vector2d p0_pix =
            predict_pixel(landmark, state.clones_IMU[0], state.calib_IMUtoCAM[0], obs_clone, state.calib_IMUtoCAM[0], cam);

        // H_f columns (lambda)
        for (int col = 0; col < 3; ++col) {
            type::Variable lm2 = landmark;
            lm2.value[col] += eps;
            const Eigen::Vector2d p1_pix = predict_pixel(lm2, state.clones_IMU[0], state.calib_IMUtoCAM[0], obs_clone,
                                                          state.calib_IMUtoCAM[0], cam);
            const Eigen::Vector2d numeric = (p1_pix - p0_pix) / eps;
            const Eigen::Vector2d analytic = H_f_full.block<2, 1>(2 * m, col);
            const double err = (numeric - analytic).norm();
            max_err_full = std::max(max_err_full, err);
            std::cerr << "meas " << m << " H_f col " << col << " numeric=[" << numeric.transpose() << "] analytic=["
                       << analytic.transpose() << "] err=" << err << "\n";
        }

        // H_x columns for each Hx_order variable (anchor clone, and whichever
        // clone(s) actually observe this measurement).
        for (int v = 0; v < num_Hx; ++v) {
            type::Variable* var = Hx_order[v];
            for (int col = 0; col < var->size; ++col) {
                Eigen::VectorXd dxp = Eigen::VectorXd::Zero(var->size);
                dxp[col] = eps;
                type::Variable var2 = *var;
                type::update_variable(var2, dxp);

                const type::Variable& anchor_used = (var == &state.clones_IMU[0]) ? var2 : state.clones_IMU[0];
                const type::Variable& obs_used = (var == &obs_clone) ? var2 : obs_clone;
                // Skip columns for variables that don't affect this measurement at all
                // (not the anchor and not the observing clone).
                if (var != &state.clones_IMU[0] && var != &obs_clone) continue;

                const Eigen::Vector2d p1_pix =
                    predict_pixel(landmark, anchor_used, state.calib_IMUtoCAM[0], obs_used, state.calib_IMUtoCAM[0], cam);
                const Eigen::Vector2d numeric = (p1_pix - p0_pix) / eps;
                const Eigen::Vector2d analytic = H_x_full.block<2, 1>(2 * m, col_offsets[v] + col);
                const double err = (numeric - analytic).norm();
                max_err_full = std::max(max_err_full, err);
                std::cerr << "meas " << m << " H_x var=" << v << " col " << col << " numeric=[" << numeric.transpose()
                           << "] analytic=[" << analytic.transpose() << "] err=" << err << "\n";
            }
        }
    }

    std::cout << "max full-chain error: " << max_err_full << "\n";
    if (max_err_full > 1e-3) {
        std::cerr << "FULL-CHAIN JACOBIAN MISMATCH DETECTED\n";
        return 1;
    }
    std::cout << "get_feature_jacobian_slam() verified against finite differences\n";
    return 0;
}
