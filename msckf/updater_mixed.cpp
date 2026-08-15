#include "updater_mixed.hpp"
#include "../type/quat_ops.hpp"
#include <cmath>

namespace msckf {

// MSCKF feature Jacobian using official OpenVINS's ANCHORED_MSCKF_INVERSE_DEPTH
// representation (matches ov_msckf UpdaterHelper::get_feature_jacobian_full +
// get_feature_jacobian_representation). The feature is parameterized as
// (alpha=x/z, beta=y/z, rho=1/z) in its anchor camera frame; its global
// position p_FinG is recomputed from p_FinA + the anchor clone's current pose,
// which introduces an extra Jacobian coupling to the anchor clone (H_anc) that
// the plain GLOBAL_3D representation lacks. This better conditions low-parallax
// / degenerate-motion features (e.g. KAIST circle.bag) exactly as official does.
// PARITY NOTE (verified by tests/bitdiff_jac.cpp): for MONO features this
// function is bit-exact against official's UpdaterHelper::get_feature_jacobian_full
// -- res, H_f and H_x all at 0 ULP. For STEREO features the two differ by a
// permutation of rows and of Hx_order columns, because official iterates a
// feature's per-camera lists through an unordered_map (camera 1 first under
// libstdc++) while this walks measurements in insertion order. Permuting rows
// and the matching columns leaves the EKF update mathematically unchanged, so
// this is a bit-parity gap, not an accuracy one. Reordering the loops to match
// was tried and did not close it (the column order follows a separate path),
// so the simpler loop is kept and the difference is documented instead.

void get_feature_jacobian_mixed(const State& state, const core::Feature& feature,
                                const core::CameraModel* camera_models,
                                Eigen::MatrixXd& H_f, Eigen::MatrixXd& H_x,
                                Eigen::VectorXd& residual,
                                type::Variable** Hx_order, int& num_Hx,
                                type::LandmarkRepresentation representation) {
    num_Hx = 0;
    // GLOBAL_3D keeps the feature in global XYZ: no anchor variable, no
    // representation chain, no anchor-clone coupling -- official's default for
    // MSCKF features.
    const bool want_anchored = representation != type::LandmarkRepresentation::GLOBAL_3D;
    auto add_variable = [&](type::Variable* variable) {
        for (int i = 0; i < num_Hx; ++i) {
            if (Hx_order[i] == variable) return;
        }
        Hx_order[num_Hx++] = variable;
    };

    // Anchor clone/camera for this feature (set by single_triangulation).
    const int acam = feature.anchor_cam_id;
    type::Variable* anchor_clone = nullptr;
    for (int i = 0; i < state.num_clones; ++i) {
        if (state.clones_IMU[i].timestamp == feature.anchor_clone_timestamp) {
            anchor_clone = const_cast<type::Variable*>(&state.clones_IMU[i]);
            break;
        }
    }

    for (int measurement = 0; measurement < feature.num_measurements; ++measurement) {
        const int camera = feature.measurements[measurement].cam_id;
        if (camera < 0 || camera >= state.options.num_cameras) continue;
        if (state.options.do_calib_camera_pose)
            add_variable(const_cast<type::Variable*>(&state.calib_IMUtoCAM[camera]));
        if (state.options.do_calib_camera_intrinsics)
            add_variable(const_cast<type::Variable*>(&state.cam_intrinsics[camera]));
        for (int clone = 0; clone < state.num_clones; ++clone) {
            if (state.clones_IMU[clone].timestamp == feature.measurements[measurement].timestamp) {
                add_variable(const_cast<type::Variable*>(&state.clones_IMU[clone]));
                break;
            }
        }
    }
    // Anchored representation: the anchor clone (and its extrinsics) must be in
    // the Jacobian even if the anchor timestamp is not itself a measurement.
    if (want_anchored && anchor_clone) add_variable(anchor_clone);
    if (want_anchored && acam >= 0 && state.options.do_calib_camera_pose)
        add_variable(const_cast<type::Variable*>(&state.calib_IMUtoCAM[acam]));

    int column_offsets[100];
    int total_columns = 0;
    for (int i = 0; i < num_Hx; ++i) {
        column_offsets[i] = total_columns;
        total_columns += Hx_order[i]->size;
    }
    H_f.setZero(2 * feature.num_measurements, 3);
    H_x.setZero(2 * feature.num_measurements, total_columns);
    residual.setZero(2 * feature.num_measurements);

    // ----- Representation Jacobians (computed once) -----
    // dpfg_dlambda (3x3): d(p_FinG)/d(inverse-depth params).
    // H_anc (3x6): d(p_FinG)/d(anchor clone pose).
    // p_FinG recomputed from p_FinA + anchor pose.
    Eigen::Matrix3d dpfg_dlambda = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 3, 6> H_anc = Eigen::Matrix<double, 3, 6>::Zero();
    Eigen::Vector3d p_FinG = feature.p_FinG;
    bool anchored = want_anchored && (anchor_clone != nullptr && acam >= 0);
    if (anchored) {
        const type::Variable& acalib = state.calib_IMUtoCAM[acam];
        const Eigen::Matrix3d R_ItoC_a = acalib.Rot();
        const Eigen::Vector3d p_IinC_a = acalib.pos();
        Eigen::Matrix3d R_GtoA = anchor_clone->Rot();
        Eigen::Vector3d p_AinG = anchor_clone->pos();

        // Global position from current anchor pose (used for the residual).
        p_FinG = R_GtoA.transpose() * R_ItoC_a.transpose() * (feature.p_FinA - p_IinC_a) + p_AinG;

        // For the representation Jacobian, FEJ the anchor pose (official does).
        Eigen::Vector3d p_FinA = feature.p_FinA;
        if (state.options.do_fej) {
            const Eigen::Vector3d p_FinG_best = p_FinG;
            R_GtoA = anchor_clone->Rot_fej();
            p_AinG = anchor_clone->pos_fej();
            p_FinA = R_ItoC_a * R_GtoA * (p_FinG_best - p_AinG) + p_IinC_a;
        }
        const Eigen::Matrix3d R_CtoG = R_GtoA.transpose() * R_ItoC_a.transpose();

        // Anchor clone Jacobian.
        H_anc.leftCols<3>().noalias() = -R_GtoA.transpose() * type::skew_x(R_ItoC_a.transpose() * (p_FinA - p_IinC_a));
        H_anc.rightCols<3>().setIdentity();

        // MSCKF inverse-depth chain: d(p_FinA)/d(alpha,beta,rho).
        const double az = p_FinA.z();
        const double alpha = p_FinA.x() / az;
        const double beta = p_FinA.y() / az;
        const double rho = 1.0 / az;
        Eigen::Matrix3d d_pfinA_dpinv;
        d_pfinA_dpinv << 1.0 / rho, 0.0, -(1.0 / (rho * rho)) * alpha,
                         0.0, 1.0 / rho, -(1.0 / (rho * rho)) * beta,
                         0.0, 0.0, -(1.0 / (rho * rho));
        dpfg_dlambda.noalias() = R_CtoG * d_pfinA_dpinv;
    }

    // Column offset of the anchor clone in H_x (for the coupling term).
    int anchor_col = -1;
    if (anchored) {
        for (int v = 0; v < num_Hx; ++v)
            if (Hx_order[v] == anchor_clone) { anchor_col = column_offsets[v]; break; }
    }

    for (int measurement = 0; measurement < feature.num_measurements; ++measurement) {
        const core::FeatureMeasurement& observation = feature.measurements[measurement];
        const int camera = observation.cam_id;
        if (camera < 0 || camera >= state.options.num_cameras) continue;
        const type::Variable* clone = nullptr;
        for (int i = 0; i < state.num_clones; ++i) {
            if (state.clones_IMU[i].timestamp == observation.timestamp) {
                clone = &state.clones_IMU[i];
                break;
            }
        }
        if (!clone) continue;

        const type::Variable& calibration = state.calib_IMUtoCAM[camera];
        const Eigen::Matrix3d R_ItoC = calibration.Rot();
        const Eigen::Vector3d p_IinC = calibration.pos();
        Eigen::Matrix3d R_GtoI = clone->Rot();
        Eigen::Vector3d p_IinG = clone->pos();
        Eigen::Vector3d p_FinI = R_GtoI * (p_FinG - p_IinG);
        Eigen::Vector3d p_FinC = R_ItoC * p_FinI + p_IinC;
        if (std::abs(p_FinC.z()) < 1e-10) continue;
        Eigen::Vector2d normalized = p_FinC.head<2>() / p_FinC.z();
        residual.segment<2>(2 * measurement) = observation.uv -
            core::distort(camera_models[camera], normalized);

        if (state.options.do_fej) {
            R_GtoI = clone->Rot_fej();
            p_IinG = clone->pos_fej();
            // Anchored: p_FinG_fej == p_FinG (recomputed), per official.
            p_FinI = R_GtoI * (p_FinG - p_IinG);
            p_FinC = R_ItoC * p_FinI + p_IinC;
            normalized = p_FinC.head<2>() / p_FinC.z();
        }
        Eigen::Matrix2d dz_dzn;
        Eigen::Matrix<double, 2, 8> dz_dzeta;
        core::compute_distort_jacobian(camera_models[camera], normalized, dz_dzn, dz_dzeta);
        Eigen::Matrix<double, 2, 3> dzn_dpfc;
        const double z = p_FinC.z();
        dzn_dpfc << 1.0 / z, 0.0, -p_FinC.x() / (z * z),
                     0.0, 1.0 / z, -p_FinC.y() / (z * z);
        const Eigen::Matrix<double, 2, 3> dz_dpfc = dz_dzn * dzn_dpfc;
        const Eigen::Matrix3d dpfc_dpfg = R_ItoC * R_GtoI;
        const Eigen::Matrix<double, 2, 3> dz_dpfg = dz_dpfc * dpfc_dpfg;

        // Feature Jacobian: chain through the representation.
        H_f.block<2, 3>(2 * measurement, 0) = dz_dpfg * dpfg_dlambda;

        for (int variable = 0; variable < num_Hx; ++variable) {
            const int column = column_offsets[variable];
            if (Hx_order[variable] == clone) {
                Eigen::Matrix<double, 3, 6> dpfc_dclone;
                dpfc_dclone.leftCols<3>() = R_ItoC * type::skew_x(p_FinI);
                dpfc_dclone.rightCols<3>() = -dpfc_dpfg;
                H_x.block<2, 6>(2 * measurement, column) = dz_dpfc * dpfc_dclone;
            } else if (Hx_order[variable] == &state.calib_IMUtoCAM[camera]) {
                Eigen::Matrix<double, 3, 6> dpfc_dcalibration;
                dpfc_dcalibration.leftCols<3>() = type::skew_x(p_FinC - p_IinC);
                dpfc_dcalibration.rightCols<3>() = Eigen::Matrix3d::Identity();
                H_x.block<2, 6>(2 * measurement, column) = dz_dpfc * dpfc_dcalibration;
            } else if (Hx_order[variable] == &state.cam_intrinsics[camera]) {
                H_x.block<2, 8>(2 * measurement, column) = dz_dzeta;
            }
        }

        // Anchor-clone coupling: added (+=) since the anchor may also be this
        // measurement's clone (then its column already holds the clone term).
        if (anchored && anchor_col >= 0) {
            H_x.block<2, 6>(2 * measurement, anchor_col).noalias() += dz_dpfg * H_anc;
        }
    }
}

} // namespace msckf
