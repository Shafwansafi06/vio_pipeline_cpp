#include "updater_slam_helper.hpp"
#include "../type/quat_ops.hpp"
#include <cmath>

namespace msckf {

bool get_feature_jacobian_representation(const State& state, const type::Variable& landmark,
                                          Eigen::Matrix3d& H_f,
                                          Eigen::Matrix<double, 3, 6>& H_anchor_clone,
                                          type::Variable*& anchor_clone_out) {
    const type::Variable* anchor_clone = nullptr;
    for (int c = 0; c < state.num_clones; ++c) {
        if (state.clones_IMU[c].timestamp == landmark.timestamp) {
            anchor_clone = &state.clones_IMU[c];
            break;
        }
    }
    if (!anchor_clone) return false;
    anchor_clone_out = const_cast<type::Variable*>(anchor_clone);

    const type::Variable& calib_anchor = state.calib_IMUtoCAM[landmark.anchor_cam_id];
    const Eigen::Matrix3d R_ItoC = calib_anchor.Rot();
    const Eigen::Vector3d p_IinC = calib_anchor.pos();
    Eigen::Matrix3d R_GtoI = anchor_clone->Rot();
    Eigen::Vector3d p_IinG = anchor_clone->pos();
    Eigen::Vector3d p_FinA = type::get_landmark_xyz(landmark, false);

    if (state.options.do_fej) {
        // Evaluate the Jacobian at the anchor's first-estimate pose, but keep
        // the feature parameters consistent with that pose by mapping the
        // current best-estimate global point back through it.
        const Eigen::Matrix3d R_GtoI_best = anchor_clone->Rot();
        const Eigen::Vector3d p_IinG_best = anchor_clone->pos();
        const Eigen::Vector3d p_FinG_best =
            R_GtoI_best.transpose() * R_ItoC.transpose() * (p_FinA - p_IinC) + p_IinG_best;
        R_GtoI = anchor_clone->Rot_fej();
        p_IinG = anchor_clone->pos_fej();
        const Eigen::Matrix3d R_CtoG_fej = R_GtoI.transpose() * R_ItoC.transpose();
        p_FinA = R_CtoG_fej.transpose() * (p_FinG_best - p_IinG) + p_IinC;
    }

    const Eigen::Matrix3d R_CtoG = R_GtoI.transpose() * R_ItoC.transpose();

    const Eigen::Vector3d term = R_ItoC.transpose() * (p_FinA - p_IinC);
    H_anchor_clone.setZero();
    H_anchor_clone.block<3, 3>(0, 0) = -R_GtoI.transpose() * type::skew_x(term);
    H_anchor_clone.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

    const double alpha = p_FinA[0] / p_FinA[2];
    const double beta = p_FinA[1] / p_FinA[2];
    const double rho = 1.0 / p_FinA[2];
    Eigen::Matrix3d d_pfinA_dpinv = Eigen::Matrix3d::Zero();
    d_pfinA_dpinv(0, 0) = 1.0 / rho;
    d_pfinA_dpinv(0, 2) = -alpha / (rho * rho);
    d_pfinA_dpinv(1, 1) = 1.0 / rho;
    d_pfinA_dpinv(1, 2) = -beta / (rho * rho);
    d_pfinA_dpinv(2, 2) = -1.0 / (rho * rho);

    H_f = R_CtoG * d_pfinA_dpinv;
    return true;
}

bool get_feature_jacobian_slam(const State& state, const type::Variable& landmark,
                                const core::Feature& feature, const core::CameraModel* cam_models,
                                Eigen::MatrixXd& H_f, Eigen::MatrixXd& H_x, Eigen::VectorXd& res,
                                type::Variable** Hx_order, int& num_Hx) {
    Eigen::Matrix3d H_f_repr;
    Eigen::Matrix<double, 3, 6> H_anchor_clone;
    type::Variable* anchor_clone = nullptr;
    if (!get_feature_jacobian_representation(state, landmark, H_f_repr, H_anchor_clone, anchor_clone)) {
        num_Hx = 0;
        return false;
    }

    const type::Variable& calib_anchor = state.calib_IMUtoCAM[landmark.anchor_cam_id];
    const Eigen::Vector3d p_FinA = type::get_landmark_xyz(landmark, false);
    // Current-estimate global position, using the anchor's CURRENT (non-FEJ)
    // pose -- matches the reference implementation's approximation of
    // treating p_FinG_fej == p_FinG for the measurement residual chain.
    const Eigen::Vector3d p_FinG = anchor_clone->Rot().transpose() * calib_anchor.Rot().transpose() *
                                        (p_FinA - calib_anchor.pos()) +
                                    anchor_clone->pos();

    num_Hx = 0;
    auto add_variable = [&](type::Variable* variable) {
        for (int i = 0; i < num_Hx; ++i) {
            if (Hx_order[i] == variable) return;
        }
        Hx_order[num_Hx++] = variable;
    };
    add_variable(anchor_clone);
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

    int column_offsets[100];
    int total_columns = 0;
    for (int i = 0; i < num_Hx; ++i) {
        column_offsets[i] = total_columns;
        total_columns += Hx_order[i]->size;
    }

    int anchor_col = -1;
    for (int i = 0; i < num_Hx; ++i) {
        if (Hx_order[i] == anchor_clone) {
            anchor_col = column_offsets[i];
            break;
        }
    }

    H_f.setZero(2 * feature.num_measurements, 3);
    H_x.setZero(2 * feature.num_measurements, total_columns);
    res.setZero(2 * feature.num_measurements);

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
        res.segment<2>(2 * measurement) = observation.uv - core::distort(cam_models[camera], normalized);

        if (state.options.do_fej) {
            R_GtoI = clone->Rot_fej();
            p_IinG = clone->pos_fej();
            p_FinI = R_GtoI * (p_FinG - p_IinG);
            p_FinC = R_ItoC * p_FinI + p_IinC;
            normalized = p_FinC.head<2>() / p_FinC.z();
        }

        Eigen::Matrix2d dz_dzn;
        Eigen::Matrix<double, 2, 8> dz_dzeta;
        core::compute_distort_jacobian(cam_models[camera], normalized, dz_dzn, dz_dzeta);
        Eigen::Matrix<double, 2, 3> dzn_dpfc;
        const double z = p_FinC.z();
        dzn_dpfc << 1.0 / z, 0.0, -p_FinC.x() / (z * z),
                     0.0, 1.0 / z, -p_FinC.y() / (z * z);
        const Eigen::Matrix<double, 2, 3> dz_dpfc = dz_dzn * dzn_dpfc;
        const Eigen::Matrix3d dpfc_dpfg = R_ItoC * R_GtoI;
        const Eigen::Matrix<double, 2, 3> dz_dpfg = dz_dpfc * dpfc_dpfg;

        H_f.block<2, 3>(2 * measurement, 0) = dz_dpfg * H_f_repr;

        for (int variable = 0; variable < num_Hx; ++variable) {
            const int column = column_offsets[variable];
            if (Hx_order[variable] == clone) {
                Eigen::Matrix<double, 3, 6> dpfc_dclone;
                dpfc_dclone.leftCols<3>() = R_ItoC * type::skew_x(p_FinI);
                dpfc_dclone.rightCols<3>() = -dpfc_dpfg;
                H_x.block<2, 6>(2 * measurement, column) += dz_dpfc * dpfc_dclone;
            }
            if (state.options.do_calib_camera_pose && Hx_order[variable] == &state.calib_IMUtoCAM[camera]) {
                Eigen::Matrix<double, 3, 6> dpfc_dcalibration;
                dpfc_dcalibration.leftCols<3>() = type::skew_x(p_FinC - p_IinC);
                dpfc_dcalibration.rightCols<3>() = Eigen::Matrix3d::Identity();
                H_x.block<2, 6>(2 * measurement, column) += dz_dpfc * dpfc_dcalibration;
            }
            if (state.options.do_calib_camera_intrinsics && Hx_order[variable] == &state.cam_intrinsics[camera]) {
                H_x.block<2, 8>(2 * measurement, column) += dz_dzeta;
            }
        }
        // Anchor clone contribution (dp_FinG/d_anchor_pose); accumulates on
        // top of the observing-clone term above whenever the anchor clone IS
        // the observing clone for this measurement.
        if (anchor_col >= 0) {
            H_x.block<2, 6>(2 * measurement, anchor_col) += dz_dpfg * H_anchor_clone;
        }
    }
    return true;
}

} // namespace msckf
