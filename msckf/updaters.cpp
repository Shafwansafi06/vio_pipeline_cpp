#include "updaters.hpp"

#include <chrono>
#include "state_helper.hpp"
#include "updater_mixed.hpp"
#include "updater_slam_helper.hpp"
#include "chi2_table.hpp"
#include "../type/quat_ops.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace msckf {

static core::ClonesCamera build_clones_camera(const State& state) {
    core::ClonesCamera clones_cam;
    clones_cam.num_clones = state.num_clones;
    for (int c = 0; c < state.num_clones; ++c) {
        clones_cam.timestamps[c] = state.clones_IMU[c].timestamp;
        for (int i = 0; i < state.options.num_cameras; ++i) {
            Eigen::Matrix3d R_GtoCi = state.calib_IMUtoCAM[i].Rot() * state.clones_IMU[c].Rot();
            Eigen::Vector3d p_CioinG = state.clones_IMU[c].pos() - R_GtoCi.transpose() * state.calib_IMUtoCAM[i].pos();
            clones_cam.poses[i][c] = {R_GtoCi, p_CioinG};
        }
    }
    return clones_cam;
}

// Chi-squared 0.95 gate. Official precomputes dof 1..499 with
// boost::math::quantile at construction (UpdaterMSCKF.cpp:52-55); we embed the
// identical values as hex-float literals so no boost dependency leaks into this
// library and the threshold is bit-identical rather than merely close.
//
// This replaced a Wilson-Hilferty approximation plus 3-decimal constants, which
// was wrong in the 4th decimal at every dof (dof=1: 3.841 vs 3.8414588...) and
// so accepted/rejected a different set of features than official does.
// Anchor-change failure accounting. change_anchors() runs whenever a
// landmark's anchor clone is about to be marginalized; each of these paths
// silently kills the landmark, so they need to be visible.
long anchor_marg_no_old = 0;
long anchor_marg_depth = 0;
long anchor_marg_transform = 0;
long anchor_change_ok = 0;
long dinit_seen = 0, dinit_skip_cap = 0, dinit_skip_meas = 0, dinit_skip_tri = 0, dinit_ok = 0;

static double chi2_ppf_95(int dof) {
    if (dof <= 0) return 0.0;
    if (dof < CHI2_TABLE_SIZE) return CHI2_TABLE_95[dof];
    // ponytail: dof >= 500 is unreachable here -- residual rows are bounded by
    // 4 * max_clone_size (~44). Official calls boost at runtime for this case;
    // if the clone window ever grows past 125, regenerate the table instead.
    double k = static_cast<double>(dof);
    double term = 1.0 - 2.0 / (9.0 * k);
    double sd = std::sqrt(2.0 / (9.0 * k));
    double z = 1.64485362695;
    return k * std::pow(term + z * sd, 3.0);
}

void nullspace_project_inplace(Eigen::MatrixXd& H_f, Eigen::MatrixXd& H_x, Eigen::VectorXd& res) {
    // Official does this with in-place GIVENS rotations, never forming Q.
    // This used to be a HouseholderQR with an explicit Q and a rightCols()
    // slice. Both are mathematically valid left-nullspace projections, but they
    // produce DIFFERENT orthonormal bases -- so H_x and res come out different
    // (not merely off in the last bits), and every downstream chi2 gate and EKF
    // update sees different numbers. Verified against the oracle by
    // bitdiff_update.
    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n = 0; n < H_f.cols(); ++n) {
        for (int m = (int)H_f.rows() - 1; m > n; m--) {
            tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n));
            (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (H_x.block(m - 1, 0, 2, H_x.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
        }
    }
    H_x = H_x.block(H_f.cols(), 0, H_x.rows() - H_f.cols(), H_x.cols()).eval();
    res = res.block(H_f.cols(), 0, res.rows() - H_f.cols(), res.cols()).eval();
}

void measurement_compress_inplace(Eigen::MatrixXd& H_x, Eigen::VectorXd& res) {
    // Same story: official compresses in place with Givens rotations and then
    // conservativeResize()s down to rank rows, rather than building R and Q
    // from a HouseholderQR. Note the early return when the system is already
    // thin -- official leaves such a system untouched.
    if (H_x.rows() <= H_x.cols())
        return;
    // The EKF update is invariant to multiplying [H | res] on the left by any
    // orthogonal matrix, which is the whole point of this compression: keep the
    // top rows of the QR and throw the rest away. Official sweeps the system
    // with ~23k individual Givens rotations, each a separate 2 x (cols - n)
    // Eigen call; one blocked Householder QR of the augmented matrix produces
    // the same R (up to row signs, which the update does not see) in a fraction
    // of the time. Augmenting res as the last column is what avoids ever
    // forming Q.
    const int rows = int(H_x.rows());
    const int cols = int(H_x.cols());
    Eigen::MatrixXd augmented(rows, cols + 1);
    augmented.leftCols(cols) = H_x;
    augmented.col(cols) = res;

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(augmented);
    const Eigen::MatrixXd R = qr.matrixQR().topRows(cols).triangularView<Eigen::Upper>();

    H_x = R.leftCols(cols);
    res = R.col(cols);
}

void get_feature_jacobian_full(const State& state, const core::Feature& feature,
                               const core::CameraModel& cam_model, int cam_id,
                               Eigen::MatrixXd& H_f, Eigen::MatrixXd& H_x, Eigen::VectorXd& res,
                               type::Variable** Hx_order, int& num_Hx) {
    num_Hx = 0;
    
    auto add_var = [&](type::Variable* var) {
        for (int i = 0; i < num_Hx; ++i) {
            if (Hx_order[i] == var) return;
        }
        Hx_order[num_Hx++] = var;
    };
    
    if (state.options.do_calib_camera_pose) {
        add_var(const_cast<type::Variable*>(&state.calib_IMUtoCAM[cam_id]));
    }
    if (state.options.do_calib_camera_intrinsics) {
        add_var(const_cast<type::Variable*>(&state.cam_intrinsics[cam_id]));
    }
    
    for (int m = 0; m < feature.num_measurements; ++m) {
        if (feature.measurements[m].cam_id == cam_id) {
            double ts = feature.measurements[m].timestamp;
            // Lookup clone
            for (int c = 0; c < state.num_clones; ++c) {
                if (state.clones_IMU[c].timestamp == ts) {
                    add_var(const_cast<type::Variable*>(&state.clones_IMU[c]));
                    break;
                }
            }
        }
    }
    
    int total_hx = 0;
    int var_col_offset[100];
    for (int i = 0; i < num_Hx; ++i) {
        var_col_offset[i] = total_hx;
        total_hx += Hx_order[i]->size;
    }
    
    int M = 0;
    int meas_idx[48];
    for (int m = 0; m < feature.num_measurements; ++m) {
        if (feature.measurements[m].cam_id == cam_id) {
            meas_idx[M++] = m;
        }
    }
    
    H_f.setZero(2 * M, 3);
    H_x.setZero(2 * M, total_hx);
    res.setZero(2 * M);
    
    Eigen::Matrix3d R_ItoC = state.calib_IMUtoCAM[cam_id].Rot();
    Eigen::Vector3d p_IinC = state.calib_IMUtoCAM[cam_id].pos();
    
    Eigen::Vector3d p_FinG = feature.p_FinG;
    Eigen::Vector3d p_FinG_fej = feature.p_FinG; // default
    
    for (int m = 0; m < M; ++m) {
        int idx = meas_idx[m];
        double ts = feature.measurements[idx].timestamp;
        
        // Find clone
        const type::Variable* clone = nullptr;
        for (int c = 0; c < state.num_clones; ++c) {
            if (state.clones_IMU[c].timestamp == ts) {
                clone = &state.clones_IMU[c];
                break;
            }
        }
        if (!clone) continue;
        
        Eigen::Matrix3d R_GtoIi = clone->Rot();
        Eigen::Vector3d p_IiinG = clone->pos();
        
        Eigen::Vector3d p_FinIi = R_GtoIi * (p_FinG - p_IiinG);
        Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
        
        Eigen::Vector2d uv_norm = p_FinCi.head<2>() / p_FinCi[2];
        Eigen::Vector2d uv_dist = core::distort(cam_model, uv_norm);
        
        res.segment<2>(2 * m) = feature.measurements[idx].uv - uv_dist;
        
        if (state.options.do_fej) {
            R_GtoIi = clone->Rot_fej();
            p_IiinG = clone->pos_fej();
            p_FinIi = R_GtoIi * (p_FinG_fej - p_IiinG);
            p_FinCi = R_ItoC * p_FinIi + p_IinC;
            uv_norm = p_FinCi.head<2>() / p_FinCi[2];
        }
        
        Eigen::Matrix2d dz_dzn;
        Eigen::Matrix<double, 2, 8> dz_dzeta;
        core::compute_distort_jacobian(cam_model, uv_norm, dz_dzn, dz_dzeta);
        
        double z = p_FinCi[2];
        Eigen::Matrix<double, 2, 3> dzn_dpfc;
        dzn_dpfc << 1.0 / z, 0.0, -p_FinCi[0] / (z * z),
                    0.0, 1.0 / z, -p_FinCi[1] / (z * z);
                    
        Eigen::Matrix<double, 2, 3> dz_dpfc = dz_dzn * dzn_dpfc;
        Eigen::Matrix3d dpfc_dpfg = R_ItoC * R_GtoIi;
        Eigen::Matrix<double, 2, 3> dz_dpfg = dz_dpfc * dpfc_dpfg;
        
        H_f.block<2, 3>(2 * m, 0) = dz_dpfg;
        
        // Find clone index in Hx_order
        int clone_idx = -1;
        for (int i = 0; i < num_Hx; ++i) {
            if (Hx_order[i] == clone) {
                clone_idx = i;
                break;
            }
        }
        if (clone_idx != -1) {
            int col = var_col_offset[clone_idx];
            Eigen::Matrix<double, 3, 6> dpfc_dclone;
            dpfc_dclone.leftCols<3>() = R_ItoC * type::skew_x(p_FinIi);
            dpfc_dclone.rightCols<3>() = -dpfc_dpfg;
            H_x.block<2, 6>(2 * m, col) = dz_dpfc * dpfc_dclone;
        }
        
        // Find cam pose calibration index
        int calib_idx = -1;
        for (int i = 0; i < num_Hx; ++i) {
            if (Hx_order[i] == &state.calib_IMUtoCAM[cam_id]) {
                calib_idx = i;
                break;
            }
        }
        if (calib_idx != -1) {
            int col = var_col_offset[calib_idx];
            Eigen::Matrix<double, 3, 6> dpfc_dcalib;
            dpfc_dcalib.leftCols<3>() = type::skew_x(p_FinCi - p_IinC);
            dpfc_dcalib.rightCols<3>() = Eigen::Matrix3d::Identity();
            H_x.block<2, 6>(2 * m, col) += dz_dpfc * dpfc_dcalib;
        }
        
        // Find cam intrinsics calibration index
        int intrin_idx = -1;
        for (int i = 0; i < num_Hx; ++i) {
            if (Hx_order[i] == &state.cam_intrinsics[cam_id]) {
                intrin_idx = i;
                break;
            }
        }
        if (intrin_idx != -1) {
            int col = var_col_offset[intrin_idx];
            H_x.block<2, 8>(2 * m, col) = dz_dzeta;
        }
    }
}

void init_updater_msckf(UpdaterMSCKFData& updater, const UpdaterOptions& options,
                        const core::FeatureInitializerOptions& feat_init_options) {
    updater.options = options;
    updater.options.sigma_pix_sq = updater.options.sigma_pix * updater.options.sigma_pix;
    updater.feat_init_options = feat_init_options;
}

// Sub-stage timing for the MSCKF update, reported by the runners.
double msckf_ms_tri = 0.0, msckf_ms_jac = 0.0, msckf_ms_chi2 = 0.0,
       msckf_ms_assemble = 0.0, msckf_ms_alloc = 0.0, msckf_ms_compress = 0.0, msckf_ms_ekf = 0.0;
long msckf_calls = 0, msckf_rows_pre = 0, msckf_cols = 0, msckf_rows_post = 0, msckf_cov = 0;

namespace {
struct SubTimer {
    double& sink;
    std::chrono::steady_clock::time_point start;
    explicit SubTimer(double& target) : sink(target), start(std::chrono::steady_clock::now()) {}
    ~SubTimer() { sink += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); }
};
} // namespace

void update_msckf(UpdaterMSCKFData& updater, State& state, core::Feature** feature_vec, int feature_count, const core::CameraModel* cam_models) {
    // Collect camera clones poses for triangulation
    core::ClonesCamera clones_cam = build_clones_camera(state);

#ifdef DOD_MSCKF_ACCEPT_DEBUG
    static long dbg_calls = 0, dbg_seen = 0, dbg_rej_tri = 0, dbg_rej_chi2 = 0;
    ++dbg_calls;
    dbg_seen += feature_count;
#endif

    // Process triangulation and Gauss-Newton refinement
    int num_triangulated = 0;
    for (int i = 0; i < feature_count; ++i) {
        core::Feature* feat = feature_vec[i];
        SubTimer tri_timer(msckf_ms_tri);
        bool success_tri = core::single_triangulation(*feat, clones_cam, updater.feat_init_options);
        bool success_refine = true;
        if (success_tri) {
            success_refine = core::single_gaussnewton(*feat, clones_cam, updater.feat_init_options);
        }
        if (!success_tri || !success_refine) {
            feat->to_delete = true;
#ifdef DOD_MSCKF_ACCEPT_DEBUG
            ++dbg_rej_tri;
#endif
        } else {
            num_triangulated++;
        }
    }
    
    if (num_triangulated < updater.options.min_features_for_update) {
        for (int i = 0; i < feature_count; ++i) feature_vec[i]->to_delete = true;
        return;
    }
    
    // Accumulate linear system
    // Reused across frames: this used to allocate and zero a 2000x300 matrix
    // (4.8 MB) on every call, of which only the first ~280 rows are ever
    // touched. Only the rows that can actually be written are cleared.
    static thread_local Eigen::MatrixXd Hx_big;
    static thread_local Eigen::VectorXd res_big;
    {
        SubTimer alloc_timer(msckf_ms_alloc);
        if (Hx_big.rows() < 2000 || Hx_big.cols() < 300) {
            Hx_big.setZero(2000, 300);
            res_big.setZero(2000);
        }
        int rows_upper_bound = 0;
        for (int i = 0; i < feature_count; ++i) {
            rows_upper_bound += 2 * feature_vec[i]->num_measurements;
        }
        rows_upper_bound = std::min(rows_upper_bound, int(Hx_big.rows()));
        Hx_big.topRows(rows_upper_bound).setZero();
        res_big.head(rows_upper_bound).setZero();
    }
    int total_rows = 0;
    
    type::Variable* hx_order_big[100];
    int num_hx_order_big = 0;
    
    auto add_to_big = [&](type::Variable* var) -> int {
        for (int i = 0; i < num_hx_order_big; ++i) {
            if (hx_order_big[i] == var) return i;
        }
        hx_order_big[num_hx_order_big++] = var;
        return num_hx_order_big - 1;
    };
    
    int var_col_idx[100];
    int current_col_big = 0;
    
    for (int idx = 0; idx < feature_count; ++idx) {
        core::Feature* feat = feature_vec[idx];
        if (feat->to_delete) continue;
        
        // MSCKF updates assume camera 0
        Eigen::MatrixXd H_f, H_x;
        Eigen::VectorXd res;
        type::Variable* Hx_order[100];
        int num_Hx = 0;
        
        {
            SubTimer jac_timer(msckf_ms_jac);
            get_feature_jacobian_mixed(state, *feat, cam_models, H_f, H_x, res, Hx_order, num_Hx,
                                       state.options.feat_rep_msckf);
            nullspace_project_inplace(H_f, H_x, res);
        }
        
        if (H_x.rows() == 0 || res.size() == 0) {
            feat->to_delete = true;
            continue;
        }
        
        // Chi-Square Check
        SubTimer chi2_timer(msckf_ms_chi2);
        // Official: S = H P H^T, add sigma^2 to the diagonal, chi2 =
        // res . S.llt().solve(res). What was here additionally formed
        // 0.5*(S + S^T) -- a full temporary of a matrix that is already
        // symmetric by construction -- and solved with an LDLT.
        // Gather the marginal covariance into a reused buffer. Building S block
        // by block instead (avoiding the gather entirely) was tried and is
        // SLOWER here -- 0.285 -> 0.404 ms/frame -- because it turns two dense
        // products into ~13x13 small ones per feature. The gather wins; only its
        // allocation is worth removing.
        static thread_local Eigen::MatrixXd P_marg;
        get_marginal_covariance_into(state, Hx_order, num_Hx, P_marg);
        Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
        S.diagonal().array() += updater.options.sigma_pix_sq;

        double chi2 = 99999.0;
        Eigen::LLT<Eigen::MatrixXd> solver(S);
        if (solver.info() == Eigen::Success) {
            chi2 = res.dot(solver.solve(res));
        }
        
        double chi2_check = chi2_ppf_95(res.size());
        if (chi2 > updater.options.chi2_multipler * chi2_check) {
            feat->to_delete = true;
#ifdef DOD_MSCKF_ACCEPT_DEBUG
            ++dbg_rej_chi2;
#endif
            continue;
        }
        
        // Add to big H
        SubTimer assemble_timer(msckf_ms_assemble);
        int start_row = total_rows;
        int num_rows = res.size();
        
        if (total_rows + num_rows > Hx_big.rows()) {
            // Resize big system buffers. Eigen's conservativeResize does NOT
            // zero-initialize the newly added rows -- they contain whatever
            // was previously in that memory. Without explicitly zeroing them,
            // the "+=" accumulation below into a newly-grown region adds onto
            // garbage instead of zero, silently corrupting the linear system
            // fed to the EKF update whenever this resize path is taken.
            const int old_rows = int(Hx_big.rows());
            Hx_big.conservativeResize(old_rows + 1000, Eigen::NoChange);
            Hx_big.bottomRows(1000).setZero();
            res_big.conservativeResize(res_big.size() + 1000);
        }

        res_big.segment(start_row, num_rows) = res;

        int col_offset = 0;
        for (int k = 0; k < num_Hx; ++k) {
            type::Variable* var = Hx_order[k];
            int global_var_idx = add_to_big(var);
            // Update mapping
            current_col_big = 0;
            for (int c = 0; c < num_hx_order_big; ++c) {
                var_col_idx[c] = current_col_big;
                current_col_big += hx_order_big[c]->size;
            }

            if (current_col_big > Hx_big.cols()) {
                // Same uninitialized-memory hazard as the row resize above.
                Hx_big.conservativeResize(Eigen::NoChange, Hx_big.cols() + 100);
                Hx_big.rightCols(100).setZero();
            }

            int target_col = var_col_idx[global_var_idx];
            Hx_big.block(start_row, target_col, num_rows, var->size) += H_x.block(0, col_offset, num_rows, var->size);
            col_offset += var->size;
        }
        
        total_rows += num_rows;
    }
    
    // Mark features as deleted after update
    for (int i = 0; i < feature_count; ++i) feature_vec[i]->to_delete = true;

#ifdef DOD_MSCKF_ACCEPT_DEBUG
    if (dbg_calls % 100 == 0) {
        std::fprintf(stderr, "[DOD-MSCKF-DEBUG] calls=%ld seen=%ld rej_tri=%ld rej_chi2=%ld accept_rate=%.3f\n",
            dbg_calls, dbg_seen, dbg_rej_tri, dbg_rej_chi2,
            1.0 - double(dbg_rej_tri + dbg_rej_chi2) / double(dbg_seen > 0 ? dbg_seen : 1));
    }
#endif

    if (total_rows == 0) return;
    
    Eigen::MatrixXd Hx_valid = Hx_big.topLeftCorner(total_rows, current_col_big);
    Eigen::VectorXd res_valid = res_big.head(total_rows);
    
    ++msckf_calls;
    msckf_rows_pre += Hx_valid.rows();
    msckf_cols += Hx_valid.cols();
    { SubTimer compress_timer(msckf_ms_compress); measurement_compress_inplace(Hx_valid, res_valid); }
    msckf_rows_post += Hx_valid.rows();
    msckf_cov += state.cov_size;

    if (Hx_valid.rows() < updater.options.min_update_rows) return;

    Eigen::MatrixXd R_big = updater.options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_valid.size(), res_valid.size());
    { SubTimer ekf_timer(msckf_ms_ekf); EKFUpdate(state, hx_order_big, num_hx_order_big, Hx_valid, res_valid, R_big); }
}

void init_updater_slam(UpdaterSLAMData& updater, const UpdaterOptions& options_slam, const UpdaterOptions& options_aruco,
                       const core::FeatureInitializerOptions& feat_init_options) {
    updater.options_slam = options_slam;
    updater.options_aruco = options_aruco;
    updater.feat_init_options = feat_init_options;
    updater.options_slam.sigma_pix_sq = updater.options_slam.sigma_pix * updater.options_slam.sigma_pix;
    updater.options_aruco.sigma_pix_sq = updater.options_aruco.sigma_pix * updater.options_aruco.sigma_pix;
}

void delayed_init_slam(UpdaterSLAMData& updater, State& state, core::Feature** feature_vec, int feature_count, const core::CameraModel* cam_models) {
    if (feature_count == 0) return;

    double clonetimes[20];
    for (int c = 0; c < state.num_clones; ++c) clonetimes[c] = state.clones_IMU[c].timestamp;
    core::ClonesCamera clones_cam = build_clones_camera(state);

    for (int i = 0; i < feature_count; ++i) {
        core::Feature* feat = feature_vec[i];
        feat->to_delete = true; // consumed either way, matching the reference implementation

        ++dinit_seen;
        if (state.num_slam_features >= state.options.max_slam_features) { ++dinit_skip_cap; continue; }

        core::clean_old_measurements(*feat, clonetimes, state.num_clones);
        if (feat->num_measurements < 2) { ++dinit_skip_meas; continue; }

        const bool success_tri = core::single_triangulation(*feat, clones_cam, updater.feat_init_options);
        const bool success_refine = success_tri && core::single_gaussnewton(*feat, clones_cam, updater.feat_init_options);
        if (!success_tri || !success_refine) { ++dinit_skip_tri; continue; }

        // Stage the candidate directly in its future state slot so
        // get_feature_jacobian_slam()/initialize() reference a stable
        // address; initialize() only mutates it on chi2 success.
        type::Variable& candidate = state.features_SLAM[state.num_slam_features];
        candidate = type::Variable{};
        candidate.type = type::VariableType::LANDMARK;
        candidate.representation = state.options.feat_rep_slam;
        candidate.size = 3;
        candidate.storage_size = 3;
        candidate.feat_id = feat->featid;
        candidate.anchor_cam_id = feat->anchor_cam_id;
        candidate.timestamp = feat->anchor_clone_timestamp; // repurposed: anchor clone ts
        type::set_landmark_from_xyz(candidate, feat->p_FinA, false);
        type::set_landmark_from_xyz(candidate, feat->p_FinA, true);

        Eigen::MatrixXd H_f, H_x;
        Eigen::VectorXd res;
        type::Variable* Hx_order[100];
        int num_Hx = 0;
        if (!get_feature_jacobian_slam(state, candidate, *feat, cam_models, H_f, H_x, res, Hx_order, num_Hx)) continue;
        if (res.size() == 0) continue;

        const Eigen::MatrixXd R = updater.options_slam.sigma_pix_sq * Eigen::MatrixXd::Identity(res.size(), res.size());
        if (initialize(state, &candidate, Hx_order, num_Hx, H_x, H_f, R, res, updater.options_slam.chi2_multipler)) {
            state.num_slam_features++;
            ++dinit_ok;
        }
    }
}

void update_slam(UpdaterSLAMData& updater, State& state, core::Feature** feature_vec, int feature_count, const core::CameraModel* cam_models) {
    if (feature_count == 0) return;

    double clonetimes[20];
    for (int c = 0; c < state.num_clones; ++c) clonetimes[c] = state.clones_IMU[c].timestamp;

    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(2000, 300);
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(2000);
    int total_rows = 0;

    type::Variable* hx_order_big[150];
    int num_hx_order_big = 0;
    auto add_to_big = [&](type::Variable* var) -> int {
        for (int i = 0; i < num_hx_order_big; ++i) {
            if (hx_order_big[i] == var) return i;
        }
        hx_order_big[num_hx_order_big++] = var;
        return num_hx_order_big - 1;
    };

    for (int i = 0; i < feature_count; ++i) {
        core::Feature* feat = feature_vec[i];
        feat->to_delete = true;

        type::Variable* landmark = nullptr;
        for (int s = 0; s < state.num_slam_features; ++s) {
            if (state.features_SLAM[s].feat_id == feat->featid) {
                landmark = &state.features_SLAM[s];
                break;
            }
        }
        if (!landmark) continue;

        core::clean_old_measurements(*feat, clonetimes, state.num_clones);
        if (feat->num_measurements < 1) continue;

        // Build the Jacobian through the SAME function official uses --
        // get_feature_jacobian_full, whose DOD equivalent is
        // get_feature_jacobian_mixed and is proven bit-exact against it by
        // tests/bitdiff_jac.cpp for both GLOBAL_3D and the anchored
        // inverse-depth representation this actually runs.
        //
        // This replaced get_feature_jacobian_slam(), a separate hand-written
        // implementation with its own variable ordering (anchor clone pushed
        // first rather than in measurement order), its own column layout and
        // its own residual assembly. Official has no such function: it copies
        // the landmark's anchor and position into the feature, calls
        // get_feature_jacobian_full, and appends H_f as extra columns -- which
        // is exactly what happens below. Keeping a bespoke second
        // implementation of an already-verified function was the last place an
        // unverified divergence could hide.
        core::Feature feat_slam = *feat;
        feat_slam.anchor_cam_id = landmark->anchor_cam_id;
        feat_slam.anchor_clone_timestamp = landmark->timestamp; // repurposed: anchor clone ts
        feat_slam.p_FinA = type::get_landmark_xyz(*landmark, false);

        Eigen::MatrixXd H_f, H_x;
        Eigen::VectorXd res;
        type::Variable* Hx_order[100];
        int num_Hx = 0;
        get_feature_jacobian_mixed(state, feat_slam, cam_models, H_f, H_x, res, Hx_order, num_Hx,
                                   state.options.feat_rep_slam);
        if (res.size() == 0 || num_Hx == 0) {
            landmark->update_fail_count++;
            continue;
        }

        // Full order = state vars + this landmark, H_xf = [H_x | H_f].
        type::Variable* Hxf_order[101];
        for (int k = 0; k < num_Hx; ++k) Hxf_order[k] = Hx_order[k];
        Hxf_order[num_Hx] = landmark;
        const int num_Hxf = num_Hx + 1;

        Eigen::MatrixXd H_xf(H_x.rows(), H_x.cols() + H_f.cols());
        H_xf << H_x, H_f;

        // Official: S = H_xf P H_xf^T, S.diagonal() += sigma_pix_sq, then
        // res.dot(S.llt().solve(res)). No symmetrisation, and an LLT rather
        // than an LDLT -- both were additions here.
        Eigen::MatrixXd P_marg = get_marginal_covariance(state, Hxf_order, num_Hxf);
        Eigen::MatrixXd S = H_xf * P_marg * H_xf.transpose();
        S.diagonal() += updater.options_slam.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
        double chi2 = res.dot(S.llt().solve(res));
        const double chi2_check = chi2_ppf_95(static_cast<int>(res.size()));
        if (chi2 > updater.options_slam.chi2_multipler * chi2_check) {
            landmark->update_fail_count++;
            continue;
        }

        const int start_row = total_rows;
        const int num_rows = static_cast<int>(res.size());
        if (total_rows + num_rows > Hx_big.rows()) {
            // Same uninitialized-memory hazard as update_msckf above:
            // conservativeResize does not zero new rows before the "+="
            // accumulation below writes into them.
            Hx_big.conservativeResize(Hx_big.rows() + 1000, Eigen::NoChange);
            Hx_big.bottomRows(1000).setZero();
            res_big.conservativeResize(res_big.size() + 1000);
        }
        res_big.segment(start_row, num_rows) = res;

        int col_offset = 0;
        int var_col_idx[150];
        for (int k = 0; k < num_Hxf; ++k) {
            type::Variable* var = Hxf_order[k];
            const int global_idx = add_to_big(var);
            int running_col = 0;
            for (int c = 0; c < num_hx_order_big; ++c) {
                var_col_idx[c] = running_col;
                running_col += hx_order_big[c]->size;
            }
            if (running_col > Hx_big.cols()) {
                Hx_big.conservativeResize(Eigen::NoChange, Hx_big.cols() + 100);
                Hx_big.rightCols(100).setZero();
            }
            const int target_col = var_col_idx[global_idx];
            Hx_big.block(start_row, target_col, num_rows, var->size) += H_xf.block(0, col_offset, num_rows, var->size);
            col_offset += var->size;
        }
        total_rows += num_rows;
    }

    if (total_rows == 0) return;
    int final_cols = 0;
    for (int c = 0; c < num_hx_order_big; ++c) final_cols += hx_order_big[c]->size;

    Eigen::MatrixXd Hx_valid = Hx_big.topLeftCorner(total_rows, final_cols);
    Eigen::VectorXd res_valid = res_big.head(total_rows);
    Eigen::MatrixXd R_big = updater.options_slam.sigma_pix_sq * Eigen::MatrixXd::Identity(total_rows, total_rows);
    EKFUpdate(state, hx_order_big, num_hx_order_big, Hx_valid, res_valid, R_big);
}

void perform_anchor_change(State& state, type::Variable& landmark, double new_anchor_timestamp, int new_cam_id) {
    Eigen::Matrix3d H_f_old;
    Eigen::Matrix<double, 3, 6> H_x_old_anchor;
    type::Variable* old_anchor_clone = nullptr;
    if (!get_feature_jacobian_representation(state, landmark, H_f_old, H_x_old_anchor, old_anchor_clone)) {
        ++anchor_marg_no_old; // old anchor already gone; drop next frame
        landmark.should_marg = true;
        return;
    }
    const Eigen::Vector3d p_FinA_old = type::get_landmark_xyz(landmark, false);

    const type::Variable& calib_old = state.calib_IMUtoCAM[landmark.anchor_cam_id];
    const Eigen::Matrix3d R_GtoOLD = calib_old.Rot() * old_anchor_clone->Rot();
    const Eigen::Vector3d p_OLDinG = old_anchor_clone->pos() - R_GtoOLD.transpose() * calib_old.pos();

    type::Variable* new_anchor_clone = nullptr;
    for (int c = 0; c < state.num_clones; ++c) {
        if (state.clones_IMU[c].timestamp == new_anchor_timestamp) {
            new_anchor_clone = &state.clones_IMU[c];
            break;
        }
    }
    if (!new_anchor_clone) return;
    const type::Variable& calib_new = state.calib_IMUtoCAM[new_cam_id];
    const Eigen::Matrix3d R_GtoNEW = calib_new.Rot() * new_anchor_clone->Rot();
    const Eigen::Vector3d p_NEWinG = new_anchor_clone->pos() - R_GtoNEW.transpose() * calib_new.pos();

    const Eigen::Matrix3d R_OLDtoNEW = R_GtoNEW * R_GtoOLD.transpose();
    const Eigen::Vector3d p_OLDinNEW = R_GtoNEW * (p_OLDinG - p_NEWinG);
    const Eigen::Vector3d p_FinA_new = R_OLDtoNEW * p_FinA_old + p_OLDinNEW;

    // Reject reanchoring into a degenerate depth: H_f_new's inverse (used
    // below to build Phi) blows up as z->0, and a huge, poorly-conditioned
    // Phi can push the propagated covariance numerically non-PSD even though
    // the transform is exact in infinite precision. Dropping the landmark
    // (should_marg via the caller's failure handling) is safer than
    // corrupting the whole filter's covariance.
    if (!(p_FinA_new.z() > 0.05 && p_FinA_new.z() < 100.0)) {
        ++anchor_marg_depth;
        landmark.should_marg = true;
        return;
    }

    const int old_anchor_cam_id = landmark.anchor_cam_id;
    const double old_timestamp = landmark.timestamp;
    landmark.anchor_cam_id = new_cam_id;
    landmark.timestamp = new_anchor_timestamp;
    type::set_landmark_from_xyz(landmark, p_FinA_new, false);

    Eigen::Matrix3d H_f_new;
    Eigen::Matrix<double, 3, 6> H_x_new_anchor;
    type::Variable* new_anchor_check = nullptr;
    const bool new_ok = get_feature_jacobian_representation(state, landmark, H_f_new, H_x_new_anchor, new_anchor_check);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H_f_new, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const double cond = (new_ok && svd.singularValues()(2) > 1e-9)
                             ? svd.singularValues()(0) / svd.singularValues()(2)
                             : 1e18;
    if (!new_ok || cond > 1e6) {
        // Roll back; the old anchor is about to be marginalized regardless,
        // so drop the landmark next frame rather than leave it dangling.
        landmark.anchor_cam_id = old_anchor_cam_id;
        landmark.timestamp = old_timestamp;
        type::set_landmark_from_xyz(landmark, p_FinA_old, false);
        ++anchor_marg_transform;
        landmark.should_marg = true;
        return;
    }

    const Eigen::Matrix3d H_f_new_inv = H_f_new.inverse();

    type::Variable* all_vars[3];
    int num_all = 0;
    all_vars[num_all++] = old_anchor_clone;
    if (new_anchor_clone != old_anchor_clone) all_vars[num_all++] = new_anchor_clone;
    all_vars[num_all++] = &landmark;

    int col_of[3];
    int total_cols = 0;
    for (int i = 0; i < num_all; ++i) {
        col_of[i] = total_cols;
        total_cols += all_vars[i]->size;
    }
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(3, total_cols);
    for (int i = 0; i < num_all; ++i) {
        if (all_vars[i] == old_anchor_clone) Phi.block<3, 6>(0, col_of[i]) += H_f_new_inv * H_x_old_anchor;
        if (all_vars[i] == &landmark) Phi.block<3, 3>(0, col_of[i]) += H_f_new_inv * H_f_old;
        if (all_vars[i] == new_anchor_clone) Phi.block<3, 6>(0, col_of[i]) -= H_f_new_inv * H_x_new_anchor;
    }

    type::Variable* order_NEW[1] = {&landmark};
    const Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(3, 3);
    EKFPropagation(state, order_NEW, 1, all_vars, num_all, Phi, Q);
}

void change_anchors(UpdaterSLAMData& updater, State& state) {
    (void)updater;
    if (state.num_clones <= state.options.max_clone_size) return;
    const double marg_time = margtimestep(state);
    if (marg_time < 0.0) return;

    for (int s = 0; s < state.num_slam_features; ++s) {
        type::Variable& landmark = state.features_SLAM[s];
        if (landmark.representation == type::LandmarkRepresentation::GLOBAL_3D ||
            landmark.representation == type::LandmarkRepresentation::GLOBAL_FULL_INVERSE_DEPTH) {
            continue;
        }
        if (landmark.timestamp == marg_time) {
            const bool was_marg = landmark.should_marg;
            perform_anchor_change(state, landmark, state.timestamp, landmark.anchor_cam_id);
            if (!was_marg && !landmark.should_marg) ++anchor_change_ok;
        }
    }
}

void init_updater_zupt(UpdaterZeroVelocityData& updater, const PropagatorNoises& noises, double gravity_mag,
                       double zupt_max_velocity, double zupt_noise_multiplier,
                       double zupt_max_disparity, double zupt_chi2_multiplier) {
    updater.noises = noises;
    updater.gravity << 0.0, 0.0, gravity_mag;
    updater.zupt_max_velocity = zupt_max_velocity;
    updater.zupt_noise_multiplier = zupt_noise_multiplier;
    updater.zupt_max_disparity = zupt_max_disparity;
    updater.zupt_chi2_multiplier = zupt_chi2_multiplier;
    updater.last_prop_time_offset = 0.0;
    updater.have_last_prop_time_offset = false;
    updater.last_zupt_state_timestamp = 0.0;
    updater.last_zupt_count = 0;
}

bool try_update_zupt(UpdaterZeroVelocityData& updater, State& state, double timestamp, const core::ImuData* imu_data, int imu_count,
                                     const core::FeatureDatabase& db, bool wait_for_jerk) {
    if (imu_count < 2) return false;
    if (state.timestamp == timestamp) return false;
    
    if (!updater.have_last_prop_time_offset) {
        updater.last_prop_time_offset = state.calib_dt_CAMtoIMU.value[0];
        updater.have_last_prop_time_offset = true;
    }
    
    double t_off_new = state.calib_dt_CAMtoIMU.value[0];
    double time0 = state.timestamp + updater.last_prop_time_offset;
    double time1 = timestamp + t_off_new;
    
    core::ImuData imu_recent[1000];
    int recent_count = select_imu_readings(imu_data, imu_count, time0, time1, imu_recent, 1000);
    
    updater.last_prop_time_offset = t_off_new;
    if (recent_count < 2) return false;
    
    // Set up ZUPT EKF Jacobians (H) and residual (res)
    int m_size = 6 * (recent_count - 1);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m_size, 9);
    Eigen::VectorXd res = Eigen::VectorXd::Zero(m_size);
    
    Eigen::Matrix3d Dw = Dm(state.options.imu_model, state.calib_imu_dw.value);
    Eigen::Matrix3d Da = Dm(state.options.imu_model, state.calib_imu_da.value);
    Eigen::Matrix3d Tg = msckf::Tg(state.calib_imu_tg.value);
    
    double dt_summed = 0.0;
    int valid_count = 0;
    
    for (int i = 0; i < recent_count - 1; ++i) {
        double dt = imu_recent[i+1].timestamp - imu_recent[i].timestamp;
        if (dt <= 0.0) continue;
        
        Eigen::Vector3d a_hat = state.calib_imu_ACCtoIMU.Rot() * Da * (imu_recent[i].am - state.imu.bias_a());
        Eigen::Vector3d w_hat = state.calib_imu_GYROtoIMU.Rot() * Dw * (imu_recent[i].wm - state.imu.bias_g() - Tg * a_hat);
        
        double w_omega = std::sqrt(dt) / updater.noises.sigma_w;
        double w_accel = std::sqrt(dt) / updater.noises.sigma_a;
        
        int row = 6 * valid_count;
        
        Eigen::Matrix3d R_GtoI = state.imu.Rot();
        Eigen::Matrix3d R_jacob = state.options.do_fej ? state.imu.Rot_fej() : R_GtoI;
        
        res.segment<3>(row) = -w_omega * w_hat;
        res.segment<3>(row + 3) = -w_accel * (a_hat - R_GtoI * updater.gravity);
        
        H.block<3, 3>(row, 3) = -w_omega * Eigen::Matrix3d::Identity(); // w.r.t bg
        H.block<3, 3>(row + 3, 0) = -w_accel * type::skew_x(R_jacob * updater.gravity); // w.r.t theta
        H.block<3, 3>(row + 3, 6) = -w_accel * Eigen::Matrix3d::Identity(); // w.r.t ba
        
        dt_summed += dt;
        valid_count++;
    }
    
    if (valid_count == 0) return false;
    
    int actual_rows = 6 * valid_count;
    Eigen::MatrixXd H_valid = H.topRows(actual_rows);
    Eigen::VectorXd res_valid = res.head(actual_rows);
    
    measurement_compress_inplace(H_valid, res_valid);

    if (H_valid.rows() < 1) return false;

    Eigen::MatrixXd R = updater.zupt_noise_multiplier * Eigen::MatrixXd::Identity(res_valid.size(), res_valid.size());
    
    Eigen::Matrix<double, 6, 6> Q_bias = Eigen::Matrix<double, 6, 6>::Identity();
    Q_bias.block<3, 3>(0, 0) *= dt_summed * updater.noises.sigma_wb_2;
    Q_bias.block<3, 3>(3, 3) *= dt_summed * updater.noises.sigma_ab_2;
    
    type::Variable* Hx_order[3];
    // ZUPT variables: imu (which has id starting at 0, size 15)
    // Wait, Hx_order in Python has: [imu.q(), imu.bg(), imu.ba()]
    // Since they are all subvariables of state.imu, we can just pass state.imu!
    // But since the Jacobians H are mapped w.r.t the state error vector of IMU:
    // theta (0:3), bg (9:12), ba (12:15).
    // In our variable system, the active variable is state.imu.
    // The EKF covariance contains state.imu.
    // So the Jacobian H has columns corresponding to the IMU state!
    // Let's verify: H has size (M_compressed, 9).
    // The 9 columns correspond to: theta (0:3), bg (3:6), ba (6:9).
    // But EKF state has size 15 for IMU: theta (0:3), pos (3:6), vel (6:9), bg (9:12), ba (12:15).
    // So we must expand H to map to the full 15 error states of IMU!
    // Specifically:
    // H_full has size (M_compressed, 15).
    // Columns of H_full:
    // 0:3 -> H.block(0, 0) (theta)
    // 3:6 -> 0 (pos)
    // 6:9 -> 0 (vel)
    // 9:12 -> H.block(0, 3) (bg)
    // 12:15 -> H.block(0, 6) (ba).
    // This is a beautiful and critical correction!
    Eigen::MatrixXd H_full = Eigen::MatrixXd::Zero(res_valid.size(), 15);
    H_full.leftCols<3>() = H_valid.leftCols<3>();
    H_full.block(0, 9, H_valid.rows(), 3) = H_valid.block(0, 3, H_valid.rows(), 3);
    H_full.block(0, 12, H_valid.rows(), 3) = H_valid.block(0, 6, H_valid.rows(), 3);
    
    type::Variable* Hx_order_full[1] = { &state.imu };
    Eigen::MatrixXd P_marg = get_marginal_covariance(state, Hx_order_full, 1);
    
    // Add bias random walk noise to IMU covariance block directly
    P_marg.block<6, 6>(9, 9) += Q_bias;
    
    Eigen::MatrixXd S = H_full * P_marg * H_full.transpose() + R;
    double chi2 = 99999.0;
    Eigen::LDLT<Eigen::MatrixXd> solver(S);
    if (solver.info() == Eigen::Success) {
        chi2 = res_valid.dot(solver.solve(res_valid));
    }
    
    double chi2_check = chi2_ppf_95(res_valid.size());
    double vel_norm = state.imu.vel().norm();
    
    // Check disparity
    bool disparity_passed = false;
    double disp_avg = 0.0, disp_var = 0.0;
    int num_features = 0;
    compute_disparity_two_frames(db, state.timestamp, timestamp, disp_avg, disp_var, num_features);
    if (disp_avg != -1.0 && disp_avg < updater.zupt_max_disparity && num_features > 20) {
        disparity_passed = true;
    }
    
    if (!disparity_passed && (chi2 > updater.zupt_chi2_multiplier * chi2_check || vel_norm > updater.zupt_max_velocity)) {
        updater.last_zupt_state_timestamp = 0.0;
        updater.last_zupt_count = 0;
        return false;
    }
    
    
    // Random walk propagation of biases in EKF state covariance
    // Phi_bias is Identity, Qd_bias is Q_bias
    type::Variable* Phi_order[2] = { &state.calib_imu_dw, &state.calib_imu_da }; // wait! Bias propagation is for bg and ba, which are inside state.imu!
    // In state.Cov: bg starts at state.imu.id + 9, ba starts at state.imu.id + 12.
    // So we can directly add Q_bias to state.Cov at the indices of bg and ba!
    // This is incredibly clean and matches propagating biases via random walk!
    int bg_cov_idx = state.imu.id + 9;
    int ba_cov_idx = state.imu.id + 12;
    state.Cov.block<3, 3>(bg_cov_idx, bg_cov_idx) += Q_bias.block<3, 3>(0, 0);
    state.Cov.block<3, 3>(ba_cov_idx, ba_cov_idx) += Q_bias.block<3, 3>(3, 3);
    
    EKFUpdate(state, Hx_order_full, 1, H_full, res_valid, R);
    state.timestamp = timestamp;
    return true;
}

} // namespace msckf
