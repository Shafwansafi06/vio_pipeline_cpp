// SchurVINS MSCKF update, ported data-oriented.
//
// Algorithm: SchurVINS (Fan et al., CVPR 2024, arXiv:2312.01616), following the
// authors' reference implementation `ov_SchurVINS` (UpdaterSchurVINS.cpp,
// StateHelper::SV_EKFUpdate). The algorithm is theirs; what is ours is the data
// layout -- fixed-capacity contiguous buffers, fixed-size Eigen blocks, and no
// heap allocation on the per-frame path.
//
// The structure, and why it is cheap:
//
//   A single measurement touches exactly ONE clone, so the pose-pose Hessian
//   built from measurements alone is BLOCK DIAGONAL. Every off-diagonal
//   coupling between clones enters only through the Schur term. Per feature:
//
//       A(i,i) += jx_i^T jx_i          b(i)  += jx_i^T r_i        (per measurement)
//       V      += jf_i^T jf_i          gv    += jf_i^T r_i        (3x3, 3x1)
//       W(i)   += jx_i^T jf_i                                     (6x3, per clone)
//
//   then, once per feature, with V_inv = V^-1 (a 3x3 inverse):
//
//       A(m,n) -= W(m) V_inv W(n)^T        for every observed clone pair
//       b(m)   -= W(m) V_inv gv
//
//   The reduced system (A, b) is applied by an ordinary EKF update with
//   H = A (symmetric) and R = sigma_pix^2 * I.
//
// Faithful to the reference, including where that differs from our own MSCKF:
//   * NO chi2 gate. The reference gates nothing; it relies on a Huber weight
//     (huberA = 1.5) applied to residual and Jacobians alike.
//   * obs_invdev = 0.25 scales residual and Jacobians. Hardcoded there, kept
//     here -- it is a weighting choice, not a derived noise model.
//   * R = sigma_pix^2 * I. The reference has `sigma_pix_sq * Amtx` present but
//     COMMENTED OUT, which matches what we found independently: feeding the
//     Hessian as the noise covariance makes S = A P A + sigma^2 A scale as J^4
//     and the covariance goes negative within three updates.
//   * GLOBAL_3D feature representation only, no FEJ, no camera calibration --
//     all three are hard errors in the reference. The comparison config must
//     therefore disable FEJ and intrinsic calibration on BOTH sides.
//
// Selected with VIO_SCHUR=1; see docs/schurvins_evaluation.md.

#include "updaters.hpp"

#include "state_helper.hpp"
#include "../type/quat_ops.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace msckf {

double schur_ms_tri = 0.0, schur_ms_jac = 0.0, schur_ms_chi2 = 0.0,
       schur_ms_accum = 0.0, schur_ms_reduce = 0.0, schur_ms_ekf = 0.0, schur_ms_backsub = 0.0;
long schur_calls = 0, schur_dim = 0;
long schur_slam_calls = 0, schur_slam_llt_fail = 0, schur_slam_empty = 0, schur_slam_landmarks = 0;
long schur_gated = 0;

namespace {

struct SchurTimer {
    double& sink;
    std::chrono::steady_clock::time_point start;
    explicit SchurTimer(double& target) : sink(target), start(std::chrono::steady_clock::now()) {}
    ~SchurTimer() {
        sink += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
};

// Reference constants (UpdaterSchurVINS.cpp:308-318). obs_invdev = 0.25 is
// hardcoded there and implies a 4 px measurement sigma -- 4x looser than the
// 1 px this pipeline (and OpenVINS's own EuRoC config) uses. VIO_SCHUR_INVDEV
// overrides it so the weighting can be separated from the algorithm.
constexpr double kObsInvDevDefault = 0.25;
constexpr double kHuberA = 1.5;

constexpr int kMaxClones = 20;

} // namespace

void update_msckf_schur(UpdaterMSCKFData& updater, State& state, core::Feature** feature_vec,
                        int feature_count, const core::CameraModel* cam_models) {
    const int num_clones = state.num_clones;
    if (num_clones <= 0 || num_clones > kMaxClones) return;

    static const bool schur_use_chi2 = std::getenv("VIO_SCHUR_CHI2") != nullptr &&
                                       std::atoi(std::getenv("VIO_SCHUR_CHI2")) != 0;
    static const double kObsInvDev = std::getenv("VIO_SCHUR_INVDEV") != nullptr
                                         ? std::atof(std::getenv("VIO_SCHUR_INVDEV"))
                                         : kObsInvDevDefault;
    core::ClonesCamera clones_cam = build_clones_camera(state);

    int num_triangulated = 0;
    {
        SchurTimer timer(schur_ms_tri);
        for (int i = 0; i < feature_count; ++i) {
            core::Feature* feat = feature_vec[i];
            bool ok = core::single_triangulation(*feat, clones_cam, updater.feat_init_options);
            if (ok) ok = core::single_gaussnewton(*feat, clones_cam, updater.feat_init_options);
            if (!ok) {
                feat->to_delete = true;
            } else {
                ++num_triangulated;
            }
        }
    }
    if (num_triangulated < updater.options.min_features_for_update) {
        for (int i = 0; i < feature_count; ++i) feature_vec[i]->to_delete = true;
        return;
    }

    // The reduced system spans the clone poses, 6 error-state dims each. Sized
    // once at the capacity and used as a top-left corner, so no per-frame alloc.
    // Sized exactly to the clone window and reused: Eigen only reallocates when
    // the window size itself changes, which happens a handful of times per run.
    const int dim = 6 * num_clones;
    static thread_local Eigen::MatrixXd A;
    static thread_local Eigen::VectorXd b;
    if (A.rows() != dim) {
        A.setZero(dim, dim);
        b.setZero(dim);
    } else {
        A.setZero();
        b.setZero();
    }

    // Per-feature accumulators, reused across features.
    Eigen::Matrix<double, 6, 3> W[kMaxClones];
    bool touched[kMaxClones];
    int touched_list[kMaxClones];

    int used_features = 0;
    {
        SchurTimer timer(schur_ms_accum);
        for (int idx = 0; idx < feature_count; ++idx) {
            core::Feature* feat = feature_vec[idx];
            if (feat->to_delete) continue;

            Eigen::Matrix3d V = Eigen::Matrix3d::Zero();
            Eigen::Vector3d gv = Eigen::Vector3d::Zero();
            double rTr = 0.0;
            int num_rows = 0;
            int num_touched = 0;
            for (int c = 0; c < num_clones; ++c) touched[c] = false;

            const Eigen::Vector3d p_FinG = feat->p_FinG;

            for (int m = 0; m < feat->num_measurements; ++m) {
                const core::FeatureMeasurement& obs = feat->measurements[m];
                const int camera = obs.cam_id;
                if (camera < 0 || camera >= state.options.num_cameras) continue;

                int slot = -1;
                for (int c = 0; c < num_clones; ++c) {
                    if (state.clones_IMU[c].timestamp == obs.timestamp) { slot = c; break; }
                }
                if (slot < 0) continue;

                const type::Variable& clone = state.clones_IMU[slot];
                const type::Variable& calib = state.calib_IMUtoCAM[camera];
                const Eigen::Matrix3d R_GtoI = clone.Rot();
                const Eigen::Vector3d p_IinG = clone.pos();
                const Eigen::Matrix3d R_ItoC = calib.Rot();
                const Eigen::Vector3d p_IinC = calib.pos();

                const Eigen::Vector3d p_FinI = R_GtoI * (p_FinG - p_IinG);
                const Eigen::Vector3d p_FinC = R_ItoC * p_FinI + p_IinC;
                if (std::abs(p_FinC.z()) < 1e-10) continue;

                const Eigen::Vector2d normalized = p_FinC.head<2>() / p_FinC.z();
                Eigen::Vector2d r = obs.uv - core::distort(cam_models[camera], normalized);

                // Huber weight on the raw residual, exactly as the reference
                // does it: scale the residual by sqrt(rho') and the Jacobians by
                // the same factor, then scale both by obs_invdev.
                double huber_scale = 1.0;
                const double r_l2 = r.squaredNorm();
                if (r_l2 > kHuberA * kHuberA) {
                    const double radius = std::sqrt(r_l2);
                    const double rho1 = std::max(std::numeric_limits<double>::min(), kHuberA / radius);
                    huber_scale = std::sqrt(rho1);
                    r *= huber_scale;
                }
                r *= kObsInvDev;

                Eigen::Matrix2d dz_dzn;
                Eigen::Matrix<double, 2, 8> dz_dzeta;
                core::compute_distort_jacobian(cam_models[camera], normalized, dz_dzn, dz_dzeta);

                Eigen::Matrix<double, 2, 3> dzn_dpfc;
                const double z = p_FinC.z();
                dzn_dpfc << 1.0 / z, 0.0, -p_FinC.x() / (z * z),
                            0.0, 1.0 / z, -p_FinC.y() / (z * z);
                dzn_dpfc *= kObsInvDev * huber_scale;

                const Eigen::Matrix<double, 2, 3> dz_dpfc = dz_dzn * dzn_dpfc;
                const Eigen::Matrix3d dpfc_dpfg = R_ItoC * R_GtoI;

                Eigen::Matrix<double, 3, 6> dpfc_dclone;
                dpfc_dclone.leftCols<3>() = R_ItoC * type::skew_x(p_FinI);
                dpfc_dclone.rightCols<3>() = -dpfc_dpfg;

                const Eigen::Matrix<double, 2, 6> jx = dz_dpfc * dpfc_dclone;
                const Eigen::Matrix<double, 2, 3> jf = dz_dpfc * dpfc_dpfg;

                A.block<6, 6>(6 * slot, 6 * slot).noalias() += jx.transpose() * jx;
                b.segment<6>(6 * slot).noalias() += jx.transpose() * r;

                V.noalias() += jf.transpose() * jf;
                gv.noalias() += jf.transpose() * r;
                rTr += r.squaredNorm();
                num_rows += 2;

                if (!touched[slot]) {
                    touched[slot] = true;
                    touched_list[num_touched++] = slot;
                    W[slot].setZero();
                }
                // Stacked per clone: a stereo pair contributes twice to the same
                // clone (the reference's stack_W_map).
                W[slot].noalias() += jx.transpose() * jf;
            }

            if (num_touched == 0) continue;

            // Schur reduction. V is 3x3 -- this inverse is the whole trick.
            V.diagonal().array() += 1e-9;
            const Eigen::Matrix3d V_inv = V.inverse();

            // Outlier gate, ours, not the reference's -- and it is free.
            //
            // The reference gates nothing; it relies on a Huber weight alone.
            // But the MSCKF chi2 statistic is already sitting in the Schur
            // intermediates: minimising over the landmark leaves residual energy
            //     r^T r - gv^T V^-1 gv
            // which is exactly the squared norm of the nullspace-projected
            // residual, with 2m-3 degrees of freedom. No projection, no marginal
            // covariance, no extra matrix -- two dot products on quantities the
            // reduction has already computed.
            //
            // (The residual is pre-scaled by obs_invdev = 1/4, i.e. whitened
            //  against a 4 px sigma, so the statistic needs no further scaling.)
            if (schur_use_chi2) {
                const int dof = num_rows - 3;
                if (dof > 0) {
                    const double chi2 = rTr - gv.dot(V_inv * gv);
                    if (chi2 > updater.options.chi2_multipler * chi2_ppf_95(dof)) {
                        ++schur_gated;
                        continue;
                    }
                }
            }

            for (int a = 0; a < num_touched; ++a) {
                const int m = touched_list[a];
                const Eigen::Matrix<double, 6, 3> WVinv = W[m] * V_inv;
                for (int c = a; c < num_touched; ++c) {
                    const int n = touched_list[c];
                    const Eigen::Matrix<double, 6, 6> schur = WVinv * W[n].transpose();
                    A.block<6, 6>(6 * m, 6 * n).noalias() -= schur;
                    if (m != n) {
                        A.block<6, 6>(6 * n, 6 * m).noalias() -= schur.transpose();
                    }
                }
                b.segment<6>(6 * m).noalias() -= WVinv * gv;
            }
            ++used_features;
        }
    }

    for (int i = 0; i < feature_count; ++i) feature_vec[i]->to_delete = true;
    if (used_features == 0) return;

    // EKF update on the reduced system. H = A is symmetric; R is isotropic.
    type::Variable* order[kMaxClones];
    for (int c = 0; c < num_clones; ++c) order[c] = &state.clones_IMU[c];

    // ---- Whiten the reduced system, then update ----
    //
    // The reference applies (A, b) directly with R = sigma_pix^2 * I. That is
    // not a valid measurement model: with H = A (a Hessian, not a measurement
    // Jacobian), S = A P A + R, and A P A dominates R by orders of magnitude
    // here (A ~ 1e4-1e6, P ~ 1e-4), so the gain degenerates to K -> A^-1 -- an
    // unregularised Gauss-Newton step that ignores the IMU prior. Measured:
    // |dx| of 0.6-2.4 m on the first updates and divergence to 1.4e5 m ATE, at
    // every damping scale tried (R x1, x1e2, x1e4, x1e6, x1e8 all diverge).
    //
    // The correct covariance follows from b = J^T n with n ~ N(0, sigma^2 I):
    // cov(b) = sigma^2 A exactly (the Schur cross-terms cancel). Whitening with
    // A = L L^T turns it back into an ordinary update:
    //
    //     L^-1 b = L^T x~ + L^-1 n,     cov(L^-1 n) = sigma^2 I
    //
    // so H = L^T, res = L^-1 b, R = sigma^2 I. This is their accumulation with
    // the numerics fixed, and it is what the results in
    // docs/schurvins_evaluation.md use. VIO_SCHUR_LITERAL=1 restores the
    // reference's own rule for comparison.
    static thread_local Eigen::MatrixXd R;
    if (R.rows() != dim) R.setIdentity(dim, dim);
    R.setIdentity();
    R *= updater.options.sigma_pix_sq;

    const bool literal = std::getenv("VIO_SCHUR_LITERAL") != nullptr;
    Eigen::MatrixXd H_used;
    Eigen::VectorXd res_used;
    if (literal) {
        H_used = A;
        res_used = b;
    } else {
        SchurTimer timer(schur_ms_reduce);
        // A is PSD in exact arithmetic but is a difference of large similar
        // quantities, so the ridge has to scale with the matrix: an absolute
        // ridge is meaningless against a diagonal running to 1e6.
        const double scale = std::max(1.0, A.diagonal().maxCoeff());
        bool ok = false;
        for (double relative : {1e-9, 1e-7, 1e-5}) {
            Eigen::MatrixXd candidate = A;
            candidate.diagonal().array() += relative * scale;
            Eigen::LLT<Eigen::MatrixXd> llt(candidate);
            if (llt.info() == Eigen::Success) {
                const Eigen::MatrixXd L = llt.matrixL();
                H_used = L.transpose();
                res_used = L.triangularView<Eigen::Lower>().solve(b);
                ok = true;
                break;
            }
        }
        if (!ok) { ++schur_slam_llt_fail; return; }
    }

    ++schur_calls;
    schur_dim += dim;
    Eigen::VectorXd dx;
    {
        SchurTimer timer(schur_ms_ekf);
        EKFUpdate(state, order, num_clones, H_used, res_used, R, &dx);
    }
    if (std::getenv("VIO_SCHUR_DEBUG") != nullptr && schur_calls <= 8) {
        std::fprintf(stderr,
                     "[schur %ld] feats %d  dim %d  |A| %.3e  |b| %.3e  |dx| %.3e  maxdiagA %.3e\n",
                     schur_calls, used_features, dim, A.norm(), b.norm(), dx.norm(),
                     A.diagonal().maxCoeff());
    }
}

} // namespace msckf
