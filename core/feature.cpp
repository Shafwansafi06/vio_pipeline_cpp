#include "feature.hpp"
#include "../type/quat_ops.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

namespace core {

long dbg_max_meas = 0, dbg_max_count = 0, dbg_shift_elems = 0, dbg_compact_elems = 0;

// Triangulation rejection counters (see single_triangulation). Reported by the
// runners; zero-cost to leave in, and the alternative is guessing.
long tri_reject_cond = 0;
long tri_reject_mindist = 0;
long tri_reject_maxdist = 0;
long tri_reject_nan = 0;
long tri_accept = 0;
long db_full_refusals = 0;
long feat_meas_overflow = 0;
long tri_reject_cond_meas = 0;
long tri_accept_meas = 0;
long tri_cond_hist[32] = {0};
long tri_accept_hist[32] = {0};
long tri_depth_hist[20] = {0};
double tri_depth_sum = 0.0;
long epi_computed = 0;
long epi_no_baseline = 0;
long epi_score_hist[20] = {0};
long gn_reject_dist = 0;
long gn_reject_baseline = 0;
long gn_accept = 0;
// parallax_noise_scale() accounting. Every early-out has its own counter so a
// silently saturated weight cannot hide behind the mean.
long pnw_computed = 0;
long pnw_no_depth = 0;
long pnw_no_baseline = 0;
double pnw_scale_sum = 0.0;
double pnw_scale_max = 0.0;

// Official iterates features' per-camera measurement lists through an
// unordered_map<size_t, ...> keyed by camera id. With ids inserted 0 then 1,
// libstdc++ walks them in reverse: camera 1 first. Every accumulation order and
// the anchor tie-break inherit that, so DOD has to walk them the same way.
// These names exist so the intent survives; do not "fix" them to ascending.
static constexpr int OV_CAM_FIRST = 1;
static constexpr int OV_CAM_END = -1;
static constexpr int OV_CAM_STEP = -1;

const ClonePose* find_clone_pose(const ClonesCamera& clones, int cam_id, double ts) {
    for (int i = 0; i < clones.num_clones; ++i) {
        if (clones.timestamps[i] == ts) {
            return &clones.poses[cam_id][i];
        }
    }
    return nullptr;
}

double epipolar_consistency_score(const Feature& feat, const ClonesCamera& clones) {
    int oldest_idx = -1, newest_idx = -1;
    double oldest_ts = std::numeric_limits<double>::infinity();
    double newest_ts = -std::numeric_limits<double>::infinity();
    for (int m = 0; m < feat.num_measurements; ++m) {
        if (feat.measurements[m].cam_id != 0) continue;
        const double ts = feat.measurements[m].timestamp;
        if (ts < oldest_ts) { oldest_ts = ts; oldest_idx = m; }
        if (ts > newest_ts) { newest_ts = ts; newest_idx = m; }
    }
    if (oldest_idx < 0 || newest_idx < 0 || oldest_idx == newest_idx) return -1.0;

    const ClonePose* pose_i = find_clone_pose(clones, 0, oldest_ts);
    const ClonePose* pose_j = find_clone_pose(clones, 0, newest_ts);
    if (pose_i == nullptr || pose_j == nullptr) return -1.0;

    Eigen::Vector3d bi(feat.measurements[oldest_idx].uv_norm[0], feat.measurements[oldest_idx].uv_norm[1], 1.0);
    Eigen::Vector3d bj(feat.measurements[newest_idx].uv_norm[0], feat.measurements[newest_idx].uv_norm[1], 1.0);
    bi.normalize();
    bj.normalize();

    // Bearings rotated camera -> global. ClonesCamera::poses already stores
    // GLOBAL camera poses with extrinsics folded in, so unlike the
    // featureless initializer's I0-frame version of this identity, there is
    // no separate IMU-frame step: pose->R is R_GtoC directly.
    const Eigen::Vector3d b_i_G = pose_i->R.transpose() * bi;
    const Eigen::Vector3d b_j_G = pose_j->R.transpose() * bj;

    const Eigen::Vector3d n = b_i_G.cross(b_j_G);
    const double n_norm = n.norm();
    if (n_norm < 1e-9) { ++epi_no_baseline; return -1.0; }  // parallel bearings: undefined plane
    const Eigen::Vector3d n_hat = n / n_norm;

    const Eigen::Vector3d t = pose_j->p - pose_i->p;
    const double t_norm = t.norm();
    if (t_norm < 1e-9) { ++epi_no_baseline; return -1.0; }  // clones effectively co-located
    const Eigen::Vector3d t_hat = t / t_norm;

    const double score = std::abs(n_hat.dot(t_hat));
    ++epi_computed;
    int bucket = int(score * 20.0);
    if (bucket < 0) bucket = 0;
    if (bucket > 19) bucket = 19;
    ++epi_score_hist[bucket];
    return score;
}

// Parallax-limited measurement noise scale (see feature.hpp for the model).
//
// Derivation. A stereo pair with baseline b and focal length f recovers depth
// as Z = f*b/d, so a disparity error sigma_d propagates to
//
//     sigma_Z = Z^2 * sigma_d / (f*b).                                    (1)
//
// That is the Z^2 growth that makes high-altitude stereo weak. On its own it
// is NOT a reason to inflate R: re-projecting sigma_Z into a clone displaced
// by a translational baseline B gives a pixel shift f*B*sigma_Z/Z^2 =
// B*sigma_d/b, in which Z cancels. The filter already sees that through the
// Jacobians. What the filter does NOT see is that (1) is a first-order
// propagation, and the triangulation it linearises about is only trustworthy
// while the depth-to-parallax ratio stays small. The quantity that actually
// degrades is
//
//     rho = Z / B,       B = widest translational baseline that saw the feature,
//
// and the extra pixel-domain uncertainty from linearising at a badly
// conditioned triangulation is modelled here as proportional to rho:
//
//     sigma_eff^2 = sigma_pix^2 * (1 + (lambda*rho)^2).                   (2)
//
// lambda is dimensionless and is the only tuned quantity; lambda = 0 disables
// the model exactly (the returned 1.0 makes the caller's division a division
// by 1.0, which is exact in IEEE-754, so trajectories stay bit-identical).
// The result is clamped to max_scale so a near-zero baseline cannot produce an
// unbounded weight.
//
// NOTE the model is written in Z/B, not in altitude. An altitude threshold
// would be a constant tuned on one dataset (CLAUDE.md, rule 8.5); Z/B is the
// geometry itself and needs no per-dataset value.
double parallax_noise_scale(const Feature& feat, const ClonesCamera& clones,
                            double lambda, double max_scale) {
    if (lambda <= 0.0) return 1.0;

    // Depth of the linearisation point in its anchor camera. p_FinA is written
    // by single_triangulation/single_gaussnewton immediately before every
    // caller of this function.
    const double Z = feat.p_FinA.z();
    if (!(Z > 0.0)) { ++pnw_no_depth; return max_scale; }

    // Widest translational baseline among the clones that observed it. O(n^2)
    // over at most FEATURE_MAX_MEASUREMENTS rows with the clone lookup hoisted
    // out; the alternative (an axis-aligned extent) underestimates the spread.
    Eigen::Vector3d pos[FEATURE_MAX_MEASUREMENTS];
    int n_pos = 0;
    for (int m = 0; m < feat.num_measurements; ++m) {
        const ClonePose* pose = find_clone_pose(clones, feat.measurements[m].cam_id,
                                                feat.measurements[m].timestamp);
        if (pose == nullptr) continue;
        pos[n_pos++] = pose->p;
    }
    double B_sq = 0.0;
    for (int i = 0; i < n_pos; ++i) {
        for (int j = i + 1; j < n_pos; ++j) {
            const double d_sq = (pos[i] - pos[j]).squaredNorm();
            if (d_sq > B_sq) B_sq = d_sq;
        }
    }
    if (!(B_sq > 0.0)) { ++pnw_no_baseline; return max_scale; }

    const double rho_sq = (Z * Z) / B_sq;
    double scale = 1.0 + lambda * lambda * rho_sq;
    if (scale > max_scale) scale = max_scale;

    ++pnw_computed;
    pnw_scale_sum += scale;
    if (scale > pnw_scale_max) pnw_scale_max = scale;
    return scale;
}

int find_feature_index(const FeatureDatabase& db, int feat_id) {
    int low = 0;
    int high = (int)db.count - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        const int mid_id = db.features[db.order[mid]].featid;
        if (mid_id == feat_id) {
            return mid;
        } else if (mid_id < feat_id) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return -1;
}

Feature* get_feature(FeatureDatabase& db, int feat_id) {
    int idx = find_feature_index(db, feat_id);
    if (idx != -1) {
        return &feature_at(db, idx);
    }
    return nullptr;
}

bool get_feature_clone(const FeatureDatabase& db, int feat_id, Feature& feat_out) {
    int idx = find_feature_index(db, feat_id);
    if (idx == -1) {
        return false;
    }
    feat_out = feature_at(db, idx);
    return true;
}

void update_feature(FeatureDatabase& db, int feat_id, double timestamp, int cam_id, double u, double v, double u_n, double v_n) {
    int idx = find_feature_index(db, feat_id);
    if (idx != -1) {
        Feature& feat = feature_at(db, idx);
        if (feat.num_measurements < FEATURE_MAX_MEASUREMENTS) {
            feat.measurements[feat.num_measurements++] = {cam_id, timestamp, Eigen::Vector2d(u, v), Eigen::Vector2d(u_n, v_n)};
            if (feat.num_measurements > dbg_max_meas) dbg_max_meas = feat.num_measurements;
        } else {
            // The array is full and this observation is thrown away. Measured
            // nonzero on both KAIST sequences -- see the sizing note on
            // FEATURE_MAX_MEASUREMENTS for why, and why the fix is the
            // initialisation latency rather than a bigger array.
            ++feat_meas_overflow;
        }
        return;
    }
    
    if (db.count >= FEATURE_DB_CAPACITY) {
        // Silently refusing new features once full is not the same as
        // official, whose database is an unbounded map pruned by age. If this
        // fires the tracker is effectively capped and tracks cannot be
        // replaced, so it must be visible rather than silent.
        ++db_full_refusals;
        return;
    }
    
    int insert_idx = 0;
    while (insert_idx < (int)db.count && db.features[db.order[insert_idx]].featid < feat_id) {
        insert_idx++;
    }

    // Take a slot: recycled if one is free, otherwise the next fresh one. The
    // payload stays where it is for its whole life.
    int slot;
    if (db.num_free > 0) {
        slot = db.free_slots[--db.num_free];
    } else {
        if (db.num_slots_used >= FEATURE_DB_CAPACITY) { ++db_full_refusals; return; }
        slot = db.num_slots_used++;
    }

    dbg_shift_elems += (long)db.count - insert_idx;
    for (int i = (int)db.count; i > insert_idx; --i) {
        db.order[i] = db.order[i - 1];
    }
    db.order[insert_idx] = slot;

    Feature& feat = db.features[slot];
    feat.featid = feat_id;
    feat.to_delete = false;
    feat.num_measurements = 1;
    feat.measurements[0] = {cam_id, timestamp, Eigen::Vector2d(u, v), Eigen::Vector2d(u_n, v_n)};
    feat.p_FinG.setZero();
    feat.p_FinA.setZero();
    feat.anchor_cam_id = -1;
    feat.anchor_clone_timestamp = 0.0;
    
    db.count++;
    if ((long)db.count > dbg_max_count) dbg_max_count = (long)db.count;
}

void cleanup_db(FeatureDatabase& db) {
    dbg_compact_elems += (long)db.count;
    int write_idx = 0;
    for (int i = 0; i < (int)db.count; ++i) {
        const int slot = db.order[i];
        if (!db.features[slot].to_delete) {
            db.order[write_idx++] = slot;
        } else {
            db.free_slots[db.num_free++] = slot;
        }
    }
    db.count = write_idx;
}

void cleanup_db_measurements(FeatureDatabase& db, double timestamp) {
    int write_idx = 0;
    for (int i = 0; i < (int)db.count; ++i) {
        const int slot = db.order[i];
        Feature& feat = db.features[slot];
        clear_older_measurements(feat, timestamp);
        if (feat.num_measurements >= 1) {
            db.order[write_idx++] = slot;
        } else {
            db.free_slots[db.num_free++] = slot;
        }
    }
    db.count = write_idx;
}

void cleanup_db_measurements_exact(FeatureDatabase& db, double timestamp) {
    int write_idx = 0;
    for (int i = 0; i < (int)db.count; ++i) {
        const int slot = db.order[i];
        Feature& feat = db.features[slot];
        clean_invalid_measurements(feat, &timestamp, 1);
        if (feat.num_measurements >= 1) {
            db.order[write_idx++] = slot;
        } else {
            db.free_slots[db.num_free++] = slot;
        }
    }
    db.count = write_idx;
}

double get_oldest_db_timestamp(const FeatureDatabase& db) {
    double oldest_time = -1.0;
    for (int i = 0; i < (int)db.count; ++i) {
        const Feature& feat = feature_at(db, i);
        for (int m = 0; m < feat.num_measurements; ++m) {
            double t = feat.measurements[m].timestamp;
            if (oldest_time == -1.0 || t < oldest_time) {
                oldest_time = t;
            }
        }
    }
    return oldest_time;
}

void clean_old_measurements(Feature& feat, const double* valid_times, int num_valid) {
    int write_idx = 0;
    for (int i = 0; i < feat.num_measurements; ++i) {
        bool is_valid = false;
        for (int k = 0; k < num_valid; ++k) {
            if (feat.measurements[i].timestamp == valid_times[k]) {
                is_valid = true;
                break;
            }
        }
        if (is_valid) {
            feat.measurements[write_idx++] = feat.measurements[i];
        }
    }
    feat.num_measurements = write_idx;
}

void clean_invalid_measurements(Feature& feat, const double* invalid_times, int num_invalid) {
    int write_idx = 0;
    for (int i = 0; i < feat.num_measurements; ++i) {
        bool is_invalid = false;
        for (int k = 0; k < num_invalid; ++k) {
            if (feat.measurements[i].timestamp == invalid_times[k]) {
                is_invalid = true;
                break;
            }
        }
        if (!is_invalid) {
            feat.measurements[write_idx++] = feat.measurements[i];
        }
    }
    feat.num_measurements = write_idx;
}

void clear_older_measurements(Feature& feat, double timestamp) {
    int write_idx = 0;
    for (int i = 0; i < feat.num_measurements; ++i) {
        if (feat.measurements[i].timestamp > timestamp) {
            feat.measurements[write_idx++] = feat.measurements[i];
        }
    }
    feat.num_measurements = write_idx;
}

bool single_triangulation(Feature& feat, const ClonesCamera& clones, const FeatureInitializerOptions& options) {
    // Anchor selection must follow official's, INCLUDING its tie-break.
    //
    // Official loops `for (auto const &pair : feat->timestamps)` -- an
    // unordered_map keyed by camera id -- and takes a new anchor only on a
    // strict `>`. With cam ids inserted 0 then 1, libstdc++ iterates them 1
    // then 0, so on an equal-count tie official keeps camera 1, not camera 0.
    // Picking 0 here (as this did) anchors stereo features to the other camera
    // and changes p_FinA, the residuals, and the Jacobians for every one of
    // them. Verified against the oracle by bitdiff_tri's [6,6] and [3,3] cases.
    int count_cam[2] = {0, 0};
    double latest_ts[2] = {-1.0, -1.0};

    for (int m = 0; m < feat.num_measurements; ++m) {
        int cid = feat.measurements[m].cam_id;
        count_cam[cid]++;
        if (feat.measurements[m].timestamp > latest_ts[cid]) {
            latest_ts[cid] = feat.measurements[m].timestamp;
        }
    }

    int anchor_cam = 0;
    int most_meas = 0;
    for (int cid = OV_CAM_FIRST; cid != OV_CAM_END; cid += OV_CAM_STEP) {
        if (count_cam[cid] > most_meas) {
            anchor_cam = cid;
            most_meas = count_cam[cid];
        }
    }

    if (count_cam[anchor_cam] == 0) {
        return false;
    }
    
    feat.anchor_cam_id = anchor_cam;
    feat.anchor_clone_timestamp = latest_ts[anchor_cam];
    
    const ClonePose* anchor_pose = find_clone_pose(clones, anchor_cam, feat.anchor_clone_timestamp);
    if (!anchor_pose) return false;
    
    Eigen::Matrix3d R_GtoA = anchor_pose->R;
    Eigen::Vector3d p_AinG = anchor_pose->p;
    
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    
    // Summation order follows official's unordered_map iteration (camera 1 then
    // camera 0); a different order sums the same terms to different last bits.
    for (int cid = OV_CAM_FIRST; cid != OV_CAM_END; cid += OV_CAM_STEP)
    for (int m = 0; m < feat.num_measurements; ++m) {
        if (feat.measurements[m].cam_id != cid) continue;
        double ts = feat.measurements[m].timestamp;
        
        const ClonePose* curr_pose = find_clone_pose(clones, cid, ts);
        if (!curr_pose) continue;
        
        Eigen::Matrix3d R_GtoCi = curr_pose->R;
        Eigen::Vector3d p_CiinG = curr_pose->p;
        
        Eigen::Matrix3d R_AtoCi = R_GtoCi * R_GtoA.transpose();
        Eigen::Vector3d p_CiinA = R_GtoA * (p_CiinG - p_AinG);
        
        Eigen::Vector3d b_i(feat.measurements[m].uv_norm[0], feat.measurements[m].uv_norm[1], 1.0);
        b_i = R_AtoCi.transpose() * b_i;
        b_i.normalize();
        
        Eigen::Matrix3d Bperp = type::skew_x(b_i);
        Eigen::Matrix3d Ai = Bperp.transpose() * Bperp;
        A += Ai;
        b += Ai * p_CiinA;
    }
    
    // NOTE: official applies no determinant test here -- it relies solely on the
    // condition-number / depth checks below. The |det(A)| < 1e-12 early return
    // that used to sit here was an invention, and it rejected features official
    // accepts, depressing the MSCKF accept rate.
    Eigen::Vector3d p_f = A.colPivHouseholderQr().solve(b);
    
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(A);
    Eigen::Vector3d S = svd.singularValues();
    double condA = (S[2] == 0.0) ? 1e9 : (S[0] / S[2]);

    // Per-criterion rejection accounting. 47% of features were being rejected
    // here and "triangulation" is four different tests; knowing which one fires
    // is the difference between tuning a threshold and rewriting a frontend.
    if (std::abs(condA) > options.max_cond_number) {
        ++tri_reject_cond;
        tri_reject_cond_meas += feat.num_measurements;
        if (feat.num_measurements < 32) ++tri_cond_hist[feat.num_measurements];
        return false;
    }
    if (p_f[2] < options.min_dist) { ++tri_reject_mindist; return false; }
    if (p_f[2] > options.max_dist) { ++tri_reject_maxdist; return false; }
    if (std::isnan(p_f.norm())) { ++tri_reject_nan; return false; }
    ++tri_accept;
    tri_accept_meas += feat.num_measurements;
    if (feat.num_measurements < 32) ++tri_accept_hist[feat.num_measurements];
    {
        int bucket = static_cast<int>(p_f[2] / 5.0);
        tri_depth_hist[std::clamp(bucket, 0, 19)] += 1;
        tri_depth_sum += p_f[2];
    }

    feat.p_FinA = p_f;
    feat.p_FinG = R_GtoA.transpose() * p_f + p_AinG;
    return true;
}

bool single_triangulation_1d(Feature& feat, const ClonesCamera& clones, const FeatureInitializerOptions& options) {
    int count_cam[2] = {0, 0};
    double latest_ts[2] = {-1.0, -1.0};
    int anchor_meas_idx = -1;
    
    for (int m = 0; m < feat.num_measurements; ++m) {
        int cid = feat.measurements[m].cam_id;
        count_cam[cid]++;
        if (feat.measurements[m].timestamp > latest_ts[cid]) {
            latest_ts[cid] = feat.measurements[m].timestamp;
        }
    }
    
    int anchor_cam = -1;
    if (count_cam[0] > count_cam[1]) {
        anchor_cam = 0;
    } else if (count_cam[1] > count_cam[0]) {
        anchor_cam = 1;
    } else {
        anchor_cam = 0;
    }
    
    if (count_cam[anchor_cam] == 0) return false;
    
    feat.anchor_cam_id = anchor_cam;
    feat.anchor_clone_timestamp = latest_ts[anchor_cam];
    
    for (int m = 0; m < feat.num_measurements; ++m) {
        if (feat.measurements[m].cam_id == anchor_cam && feat.measurements[m].timestamp == feat.anchor_clone_timestamp) {
            anchor_meas_idx = m;
            break;
        }
    }
    
    if (anchor_meas_idx == -1) return false;
    
    const ClonePose* anchor_pose = find_clone_pose(clones, anchor_cam, feat.anchor_clone_timestamp);
    if (!anchor_pose) return false;
    
    Eigen::Matrix3d R_GtoA = anchor_pose->R;
    Eigen::Vector3d p_AinG = anchor_pose->p;
    
    Eigen::Vector3d bearing_inA(feat.measurements[anchor_meas_idx].uv_norm[0],
                                feat.measurements[anchor_meas_idx].uv_norm[1],
                                1.0);
    bearing_inA.normalize();
    
    double A = 0.0;
    double b = 0.0;
    
    for (int m = 0; m < feat.num_measurements; ++m) {
        if (m == anchor_meas_idx) continue;
        
        int cid = feat.measurements[m].cam_id;
        double ts = feat.measurements[m].timestamp;
        
        const ClonePose* curr_pose = find_clone_pose(clones, cid, ts);
        if (!curr_pose) continue;
        
        Eigen::Matrix3d R_GtoCi = curr_pose->R;
        Eigen::Vector3d p_CiinG = curr_pose->p;
        
        Eigen::Matrix3d R_AtoCi = R_GtoCi * R_GtoA.transpose();
        Eigen::Vector3d p_CiinA = R_GtoA * (p_CiinG - p_AinG);
        
        Eigen::Vector3d b_i(feat.measurements[m].uv_norm[0], feat.measurements[m].uv_norm[1], 1.0);
        b_i = R_AtoCi.transpose() * b_i;
        b_i.normalize();
        
        Eigen::Matrix3d Bperp = type::skew_x(b_i);
        Eigen::Vector3d BperpBanchor = Bperp * bearing_inA;
        
        A += BperpBanchor.dot(BperpBanchor);
        b += BperpBanchor.dot(Bperp * p_CiinA);
    }
    
    if (A == 0.0) return false;
    
    double depth = b / A;
    Eigen::Vector3d p_f = depth * bearing_inA;
    
    if (p_f[2] < options.min_dist ||
        p_f[2] > options.max_dist ||
        std::isnan(p_f.norm())) {
        return false;
    }
    
    feat.p_FinA = p_f;
    feat.p_FinG = R_GtoA.transpose() * p_f + p_AinG;
    return true;
}

double compute_error(const ClonesCamera& clones, const Feature& feat, double alpha, double beta, double rho) {
    // Two parity details, both verified by bitdiff_tri:
    //   - Official forms the predicted measurement and the residual in FLOAT32
    //     (Eigen::Matrix<float,2,1>), because uvs_norm is stored as VectorXf.
    //     Doing this in double gives a different cost, which steers the
    //     Gauss-Newton refinement down a different path.
    //   - Official accumulates pow(res.norm(), 2) -- a square root followed by
    //     a square -- not res.squaredNorm(). They differ in the last bits.
    double err = 0;

    const ClonePose* anchor_pose = find_clone_pose(clones, feat.anchor_cam_id, feat.anchor_clone_timestamp);
    if (!anchor_pose) return 1e9;

    const Eigen::Matrix3d& R_GtoA = anchor_pose->R;
    const Eigen::Vector3d& p_AinG = anchor_pose->p;

    for (int cid = OV_CAM_FIRST; cid != OV_CAM_END; cid += OV_CAM_STEP)
    for (int m = 0; m < feat.num_measurements; ++m) {
        if (feat.measurements[m].cam_id != cid) continue;
        const double ts = feat.measurements[m].timestamp;

        const ClonePose* curr_pose = find_clone_pose(clones, cid, ts);
        if (!curr_pose) continue;

        const Eigen::Matrix3d& R_GtoCi = curr_pose->R;
        const Eigen::Vector3d& p_CiinG = curr_pose->p;

        Eigen::Matrix<double, 3, 3> R_AtoCi;
        R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
        Eigen::Matrix<double, 3, 1> p_CiinA;
        p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);
        Eigen::Matrix<double, 3, 1> p_AinCi;
        p_AinCi.noalias() = -R_AtoCi * p_CiinA;

        double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
        double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
        double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);

        Eigen::Matrix<float, 2, 1> z;
        z << hi1 / hi3, hi2 / hi3;
        Eigen::Matrix<float, 2, 1> uv_norm_f;
        uv_norm_f << (float)feat.measurements[m].uv_norm[0], (float)feat.measurements[m].uv_norm[1];
        Eigen::Matrix<float, 2, 1> res = uv_norm_f - z;
        err += std::pow(res.norm(), 2);
    }
    return err;
}

bool single_gaussnewton(Feature& feat, const ClonesCamera& clones, const FeatureInitializerOptions& options) {
    if (feat.p_FinA[2] == 0.0) return false;

    // alpha/beta DIVIDE by p_FinA[2]; they do not multiply by rho. The two are
    // equal in exact arithmetic and differ by 1 ULP in doubles, and that
    // difference is the seed of the whole refinement -- it propagates through
    // every Gauss-Newton iteration.
    double rho = 1 / feat.p_FinA[2];
    double alpha = feat.p_FinA[0] / feat.p_FinA[2];
    double beta = feat.p_FinA[1] / feat.p_FinA[2];
    
    double lam = options.init_lamda;
    double eps = 10000.0;
    int runs = 0;
    
    bool recompute = true;
    Eigen::Matrix3d Hess = Eigen::Matrix3d::Zero();
    Eigen::Vector3d grad = Eigen::Vector3d::Zero();
    
    double cost_old = compute_error(clones, feat, alpha, beta, rho);
    
    const ClonePose* anchor_pose = find_clone_pose(clones, feat.anchor_cam_id, feat.anchor_clone_timestamp);
    if (!anchor_pose) return false;
    
    Eigen::Matrix3d R_GtoA = anchor_pose->R;
    Eigen::Vector3d p_AinG = anchor_pose->p;
    
    while (runs < options.max_runs && lam < options.max_lamda && eps > options.min_dx) {
        if (recompute) {
            Hess.setZero();
            grad.setZero();
            
            for (int cid = OV_CAM_FIRST; cid != OV_CAM_END; cid += OV_CAM_STEP)
            for (int m = 0; m < feat.num_measurements; ++m) {
                if (feat.measurements[m].cam_id != cid) continue;
                double ts = feat.measurements[m].timestamp;
                
                const ClonePose* curr_pose = find_clone_pose(clones, cid, ts);
                if (!curr_pose) continue;
                
                Eigen::Matrix3d R_GtoCi = curr_pose->R;
                Eigen::Vector3d p_CiinG = curr_pose->p;
                
                Eigen::Matrix3d R_AtoCi = R_GtoCi * R_GtoA.transpose();
                Eigen::Vector3d p_CiinA = R_GtoA * (p_CiinG - p_AinG);
                Eigen::Vector3d p_AinCi = -R_AtoCi * p_CiinA;
                
                double hi1 = R_AtoCi(0, 0)*alpha + R_AtoCi(0, 1)*beta + R_AtoCi(0, 2) + rho*p_AinCi[0];
                double hi2 = R_AtoCi(1, 0)*alpha + R_AtoCi(1, 1)*beta + R_AtoCi(1, 2) + rho*p_AinCi[1];
                double hi3 = R_AtoCi(2, 0)*alpha + R_AtoCi(2, 1)*beta + R_AtoCi(2, 2) + rho*p_AinCi[2];
                
                // pow(hi3, 2) rather than hi3*hi3, and a FLOAT residual cast
                // back to double for the gradient -- both are official's, both
                // change the last bits. No hi3 == 0 guard either; official has
                // none, and adding one silently drops a measurement.
                double d_z1_d_alpha = (R_AtoCi(0, 0) * hi3 - hi1 * R_AtoCi(2, 0)) / (std::pow(hi3, 2));
                double d_z1_d_beta = (R_AtoCi(0, 1) * hi3 - hi1 * R_AtoCi(2, 1)) / (std::pow(hi3, 2));
                double d_z1_d_rho = (p_AinCi[0] * hi3 - hi1 * p_AinCi[2]) / (std::pow(hi3, 2));
                double d_z2_d_alpha = (R_AtoCi(1, 0) * hi3 - hi2 * R_AtoCi(2, 0)) / (std::pow(hi3, 2));
                double d_z2_d_beta = (R_AtoCi(1, 1) * hi3 - hi2 * R_AtoCi(2, 1)) / (std::pow(hi3, 2));
                double d_z2_d_rho = (p_AinCi[1] * hi3 - hi2 * p_AinCi[2]) / (std::pow(hi3, 2));

                Eigen::Matrix<double, 2, 3> H;
                H << d_z1_d_alpha, d_z1_d_beta, d_z1_d_rho, d_z2_d_alpha, d_z2_d_beta, d_z2_d_rho;

                Eigen::Matrix<float, 2, 1> z;
                z << hi1 / hi3, hi2 / hi3;
                Eigen::Matrix<float, 2, 1> uv_norm_f;
                uv_norm_f << (float)feat.measurements[m].uv_norm[0], (float)feat.measurements[m].uv_norm[1];
                Eigen::Matrix<float, 2, 1> res = uv_norm_f - z;

                grad.noalias() += H.transpose() * res.cast<double>();
                Hess.noalias() += H.transpose() * H;
            }
        }
        
        Eigen::Matrix3d Hess_l = Hess;
        for (int r = 0; r < 3; ++r) {
            Hess_l(r, r) *= (1.0 + lam);
        }
        
        // Official has no determinant guard here -- it always takes the step and
        // lets the cost comparison below decide. The guard that used to sit
        // here turned well-conditioned iterations into lambda increases.
        Eigen::Vector3d dx = Hess_l.colPivHouseholderQr().solve(grad);
        double cost = compute_error(clones, feat, alpha + dx[0], beta + dx[1], rho + dx[2]);
        
        if (cost <= cost_old && (cost_old - cost) / cost_old < options.min_dcost) {
            alpha += dx[0];
            beta += dx[1];
            rho += dx[2];
            break;
        }
        
        if (cost <= cost_old) {
            recompute = true;
            cost_old = cost;
            alpha += dx[0];
            beta += dx[1];
            rho += dx[2];
            runs++;
            lam /= options.lam_mult;
            eps = dx.norm();
        } else {
            recompute = false;
            lam *= options.lam_mult;
            continue;
        }
    }
    
    if (rho == 0.0) return false;
    
    feat.p_FinA[0] = alpha / rho;
    feat.p_FinA[1] = beta / rho;
    feat.p_FinA[2] = 1.0 / rho;
    
    // Baseline Check
    Eigen::Vector3d p_vec = feat.p_FinA;
    Eigen::Vector3d v1 = p_vec.normalized();
    
    // Gram-Schmidt for tangent basis
    Eigen::Vector3d u(1.0, 0.0, 0.0);
    if (std::abs(v1[0]) > 0.9) {
        u = Eigen::Vector3d(0.0, 1.0, 0.0);
    }
    Eigen::Vector3d v2 = (u - v1.dot(u) * v1).normalized();
    Eigen::Vector3d v3 = v1.cross(v2);
    
    Eigen::Matrix<double, 3, 2> Q_tangent;
    Q_tangent.col(0) = v2;
    Q_tangent.col(1) = v3;
    
    double base_line_max = 0.0;
    
    for (int m = 0; m < feat.num_measurements; ++m) {
        int cid = feat.measurements[m].cam_id;
        double ts = feat.measurements[m].timestamp;
        
        const ClonePose* curr_pose = find_clone_pose(clones, cid, ts);
        if (!curr_pose) continue;
        
        Eigen::Vector3d p_CiinG = curr_pose->p;
        Eigen::Vector3d p_CiinA = R_GtoA * (p_CiinG - p_AinG);
        
        double base_line = (Q_tangent.transpose() * p_CiinA).norm();
        if (base_line > base_line_max) {
            base_line_max = base_line;
        }
    }
    
    double dist = feat.p_FinA[2];
    double norm = feat.p_FinA.norm();
    
    if (dist < options.min_dist || dist > options.max_dist || std::isnan(norm)) {
        ++gn_reject_dist;
        return false;
    }
    if (base_line_max == 0.0 || (norm / base_line_max) > options.max_baseline) {
        ++gn_reject_baseline;
        return false;
    }
    ++gn_accept;
    
    feat.p_FinG = R_GtoA.transpose() * feat.p_FinA + p_AinG;
    return true;
}

bool compute_disparity(const FeatureDatabase& db, double newest_time, double oldest_time, double& mean, double& var, int& count) {
    double disparities[FEATURE_DB_CAPACITY * 2];
    int disparities_count = 0;
    
    for (int i = 0; i < (int)db.count; ++i) {
        const Feature& feat = feature_at(db, i);
        
        for (int cid = 0; cid < 2; ++cid) {
            int cam_meas[FEATURE_MAX_MEASUREMENTS];
            int num_cam_meas = 0;
            for (int m = 0; m < feat.num_measurements; ++m) {
                if (feat.measurements[m].cam_id == cid) {
                    cam_meas[num_cam_meas++] = m;
                }
            }
            
            if (num_cam_meas < 2) continue;
            
            bool found0 = false;
            bool found1 = false;
            Eigen::Vector2d uv0, uv1;
            
            for (int k = 0; k < num_cam_meas; ++k) {
                int m = cam_meas[k];
                double time = feat.measurements[m].timestamp;
                
                if ((oldest_time == -1.0 || time > oldest_time) && !found0) {
                    uv0 = feat.measurements[m].uv;
                    found0 = true;
                    continue;
                }
                
                if ((newest_time == -1.0 || time < newest_time) && found0) {
                    uv1 = feat.measurements[m].uv;
                    found1 = true;
                    continue;
                }
            }
            
            if (found0 && found1) {
                double disp = (uv1 - uv0).norm();
                disparities[disparities_count++] = disp;
            }
        }
    }
    
    count = disparities_count;
    if (disparities_count < 2) {
        mean = -1.0;
        var = -1.0;
        return false;
    }
    
    double sum = 0.0;
    for (int i = 0; i < disparities_count; ++i) sum += disparities[i];
    mean = sum / disparities_count;
    
    double sq_sum = 0.0;
    for (int i = 0; i < disparities_count; ++i) {
        double d = disparities[i] - mean;
        sq_sum += d * d;
    }
    var = std::sqrt(sq_sum / (disparities_count - 1));
    return true;
}

bool compute_disparity_two_frames(const FeatureDatabase& db, double time0, double time1, double& mean, double& var, int& count) {
    double disparities[FEATURE_DB_CAPACITY * 2];
    int disparities_count = 0;
    
    for (int i = 0; i < (int)db.count; ++i) {
        const Feature& feat = feature_at(db, i);
        
        for (int cid = 0; cid < 2; ++cid) {
            int idx0 = -1;
            int idx1 = -1;
            
            for (int m = 0; m < feat.num_measurements; ++m) {
                if (feat.measurements[m].cam_id == cid) {
                    if (feat.measurements[m].timestamp == time0) idx0 = m;
                    if (feat.measurements[m].timestamp == time1) idx1 = m;
                }
            }
            
            if (idx0 != -1 && idx1 != -1) {
                double disp = (feat.measurements[idx1].uv - feat.measurements[idx0].uv).norm();
                disparities[disparities_count++] = disp;
            }
        }
    }
    
    count = disparities_count;
    if (disparities_count < 2) {
        mean = -1.0;
        var = -1.0;
        return false;
    }
    
    double sum = 0.0;
    for (int i = 0; i < disparities_count; ++i) sum += disparities[i];
    mean = sum / disparities_count;
    
    double sq_sum = 0.0;
    for (int i = 0; i < disparities_count; ++i) {
        double d = disparities[i] - mean;
        sq_sum += d * d;
    }
    var = std::sqrt(sq_sum / (disparities_count - 1));
    return true;
}

} // namespace core
