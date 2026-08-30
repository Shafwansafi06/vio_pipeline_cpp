#pragma once
#include <Eigen/Core>

namespace core {

// Triangulation rejection accounting, defined in feature.cpp.
extern long tri_reject_cond, tri_reject_mindist, tri_reject_maxdist, tri_reject_nan, tri_accept;
extern long gn_reject_dist, gn_reject_baseline, gn_accept;
extern long tri_reject_cond_meas, tri_accept_meas;
extern long db_full_refusals;
// Measurements discarded because a feature's array was already full.
extern long feat_meas_overflow;
extern long dbg_max_meas, dbg_max_count, dbg_shift_elems, dbg_compact_elems;
extern long tri_cond_hist[32], tri_accept_hist[32];
// Diagnostic-only: accepted triangulated depth (p_f[2], meters), bucketed in
// 5m bins 0-100m (bin 19 catches everything >=95m). Never read by any
// decision -- write-only accumulator per dod-style Rule 5 exception. Added to
// answer "what does this pipeline actually triangulate to" on datasets whose
// scene depth is unknown (indoor EuRoC calibration constants assumed
// max_dist=75.0 is right; this is how that assumption gets checked).
extern long tri_depth_hist[20];
extern double tri_depth_sum;

// Epipolar bearing-consistency diagnostic, defined in feature.cpp.
// epi_computed: how many features had two cam-0 measurements with enough
// baseline to run the check at all. epi_no_baseline: skipped because the two
// clones were too close together (parallel bearings, undefined epipolar
// plane) or too near each other in position -- not evidence of anything,
// just insufficient parallax. epi_score_hist buckets the computed score
// (0.0-1.0) into 20 bins of width 0.05; a genuinely single, static 3D point
// should score near 0 in every bucket except the first.
extern long epi_computed, epi_no_baseline;
extern long epi_score_hist[20];

struct ClonePose {
    Eigen::Matrix3d R;
    Eigen::Vector3d p;
};

struct ClonesCamera {
    int num_clones = 0;
    double timestamps[20];
    ClonePose poses[2][20]; // [camera_id][clone_index]
};

const ClonePose* find_clone_pose(const ClonesCamera& clones, int cam_id, double ts);

struct FeatureMeasurement {
    int cam_id;
    double timestamp;
    Eigen::Vector2d uv;
    Eigen::Vector2d uv_norm;
};

// Once the filter is running, a feature holds at most one measurement per
// camera per live clone plus the frame being processed: 2 * max_clone_size + 2,
// which is 24 for the shipped 11-clone window. Measured 2026-08-18 over the ten
// sequences, peak observed measurements per feature (dbg_max_meas):
//
//   all eight EuRoC sequences   24   <- exactly the clone-derived bound
//   KAIST circle, KAIST infinity 48   <- SATURATED, measurements were dropped
//
// KAIST exceeds the bound because measurements are only pruned when a clone is
// marginalised, and before initialisation there are no clones: a feature the
// tracker holds across the pre-init stretch accumulates one measurement per
// camera per frame with nothing to trim it. DOD initialises at t+25.4 s on
// KAIST circle against official's t+3.7 s, so KAIST spends long enough there to
// fill the array; EuRoC initialises inside 2 s and never approaches it.
//
// So this is 2x the *post-init* bound and NOT slack that can be reclaimed: an
// earlier attempt to cut it to 24 (whose comment survived here for a while
// claiming the cut had been made) would have truncated KAIST's pre-init
// accumulation harder than it already is. Cutting it is worth ~1.2 kB per
// feature and halves every database scan, but it is blocked on the
// initialisation latency, not on this constant. update_feature() counts the
// drops as feat_meas_overflow so the saturation is visible rather than silent.
//
// init_state() validates max_clone_size against FEATURE_MAX_CLONES_SUPPORTED.
constexpr int FEATURE_MAX_MEASUREMENTS = 48;
constexpr int FEATURE_MAX_CLONES_SUPPORTED = (FEATURE_MAX_MEASUREMENTS - 2) / 2;

struct Feature {
    int featid = -1;
    bool to_delete = false;
    FeatureMeasurement measurements[FEATURE_MAX_MEASUREMENTS];
    int num_measurements = 0;
    
    // Triangulated coordinates
    Eigen::Vector3d p_FinG = Eigen::Vector3d::Zero();
    int anchor_cam_id = -1;
    double anchor_clone_timestamp = 0.0;
    Eigen::Vector3d p_FinA = Eigen::Vector3d::Zero();
};

// Multiplier on sigma_pix^2 for one feature, from the depth-to-parallax ratio
// of its own triangulation: sigma_eff^2 = sigma_pix^2 * (1 + (lambda*Z/B)^2),
// clamped to `max_scale`. Returns exactly 1.0 when lambda <= 0, which makes
// the model a no-op bit-for-bit. Derivation and rationale in feature.cpp.
// Requires feat.p_FinA to hold the current triangulation.
double parallax_noise_scale(const Feature& feat, const ClonesCamera& clones,
                            double lambda, double max_scale);
extern long pnw_computed, pnw_no_depth, pnw_no_baseline, pnw_clamped;
extern double pnw_scale_sum, pnw_scale_max;

// Epipolar-plane bearing consistency between this feature's OLDEST and
// NEWEST cam-0 measurement, using clone poses the filter has ALREADY
// estimated -- unlike the featureless dynamic initializer (initialize/
// initialization.cpp), which solves the epipolar-normal null-space for an
// UNKNOWN relative translation direction via an eigendecomposition over many
// features, this reuses the same bearing-rotate-and-cross-product identity
// where the translation direction is already known from the state, so it
// collapses to one cross product and one dot product per feature: no
// eigensolve, no batching.
//
// Returns |n_hat . t_hat| in [0,1], where n is the epipolar-plane normal
// (perpendicular to the true relative translation for any genuinely single,
// static 3D point) and t is the known translation between the two clones.
// Near 0 is consistent; large means these two measurements cannot be the
// same physical point given the filter's own current pose estimates --
// independent of chi2, since chi2 only asks "does this fit the linearized
// state," never "is this the same point." Returns -1.0 (not computable) if
// there are fewer than two cam-0 measurements or the two clones don't have
// enough baseline to define an epipolar plane.
//
// DIAGNOSTIC ONLY as of this commit: computed and counted, does not reject
// anything. See core/feature.cpp's definition for why.
double epipolar_consistency_score(const Feature& feat, const ClonesCamera& clones);

// Feature payloads are 2400 bytes each. Keeping them physically sorted by
// featid meant every new feature memmove'd the tail of the array: measured
// 9,556 Feature copies per frame = 22.9 MB/s of pure memcpy to maintain an
// ordering over a 4-byte key, plus a full compaction on every cleanup_db (3-4
// times a frame).
//
// The payloads now never move. `order` holds slot indices sorted by featid, so
// insertion shifts 4-byte ints instead of 2400-byte structs, and cleanup only
// rewrites `order`. Logical iteration order is unchanged -- still ascending
// featid -- so results are bit-identical. Freed slots are recycled through a
// stack. Payload addresses are now stable across cleanup, which is strictly
// safer than before (callers used to re-resolve pointers by id after every
// compaction).
//
// Access live feature k (0 <= k < count) with feature_at(db, k), never
// db.features[k] -- the latter is a storage slot, not a logical position.
// Measured peak occupancy over the ten-sequence sweep is 250 live features
// (MH_04); the tracker is configured for 200 per camera and the database holds
// the union across cameras plus whatever has not yet been marginalised. 2048 is
// therefore ~8x the observed peak, and db_full_refusals has been zero on every
// sequence. It is not sized down because the cost of the slack is address space,
// not bandwidth: every scan runs over `count`, never over the capacity.
constexpr int FEATURE_DB_CAPACITY = 2048;

struct FeatureDatabase {
    Feature features[FEATURE_DB_CAPACITY];
    int order[FEATURE_DB_CAPACITY];
    int free_slots[FEATURE_DB_CAPACITY];
    int num_free = 0;
    int num_slots_used = 0;
    std::size_t count = 0;
};

inline Feature& feature_at(FeatureDatabase& db, int k) { return db.features[db.order[k]]; }
inline const Feature& feature_at(const FeatureDatabase& db, int k) { return db.features[db.order[k]]; }

// Database functions
int find_feature_index(const FeatureDatabase& db, int feat_id);
Feature* get_feature(FeatureDatabase& db, int feat_id);
bool get_feature_clone(const FeatureDatabase& db, int feat_id, Feature& feat_out);
void update_feature(FeatureDatabase& db, int feat_id, double timestamp, int cam_id, double u, double v, double u_n, double v_n);
void cleanup_db(FeatureDatabase& db);
void cleanup_db_measurements(FeatureDatabase& db, double timestamp);
void cleanup_db_measurements_exact(FeatureDatabase& db, double timestamp);
double get_oldest_db_timestamp(const FeatureDatabase& db);

// Feature measurement filtering
void clean_old_measurements(Feature& feat, const double* valid_times, int num_valid);
void clean_invalid_measurements(Feature& feat, const double* invalid_times, int num_invalid);
void clear_older_measurements(Feature& feat, double timestamp);

// Triangulation Options
struct FeatureInitializerOptions {
    bool triangulate_1d = false;
    bool refine_features = true;
    double max_cond_number = 10000.0;
    double min_dist = 0.25;
    double max_dist = 40.0;
    double max_baseline = 40.0;
    double init_lamda = 1e-3;
    double max_lamda = 1e10;
    int max_runs = 10;
    double min_dx = 1e-6;
    double min_dcost = 1e-6;
    double lam_mult = 10.0;
};

// Triangulation Math
bool single_triangulation(Feature& feat, const ClonesCamera& clones, const FeatureInitializerOptions& options);
bool single_triangulation_1d(Feature& feat, const ClonesCamera& clones, const FeatureInitializerOptions& options);
bool single_gaussnewton(Feature& feat, const ClonesCamera& clones, const FeatureInitializerOptions& options);
double compute_error(const ClonesCamera& clones, const Feature& feat, double alpha, double beta, double rho);

bool compute_disparity(const FeatureDatabase& db, double newest_time, double oldest_time, double& mean, double& var, int& count);
bool compute_disparity_two_frames(const FeatureDatabase& db, double time0, double time1, double& mean, double& var, int& count);

} // namespace core
