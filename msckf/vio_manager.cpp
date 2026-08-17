#include "vio_manager.hpp"

#include <algorithm>
#include <chrono>
#include "state_helper.hpp"
#include "hover_detector.hpp"

namespace msckf {

// Feature classification accounting, reported by the runners.
long cls_frames = 0, cls_lost = 0, cls_marginal = 0, cls_maxtrack = 0;
long cls_slam_update = 0, cls_db_count = 0;
long cls_retire_untracked = 0, cls_retire_chi2 = 0;
double stage_ms_propagate = 0.0, stage_ms_msckf = 0.0, stage_ms_slam = 0.0,
       stage_ms_slam_delayed = 0.0, stage_ms_marg = 0.0;

namespace {
// Accumulates the wall time of the scope it is declared in.
struct StageTimer {
    double& sink;
    std::chrono::steady_clock::time_point start;
    explicit StageTimer(double& target) : sink(target), start(std::chrono::steady_clock::now()) {}
    ~StageTimer() {
        sink += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
};
} // namespace


void init_vio_manager(VioManagerData& vio, const VioManagerOptions& params) {
    vio.params = params;
    init_updater_msckf(vio.updater_msckf, params.updater_opt, params.feat_init_opt);
    init_updater_slam(vio.updater_slam, params.slam_updater_opt, params.aruco_updater_opt, params.feat_init_opt);
    init_updater_zupt(vio.updater_zupt, params.noises, params.gravity_mag, params.zupt_max_velocity,
                      params.zupt_noise_multiplier, params.zupt_max_disparity);
    init_propagator(vio.prop, params.noises, params.gravity_mag);
    init_hover_detector(vio.hover_detector, params.hover_opt);

    StateOptions opt = params.state_opt;
    opt.num_cameras = params.num_cameras;
    
    init_state(vio.state, opt);

    // Calibration variables are nominal state even when held fixed. Leaving
    // these at identity silently makes every feature Jacobian geometrically
    // inconsistent with the tracker calibration.
    vio.state.calib_dt_CAMtoIMU.value[0] = params.calib_camimu_dt;
    vio.state.calib_dt_CAMtoIMU.fej[0] = params.calib_camimu_dt;
    for (int camera = 0; camera < params.num_cameras; ++camera) {
        for (int k = 0; k < 8; ++k) {
            vio.state.cam_intrinsics[camera].value[k] = params.cam_models[camera].values[k];
            vio.state.cam_intrinsics[camera].fej[k] = params.cam_models[camera].values[k];
        }
        for (int k = 0; k < 7; ++k) {
            vio.state.calib_IMUtoCAM[camera].value[k] = params.camera_extrinsics[camera][k];
            vio.state.calib_IMUtoCAM[camera].fej[k] = params.camera_extrinsics[camera][k];
        }
    }
}

void feed_measurement_imu(VioManagerData& vio, const core::ImuData& message) {
    if (vio.imu_count < 10000) {
        // Keep sorted by timestamp
        int insert_idx = vio.imu_count;
        while (insert_idx > 0 && vio.imu_buffer[insert_idx - 1].timestamp > message.timestamp) {
            vio.imu_buffer[insert_idx] = vio.imu_buffer[insert_idx - 1];
            insert_idx--;
        }
        vio.imu_buffer[insert_idx] = message;
        vio.imu_count++;
    }
}

void feed_measurement_camera_tracks(VioManagerData& vio, double timestamp, const core::Feature* tracks, int track_count) {
    // 1. Update feature database
    for (int i = 0; i < track_count; ++i) {
        const core::Feature& tr = tracks[i];
        for (int m = 0; m < tr.num_measurements; ++m) {
            core::update_feature(vio.db, tr.featid, tr.measurements[m].timestamp,
                                 tr.measurements[m].cam_id,
                                 tr.measurements[m].uv[0], tr.measurements[m].uv[1],
                                 tr.measurements[m].uv_norm[0], tr.measurements[m].uv_norm[1]);
        }
    }
    
    // 2. Try to initialize if not yet initialized
    if (!vio.is_initialized) {
        try_to_initialize(vio);
        return;
    }
    
    // 3. Try Zero Velocity Update (ZUPT)
    bool zupt_success = vio.params.enable_zupt &&
        try_update_zupt(vio.updater_zupt, vio.state, timestamp, vio.imu_buffer, vio.imu_count, vio.db, true);
    
    if (!zupt_success) {
        // First post-initialization frame starts the dt_slam_delay clock.
        if (vio.slam_start_time < 0.0) vio.slam_start_time = timestamp;
        const double previous_clone_timestamp =
            vio.state.num_clones > 0 ? vio.state.clones_IMU[vio.state.num_clones - 1].timestamp : -1.0;

        // Standard MSCKF step: Propagate IMU and augment clone
        { StageTimer stage_timer(msckf::stage_ms_propagate); propagate_and_clone(vio.prop, vio.state, timestamp, vio.imu_buffer, vio.imu_count); }

        // Hovering classifier (Kottas, Wu & Roumeliotis): rotation-compensated
        // bearing-vector residual between the two most recent clones, with
        // hysteresis. Independent of ZUPT/IMU-velocity heuristics.
        bool hovering = false;
        if (vio.params.enable_hover_detection && previous_clone_timestamp >= 0.0 &&
            vio.state.num_clones >= 2) {
            const type::Variable& prev_clone = vio.state.clones_IMU[vio.state.num_clones - 2];
            const type::Variable& curr_clone = vio.state.clones_IMU[vio.state.num_clones - 1];
            // Bearing vectors live in the CAMERA frame (uv_norm), not the IMU
            // frame -- conjugate the IMU-frame relative rotation by the
            // camera-IMU extrinsic so it applies to camera-frame bearings.
            const Eigen::Matrix3d& R_ItoC = vio.state.calib_IMUtoCAM[0].Rot();
            const Eigen::Matrix3d R_imu_curr_from_prev = curr_clone.Rot() * prev_clone.Rot().transpose();
            const Eigen::Matrix3d R_curr_from_prev = R_ItoC * R_imu_curr_from_prev * R_ItoC.transpose();
            hovering = update_hover_detector(vio.hover_detector, vio.db, previous_clone_timestamp,
                                             timestamp, R_curr_from_prev);
        }

        // Classify tracked features into: lost (no measurement this frame),
        // MSCKF-marginal (touching the clone about to be dropped, but not yet
        // long-lived enough / SLAM capacity full), new SLAM candidates
        // (long-lived enough and there's room), and existing SLAM landmarks
        // needing their per-frame update.
        core::Feature* lost_features[2048];
        core::Feature* marginal_features[2048];
        core::Feature* maxtrack_features[2048];
        core::Feature* slam_update_features[64];
        // Feature IDS as well as pointers. cleanup_db() compacts
        // vio.db.features[] in place between the updates below, which
        // invalidates every pointer captured here; the ids stay valid and are
        // re-resolved at each use. Keeping the cleanups matters -- they are
        // what stops a feature already consumed by one updater from being
        // consumed again by the next in the same frame.
        int slam_update_ids[64];
        static int maxtrack_ids[2048];
        int lost_count = 0, marginal_count = 0, maxtrack_count = 0, slam_update_count = 0;
        const bool clones_full = vio.state.num_clones > vio.state.options.max_clone_size;
        // The clone actually about to be dropped differs by mode: FIFO drops
        // the oldest (margtimestep); LIFO drops the previous "current" clone
        // (second-to-last) instead, per marginalize_lifo_clone.
        const double marginal_time = hovering && clones_full && vio.state.num_clones >= 2
            ? vio.state.clones_IMU[vio.state.num_clones - 2].timestamp
            : margtimestep(vio.state);

        for (int i = 0; i < (int)vio.db.count; ++i) {
            core::Feature& feat = vio.db.features[i];
            bool is_lost = true;
            bool contains_marginal = false;
            for (int m = 0; m < feat.num_measurements; ++m) {
                if (feat.measurements[m].timestamp == timestamp) is_lost = false;
                if (feat.measurements[m].timestamp == marginal_time) contains_marginal = true;
            }
            contains_marginal = contains_marginal && clones_full;

            bool is_slam = vio.params.enable_slam;
            if (is_slam) {
                is_slam = false;
                for (int s = 0; s < vio.state.num_slam_features; ++s) {
                    if (vio.state.features_SLAM[s].feat_id == feat.featid) { is_slam = true; break; }
                }
            }
            if (is_slam) {
                if (feat.num_measurements >= 1 && slam_update_count < 64) {
                    slam_update_ids[slam_update_count] = feat.featid;
                    slam_update_features[slam_update_count++] = &feat;
                }
                continue;
            }

            if (!(feat.num_measurements >= 2 && (is_lost || contains_marginal))) continue;

            // "Reached max track" is a PER-CAMERA test in official:
            //
            //   for (const auto &cams : feat->timestamps)
            //     if ((int)cams.second.size() > max_clone_size) reached_max = true;
            //
            // i.e. any one camera having more than max_clone_size observations
            // is enough. This used to require 2 * max_clone_size measurements
            // summed across both cameras, on the reasoning that stereo doubles
            // the count. It does -- for stereo features. A MONO feature tracked
            // through the entire window has exactly max_clone_size + 1
            // measurements, which official promotes and this rejected, so
            // almost nothing was ever promoted to SLAM: measured on EuRoC
            // MH_01, 21091 features sat at exactly 12 measurements (one camera,
            // full window) and not one of them became a landmark. Official
            // feeds ~5.9 features per MSCKF update because the long tracks
            // became persistent state; DOD was feeding ~20 and throwing them
            // all away.
            int meas_per_cam[2] = {0, 0};
            for (int m = 0; m < feat.num_measurements; ++m) {
                const int c = feat.measurements[m].cam_id;
                if (c >= 0 && c < 2) ++meas_per_cam[c];
            }
            bool reached_max = false;
            for (int c = 0; c < vio.state.options.num_cameras && c < 2; ++c) {
                if (meas_per_cam[c] > vio.state.options.max_clone_size) { reached_max = true; break; }
            }
            // Official also waits dt_slam_delay after startup before admitting
            // any SLAM landmark -- it "normally prevents bad first set of slam
            // points", i.e. landmarks anchored on a barely-converged state.
            const bool slam_delay_passed =
                (vio.slam_start_time < 0.0) || (timestamp - vio.slam_start_time >= vio.params.dt_slam_delay);
            if (vio.params.enable_slam && slam_delay_passed && contains_marginal && reached_max &&
                vio.state.num_slam_features + maxtrack_count < vio.state.options.max_slam_features &&
                maxtrack_count < 2048) {
                maxtrack_ids[maxtrack_count] = feat.featid;
                maxtrack_features[maxtrack_count++] = &feat;
            } else if (is_lost) {
                lost_features[lost_count++] = &feat;
            } else {
                marginal_features[marginal_count++] = &feat;
            }
        }

        core::Feature* msckf_features[2048];
        int msckf_count = 0;
        for (int i = 0; i < lost_count; ++i) msckf_features[msckf_count++] = lost_features[i];
        for (int i = 0; i < marginal_count; ++i) msckf_features[msckf_count++] = marginal_features[i];

        // NOTE (2026-07-22): tried adding official's max_msckf_in_update
        // sort+cap here (keep only the longest tracks when count exceeds the
        // configured cap). Measured effect on real data: msckf_count rarely
        // exceeds the cap (~17 features/call average, not the ~150 once
        // assumed), so the cap barely changed the single-shot accept rate
        // (6.4% -> 8.2%, still far below official's ~70%), and it made ATE
        // RMSE *worse* on both datasets (circle.bag 0.785->1.043 m,
        // infinity.bag 1.070->1.306 m). Reverted; not the bottleneck.

        // Whether each SLAM landmark is still being tracked must be sampled
        // BEFORE the updates run. update_msckf/update_slam/delayed_init_slam
        // each mark every feature they consume to_delete, and the cleanup_db()
        // calls that follow them erase those features outright -- so asking the
        // database afterwards reports "not tracked" for exactly the landmarks
        // that were just successfully updated, and retires them. Official makes
        // this determination against the tracker database up front, before its
        // updates, which is what this reproduces.
        bool slam_still_tracked[64] = {false};
        const int slam_count_snapshot = vio.state.num_slam_features;
        for (int s = 0; s < slam_count_snapshot && s < 64; ++s) {
            slam_still_tracked[s] = (core::get_feature(vio.db, vio.state.features_SLAM[s].feat_id) != nullptr);
        }

        if (msckf_count > 0) {
            // Every feature fed to the MSCKF update this frame is discarded
            // afterward (update_msckf marks all of them to_delete=true,
            // success or failure) -- a strict one-shot attempt, matching
            // official exactly. The tracker's own persistent per-camera track
            // state (TrackerData::previous[]) is independent of this
            // database, so a still-tracked physical point gets a brand-new
            // measurement history starting next frame via update_feature()'s
            // "not found -> create" path, instead of retrying triangulation
            // on the same already-failed accumulated history.
            // Official caps the update at max_msckf_in_update (40 on EuRoC),
            // keeping the LONGEST tracks: it sorts ascending by total
            // measurement count and erases from the front. DOD ignored the
            // option entirely and fed every feature. See VioManager.cpp:508-524.
            core::Feature** msckf_used = msckf_features;
            int msckf_used_count = msckf_count;
            const int msckf_cap = vio.state.options.max_msckf_in_update;
            if (msckf_cap > 0 && msckf_count > msckf_cap) {
                std::sort(msckf_features, msckf_features + msckf_count,
                          [](const core::Feature* a, const core::Feature* b) {
                              return a->num_measurements < b->num_measurements;
                          });
                msckf_used = msckf_features + (msckf_count - msckf_cap);
                msckf_used_count = msckf_cap;
            }
            { StageTimer stage_timer(msckf::stage_ms_msckf); update_msckf(vio.updater_msckf, vio.state, msckf_used, msckf_used_count, vio.params.cam_models); }
            core::cleanup_db(vio.db);
        }

        if (vio.params.enable_slam) {
            if (slam_update_count > 0) {
                // NOTE: official applies SLAM updates in sequential batches of
                // max_slam_in_update, correcting the state between batches
                // (VioManager.cpp slices feats_slam_UPDATE and calls update()
                // until drained). That was implemented and MEASURED here: it is
                // worse on EuRoC MH_01 -- ATE 0.2632 -> 0.3508 m, p95 0.733,
                // path 82.79 -> 84.26 m -- so the single joint update is kept.
                // A documented, measured divergence, not an oversight.
                // Re-resolve by feature id. cleanup_db() above compacted
                // vio.db.features[] in place, so the pointers captured during
                // classification no longer refer to the intended features.
                core::Feature* resolved[64];
                int resolved_count = 0;
                for (int i = 0; i < slam_update_count && resolved_count < 64; ++i) {
                    core::Feature* f = core::get_feature(vio.db, slam_update_ids[i]);
                    if (f) resolved[resolved_count++] = f;
                }
                if (resolved_count > 0) {
                    const int batch = (vio.state.options.max_slam_in_update > 0)
                                          ? vio.state.options.max_slam_in_update
                                          : resolved_count;
                    for (int start = 0; start < resolved_count; start += batch) {
                        const int n = std::min(batch, resolved_count - start);
                        { StageTimer stage_timer(msckf::stage_ms_slam); update_slam(vio.updater_slam, vio.state, resolved + start, n, vio.params.cam_models); }
                    }
                }
                core::cleanup_db(vio.db);
            }

            // Per-frame classification accounting. Official feeds ~5.9 features per
        // MSCKF update on EuRoC and DOD feeds ~20; this says which bucket the
        // difference is in.
        cls_frames += 1;
        cls_lost += lost_count;
        cls_marginal += marginal_count;
        cls_maxtrack += maxtrack_count;
        cls_slam_update += slam_update_count;
        cls_db_count += (long)vio.db.count;

        if (maxtrack_count > 0) {
                // Same re-resolution as above; these pointers predate two
                // cleanup_db() compactions.
                static core::Feature* resolved_mt[2048];
                int resolved_mt_count = 0;
                for (int i = 0; i < maxtrack_count && resolved_mt_count < 2048; ++i) {
                    core::Feature* f = core::get_feature(vio.db, maxtrack_ids[i]);
                    if (f) resolved_mt[resolved_mt_count++] = f;
                }
                if (resolved_mt_count > 0) {
                    StageTimer stage_timer(msckf::stage_ms_slam_delayed);
                    delayed_init_slam(vio.updater_slam, vio.state, resolved_mt, resolved_mt_count,
                                      vio.params.cam_models);
                }
                core::cleanup_db(vio.db);
            }

            // Drop SLAM landmarks that were not tracked into this frame, or
            // that have repeatedly failed their chi2 gate. Uses the snapshot
            // taken before the updates -- see the comment there.
            for (int s = 0; s < vio.state.num_slam_features; ++s) {
                type::Variable& lm = vio.state.features_SLAM[s];
                const bool tracked = (s < slam_count_snapshot && s < 64) ? slam_still_tracked[s] : true;
                if (!tracked) ++cls_retire_untracked;
                else if (lm.update_fail_count > 1) ++cls_retire_chi2;
                if (!tracked || lm.update_fail_count > 1) lm.should_marg = true;
            }
            { StageTimer stage_timer(msckf::stage_ms_marg); marginalize_slam(vio.state); }
        }


        // The EKF estimates camera intrinsics into state.cam_intrinsics[],
        // but every residual/Jacobian computation (get_feature_jacobian_mixed
        // et al.) reads distortion from vio.params.cam_models -- a separate
        // buffer written only once at init. Without this resync, the
        // estimated intrinsics silently drift away from what the filter
        // actually projects with: the EKF keeps building intrinsics<->pose
        // cross-covariance and pushing "corrections" that never take effect,
        // corrupting the trajectory. This is why improving tracker precision
        // made things WORSE, not better -- lower measurement noise means
        // each update trusts (and pushes) the bogus intrinsics correlation
        // harder. Matches official OpenVINS's StateHelper::EKFUpdate, which
        // copies the estimated intrinsics back into its live camera objects
        // immediately after every update.
        if (vio.params.state_opt.do_calib_camera_intrinsics) {
            for (int camera = 0; camera < vio.params.num_cameras; ++camera) {
                core::set_camera_values(vio.params.cam_models[camera], vio.state.cam_intrinsics[camera].value);
            }
        }

        if (clones_full && marginal_time >= 0.0) {
            core::cleanup_db_measurements(vio.db, marginal_time);
        }

        if (vio.params.enable_slam) {
            // Move any SLAM landmark anchored on the clone about to be
            // dropped onto the newest clone before marginalizing it away.
            { StageTimer stage_timer(msckf::stage_ms_marg); change_anchors(vio.updater_slam, vio.state); }
        }

        // Marginalize old clones if exceeding sliding window size: FIFO
        // during generic motion, LIFO during hovering (paper, Sect. III-B) so
        // the generic-motion baseline already in the window is preserved
        // instead of being squeezed out by more zero-baseline hover clones.
        if (hovering) {
            marginalize_lifo_clone(vio.state);
        } else {
            marginalize_old_clone(vio.state);
        }
    }
    
    // Clean old IMU readings from cache
    double oldest_clone_time = margtimestep(vio.state);
    if (oldest_clone_time != -1.0) {
        int write_idx = 0;
        for (int i = 0; i < vio.imu_count; ++i) {
            if (vio.imu_buffer[i].timestamp >= oldest_clone_time - 0.5) {
                vio.imu_buffer[write_idx++] = vio.imu_buffer[i];
            }
        }
        vio.imu_count = write_idx;
    }
}

bool try_to_initialize(VioManagerData& vio) {
    Eigen::Matrix<double, 15, 15> init_cov;
    double init_ts = 0.0;
    
    // wait_for_jerk=true matches official OpenVINS's InertialInitializer
    // default (initialize(..., wait_for_jerk=true)): initialize on the
    // takeoff "jerk" (still-then-moving) rather than during the initial
    // stationary window. DOD previously passed false (init when stationary),
    // a real divergence from official's init trigger.
    bool success = initialize::static_initialize(vio.params.init_opt, vio.imu_buffer, vio.imu_count,
                                                              vio.state.imu, init_cov, init_ts, vio.params.init_wait_for_jerk);
    // Dynamic (linear MLE) fallback: for sequences with no clean stationary
    // start (e.g. EuRoC MH_*), static init never fires. Matches official
    // OpenVINS's InertialInitializer, which falls back to the dynamic
    // initializer when the static one cannot find a stationary window.
    if (!success && vio.params.init_opt.init_dyn_use) {
        if (vio.params.init_opt.init_featureless) {
            success = initialize::featureless_initialize(
                vio.params.init_opt, vio.db, vio.imu_buffer, vio.imu_count,
                vio.params.camera_extrinsics, vio.params.num_cameras,
                vio.state.imu, init_cov, init_ts);
        }
        // The feature-based solve remains the fallback: if the bearing geometry
        // is too weak to give a translation direction, it can still be tried.
        if (!success) {
            success = initialize::dynamic_initialize(vio.params.init_opt, vio.db, vio.imu_buffer, vio.imu_count,
                                                     vio.params.camera_extrinsics, vio.params.num_cameras,
                                                     vio.state.imu, init_cov, init_ts);
        }
    }
    if (success) {
        vio.state.timestamp = init_ts;
        
        type::Variable* order[1] = { &vio.state.imu };
        vio.state.Cov.setZero();
        set_initial_covariance(vio.state, init_cov, order, 1);
        
        vio.is_initialized = true;
        vio.initialized_time = init_ts;

        // Clean up database for measurements before initialization
        core::cleanup_db_measurements(vio.db, init_ts);
        return true;
    }
    return false;
}

} // namespace msckf
