#pragma once
#include "state.hpp"
#include "propagator.hpp"
#include "updaters.hpp"
#include "hover_detector.hpp"
#include "../core/feature.hpp"
#include "../core/cam.hpp"
#include "../core/sensor_data.hpp"
#include "../initialize/initialization.hpp"

namespace msckf {

// Per-frame feature classification counters, defined in vio_manager.cpp.
extern long cls_frames, cls_lost, cls_marginal, cls_maxtrack, cls_slam_update, cls_db_count;
extern long cls_retire_untracked, cls_retire_chi2;

// Rejections on the measurement-input path, defined in vio_manager.cpp. Every
// one of these means data was thrown away; a nonzero value in a run that used
// to report zero is the signal that a bound is now binding.
extern long imu_buffer_evictions, imu_stale_drops, frame_unpropagated_drops;

// Per-stage accumulated wall time (ms), mirroring official OpenVINS's
// record_timing_information columns so the two can be compared stage by stage.
extern double stage_ms_propagate, stage_ms_msckf, stage_ms_slam, stage_ms_slam_delayed, stage_ms_marg,
              stage_ms_db, stage_ms_classify;

struct VioManagerOptions {
    double gravity_mag = 9.81;
    StateOptions state_opt;
    initialize::InitializerOptions init_opt;
    PropagatorNoises noises;
    UpdaterOptions updater_opt;
    UpdaterOptions slam_updater_opt;
    UpdaterOptions aruco_updater_opt;
    core::FeatureInitializerOptions feat_init_opt;
    // SLAM (delayed anchored inverse-depth landmark) updater. The Jacobians
    // are verified correct (tests/verify_slam_jacobians.cpp), but the
    // integration still has an unresolved robustness bug -- a near-degenerate
    // landmark can corrupt the covariance under real (noisy) tracking,
    // observed as either a hard EKFPropagation assert or, with a stricter
    // chi2 gate, catastrophic silent divergence. Left off by default; the
    // MSCKF-only path is the verified 0.685 m ATE baseline on circle.bag.
    bool enable_slam = false;
    // NOTE: update_msckf_schur() does NOT apply the parallax noise whitening
    // that update_msckf() does. Flipping this to true silently disables
    // UpdaterOptions::parallax_noise_lambda in the MSCKF path. Nothing sets it
    // today; wire the whitening into updater_schur.cpp before anything does.
    bool use_schur_msckf = false;
    // Seconds after the first processed image before any SLAM landmark may be
    // created. Official's dt_slam_delay (1.0 on EuRoC); guards against
    // anchoring landmarks on a state that has only just initialised.
    double dt_slam_delay = 1.0;
    // Init trigger: true = wait for the takeoff "jerk" (still->moving), matching
    // official's default; works when the sequence has a clear stationary start
    // then a distinct jerk (KAIST). EuRoC MH's takeoff is too gentle for the
    // jerk gate, so it needs false (init as soon as the platform is still).
    bool init_wait_for_jerk = true;
    // Whether to attempt the zero-velocity update at all. Official OpenVINS's
    // own kaist_vio/estimator_config.yaml ships `try_zupt: false` for this
    // dataset; a slow, gentle circular flight can look "stationary enough"
    // to a IMU-jerk/disparity gate far more often than it should, freezing
    // propagation on a large fraction of frames and producing a badly wrong
    // trajectory shape (not just extra drift). Default true (matches the
    // pre-existing behavior) but every runner config should set this
    // explicitly to match whatever the reference implementation uses for
    // that dataset.
    bool enable_zupt = true;
    // Kottas/Wu/Roumeliotis bearing-vector hover classifier + FIFO/LIFO
    // clone-window switching (msckf/hover_detector.hpp). Independent of
    // enable_zupt -- this replaces Python v2's IMU-velocity/disparity
    // hover heuristic with the paper's rotation-compensated bearing
    // residual test, which does not depend on getting IMU noise/threshold
    // tuning right to detect a real hover.
    bool enable_hover_detection = false;
    HoverDetectorOptions hover_opt;
    double zupt_max_velocity = 0.1;
    double zupt_noise_multiplier = 1.0;
    double zupt_max_disparity = 1.0;
    core::CameraModel cam_models[2];
    // [qx,qy,qz,qw,tx,ty,tz], representing Kalibr T_cam_imu.
    double camera_extrinsics[2][7] = {
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}
    };
    double calib_camimu_dt = 0.0;
    int num_cameras = 1;
};

struct VioManagerData {
    VioManagerOptions params;
    State state;
    PropagatorData prop;
    UpdaterMSCKFData updater_msckf;
    UpdaterSLAMData updater_slam;
    UpdaterZeroVelocityData updater_zupt;
    HoverDetectorData hover_detector;
    core::FeatureDatabase db;
    // 50 s of EuRoC IMU at 200 Hz. The buffer only has to span the time from
    // the oldest live clone to now (trimmed every frame in
    // feed_measurement_camera_tracks), plus, before initialisation, whatever
    // window the initialiser needs -- init_opt.init_window_time, 2 s in every
    // shipped config. 50 s is two orders of magnitude of headroom on both, and
    // overflow now evicts the oldest sample rather than refusing the newest,
    // so a long pre-init stretch degrades to a sliding window instead of
    // freezing the buffer permanently.
    static constexpr int IMU_BUFFER_CAPACITY = 10000;
    core::ImuData imu_buffer[IMU_BUFFER_CAPACITY];
    int imu_count = 0;
    bool is_initialized = false;
    double initialized_time = -1.0;
    // Timestamp of the first camera frame processed after initialization, used
    // for the dt_slam_delay gate. Negative means "not started yet".
    double slam_start_time = -1.0;
    int init_attempt = 0;
};

void init_vio_manager(VioManagerData& vio, const VioManagerOptions& params);
void feed_measurement_imu(VioManagerData& vio, const core::ImuData& message);
void feed_measurement_camera_tracks(VioManagerData& vio, double timestamp, const core::Feature* tracks, int track_count);
bool try_to_initialize(VioManagerData& vio);

inline bool is_initialized(const VioManagerData& vio) { return vio.is_initialized; }
inline double initialized_time(const VioManagerData& vio) { return vio.initialized_time; }
inline const State& get_state(const VioManagerData& vio) { return vio.state; }
inline State& get_state(VioManagerData& vio) { return vio.state; }
inline const core::FeatureDatabase& get_db(const VioManagerData& vio) { return vio.db; }
inline core::FeatureDatabase& get_db(VioManagerData& vio) { return vio.db; }

} // namespace msckf
