// FGI Masala Stereo-Visual-Inertial Dataset 2021 configuration (George et al.,
// "Visual-Inertial Odometry Using High Flying Altitude Drone Datasets", Drones
// 2023, 7, 36). 12 bags at 40/60/80/100 m altitude, 2/3/4 m/s, fixed 30 cm
// stereo baseline. Topics and rates measured via `rosbag info` on 40_4.bag
// (241s, 32871 msgs), not guessed:
//   /imu/imu/data          24170 msgs -> ~100.3 Hz
//   /left/downsample_raw    3867 msgs -> ~16.0 Hz
//   /right/downsample_raw   3867 msgs -> ~16.0 Hz  (EuRoC is 20 Hz stereo)
//
// Calibration read directly from the dataset's own Kalibr-format
// "sensor parameters"/{imu,left_camera,right_camera}.yaml. imu.yaml's T_BS is
// identity (body frame == IMU frame), same convention as EuRoC, so
// T_cam_imu = inverse(T_BS) exactly as tools/euroc_options.hpp derives it.
// Baseline recovered from T1's translation (~0.300 m in y) matches the
// paper's stated 30 cm rig exactly -- a check that this conversion is right,
// not an assumption.
//
// Every numeric knob below other than calibration (init thresholds,
// triangulation gates) is a placeholder copied from EuRoC and marked as such.
// Per CLAUDE.md/dod-style Rule 8.5, none of these are trustworthy until
// measured on this dataset's own sequences -- max_dist=75.0 in particular was
// tuned against EuRoC's <5m indoor depths and is a strong suspect at 40-100m
// scene depth.
#pragma once
#include <cstdlib>

#include "../core/tracker.hpp"
#include "../msckf/vio_manager.hpp"
#include "../type/quat_ops.hpp"

namespace mid_altitude {

constexpr int kImageWidth = 612;
constexpr int kImageHeight = 512;
constexpr const char* kLeftTopic = "/left/downsample_raw";
constexpr const char* kRightTopic = "/right/downsample_raw";
constexpr const char* kImuTopic = "/imu/imu/data";

inline void set_pose(double target[7], const double matrix[16]) {
    Eigen::Matrix3d rotation;
    rotation << matrix[0], matrix[1], matrix[2],
                matrix[4], matrix[5], matrix[6],
                matrix[8], matrix[9], matrix[10];
    const Eigen::Vector4d quaternion = type::rot_2_quat(rotation);
    for (int i = 0; i < 4; ++i) target[i] = quaternion[i];
    target[4] = matrix[3];
    target[5] = matrix[7];
    target[6] = matrix[11];
}

inline msckf::VioManagerOptions make_mid_altitude_options() {
    msckf::VioManagerOptions options;
    options.gravity_mag = 9.81;
    options.num_cameras = 2;
    options.state_opt.do_fej = true;
    options.state_opt.integration_method = msckf::IntegrationMethod::RK4;
    options.state_opt.do_calib_camera_pose = false;
    options.state_opt.do_calib_camera_intrinsics = true;
    // Tested do_calib_camera_timeoffset=true on 40_4.bag: converged to ~0,
    // ATE unchanged to 4 decimal places (12.8143 vs 12.8143), 2.4x the
    // estimator cost for zero benefit. This dataset's cam-imu offset really
    // is ~0; leave calibration off.
    options.state_opt.do_calib_camera_timeoffset = false;
    options.state_opt.max_clone_size = 11;
    // Temporal baseline is the only real parallax this dataset has: the stereo
    // rig is 0.30 m against 40-100 m of scene depth (Z/B of 130-330), so
    // triangulation depends on how far the platform moves across the clone
    // window. At 16 Hz and 4 m/s, 11 clones is ~0.69 s ~ 2.75 m; 20 clones is
    // ~1.25 s ~ 5 m, which halves Z/B. Clamped at 20 because State::clones_IMU,
    // ClonesCamera::poses and the clonetimes[] scratch arrays are all fixed at
    // 20 -- a larger value overflows them.
    if (const char* env = std::getenv("VIO_MAX_CLONES")) {
        const int n = std::atoi(env);
        if (n >= 3) options.state_opt.max_clone_size = n < 20 ? n : 20;
    }
    options.state_opt.max_slam_features = 50;
    options.state_opt.max_slam_in_update = 25;
    // PLACEHOLDER: copied from euroc_options.hpp's measured value, not
    // measured on this dataset. Feature supply differs (16 Hz vs 20 Hz
    // stereo, 40-100m depth) so this is a candidate for re-sweeping.
    options.state_opt.max_msckf_in_update = 75;
    options.state_opt.feat_rep_msckf = type::LandmarkRepresentation::GLOBAL_3D;
    options.state_opt.feat_rep_slam = type::LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
    options.init_opt.init_window_time = 2.0;
    // Experiment inputs (CLAUDE.md rule 8.5: sweep, don't guess). 60_4 diverges
    // from t=0 with the EuRoC-inherited window: at 60m depth the 2s window
    // gives ~8m of translation baseline against 60m of scene depth, and the
    // featureless [v,g] solve comes back with rec_cond 0.0034 (50x worse than
    // 40_4's 6.9e-05) and gravity-vs-accel 6.1 deg (40_4: 0.94). The estimate
    // then claims km-scale displacement in the first 30s. Longer window is the
    // direct lever: parallax angle grows ~linearly with window length.
    if (const char* env = std::getenv("VIO_INIT_WINDOW")) {
        const double w = std::atof(env);
        if (w > 0.0) options.init_opt.init_window_time = w;
    }
    if (const char* env = std::getenv("VIO_INIT_NUM_POSE")) {
        const int n = std::atoi(env);
        if (n >= 3) options.init_opt.init_dyn_num_pose = n;
    }
    // Conditioning gate for the featureless [v,g] solve (initialization.cpp
    // already implements it; init_dyn_min_rec_cond defaults to 0 = off). This
    // is the lever the span-gate experiment pointed at: EuRoC's 3-frame solves
    // come back at rec_cond 1.6e-4 .. 2.3e-4 (indoor parallax is huge) while
    // FGI 60_4's come back at 0.0034 and DIVERGE. A threshold between those
    // regimes rejects the early bad solves and lets the unbounded pre-init
    // database grow until the geometry supports a good one -- measured
    // rec_cond vs window on 60_4: 0.0034 (3 frames), 0.0068 (4 s), 0.065
    // (6 s). EuRoC keeps the default (off) and stays bit-identical.
    if (const char* env = std::getenv("VIO_INIT_MIN_REC_COND")) {
        const double c = std::atof(env);
        if (c > 0.0) options.init_opt.init_dyn_min_rec_cond = c;
    }
    // TESTED init_imu_thresh 1.5 -> 0.5 on 60_4.bag's divergence: byte-
    // identical output (this dataset never goes IMU-still from t=0, so the
    // static initializer this threshold gates never fires regardless of the
    // value -- confirmed, not assumed). The active path is
    // init_featureless's dynamic solve; its own conditioning
    // (rec_cond=0.00336631) is 50x worse than 40_4.bag's (6.86e-05).
    // Reverted; targeting init_window_time/init_dyn_num_pose next.
    options.init_opt.init_imu_thresh = 1.5;
    options.init_opt.init_max_disparity = 5.0;
    options.init_opt.init_max_features = 50;
    options.init_opt.gravity_mag = options.gravity_mag;
    options.init_opt.init_dyn_use = true;
    options.init_opt.init_dyn_num_pose = 5;
    options.init_opt.init_dyn_min_deg = 5.0;
    options.init_opt.init_featureless = true;
    options.init_opt.init_dyn_zero_velocity = false;
    // sensor parameters/imu.yaml (MTi-680G RTK GNSS/INS), noise density /
    // random walk as-shipped.
    options.noises.sigma_a = 9.1250479353131696e-03;
    options.noises.sigma_ab = 1.3059510861758910e-04;
    options.noises.sigma_w = 1.9425812268625707e-03;
    options.noises.sigma_wb = 3.9570963198826466e-05;
    options.noises.sigma_a_2 = options.noises.sigma_a * options.noises.sigma_a;
    options.noises.sigma_ab_2 = options.noises.sigma_ab * options.noises.sigma_ab;
    options.noises.sigma_w_2 = options.noises.sigma_w * options.noises.sigma_w;
    options.noises.sigma_wb_2 = options.noises.sigma_wb * options.noises.sigma_wb;
    options.updater_opt.sigma_pix = 1.0;
    options.updater_opt.sigma_pix_sq = 1.0;
    options.updater_opt.chi2_multipler = 1.0;
    options.slam_updater_opt.sigma_pix = 1.0;
    options.slam_updater_opt.sigma_pix_sq = 1.0;
    options.slam_updater_opt.chi2_multipler = 1.0;
    options.aruco_updater_opt = options.slam_updater_opt;

    // Parallax-limited measurement noise. Off by default so this runner still
    // reproduces the recorded trajectories bit-for-bit; set VIO_PARALLAX_LAMBDA
    // to sweep it. NOT tuned -- the value is an experiment input, and per
    // CLAUDE.md rule 8.5 a constant tuned on one altitude would not transfer to
    // another anyway.
    if (const char* env = std::getenv("VIO_PARALLAX_LAMBDA")) {
        const double lambda = std::atof(env);
        options.updater_opt.parallax_noise_lambda = lambda;
        options.slam_updater_opt.parallax_noise_lambda = lambda;
        options.aruco_updater_opt.parallax_noise_lambda = lambda;
    }
    options.zupt_max_velocity = 0.02;
    options.zupt_noise_multiplier = 10.0;
    options.zupt_max_disparity = 0.20;
    options.calib_camimu_dt = 0.0; // not provided by the dataset

    options.enable_slam = true;
    // TESTED dt_slam_delay=0.3 (vs EuRoC's 1.0) on 40_4.bag: slam_features
    // stuck at 4 either way (not the limiter), ATE got slightly worse
    // (12.92m vs 12.81m). Reverted.
    options.dt_slam_delay = 1.0;
    options.enable_zupt = false;
    options.init_wait_for_jerk = false;
    options.enable_hover_detection = false;

    // Triangulation thresholds. PLACEHOLDER: copied from euroc_options.hpp
    // verbatim. This is the constant most likely to be wrong here -- EuRoC's
    // max_dist=75.0 was tuned against <5m indoor depths, and this dataset's
    // scenes are 40-100m altitude. Must be measured against real triangulated
    // depths on this dataset before trusting any accuracy number produced
    // with it unchanged.
    options.feat_init_opt = core::FeatureInitializerOptions{};
    // TESTED max_cond_number=3000.0 (vs EuRoC's 10000.0) on 40_4.bag:
    // rejected 22147 vs 14383 triangulations (fewer, "cleaner" points), but
    // ATE got worse (13.06m vs 12.81m) and retire_chi2 was unchanged --
    // confirms the bottleneck is depth/baseline geometry, not triangulation
    // gate strictness. Reverted.
    options.feat_init_opt.max_cond_number = 10000.0;
    // T-004 ranked the triangulation reject counters on 80_4 and found
    // reject_cond=64947 against reject_maxdist=13744 -- conditioning is the
    // dominant gate at altitude by 4.7x, and this constant had never been
    // swept upward. Direction is not obvious: 3000 (stricter) was measured
    // worse on 40_4, which says nothing about what looser does at 80 m.
    if (const char* env = std::getenv("VIO_MAX_COND")) {
        const double c = std::atof(env);
        if (c > 0.0) options.feat_init_opt.max_cond_number = c;
    }
    options.feat_init_opt.min_dist = 0.10;
    if (const char* env = std::getenv("VIO_MIN_DIST")) {
        const double d = std::atof(env);
        if (d > 0.0) options.feat_init_opt.min_dist = d;
    }
    // DERIVED, not tuned. EuRoC's 75.0 was set for <5m indoor scenes and is
    // the single constant that does not survive this dataset's altitude
    // range. A downward camera at altitude h sees its frame corner at slant
    // range h/cos(theta), where theta is the half-diagonal FOV angle:
    //   theta = atan(hypot(W/2,H/2)/fx) = atan(hypot(306,256)/448.3) = 41.67 deg
    //   h= 40m ->  53.5m   (under 75 -- reject_maxdist=0 measured on 40_4)
    //   h= 60m ->  80.3m   (over  75 -- reject_maxdist=43579 on 60_4)
    //   h= 80m -> 107.1m
    //   h=100m -> 133.9m
    // The cap therefore starts silently discarding valid ground features
    // somewhere between 40m and 60m, which is exactly where 60_4.bag
    // diverges while 40_4.bag does not. 200.0 covers this dataset's full
    // 40-100m envelope (133.9m worst case) with margin for terrain relief
    // and off-nadir attitude.
    options.feat_init_opt.max_dist = 200.0;
    // Control knob for the derivation above, so the 75-vs-200 claim can be
    // measured rather than argued: VIO_MAX_DIST=75 reproduces the EuRoC
    // constant this dataset inherited and is expected to reject every ground
    // feature above ~55 m altitude.
    if (const char* env = std::getenv("VIO_MAX_DIST")) {
        const double d = std::atof(env);
        if (d > 0.0) options.feat_init_opt.max_dist = d;
    }
    options.feat_init_opt.max_baseline = 100.0;
    options.feat_init_opt.max_runs = 5;

    // sensor parameters/{left,right}_camera.yaml: intrinsics [fu,fv,cu,cv],
    // radtan distortion_coefficients.
    const double cam0[8] = {448.29308777949757, 447.9978999410104, 301.68106404856, 242.76373276701213,
        -0.1762088977915627, 0.08539238648206524, 0.0002462342497331716, 0.00032291209479618914};
    const double cam1[8] = {446.38011635796755, 445.88719476806966, 306.68683920642184, 246.3972559515397,
        -0.17142203909651055, 0.07427102814683195, 0.00013522240812824837, 7.182088104512626e-05};
    core::init_camera(options.cam_models[0], core::CameraModelType::RADTAN, kImageWidth, kImageHeight, cam0);
    core::init_camera(options.cam_models[1], core::CameraModelType::RADTAN, kImageWidth, kImageHeight, cam1);

    // T_cam_imu = inverse(T_BS), same convention as EuRoC (imu.yaml's T_BS is
    // identity, so body frame == IMU frame here too). Right camera's
    // recovered translation (~0.300 m) matches the paper's stated 30 cm
    // baseline -- confirms the inversion, not assumed.
    const double T0[16] = {
        0.00555786, 0.99994638, 0.00873793, -3.7108037843e-05,
        0.99997899, -0.00552847, -0.00338442, -0.0007379110477530999,
        -0.00333593, 0.00875655, -0.9999561, 0.0001103779992541,
        0.0, 0.0, 0.0, 1.0};
    const double T1[16] = {
        0.00332251, 0.9999335, 0.01104372, -0.30024620738862906,
        0.99999448, -0.00332146, -0.00011356, -0.001160213780142,
        -7.687e-05, 0.01104404, -0.99993901, -0.0047984970499535,
        0.0, 0.0, 0.0, 1.0};
    set_pose(options.camera_extrinsics[0], T0);
    set_pose(options.camera_extrinsics[1], T1);
    return options;
}

} // namespace mid_altitude
