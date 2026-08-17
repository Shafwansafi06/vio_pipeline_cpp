// EuRoC MAV configuration, shared by the ROS bag runner and the ROS-free ASL
// runner so the two can never drift apart. Extracted verbatim from
// ros/vio_rosbag_runner_euroc.cpp.
//
// Calibration values are read directly from the dataset's own ASL-format
// sensor.yaml files (mav0/cam0, mav0/cam1, mav0/imu0) -- the same physical
// VI-sensor rig is used across all EuRoC sequences (MH_*, V1_*, V2_*).
// T_cam_imu = inverse(T_BS) since EuRoC's body frame is defined to coincide
// with the IMU frame (imu0's own T_BS is identity).
#pragma once

#include "../core/tracker.hpp"
#include "../msckf/vio_manager.hpp"
#include "../type/quat_ops.hpp"

namespace euroc {

constexpr int kImageWidth = 752;
constexpr int kImageHeight = 480;
constexpr const char* kLeftTopic = "/cam0/image_raw";
constexpr const char* kRightTopic = "/cam1/image_raw";
constexpr const char* kImuTopic = "/imu0";
constexpr const char* kGroundTruthTopic = "/leica/position";

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

inline msckf::VioManagerOptions make_euroc_options() {
    msckf::VioManagerOptions options;
    options.gravity_mag = 9.81;
    options.num_cameras = 2;
    options.state_opt.do_fej = true;
    options.state_opt.integration_method = msckf::IntegrationMethod::RK4;
    options.state_opt.do_calib_camera_pose = false;
    options.state_opt.do_calib_camera_intrinsics = true;
    options.state_opt.do_calib_camera_timeoffset = false;
    options.state_opt.max_clone_size = 11;
    options.state_opt.max_slam_features = 50;
    options.state_opt.max_slam_in_update = 25;
    // Official caps its MSCKF update at 40 features, keeping the longest tracks
    // (VioManager.cpp:508-524). DOD ignored max_msckf_in_update entirely and fed
    // every feature until 2026-08-15. The cap is now implemented; the value is
    // measured, not copied, because DOD's feature supply is not official's:
    //
    //   cap    25     40     50     75     100    150+/uncapped
    //   MH_01  0.1334 0.1309 0.1190 0.1132 0.1168 0.1213
    //   V1_01  --     0.0506 --     0.0494 0.0509 0.0487
    //
    // 40 (official's value) is clearly wrong for DOD. 75 is the best MH_01 point
    // and within 1.4% of V1_01's best, so it is the shipped compromise.
    options.state_opt.max_msckf_in_update = 75;
    // Official's euroc_mav config: GLOBAL_3D for MSCKF, anchored inverse depth
    // for SLAM. DOD ignored feat_rep_msckf entirely and ran everything anchored
    // until 2026-08-15. Switching MSCKF to global is neutral on MH_01 (0.121345
    // either way, differing in the 8th digit) and better on V1_01
    // (0.0506 -> 0.0487).
    options.state_opt.feat_rep_msckf = type::LandmarkRepresentation::GLOBAL_3D;
    options.state_opt.feat_rep_slam = type::LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
    options.init_opt.init_window_time = 2.0;
    // Official's value for every EuRoC sequence. This was 0.25 for a long time,
    // tightened because a loose threshold made MH_01 init on a marginal window
    // and ramp its velocity unbounded. That was a symptom of the IMU-coverage
    // bug below, not of the threshold: with coverage guaranteed, MH_01 gives the
    // identical 0.1301 m at 1.5, and 0.25 *breaks* V1_01 outright -- its static
    // init can never fire, the dynamic fallback takes over and diverges to
    // 6.98 m. At 1.5, V1_01 is 0.0506 m.
    options.init_opt.init_imu_thresh = 1.5;
    options.init_opt.init_max_disparity = 5.0;
    options.init_opt.init_max_features = 50;
    options.init_opt.gravity_mag = options.gravity_mag;
    // Dynamic (linear MLE) initializer: MH_01 has no clean stationary start,
    // so the static still-window init only fires ~21 s in. The dynamic
    // initializer recovers gravity/velocity/depths from motion and fires
    // ~3 s in (matches official OpenVINS's EuRoC path). Static is still tried
    // first every frame; dynamic is the fallback that wins here.
    options.init_opt.init_dyn_use = true;
    options.init_opt.init_dyn_num_pose = 5;
    options.init_opt.init_dyn_min_deg = 5.0;
    // Feature-less dynamic initialization (sqrtVINS Sec. V-A): recover velocity
    // and gravity from bearing epipolar geometry plus preintegration, never
    // estimating a 3D point. This REPLACES the init_dyn_zero_velocity
    // workaround: the feature-based solve could not observe v0 over these short
    // windows (MH_01 recovered 0.445 m/s against a truth of 0.048), so its
    // velocity had to be thrown away, and MH_02 then survived only through an
    // unexplained asymmetry between the state value and its FEJ. The
    // feature-less solve recovers 0.042 m/s on MH_01 with gravity within
    // 0.15 deg, so the velocity is usable and the workaround is gone.
    //
    //   sequence   with workaround   feature-less
    //   MH_01      0.1131            0.1282
    //   MH_02      0.1744            0.1595
    //   MH_04      0.4580            0.4416
    //   mean(8)    0.1795            0.1782
    options.init_opt.init_featureless = true;
    options.init_opt.init_dyn_zero_velocity = false;
    // mav0/imu0/sensor.yaml (ADIS16448), noise density / random walk as-shipped.
    options.noises.sigma_a = 2.0000e-3;
    options.noises.sigma_ab = 3.0000e-3;
    options.noises.sigma_w = 1.6968e-4;
    options.noises.sigma_wb = 1.9393e-5;
    options.noises.sigma_a_2 = options.noises.sigma_a * options.noises.sigma_a;
    options.noises.sigma_ab_2 = options.noises.sigma_ab * options.noises.sigma_ab;
    options.noises.sigma_w_2 = options.noises.sigma_w * options.noises.sigma_w;
    options.noises.sigma_wb_2 = options.noises.sigma_wb * options.noises.sigma_wb;
    // 1.0 px, official's up_msckf_sigma_px / up_slam_sigma_px. Was 1.2; moving
    // to official's value improves both sequences (MH_01 0.1301 -> 0.1213,
    // V1_01 0.0528 -> 0.0506).
    options.updater_opt.sigma_pix = 1.0;
    options.updater_opt.sigma_pix_sq = 1.0;
    options.updater_opt.chi2_multipler = 1.0;
    options.slam_updater_opt.sigma_pix = 1.0;
    options.slam_updater_opt.sigma_pix_sq = 1.0;
    options.slam_updater_opt.chi2_multipler = 1.0;
    options.aruco_updater_opt = options.slam_updater_opt;
    options.zupt_max_velocity = 0.02;
    options.zupt_noise_multiplier = 10.0;
    options.zupt_max_disparity = 0.20;
    options.calib_camimu_dt = 0.0; // not provided by the dataset; same for both pipelines under test

    // SLAM ON. It was disabled for a long time on the evidence that it tripled
    // ATE (0.3947 m off, 1.1296 m on). That evidence was an artifact of the ROS
    // bag runner, which processed a stereo pair as soon as both images had
    // arrived -- before the IMU spanning that image had been fed. DOD propagates
    // with whatever is in the buffer, so those updates were built on short
    // propagation. Measured on EuRoC MH_01 through tools/dod_asl_runner.cpp,
    // which guarantees IMU coverage up to each image:
    //
    //     SLAM off                     ATE 0.1900 m
    //     SLAM on                      ATE 0.1301 m      <- official: 0.1333 m
    //
    // and the sensitivity that explains the old numbers, SLAM on:
    //
    //     IMU held back by 5 ms        ATE 9.2225 m, path 170.65 m
    //
    // Official OpenVINS is immune because VioManager queues camera messages and
    // only processes one once the IMU has passed its timestamp. Any DOD
    // transport must do the same; see the guard in ros/vio_rosbag_runner_euroc.cpp.
    options.enable_slam = true;
    options.dt_slam_delay = 1.0;
    options.enable_zupt = false;
    options.init_wait_for_jerk = false; // MH takeoff too gentle for jerk gate; init when still
    options.enable_hover_detection = false;
    // Triangulation thresholds. EuRoC scenes (machine hall / vicon room) have
    // larger, real-distortion depth range than KAIST's rectified close indoor
    // circle, so keep max_dist generous. Values from official's
    // FeatureInitializerOptions defaults, loosened max_dist/max_baseline for
    // the larger scenes.
    options.feat_init_opt = core::FeatureInitializerOptions{};
    options.feat_init_opt.max_cond_number = 10000.0;
    options.feat_init_opt.min_dist = 0.10;
    options.feat_init_opt.max_dist = 75.0;
    options.feat_init_opt.max_baseline = 100.0;
    options.feat_init_opt.max_runs = 5;

    // mav0/cam0, mav0/cam1 sensor.yaml: intrinsics [fu,fv,cu,cv], radtan distortion_coefficients.
    const double cam0[8] = {458.654, 457.296, 367.215, 248.375,
        -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};
    const double cam1[8] = {457.587, 456.134, 379.999, 255.238,
        -0.28368365, 0.07451284, -0.00010473, -3.55590700e-05};
    core::init_camera(options.cam_models[0], core::CameraModelType::RADTAN, kImageWidth, kImageHeight, cam0);
    core::init_camera(options.cam_models[1], core::CameraModelType::RADTAN, kImageWidth, kImageHeight, cam1);

    // calib_IMUtoCAM = T_cam_imu (R_ItoC, p_IinC), which for EuRoC = inverse of
    // the dataset's T_BS (T_BS maps camera->body/IMU, so T_cam_imu = inverse).
    // DOD's KAIST runner uses kalibr's T_cam_imu directly and works, confirming
    // this convention.
    const double T0[16] = {
        0.0148655429818, 0.999557249008, -0.0257744366974, 0.06522291331214665,
        -0.999880929698, 0.0149672133247, 0.00375618835797, -0.02070639072309887,
        0.00414029679422, 0.025715529948, 0.999660727178, -0.008054603453164811,
        0.0, 0.0, 0.0, 1.0};
    const double T1[16] = {
        0.0125552670891, 0.999598781151, -0.0253898008918, -0.04490198068735834,
        -0.999755099723, 0.0130119051815, 0.0179005838253, -0.02056977306809739,
        0.0182237714554, 0.0251588363115, 0.999517347078, -0.008638136949756423,
        0.0, 0.0, 0.0, 1.0};
    set_pose(options.camera_extrinsics[0], T0);
    set_pose(options.camera_extrinsics[1], T1);
    return options;
}

} // namespace euroc
