// TUM VI Benchmark configuration (Schubert et al., "The TUM VI Benchmark for
// Evaluating Visual-Inertial Odometry", IROS 2018). Independent test dataset:
// never used to tune anything in this pipeline.
//
// Calibration is the dataset's own pinhole-equi-512 release (Kalibr format,
// cvg.cit.tum.de/_media/data/datasets/visual-inertial-dataset/pinhole-equi-512.zip),
// read directly, not re-fit: camchain-imucam-imucalib.yaml gives T_cam_imu in
// the same convention DOD's KAIST runner already uses, and
// imu-imucalib.yaml gives the noise densities. Cameras are 512x512 fisheye,
// modelled as Kannala-Brandt 4 (EQUIDISTANT, k1..k4) — the model the dataset's
// "pinhole" release is fitted with.
//
// Every numeric knob below other than calibration is a PLACEHOLDER copied
// from euroc_options.hpp and marked as such. Per CLAUDE.md/dod-style rule
// 8.5, none of them are trustworthy until measured on this dataset's own
// sequences — that is the point of an independent test.
#pragma once

#include "../core/tracker.hpp"
#include "../msckf/vio_manager.hpp"
#include "../type/quat_ops.hpp"

namespace tumvi {

constexpr int kImageWidth = 512;
constexpr int kImageHeight = 512;
constexpr const char* kLeftTopic = "/cam0/image_raw";
constexpr const char* kRightTopic = "/cam1/image_raw";
constexpr const char* kImuTopic = "/imu0";

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

inline msckf::VioManagerOptions make_tumvi_options() {
    msckf::VioManagerOptions options;
    options.gravity_mag = 9.81;
    options.num_cameras = 2;
    // PLACEHOLDER: EuRoC estimator switches, not measured on TUM VI.
    options.state_opt.do_fej = true;
    options.state_opt.integration_method = msckf::IntegrationMethod::RK4;
    options.state_opt.do_calib_camera_pose = false;
    options.state_opt.do_calib_camera_intrinsics = true;
    options.state_opt.do_calib_camera_timeoffset = false;
    // PLACEHOLDER: EuRoC window sizes. TUM VI rooms are small-scale handheld
    // indoor motion, same order of scene depth as EuRoC's vicon rooms.
    options.state_opt.max_clone_size = 11;
    options.state_opt.max_slam_features = 50;
    options.state_opt.max_slam_in_update = 25;
    // PLACEHOLDER: EuRoC's measured compromise (see euroc_options.hpp).
    options.state_opt.max_msckf_in_update = 75;
    // PLACEHOLDER: EuRoC's representation split.
    options.state_opt.feat_rep_msckf = type::LandmarkRepresentation::GLOBAL_3D;
    options.state_opt.feat_rep_slam = type::LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
    options.init_opt.init_window_time = 2.0;
    // PLACEHOLDER: EuRoC init thresholds. TUM VI room sequences begin with the
    // device standing still, so the static init path should fire here.
    options.init_opt.init_imu_thresh = 1.5;
    options.init_opt.init_max_disparity = 5.0;
    options.init_opt.init_max_features = 50;
    options.init_opt.gravity_mag = options.gravity_mag;
    options.init_opt.init_dyn_use = true;
    options.init_opt.init_dyn_num_pose = 5;
    options.init_opt.init_dyn_min_deg = 5.0;
    options.init_opt.init_featureless = true;
    options.init_opt.init_dyn_zero_velocity = false;
    // imu-imucalib.yaml (BMI055), noise density / random walk as-shipped.
    options.noises.sigma_a = 2.8e-3;
    options.noises.sigma_ab = 8.6e-4;
    options.noises.sigma_w = 1.6e-4;
    options.noises.sigma_wb = 2.2e-5;
    options.noises.sigma_a_2 = options.noises.sigma_a * options.noises.sigma_a;
    options.noises.sigma_ab_2 = options.noises.sigma_ab * options.noises.sigma_ab;
    options.noises.sigma_w_2 = options.noises.sigma_w * options.noises.sigma_w;
    options.noises.sigma_wb_2 = options.noises.sigma_wb * options.noises.sigma_wb;
    // PLACEHOLDER: 1.0 px, official's / EuRoC's value.
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
    options.calib_camimu_dt = 0.0; // imu-imucalib.yaml time_offset is 0.0

    options.enable_slam = true;
    options.dt_slam_delay = 1.0;
    options.enable_zupt = false;
    options.init_wait_for_jerk = false;
    options.enable_hover_detection = false;
    // Triangulation thresholds. PLACEHOLDER: EuRoC's values; the room scenes
    // are indoor at similar depth range to EuRoC's vicon rooms.
    options.feat_init_opt = core::FeatureInitializerOptions{};
    options.feat_init_opt.max_cond_number = 10000.0;
    options.feat_init_opt.min_dist = 0.10;
    options.feat_init_opt.max_dist = 75.0;
    options.feat_init_opt.max_baseline = 100.0;
    options.feat_init_opt.max_runs = 5;

    // pinhole-equi-512/camchain-imucam-imucalib.yaml: intrinsics
    // [fu,fv,cu,cv], equidistant (Kannala-Brandt 4) distortion [k1,k2,k3,k4].
    const double cam0[8] = {190.97847715128717, 190.9733070521226, 254.93170605935475, 256.8974428996504,
        0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202, 0.00020293673591811182};
    const double cam1[8] = {190.44236969414825, 190.4344384721956, 252.59949716835982, 254.91723064636983,
        0.0034003170790442797, 0.001766278153469831, -0.00266312569781606, 0.0003299517423931039};
    core::init_camera(options.cam_models[0], core::CameraModelType::EQUIDISTANT, kImageWidth, kImageHeight, cam0);
    core::init_camera(options.cam_models[1], core::CameraModelType::EQUIDISTANT, kImageWidth, kImageHeight, cam1);

    // camchain-imucam-imucalib.yaml T_cam_imu verbatim (same convention as the
    // KAIST runner's kalibr T_cam_imu).
    const double T0[16] = {
        -0.9995250378696743, 0.029615343885863205, -0.008522328211654736, 0.04727988224914392,
        0.0075019185074052044, -0.03439736061393144, -0.9993800792498829, -0.047443232143367084,
        -0.02989013031643309, -0.998969345370175, 0.03415885127385616, -0.0681999605066297,
        0.0, 0.0, 0.0, 1.0};
    const double T1[16] = {
        -0.9995110484978581, 0.030299116376600627, -0.0077218830287333565, -0.053697434688869734,
        0.008104079263822521, 0.012511643720192351, -0.9998888851620987, -0.046131737923635924,
        -0.030199136245891378, -0.9994625667418545, -0.012751072573940885, -0.07149261284195751,
        0.0, 0.0, 0.0, 1.0};
    set_pose(options.camera_extrinsics[0], T0);
    set_pose(options.camera_extrinsics[1], T1);
    return options;
}

} // namespace tumvi
