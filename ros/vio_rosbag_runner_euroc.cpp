// EuRoC MAV variant of vio_rosbag_runner.cpp. Same estimator wiring; only the
// topics, image resolution, and calibration differ. Calibration values below
// are read directly from the dataset's own ASL-format sensor.yaml files
// (mav0/cam0, mav0/cam1, mav0/imu0) -- the same physical VI-sensor rig is
// used across all EuRoC sequences (MH_*, V1_*, V2_*), so these are valid for
// MH_01_easy.bag even though they were read from the vicon_room1/V1_01_easy
// extraction. T_cam_imu = inverse(T_BS) since EuRoC's body frame is defined
// to coincide with the IMU frame (imu0's own T_BS is identity).
#include "../arena.hpp"
#include "../core/tracker.hpp"
#include "../msckf/vio_manager.hpp"
#include "../type/quat_ops.hpp"

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PointStamped.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kImageWidth = 752;
constexpr int kImageHeight = 480;
constexpr const char* kLeftTopic = "/cam0/image_raw";
constexpr const char* kRightTopic = "/cam1/image_raw";
constexpr const char* kImuTopic = "/imu0";
constexpr const char* kGroundTruthTopic = "/leica/position";

void set_pose(double target[7], const double matrix[16]) {
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

msckf::VioManagerOptions make_euroc_options() {
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
    options.state_opt.max_msckf_in_update = 50;
    options.state_opt.feat_rep_msckf = type::LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
    options.state_opt.feat_rep_slam = type::LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
    options.init_opt.init_window_time = 2.0;
    // MH_01 is carried/jostled for the first ~6 s (a_var~2.0) and is only
    // genuinely still (a_var~0.09) around t=21-23 s. A loose thresh inits on a
    // marginal window (tilted gravity -> velocity ramps unbounded), so keep it
    // tight enough to reject everything until the true stationary window.
    options.init_opt.init_imu_thresh = 0.25;
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
    // mav0/imu0/sensor.yaml (ADIS16448), noise density / random walk as-shipped.
    options.noises.sigma_a = 2.0000e-3;
    options.noises.sigma_ab = 3.0000e-3;
    options.noises.sigma_w = 1.6968e-4;
    options.noises.sigma_wb = 1.9393e-5;
    options.noises.sigma_a_2 = options.noises.sigma_a * options.noises.sigma_a;
    options.noises.sigma_ab_2 = options.noises.sigma_ab * options.noises.sigma_ab;
    options.noises.sigma_w_2 = options.noises.sigma_w * options.noises.sigma_w;
    options.noises.sigma_wb_2 = options.noises.sigma_wb * options.noises.sigma_wb;
    options.updater_opt.sigma_pix = 1.2;
    options.updater_opt.sigma_pix_sq = 1.2 * 1.2;
    options.updater_opt.chi2_multipler = 1.0;
    options.slam_updater_opt.sigma_pix = 1.2;
    options.slam_updater_opt.sigma_pix_sq = 1.2 * 1.2;
    options.slam_updater_opt.chi2_multipler = 1.0;
    options.aruco_updater_opt = options.slam_updater_opt;
    options.zupt_max_velocity = 0.02;
    options.zupt_noise_multiplier = 10.0;
    options.zupt_max_disparity = 0.20;
    options.calib_camimu_dt = 0.0; // not provided by the dataset; same for both pipelines under test

    // Same estimator fixes that made KAIST work (this runner predated them).
    options.enable_slam = false; // SLAM made EuRoC diverge faster; MSCKF-only first
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

double stamp(const sensor_msgs::ImageConstPtr& image) {
    return image ? image->header.stamp.toSec() : -1.0;
}

} // namespace

int main(int argc, char** argv) {
    ros::init(argc, argv, "vio_dod_rosbag_runner_euroc", ros::init_options::AnonymousName);
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s BAG_PATH OUTPUT_PREFIX [max_seconds]\n", argv[0]);
        return 2;
    }
    const double max_seconds = argc > 3 ? std::atof(argv[3]) : -1.0;
    char estimate_path[1024];
    char truth_path[1024];
    char timing_path[1024];
    std::snprintf(estimate_path, sizeof(estimate_path), "%s_estimate.csv", argv[2]);
    std::snprintf(truth_path, sizeof(truth_path), "%s_groundtruth.csv", argv[2]);
    std::snprintf(timing_path, sizeof(timing_path), "%s_timing.csv", argv[2]);
    std::FILE* estimate = std::fopen(estimate_path, "w");
    std::FILE* truth = std::fopen(truth_path, "w");
    std::FILE* timing = std::fopen(timing_path, "w");
    if (!estimate || !truth || !timing) {
        std::fprintf(stderr, "failed to create output CSV files\n");
        return 3;
    }
    std::fprintf(estimate, "timestamp,px,py,pz,qx,qy,qz,qw,features,clones\n");
    // Ground truth here is Leica position-only (no orientation); qx..qw are
    // written as an identity placeholder so the CSV schema matches the
    // KAIST runner's (evaluate_trajectory.py only reads px,py,pz).
    std::fprintf(truth, "timestamp,px,py,pz,qx,qy,qz,qw\n");
    std::fprintf(timing, "timestamp,tracking_ms,estimator_ms,total_ms,observations\n");

    ArenaAllocator global_arena(MiB(96));
    auto* vio = global_arena.allocate<msckf::VioManagerData>();
    auto* tracker = global_arena.allocate<core::TrackerData>();
    void* image_storage = global_arena.allocate(core::tracker_image_storage_bytes(kImageWidth, kImageHeight), 64);
    auto* observations = static_cast<core::Feature*>(
        global_arena.allocate(sizeof(core::Feature) * core::TRACKER_MAX_FEATURES * 2U, alignof(core::Feature)));
    if (!vio || !tracker || !image_storage || !observations) return 4;

    const msckf::VioManagerOptions options = make_euroc_options();
    msckf::init_vio_manager(*vio, options);
    core::TrackerOptions tracker_options;
    if (!core::init_tracker(*tracker, tracker_options, options.cam_models, 2,
                            image_storage, core::tracker_image_storage_bytes(kImageWidth, kImageHeight))) return 5;
    cv::setNumThreads(4);
    cv::setRNGSeed(0);

    rosbag::Bag bag;
    bag.open(argv[1], rosbag::bagmode::Read);
    rosbag::View view(bag);
    sensor_msgs::ImageConstPtr left;
    sensor_msgs::ImageConstPtr right;
    double first_timestamp = -1.0;
    std::uint64_t frames = 0;
    double tracking_sum = 0.0;
    double estimator_sum = 0.0;

    for (const rosbag::MessageInstance& message : view) {
        const double message_time = message.getTime().toSec();
        if (first_timestamp < 0.0) first_timestamp = message_time;
        if (max_seconds > 0.0 && message_time - first_timestamp > max_seconds) break;
        const std::string topic = message.getTopic();
        if (topic == kImuTopic) {
            const sensor_msgs::ImuConstPtr imu = message.instantiate<sensor_msgs::Imu>();
            if (!imu) continue;
            core::ImuData data;
            data.timestamp = imu->header.stamp.toSec();
            data.wm << imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z;
            data.am << imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z;
            msckf::feed_measurement_imu(*vio, data);
        } else if (topic == kGroundTruthTopic) {
            const geometry_msgs::PointStampedConstPtr point = message.instantiate<geometry_msgs::PointStamped>();
            if (point) std::fprintf(truth, "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                point->header.stamp.toSec(), point->point.x, point->point.y, point->point.z,
                0.0, 0.0, 0.0, 1.0);
        } else if (topic == kLeftTopic) {
            left = message.instantiate<sensor_msgs::Image>();
        } else if (topic == kRightTopic) {
            right = message.instantiate<sensor_msgs::Image>();
        } else {
            continue;
        }

        if (!left || !right) continue;
        const double delta = stamp(left) - stamp(right);
        if (std::abs(delta) > 0.003) {
            if (delta < 0.0) left.reset(); else right.reset();
            continue;
        }

        const auto total_start = std::chrono::steady_clock::now();
        const cv_bridge::CvImageConstPtr left_cv = cv_bridge::toCvShare(left, "mono8");
        const cv_bridge::CvImageConstPtr right_cv = cv_bridge::toCvShare(right, "mono8");
        const auto tracking_start = std::chrono::steady_clock::now();
        const double camera_timestamp = 0.5 * (stamp(left) + stamp(right));
        const int observation_count = core::track_stereo_frame(*tracker, camera_timestamp,
            left_cv->image, right_cv->image, observations, core::TRACKER_MAX_FEATURES * 2);
        const auto estimator_start = std::chrono::steady_clock::now();
        msckf::feed_measurement_camera_tracks(*vio, camera_timestamp, observations, observation_count);
        const auto done = std::chrono::steady_clock::now();
        const double tracking_ms = std::chrono::duration<double, std::milli>(estimator_start - tracking_start).count();
        const double estimator_ms = std::chrono::duration<double, std::milli>(done - estimator_start).count();
        const double total_ms = std::chrono::duration<double, std::milli>(done - total_start).count();
        tracking_sum += tracking_ms;
        estimator_sum += estimator_ms;
        ++frames;
        std::fprintf(timing, "%.9f,%.6f,%.6f,%.6f,%d\n", camera_timestamp,
                     tracking_ms, estimator_ms, total_ms, observation_count);
        if (vio->is_initialized) {
            const double* value = vio->state.imu.value;
            std::fprintf(estimate, "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%zu,%d\n",
                vio->state.timestamp, value[4], value[5], value[6], value[0], value[1],
                value[2], value[3], vio->db.count, vio->state.num_clones);
        }
        left.reset();
        right.reset();
        if ((frames % 250U) == 0U) std::fprintf(stderr,
            "frames=%llu initialized=%d tracks=%d mean_track_ms=%.3f mean_estimator_ms=%.3f\n",
            static_cast<unsigned long long>(frames), int(vio->is_initialized), tracker->previous_count[0],
            tracking_sum / double(frames), estimator_sum / double(frames));
    }

    bag.close();
    std::fclose(estimate);
    std::fclose(truth);
    std::fclose(timing);
    std::fprintf(stderr, "complete frames=%llu initialized=%d tracking_mean_ms=%.6f estimator_mean_ms=%.6f\n",
        static_cast<unsigned long long>(frames), int(vio->is_initialized),
        frames ? tracking_sum / double(frames) : 0.0, frames ? estimator_sum / double(frames) : 0.0);
    return vio->is_initialized ? 0 : 6;
}
