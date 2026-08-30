#include "../arena.hpp"
#include "../core/tracker.hpp"
#include "../msckf/vio_manager.hpp"
#include "../type/quat_ops.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <deque>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr const char* kLeftTopic = "/camera/infra1/image_rect_raw";
constexpr const char* kRightTopic = "/camera/infra2/image_rect_raw";
constexpr const char* kImuTopic = "/mavros/imu/data";
constexpr const char* kGroundTruthTopic = "/pose_transformed";

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

msckf::VioManagerOptions make_kaist_options() {
    msckf::VioManagerOptions options;
    options.gravity_mag = 9.81;
    options.num_cameras = 2;
    options.state_opt.do_fej = true;
    options.state_opt.integration_method = msckf::IntegrationMethod::RK4;
    options.state_opt.do_calib_camera_pose = false;
    // Official's kaist config disables intrinsic calibration outright and
    // enables timeoffset instead, both because the motion is degenerate.
    options.state_opt.do_calib_camera_intrinsics =
        std::getenv("VIO_CALIB_INTR") != nullptr ? std::atoi(std::getenv("VIO_CALIB_INTR")) != 0 : true;
    options.state_opt.do_calib_camera_timeoffset = false;
    options.state_opt.max_clone_size = 11;
    options.state_opt.max_slam_features = 50;
    options.state_opt.max_slam_in_update = 25;
    // NOTE: this was dead code until 2026-08-15 -- nothing read
    // max_msckf_in_update, so this runner has always fed every MSCKF feature.
    // Now that the cap is implemented, keep KAIST uncapped unless overridden:
    // capping was measured to hurt here long before it was live (circle.bag
    // 0.785 -> 1.043 m, infinity 1.070 -> 1.306 m).
    // 50, official's own kaist value -- and NOT the 75 that EuRoC wants. The
    // best cap is dataset-dependent, which is worth remembering before copying
    // any tuned constant across datasets:
    //
    //     circle.bag, SLAM on:  uncapped 0.0650   cap 75 0.0459   cap 50 0.0374
    options.state_opt.max_msckf_in_update =
        std::getenv("VIO_MAX_MSCKF") != nullptr ? std::atoi(std::getenv("VIO_MAX_MSCKF")) : 50;
    // Match official OpenVINS kaist_vio: feat_rep_slam = ANCHORED_MSCKF_INVERSE_DEPTH.
    // The SLAM Jacobian (updater_slam_helper) is already computed in inverse-depth
    // params, but the landmark was stored/updated as raw xyz (GLOBAL_3D default) --
    // a representation mismatch (Jacobian in inverse-depth space, EKF correction
    // applied in xyz space). Setting this makes storage/update consistent with
    // the Jacobian and matches official.
    options.state_opt.feat_rep_slam = type::LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
    options.state_opt.feat_rep_msckf = type::LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
    options.state_opt.feat_rep_slam = type::LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
    options.init_opt.init_window_time = 2.0;
    options.init_opt.init_imu_thresh = 0.60;
    options.init_opt.init_max_disparity = 5.0;
    options.init_opt.init_max_features = 50;
    options.init_opt.gravity_mag = options.gravity_mag;
    options.noises.sigma_a = 0.07;
    options.noises.sigma_ab = 0.009;
    options.noises.sigma_w = 0.001;
    options.noises.sigma_wb = 0.0003;
    options.noises.sigma_a_2 = options.noises.sigma_a * options.noises.sigma_a;
    options.noises.sigma_ab_2 = options.noises.sigma_ab * options.noises.sigma_ab;
    options.noises.sigma_w_2 = options.noises.sigma_w * options.noises.sigma_w;
    options.noises.sigma_wb_2 = options.noises.sigma_wb * options.noises.sigma_wb;
    const double kaist_sigma_pix =
        std::getenv("VIO_SIGMA_PIX") != nullptr ? std::atof(std::getenv("VIO_SIGMA_PIX")) : 1.2;
    options.updater_opt.sigma_pix = kaist_sigma_pix;
    options.updater_opt.sigma_pix_sq = kaist_sigma_pix * kaist_sigma_pix;
    options.updater_opt.chi2_multipler = 1.0;
    options.slam_updater_opt.sigma_pix = kaist_sigma_pix;
    options.slam_updater_opt.sigma_pix_sq = kaist_sigma_pix * kaist_sigma_pix;
    options.slam_updater_opt.chi2_multipler = 1.0;
    options.aruco_updater_opt = options.slam_updater_opt;
    // Triangulation thresholds set to official OpenVINS's EXACT effective
    // KAIST values, for line-by-line logical parity with official's
    // FeatureInitializer. Official's kaist_vio/estimator_config.yaml +
    // FeatureInitializerOptions.h defaults:
    //   fi_max_cond_number: 25000, fi_min_dist: 0.10, fi_max_dist: 10.0,
    //   fi_max_baseline: 200, fi_max_runs (struct default): 5.
    // DOD previously used stricter values (max_cond_number=10000, min_dist=0.25,
    // max_dist=40, max_baseline=40, max_runs=10) which rejected far more
    // low-parallax triangulations than official does -- the direct cause of
    // DOD's ~9% single-shot MSCKF accept rate vs official's ~70% on this
    // dataset. Aligning to official exactly.
    // Exact official OpenVINS KAIST triangulation values (logical parity).
    // NOTE: max_cond_number=25000 gives a big infinity.bag accuracy gain
    // (1.114 -> 0.479 m) but circle.bag diverges at frame ~4030 -- a
    // DOD-specific robustness gap at that trajectory segment (official's
    // more mature upstream feature quality avoids feeding the degenerate
    // low-parallax feature that tips DOD over there). Set
    // max_cond_number=10000 (DOD guard) instead to keep circle.bag stable
    // at the cost of infinity accuracy, until the frame-4030 robustness
    // bug is fixed.
    options.feat_init_opt = core::FeatureInitializerOptions{};
    options.feat_init_opt.max_cond_number = 25000.0;
    options.feat_init_opt.min_dist = 0.10;
    options.feat_init_opt.max_dist = 10.0;
    options.feat_init_opt.max_baseline = 200.0;
    options.feat_init_opt.max_runs = 5;
    // ZUPT stays off here. Note that config/kaist_vio/estimator_config.yaml --
    // and the config official is benchmarked with -- both say try_zupt: true;
    // the comment that used to sit here claimed otherwise and was wrong. What
    // matters for parity is not the ZUPT updates themselves but what official
    // derives from them, which is the initialisation trigger below.
    options.enable_zupt = std::getenv("VIO_ENABLE_ZUPT") != nullptr
                              ? std::atoi(std::getenv("VIO_ENABLE_ZUPT")) != 0
                              : false;

    // Initialisation stays on the takeoff jerk (init_wait_for_jerk, default
    // true). Initialising while still was tried on 2026-08-18 and measured
    // worse; the note is here so it is not retried blind.
    //
    // The motivation was real. Official does not configure the trigger
    // directly: VioManagerHelper.cpp:106 sets
    // `wait_for_jerk = (updaterZUPT == nullptr)`, and the KAIST config it is
    // benchmarked with has try_zupt: true, so official initialises on the first
    // stationary window. DOD waits for accelerometer variance above
    // init_imu_thresh=0.60 while the measured variance on circle.bag is 0.048,
    // the noise floor of a stationary platform (VIO_INIT_DEBUG prints it), so
    // the first pose lands at t+25.4 s against official's t+3.7 s. Note this is
    // circle.bag alone -- infinite.bag already initialises at t+2.8 s.
    //
    // Flipping it does buy the coverage and does drain the pre-init feature
    // accumulation (meas_overflow 182,939 -> 8,960 on circle), but the
    // trajectory it produces is worse, and not merely because more of it is
    // being scored. Compared over the window BOTH versions cover:
    //
    //                        circle            infinite
    //   jerk-init (shipped)  0.0375            0.0261
    //   still-init           0.0464            0.0363
    //   still-init + ZUPT    0.0464            0.0321
    //   official             0.0317            0.0283
    //
    // So the late initialisation is not what costs the accuracy on KAIST:
    // starting earlier from a stationary window gives a worse state than
    // starting later from an excited one, by ~24% on both sequences. Official
    // beats us from a still start because it pairs that start with ZUPT
    // configured as zupt_only_at_beginning + disparity-only gating
    // (zupt_chi2_multipler: 0), neither of which this pipeline implements --
    // our ZUPT runs for the whole sequence. That is the experiment worth
    // running next, not the trigger on its own.
    //
    // VIO_ENABLE_ZUPT toggles ZUPT for that experiment; it stays off by default.
    // Official OpenVINS uses 50 SLAM landmarks for KAIST (max_slam: 50) --
    // persistent features that cut lap-to-lap drift -- and this was flipped on
    // to match, on the reasoning that the earlier crash belonged to a buggier
    // pipeline.
    //
    // MEASURED 2026-08-14: it does not hold. circle.bag with SLAM on gives
    // ATE 39.0 m and a 264 m estimated path against 29.5 m of ground truth;
    // with SLAM off the same build gives ATE 0.666 m and a 27.6 m path. A 60x
    // regression, and it is what made circle.bag stop reproducing the 0.624 m
    // recorded in Benchmark 5 -- every later investigation that used circle.bag
    // was measuring this instead of whatever it was looking for.
    //
    // Back to false, which is also VioManagerOptions' default and what the
    // parity manual documents. Re-enable only alongside the SLAM crash
    // root-cause (parity manual, open item #4), with circle.bag re-measured.
    // SLAM ON, matching official's kaist config (max_slam: 50). It was off
    // here for a long time on the evidence that it cost 39 m of ATE on
    // circle.bag -- but that was measured while this runner processed stereo
    // pairs before the IMU spanning them had arrived. With the pair queue in
    // place, SLAM is worth more on KAIST than on EuRoC:
    //
    //     circle.bag    SLAM off 0.3182 m   SLAM on 0.0650 m
    //     infinite.bag  SLAM off 0.2369 m   SLAM on 0.0261 m
    options.enable_slam = std::getenv("VIO_ENABLE_SLAM") != nullptr
                              ? std::atoi(std::getenv("VIO_ENABLE_SLAM")) != 0
                              : true;
    // Disabled: fires on ~100% of frames during a continuous slow circular
    // flight (verified via debug counter -- 0.998-1.000 hover-true rate over
    // 4758 frames), i.e. badly over-triggering. Root cause not found; the
    // camera-IMU extrinsic conjugation bug (bearings are in camera frame, not
    // IMU frame) was found and fixed, which resolved a catastrophic 43m-ATE
    // divergence down to 1.1m, but the over-triggering itself persists, so
    // this is not yet a net win over hover detection off. See
    // portdocs/Benchmark.md for the debugging trail.
    options.enable_hover_detection = false;
    options.hover_opt.epsilon = 0.01;
    options.hover_opt.min_features = 8;
    options.hover_opt.hysteresis_frames = 3;
    options.zupt_max_velocity = 0.02;
    options.zupt_noise_multiplier = 10.0;
    options.zupt_max_disparity = 0.20;
    options.calib_camimu_dt = -0.029958533056650416;

    const double cam0[8] = {380.9229090195708, 380.29264802262736,
        324.68121181846755, 224.6741321466431, 0.006896928127777268,
        -0.009144207062654397, 0.000254113977103925, 0.0021434982252719545};
    const double cam1[8] = {380.95187095303424, 380.3065956074995,
        324.0678433553536, 225.9586983198407, 0.007044055287844759,
        -0.010251485722185347, 0.0006674304399871926, 0.001678899816379666};
    core::init_camera(options.cam_models[0], core::CameraModelType::RADTAN, 640, 480, cam0);
    core::init_camera(options.cam_models[1], core::CameraModelType::RADTAN, 640, 480, cam1);

    const double T0[16] = {
        -0.04030123999740945,-0.9989998755524683,0.01936643232049068,0.02103955032447366,
        0.026311325355146964,-0.020436499663524704,-0.9994448777394171,-0.038224929976612206,
        0.9988410905708309,-0.0397693113802049,0.027108627033059024,-0.1363488241088845,
        0.0,0.0,0.0,1.0};
    const double T1[16] = {
        -0.03905752472566068,-0.9990498568899562,0.019336318430946575,-0.02909273113160158,
        0.025035478432625047,-0.020323396666370924,-0.9994799569614147,-0.03811090793611019,
        0.99892328763622,-0.03855311914877835,0.02580547271309183,-0.13656684822705098,
        0.0,0.0,0.0,1.0};
    set_pose(options.camera_extrinsics[0], T0);
    set_pose(options.camera_extrinsics[1], T1);
    return options;
}

double stamp(const sensor_msgs::ImageConstPtr& image) {
    return image ? image->header.stamp.toSec() : -1.0;
}

} // namespace

int main(int argc, char** argv) {
    ros::init(argc, argv, "vio_dod_rosbag_runner", ros::init_options::AnonymousName);
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
    std::fprintf(truth, "timestamp,px,py,pz,qx,qy,qz,qw\n");
    std::fprintf(timing, "timestamp,tracking_ms,estimator_ms,total_ms,observations\n");

    ArenaAllocator global_arena(MiB(96));
    auto* vio = global_arena.allocate<msckf::VioManagerData>();
    auto* tracker = global_arena.allocate<core::TrackerData>();
    void* image_storage = global_arena.allocate(core::tracker_image_storage_bytes(640, 480), 64);
    auto* observations = static_cast<core::Feature*>(
        global_arena.allocate(sizeof(core::Feature) * core::TRACKER_MAX_FEATURES * 2U, alignof(core::Feature)));
    if (!vio || !tracker || !image_storage || !observations) return 4;

    // Env override so a config sweep does not need a rebuild per point
    // (same pattern and same knob name as dod_asl_runner.cpp).
    auto env_int = [](const char* name, int fallback) {
        const char* value = std::getenv(name);
        return value != nullptr ? std::atoi(value) : fallback;
    };
    msckf::VioManagerOptions options = make_kaist_options();
    options.use_schur_msckf = env_int("VIO_SCHUR", options.use_schur_msckf ? 1 : 0) != 0;
    msckf::init_vio_manager(*vio, options);
    core::TrackerOptions tracker_options;
    if (!core::init_tracker(*tracker, tracker_options, options.cam_models, 2,
                            image_storage, core::tracker_image_storage_bytes(640, 480))) return 5;
    cv::setNumThreads(4);
    cv::setRNGSeed(0);

    rosbag::Bag bag;
    bag.open(argv[1], rosbag::bagmode::Read);
    rosbag::View view(bag);
    sensor_msgs::ImageConstPtr left;
    sensor_msgs::ImageConstPtr right;
    double first_timestamp = -1.0;
    double newest_imu_time = -1.0;
    // Stereo pairs waiting for the IMU to reach them. Never dropped.
    std::deque<std::pair<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr>> pending;
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
            newest_imu_time = std::max(newest_imu_time, data.timestamp);
        } else if (topic == kGroundTruthTopic) {
            const geometry_msgs::PoseStampedConstPtr pose = message.instantiate<geometry_msgs::PoseStamped>();
            if (pose) std::fprintf(truth, "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                pose->header.stamp.toSec(), pose->pose.position.x, pose->pose.position.y,
                pose->pose.position.z, pose->pose.orientation.x, pose->pose.orientation.y,
                pose->pose.orientation.z, pose->pose.orientation.w);
        } else if (topic == kLeftTopic) {
            left = message.instantiate<sensor_msgs::Image>();
        } else if (topic == kRightTopic) {
            right = message.instantiate<sensor_msgs::Image>();
        } else {
            continue;
        }

        // Form the stereo pair, then QUEUE it -- do not process it here.
        if (left && right) {
            const double delta = stamp(left) - stamp(right);
            if (std::abs(delta) > 0.003) {
                if (delta < 0.0) left.reset(); else right.reset();
            } else {
                pending.emplace_back(left, right);
                left.reset();
                right.reset();
            }
        }

        // Drain every queued pair the IMU has now passed. Bag order is RECEIVE
        // order, so an image routinely arrives before the IMU samples spanning
        // it, and DOD propagates with whatever is buffered -- 5 ms of IMU lag
        // takes EuRoC MH_01 from 0.11 m to 9.2 m. Official queues camera
        // messages for the same reason.
        //
        // The queue is the point. Holding the pair in `left`/`right` instead
        // lets the next image message OVERWRITE it, silently dropping frames:
        // that cost circle.bag 0.666 -> 60 m ATE (2456 of 3600 frames left).
        while (!pending.empty() && stamp(pending.front().first) <= newest_imu_time) {
            const sensor_msgs::ImageConstPtr pair_left = pending.front().first;
            const sensor_msgs::ImageConstPtr pair_right = pending.front().second;
            pending.pop_front();

        const auto total_start = std::chrono::steady_clock::now();
        const cv_bridge::CvImageConstPtr left_cv = cv_bridge::toCvShare(pair_left, "mono8");
        const cv_bridge::CvImageConstPtr right_cv = cv_bridge::toCvShare(pair_right, "mono8");
        const auto tracking_start = std::chrono::steady_clock::now();
        // Official OpenVINS's callback_stereo uses cam0's own timestamp as
// authoritative (not an average of the stereo pair) -- matched here to
// remove it as a variable, even though the effect is bounded (<=1.5ms,
// non-compounding through IMU integration) since the pair is already
// gated to within 3ms of each other above.
const double camera_timestamp = stamp(pair_left);
        const int observation_count = core::track_stereo_frame(*tracker, camera_timestamp,
            left_cv->image, right_cv->image, observations, core::TRACKER_MAX_FEATURES * 2);
        const auto estimator_start = std::chrono::steady_clock::now();
        msckf::feed_measurement_camera_tracks(*vio, camera_timestamp, observations, observation_count);
        // The tracker keeps its own copy of the camera intrinsics (used for
        // undistort() when computing uv_norm for every observation) separate
        // from the estimator's. When intrinsics are calibrated online, that
        // copy must be kept in sync too, or every subsequent frame's raw
        // measurements are computed with stale intrinsics even though the
        // estimator's own Jacobian/residual buffer (vio->params.cam_models)
        // is now correctly resynced.
        if (options.state_opt.do_calib_camera_intrinsics) {
            for (int camera = 0; camera < options.num_cameras; ++camera) {
                tracker->cameras[camera] = vio->params.cam_models[camera];
            }
        }
        const auto done = std::chrono::steady_clock::now();
        const double tracking_ms = std::chrono::duration<double, std::milli>(estimator_start - tracking_start).count();
        const double estimator_ms = std::chrono::duration<double, std::milli>(done - estimator_start).count();
        const double total_ms = std::chrono::duration<double, std::milli>(done - total_start).count();
        tracking_sum += tracking_ms;
        estimator_sum += estimator_ms;
        ++frames;
#ifdef DOD_DIAG_DEBUG
        if (frames >= 2900 && frames <= 4550) {
            double max_diag = -1e18, min_diag = 1e18;
            for (int i = 0; i < vio->state.cov_size; ++i) {
                const double d = vio->state.Cov(i, i);
                max_diag = std::max(max_diag, d);
                min_diag = std::min(min_diag, d);
            }
            const Eigen::MatrixXd active = vio->state.Cov.topLeftCorner(vio->state.cov_size, vio->state.cov_size);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(active, Eigen::EigenvaluesOnly);
            double min_eig = 1e18, max_eig = -1e18;
            if (solver.info() == Eigen::Success) {
                min_eig = solver.eigenvalues().minCoeff();
                max_eig = solver.eigenvalues().maxCoeff();
            }
            std::fprintf(stderr, "[DOD-DIAG] frame=%llu cov_size=%d max_diag=%.6g min_diag=%.6g "
                "min_eig=%.6g max_eig=%.6g cond=%.6g num_clones=%d db_count=%zu tracks0=%d tracks1=%d\n",
                static_cast<unsigned long long>(frames), vio->state.cov_size, max_diag, min_diag,
                min_eig, max_eig, (min_eig != 0.0 ? max_eig / min_eig : 1e18),
                vio->state.num_clones, vio->db.count, tracker->previous_count[0], tracker->previous_count[1]);

            // IMU-only block (indices 0-14: theta,p,v,bg,ba), isolated from
            // the gauge-dominated global directions, since ranks 2+ at frame
            // 3500 concentrated 85-90% of their weight here.
            if (vio->state.cov_size >= 15) {
                const Eigen::Matrix<double, 15, 15> imu_block = vio->state.Cov.topLeftCorner<15, 15>();
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> imu_solver(imu_block, Eigen::EigenvaluesOnly);
                std::fprintf(stderr, "[DOD-DIAG-IMU] frame=%llu imu_max_eig=%.6g imu_min_eig=%.6g "
                    "p_var=%.6g v_var=%.6g bg_var=%.6g ba_var=%.6g\n",
                    static_cast<unsigned long long>(frames),
                    imu_solver.eigenvalues().maxCoeff(), imu_solver.eigenvalues().minCoeff(),
                    vio->state.Cov(3,3) + vio->state.Cov(4,4) + vio->state.Cov(5,5),
                    vio->state.Cov(6,6) + vio->state.Cov(7,7) + vio->state.Cov(8,8),
                    vio->state.Cov(9,9) + vio->state.Cov(10,10) + vio->state.Cov(11,11),
                    vio->state.Cov(12,12) + vio->state.Cov(13,13) + vio->state.Cov(14,14));
            }

            if (frames == 3500) {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> full_solver(active);
                if (full_solver.info() == Eigen::Success) {
                    auto label_for = [&](const type::Variable* v) -> const char* {
                        if (v == &vio->state.imu) return "imu";
                        if (v == &vio->state.calib_imu_dw) return "calib_imu_dw";
                        if (v == &vio->state.calib_imu_da) return "calib_imu_da";
                        if (v == &vio->state.calib_imu_tg) return "calib_imu_tg";
                        if (v == &vio->state.calib_imu_GYROtoIMU) return "calib_imu_GYROtoIMU";
                        if (v == &vio->state.calib_imu_ACCtoIMU) return "calib_imu_ACCtoIMU";
                        if (v == &vio->state.calib_dt_CAMtoIMU) return "calib_dt_CAMtoIMU";
                        if (v == &vio->state.calib_IMUtoCAM[0]) return "calib_IMUtoCAM[0]";
                        if (v == &vio->state.calib_IMUtoCAM[1]) return "calib_IMUtoCAM[1]";
                        if (v == &vio->state.cam_intrinsics[0]) return "cam_intrinsics[0]";
                        if (v == &vio->state.cam_intrinsics[1]) return "cam_intrinsics[1]";
                        for (int c = 0; c < vio->state.num_clones; ++c)
                            if (v == &vio->state.clones_IMU[c]) return "clone_IMU";
                        for (int s = 0; s < vio->state.num_slam_features; ++s)
                            if (v == &vio->state.features_SLAM[s]) return "slam_feature";
                        return "unknown";
                    };
                    // Print the top 6 eigenvalues (descending) with their
                    // eigenvector's per-variable weight breakdown, not just
                    // the single dominant one -- the dominant direction in
                    // any VIO system is almost always the benign, unobservable
                    // global position/yaw gauge freedom (equal weight on
                    // every clone), which rigid ATE alignment cancels out and
                    // so cannot explain a relative-trajectory divergence. The
                    // real culprit, if any, is more likely in eigenvalue #2
                    // or later.
                    const int n = int(full_solver.eigenvalues().size());
                    std::vector<int> order(n);
                    for (int i = 0; i < n; ++i) order[i] = i;
                    std::sort(order.begin(), order.end(), [&](int a, int b) {
                        return full_solver.eigenvalues()(a) > full_solver.eigenvalues()(b);
                    });
                    for (int rank = 0; rank < std::min(6, n); ++rank) {
                        const int idx = order[rank];
                        const double val = full_solver.eigenvalues()(idx);
                        const Eigen::VectorXd evec = full_solver.eigenvectors().col(idx);
                        std::fprintf(stderr, "[DOD-DIAG-EIGVEC] frame=%llu rank=%d eigval=%.6g\n",
                            static_cast<unsigned long long>(frames), rank, val);
                        for (int v = 0; v < vio->state.num_variables; ++v) {
                            const type::Variable* var = vio->state.variables[v];
                            double sumsq = 0.0;
                            for (int k = 0; k < var->size; ++k) {
                                const int vi = var->id + k;
                                if (vi >= 0 && vi < evec.size()) sumsq += evec(vi) * evec(vi);
                            }
                            if (sumsq > 1e-3) {
                                std::fprintf(stderr, "[DOD-DIAG-EIGVEC]   var=%d label=%s id=%d size=%d weight=%.6g\n",
                                    v, label_for(var), var->id, var->size, sumsq);
                            }
                        }
                    }
                }
            }
        }
#endif
        std::fprintf(timing, "%.9f,%.6f,%.6f,%.6f,%d\n", camera_timestamp,
                     tracking_ms, estimator_ms, total_ms, observation_count);
        if (vio->is_initialized) {
            const double* value = vio->state.imu.value;
            std::fprintf(estimate, "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%zu,%d\n",
                vio->state.timestamp, value[4], value[5], value[6], value[0], value[1],
                value[2], value[3], vio->db.count, vio->state.num_clones);
        }
        if ((frames % 250U) == 0U) std::fprintf(stderr,
            "frames=%llu initialized=%d tracks=%d mean_track_ms=%.3f mean_estimator_ms=%.3f\n",
            static_cast<unsigned long long>(frames), int(vio->is_initialized), tracker->previous_count[0],
            tracking_sum / double(frames), estimator_sum / double(frames));
        }
    }

    bag.close();
    std::fclose(estimate);
    std::fclose(truth);
    std::fclose(timing);
    std::fprintf(stderr,
        "[DB] full_refusals=%ld db_count=%zu slam_features=%d max_meas=%ld meas_overflow=%ld\n",
        core::db_full_refusals, vio->db.count, vio->state.num_slam_features,
        core::dbg_max_meas, core::feat_meas_overflow);
    std::fprintf(stderr,
        "[DROP] imu_evictions=%ld imu_stale=%ld imu_window_trunc=%ld unpropagated_frames=%ld\n",
        msckf::imu_buffer_evictions, msckf::imu_stale_drops,
        msckf::imu_window_truncations, msckf::frame_unpropagated_drops);
    std::fprintf(stderr, "[EPI] computed=%ld no_baseline=%ld hist(0.00-1.00,20 bins): ",
                 core::epi_computed, core::epi_no_baseline);
    for (int i = 0; i < 20; ++i) std::fprintf(stderr, "%ld ", core::epi_score_hist[i]);
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "complete frames=%llu initialized=%d tracking_mean_ms=%.6f estimator_mean_ms=%.6f\n",
        static_cast<unsigned long long>(frames), int(vio->is_initialized),
        frames ? tracking_sum / double(frames) : 0.0, frames ? estimator_sum / double(frames) : 0.0);
    return vio->is_initialized ? 0 : 6;
}
