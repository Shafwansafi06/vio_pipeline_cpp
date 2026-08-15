// EuRoC MAV variant of vio_rosbag_runner.cpp. Same estimator wiring; only the
// topics, image resolution, and calibration differ. The configuration itself
// now lives in tools/euroc_options.hpp, shared with the ROS-free ASL runner
// (tools/dod_asl_runner.cpp) so the two can never drift apart.
#include "../arena.hpp"
#include "../core/tracker.hpp"
#include "../msckf/vio_manager.hpp"
#include "../tools/euroc_options.hpp"
#include "../type/quat_ops.hpp"

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PointStamped.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <algorithm>
#include <deque>
#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace euroc;

namespace {

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
        // order, so an image routinely arrives before the IMU samples that span
        // it; DOD propagates with whatever is buffered, and updating on short
        // propagation is catastrophic (holding the IMU back by 5 ms takes MH_01
        // from 0.11 m to 9.2 m). Official OpenVINS queues camera messages for
        // exactly this reason.
        //
        // The queue is the point. An earlier version of this held the pair in
        // `left`/`right` and skipped processing until the IMU caught up -- but
        // the next image message then OVERWROTE the waiting pair, silently
        // dropping frames. That cost KAIST circle.bag 0.666 -> 60 m ATE (2456
        // of 3600 frames survived). Nothing may be dropped.
        while (!pending.empty() &&
               0.5 * (stamp(pending.front().first) + stamp(pending.front().second)) <= newest_imu_time) {
            const sensor_msgs::ImageConstPtr pair_left = pending.front().first;
            const sensor_msgs::ImageConstPtr pair_right = pending.front().second;
            pending.pop_front();

        const auto total_start = std::chrono::steady_clock::now();
        const cv_bridge::CvImageConstPtr left_cv = cv_bridge::toCvShare(pair_left, "mono8");
        const cv_bridge::CvImageConstPtr right_cv = cv_bridge::toCvShare(pair_right, "mono8");
        const auto tracking_start = std::chrono::steady_clock::now();
        const double camera_timestamp = 0.5 * (stamp(pair_left) + stamp(pair_right));
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
    std::fprintf(stderr, "complete frames=%llu initialized=%d tracking_mean_ms=%.6f estimator_mean_ms=%.6f\n",
        static_cast<unsigned long long>(frames), int(vio->is_initialized),
        frames ? tracking_sum / double(frames) : 0.0, frames ? estimator_sum / double(frames) : 0.0);
    std::fprintf(stderr,
                 "[TRI] accept=%ld reject_cond=%ld reject_mindist=%ld reject_maxdist=%ld reject_nan=%ld | "
                 "gn_accept=%ld gn_reject_dist=%ld gn_reject_baseline=%ld\n",
                 core::tri_accept, core::tri_reject_cond, core::tri_reject_mindist,
                 core::tri_reject_maxdist, core::tri_reject_nan,
                 core::gn_accept, core::gn_reject_dist, core::gn_reject_baseline);
    std::fprintf(stderr, "[TRI] mean_meas accepted=%.2f rejected_cond=%.2f\n",
                 core::tri_accept ? (double)core::tri_accept_meas / (double)core::tri_accept : 0.0,
                 core::tri_reject_cond ? (double)core::tri_reject_cond_meas / (double)core::tri_reject_cond : 0.0);
    std::fprintf(stderr, "[DB] full_refusals=%ld db_count=%zu slam_features=%d\n",
                 core::db_full_refusals, vio->db.count, vio->state.num_slam_features);
    if (msckf::cls_frames > 0) {
        const double f = (double)msckf::cls_frames;
        std::fprintf(stderr,
                     "[CLS] frames=%ld per-frame: lost=%.2f marginal=%.2f maxtrack=%.2f "
                     "slam_update=%.2f db=%.1f retire_untracked=%.2f retire_chi2=%.2f\n",
                     msckf::cls_frames, msckf::cls_lost / f, msckf::cls_marginal / f,
                     msckf::cls_maxtrack / f, msckf::cls_slam_update / f, msckf::cls_db_count / f,
                     msckf::cls_retire_untracked / f, msckf::cls_retire_chi2 / f);
    }
    std::fprintf(stderr, "[ANCHOR] ok=%ld fail_no_old=%ld fail_depth=%ld fail_transform=%ld\n",
                 msckf::anchor_change_ok, msckf::anchor_marg_no_old, msckf::anchor_marg_depth,
                 msckf::anchor_marg_transform);
    std::fprintf(stderr, "[DINIT] seen=%ld ok=%ld skip_cap=%ld skip_meas=%ld skip_tri=%ld\n",
                 msckf::dinit_seen, msckf::dinit_ok, msckf::dinit_skip_cap, msckf::dinit_skip_meas,
                 msckf::dinit_skip_tri);
    std::fprintf(stderr, "[TRI] nmeas: ");
    for (int i = 2; i < 16; ++i)
        std::fprintf(stderr, "%d:(%ld/%ld) ", i, core::tri_accept_hist[i], core::tri_cond_hist[i]);
    std::fprintf(stderr, "\n");
    return vio->is_initialized ? 0 : 6;
}
