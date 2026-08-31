// ROS-free TUM VI runner: reads the dataset's own Euroc/DSO ASL folder export
// (512x512, https://vision.in.tum.de/tumvi/exported/euroc/512_16/) instead of a
// rosbag. Configuration comes from tools/tumvi_options.hpp (the dataset's own
// pinhole-equi-512 Kalibr calibration). Independent test dataset — nothing in
// the pipeline was tuned on it.
//
//   ./dod_asl_runner <mav0_dir> <output_prefix> [max_seconds]
//
// Writes <prefix>_estimate.csv / _groundtruth.csv / _timing.csv, the same
// schema scripts/evaluate_trajectory.py expects.

#include "../arena.hpp"
#include "../core/tracker.hpp"
#include "../msckf/state_helper.hpp"
#include "../msckf/vio_manager.hpp"
#include "tumvi_options.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tumvi;

namespace {

struct ImuSample {
    double timestamp;
    double wm[3];
    double am[3];
};

std::vector<std::pair<double, std::string>> read_image_index(const std::string& cam_dir) {
    std::vector<std::pair<double, std::string>> out;
    std::FILE* file = std::fopen((cam_dir + "/data.csv").c_str(), "r");
    if (file == nullptr) return out;
    char line[512];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        if (line[0] == '#') continue;
        long long ns = 0;
        char name[256] = {0};
        if (std::sscanf(line, "%lld,%255[^\r\n]", &ns, name) != 2) continue;
        out.emplace_back(double(ns) * 1e-9, cam_dir + "/data/" + name);
    }
    std::fclose(file);
    return out;
}

std::vector<ImuSample> read_imu(const std::string& path) {
    std::vector<ImuSample> out;
    std::FILE* file = std::fopen(path.c_str(), "r");
    if (file == nullptr) return out;
    char line[512];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        if (line[0] == '#') continue;
        long long ns = 0;
        ImuSample sample{};
        if (std::sscanf(line, "%lld,%lf,%lf,%lf,%lf,%lf,%lf", &ns, &sample.wm[0], &sample.wm[1],
                        &sample.wm[2], &sample.am[0], &sample.am[1], &sample.am[2]) != 7) continue;
        sample.timestamp = double(ns) * 1e-9;
        out.push_back(sample);
    }
    std::fclose(file);
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MAV0_DIR OUTPUT_PREFIX [max_seconds]\n", argv[0]);
        return 2;
    }
    const std::string mav0 = argv[1];
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
    // Leica ground truth is position-only; identity orientation is a placeholder
    // to keep the CSV schema identical to the other runners'.
    std::fprintf(truth, "timestamp,px,py,pz,qx,qy,qz,qw\n");
    std::fprintf(timing, "timestamp,tracking_ms,estimator_ms,total_ms,observations\n");

    ArenaAllocator global_arena(MiB(96));
    auto* vio = global_arena.allocate<msckf::VioManagerData>();
    auto* tracker = global_arena.allocate<core::TrackerData>();
    void* image_storage = global_arena.allocate(core::tracker_image_storage_bytes(kImageWidth, kImageHeight), 64);
    auto* observations = static_cast<core::Feature*>(
        global_arena.allocate(sizeof(core::Feature) * core::TRACKER_MAX_FEATURES * 2U, alignof(core::Feature)));
    if (!vio || !tracker || !image_storage || !observations) return 4;

    // Env overrides, so a config sweep does not need a rebuild per point.
    auto env_int = [](const char* name, int fallback) {
        const char* value = std::getenv(name);
        return value != nullptr ? std::atoi(value) : fallback;
    };
    msckf::VioManagerOptions options = make_tumvi_options();
    options.enable_slam = env_int("VIO_ENABLE_SLAM", options.enable_slam ? 1 : 0) != 0;
    options.use_schur_msckf = env_int("VIO_SCHUR", options.use_schur_msckf ? 1 : 0) != 0;
    options.state_opt.do_fej = env_int("VIO_FEJ", options.state_opt.do_fej ? 1 : 0) != 0;
    options.state_opt.do_calib_camera_intrinsics =
        env_int("VIO_CALIB_INTR", options.state_opt.do_calib_camera_intrinsics ? 1 : 0) != 0;
    options.state_opt.max_msckf_in_update = env_int("VIO_MAX_MSCKF", options.state_opt.max_msckf_in_update);
    options.state_opt.max_slam_features = env_int("VIO_MAX_SLAM", options.state_opt.max_slam_features);
    options.state_opt.max_slam_in_update =
        env_int("VIO_SLAM_BATCH", options.state_opt.max_slam_in_update);
    options.state_opt.do_calib_camera_pose =
        env_int("VIO_CALIB_EXTRINSICS", options.state_opt.do_calib_camera_pose ? 1 : 0) != 0;
    options.state_opt.do_calib_camera_timeoffset =
        env_int("VIO_CALIB_TIMEOFFSET", options.state_opt.do_calib_camera_timeoffset ? 1 : 0) != 0;
    auto env_double = [](const char* name, double fallback) {
        const char* value = std::getenv(name);
        return value != nullptr ? std::atof(value) : fallback;
    };
    const double sigma_pix = env_double("VIO_SIGMA_PIX", options.updater_opt.sigma_pix);
    options.updater_opt.sigma_pix = sigma_pix;
    options.updater_opt.sigma_pix_sq = sigma_pix * sigma_pix;
    options.slam_updater_opt.sigma_pix = sigma_pix;
    options.slam_updater_opt.sigma_pix_sq = sigma_pix * sigma_pix;
    options.feat_init_opt.max_cond_number = env_double("VIO_MAX_COND", options.feat_init_opt.max_cond_number);
    // "global" selects official's feat_rep_msckf: GLOBAL_3D.
    if (const char* rep = std::getenv("VIO_FEAT_REP_MSCKF")) {
        options.state_opt.feat_rep_msckf = (std::string(rep) == "global")
            ? type::LandmarkRepresentation::GLOBAL_3D
            : type::LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
    }
    options.init_opt.init_imu_thresh = env_double("VIO_INIT_IMU_THRESH", options.init_opt.init_imu_thresh);
    options.init_opt.init_dyn_use = env_int("VIO_INIT_DYN", options.init_opt.init_dyn_use ? 1 : 0) != 0;
    options.init_wait_for_jerk = env_int("VIO_WAIT_JERK", options.init_wait_for_jerk ? 1 : 0) != 0;
    options.init_opt.init_dyn_min_deg = env_double("VIO_DYN_MIN_DEG", options.init_opt.init_dyn_min_deg);
    options.init_opt.init_dyn_num_pose = env_int("VIO_DYN_NUM_POSE", options.init_opt.init_dyn_num_pose);
    options.init_opt.init_dyn_max_residual = env_double("VIO_DYN_MAX_RESIDUAL", options.init_opt.init_dyn_max_residual);
    options.init_opt.init_dyn_max_gravity_deg = env_double("VIO_DYN_MAX_GRAV_DEG", options.init_opt.init_dyn_max_gravity_deg);
    options.init_opt.init_dyn_min_rec_cond = env_double("VIO_DYN_MIN_REC_COND", options.init_opt.init_dyn_min_rec_cond);
    options.init_opt.init_dyn_min_excitation = env_double("VIO_DYN_MIN_EXCITATION", options.init_opt.init_dyn_min_excitation);
    options.init_opt.init_dyn_zero_velocity = env_int("VIO_DYN_ZERO_V0", options.init_opt.init_dyn_zero_velocity ? 1 : 0) != 0;
    options.init_opt.init_featureless = env_int("VIO_INIT_FEATURELESS", options.init_opt.init_featureless ? 1 : 0) != 0;
    // Diagnostic only: overwrite the initial velocity right after the filter
    // initialises, to test whether a bad initial velocity is what kills a run.
    const char* v0_env = std::getenv("VIO_FORCE_V0");
    if (const char* infl = std::getenv("VIO_DYN_INFLATE")) {
        const double scale = std::atof(infl);
        options.init_opt.init_dyn_inflation_ori = scale > 0.0 ? 10.0 : 1.0;
        options.init_opt.init_dyn_inflation_vel = scale > 0.0 ? 100.0 : 1.0;
        options.init_opt.init_dyn_inflation_bg = scale > 0.0 ? 10.0 : 1.0;
        options.init_opt.init_dyn_inflation_ba = scale > 0.0 ? 100.0 : 1.0;
    }
    msckf::init_vio_manager(*vio, options);
    core::TrackerOptions tracker_options;
    tracker_options.num_features = env_int("VIO_NUM_FEATURES", tracker_options.num_features);
    tracker_options.fast_threshold = env_int("VIO_FAST_THRESHOLD", tracker_options.fast_threshold);
    tracker_options.min_px_dist = env_int("VIO_MIN_PX_DIST", tracker_options.min_px_dist);
    if (!core::init_tracker(*tracker, tracker_options, options.cam_models, 2,
                            image_storage, core::tracker_image_storage_bytes(kImageWidth, kImageHeight))) return 5;
    cv::setNumThreads(4);
    cv::setRNGSeed(0);

    const auto left_index = read_image_index(mav0 + "/cam0");
    const auto right_index = read_image_index(mav0 + "/cam1");
    const auto imu = read_imu(mav0 + "/imu0/data.csv");
    if (left_index.empty() || right_index.empty() || imu.empty()) {
        std::fprintf(stderr, "bad dataset: %zu/%zu images, %zu imu samples\n",
                     left_index.size(), right_index.size(), imu.size());
        return 6;
    }
    // Pair by TIMESTAMP, not by index: some sequences ship one more image on
    // one camera than the other (MH_04 has 2033 left and 2032 right), and
    // index pairing then silently mismatches every frame after the gap. The
    // 3 ms tolerance is the ROS runner's.
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    {
        std::size_t r = 0;
        for (std::size_t l = 0; l < left_index.size(); ++l) {
            while (r + 1 < right_index.size() &&
                   std::abs(right_index[r + 1].first - left_index[l].first) <
                   std::abs(right_index[r].first - left_index[l].first)) {
                ++r;
            }
            if (std::abs(right_index[r].first - left_index[l].first) <= 0.003) {
                pairs.emplace_back(l, r);
            }
        }
        if (pairs.empty()) {
            std::fprintf(stderr, "no stereo pairs within 3 ms\n");
            return 6;
        }
        std::fprintf(stderr, "paired %zu of %zu/%zu images\n",
                     pairs.size(), left_index.size(), right_index.size());
    }

    // Ground truth is copied straight through; evaluate_trajectory.py associates
    // it with the estimate by timestamp. Machine-hall sequences ship Leica
    // position only; the vicon-room ones ship a full pose under
    // state_groundtruth_estimate0. Both start "ns,x,y,z", and the evaluator
    // reads position only, so one parse covers both.
    {
        std::FILE* gt = std::fopen((mav0 + "/leica0/data.csv").c_str(), "r");
        if (gt == nullptr) gt = std::fopen((mav0 + "/state_groundtruth_estimate0/data.csv").c_str(), "r");
        if (gt == nullptr) gt = std::fopen((mav0 + "/mocap0/data.csv").c_str(), "r");
        if (gt == nullptr) {
            std::fprintf(stderr, "no ground truth found under %s\n", mav0.c_str());
            return 9;
        }
        char line[1024];
        while (std::fgets(line, sizeof(line), gt) != nullptr) {
            if (line[0] == '#') continue;
            long long ns = 0;
            double x = 0.0, y = 0.0, z = 0.0;
            if (std::sscanf(line, "%lld,%lf,%lf,%lf", &ns, &x, &y, &z) != 4) continue;
            std::fprintf(truth, "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                         double(ns) * 1e-9, x, y, z, 0.0, 0.0, 0.0, 1.0);
        }
        std::fclose(gt);
    }

    // Skip the first N seconds of the sequence, like official's launch files do
    // (their suggested bag_start is 40 for MH_01, 35 for MH_02, 15-17.5 for
    // MH_03-05). Used to test whether a divergence is an initialisation-window
    // problem rather than a steady-state one.
    const char* skip_env = std::getenv("VIO_SKIP_S");
    const double skip_seconds = skip_env != nullptr ? std::atof(skip_env) : 0.0;
    const char* lag_env = std::getenv("VIO_IMU_LAG_S");
    const double imu_lag_seconds = lag_env != nullptr ? std::atof(lag_env) : 0.0;
    const double first_timestamp = std::min(left_index.front().first, imu.front().timestamp);
    std::size_t imu_cursor = 0;
    std::uint64_t frames = 0;
    double tracking_sum = 0.0;
    double estimator_sum = 0.0;

    for (std::size_t p = 0; p < pairs.size(); ++p) {
        const std::size_t i = pairs[p].first;
        const std::size_t j = pairs[p].second;
        const double camera_timestamp = 0.5 * (left_index[i].first + right_index[j].first);
        if (max_seconds > 0.0 && camera_timestamp - first_timestamp > max_seconds) break;
        if (camera_timestamp - first_timestamp < skip_seconds) continue;

        // Feed every IMU sample that precedes this frame, so the estimator
        // always has IMU coverage up to the image it is about to process.
        // VIO_IMU_LAG_S holds IMU back by that many seconds, reproducing a
        // transport that delivers an image before the IMU spanning it.
        const double imu_horizon = camera_timestamp - imu_lag_seconds;
        while (imu_cursor < imu.size() && imu[imu_cursor].timestamp <= imu_horizon) {
            const ImuSample& sample = imu[imu_cursor++];
            core::ImuData data;
            data.timestamp = sample.timestamp;
            data.wm << sample.wm[0], sample.wm[1], sample.wm[2];
            data.am << sample.am[0], sample.am[1], sample.am[2];
            msckf::feed_measurement_imu(*vio, data);
        }

        const cv::Mat left = cv::imread(left_index[i].second, cv::IMREAD_GRAYSCALE);
        const cv::Mat right = cv::imread(right_index[j].second, cv::IMREAD_GRAYSCALE);
        if (left.empty() || right.empty()) {
            std::fprintf(stderr, "failed to read frame %zu\n", i);
            return 7;
        }

        const auto tracking_start = std::chrono::steady_clock::now();
        const int observation_count = core::track_stereo_frame(*tracker, camera_timestamp,
            left, right, observations, core::TRACKER_MAX_FEATURES * 2);
        const auto estimator_start = std::chrono::steady_clock::now();
        msckf::feed_measurement_camera_tracks(*vio, camera_timestamp, observations, observation_count);
        const auto done = std::chrono::steady_clock::now();
        const double tracking_ms = std::chrono::duration<double, std::milli>(estimator_start - tracking_start).count();
        const double estimator_ms = std::chrono::duration<double, std::milli>(done - estimator_start).count();
        tracking_sum += tracking_ms;
        estimator_sum += estimator_ms;
        ++frames;
        std::fprintf(timing, "%.9f,%.6f,%.6f,%.6f,%d\n", camera_timestamp,
                     tracking_ms, estimator_ms, tracking_ms + estimator_ms, observation_count);
        if (vio->is_initialized && v0_env != nullptr) {
            static bool forced = false;
            if (!forced) {
                double vx = 0.0, vy = 0.0, vz = 0.0;
                if (std::sscanf(v0_env, "%lf,%lf,%lf", &vx, &vy, &vz) == 3) {
                    vio->state.imu.value[7] = vx;
                    vio->state.imu.value[8] = vy;
                    vio->state.imu.value[9] = vz;
                    std::fprintf(stderr, "[diag] forced v0 = %.3f %.3f %.3f\n", vx, vy, vz);
                }
                forced = true;
            }
        }
        if (vio->is_initialized) {
            const double* value = vio->state.imu.value;
            std::fprintf(estimate, "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%zu,%d\n",
                vio->state.timestamp, value[4], value[5], value[6], value[0], value[1],
                value[2], value[3], vio->db.count, vio->state.num_clones);
        }
        if ((frames % 500U) == 0U) std::fprintf(stderr,
            "frames=%llu initialized=%d tracks=%d mean_track_ms=%.3f mean_estimator_ms=%.3f\n",
            static_cast<unsigned long long>(frames), int(vio->is_initialized), tracker->previous_count[0],
            tracking_sum / double(frames), estimator_sum / double(frames));
    }

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
    if (frames > 0) {
        const double f = double(frames);
        std::fprintf(stderr,
                     "[STAGE ms/frame] db %.3f  propagate %.3f  msckf %.3f  slam %.3f  slam_delayed %.3f  marg %.3f\n",
                     msckf::stage_ms_db / f, msckf::stage_ms_propagate / f, msckf::stage_ms_msckf / f, msckf::stage_ms_slam / f,
                     msckf::stage_ms_slam_delayed / f, msckf::stage_ms_marg / f);
        if (msckf::msckf_calls > 0) {
            const double c = double(msckf::msckf_calls);
            std::fprintf(stderr,
                         "[MSCKF sizes] calls %ld  H rows %.0f x cols %.0f  -> after compress %.0f rows  cov %.0f\n",
                         msckf::msckf_calls, msckf::msckf_rows_pre / c, msckf::msckf_cols / c,
                         msckf::msckf_rows_post / c, msckf::msckf_cov / c);
        }
        std::fprintf(stderr, "[EKF ms/frame] M_a %.3f  S+inv %.3f  K %.3f  cov-update %.3f\n",
                     msckf::ekf_ms_ma / f, msckf::ekf_ms_s / f, msckf::ekf_ms_k / f, msckf::ekf_ms_cov / f);
        std::fprintf(stderr,
                     "[MSCKF ms/frame] tri %.3f  jac %.3f  chi2 %.3f  assemble %.3f  alloc %.3f  compress %.3f  ekf %.3f\n",
                     msckf::msckf_ms_tri / f, msckf::msckf_ms_jac / f, msckf::msckf_ms_chi2 / f,
                     msckf::msckf_ms_assemble / f, msckf::msckf_ms_alloc / f,
                     msckf::msckf_ms_compress / f, msckf::msckf_ms_ekf / f);
    }
    std::fprintf(stderr, "[DBG] max_meas=%ld max_count=%ld shift_elems/frame=%.0f compact_elems/frame=%.0f\n",
                 core::dbg_max_meas, core::dbg_max_count,
                 double(core::dbg_shift_elems) / double(frames ? frames : 1),
                 double(core::dbg_compact_elems) / double(frames ? frames : 1));
    std::fprintf(stderr,
                 "[DB] full_refusals=%ld db_count=%zu slam_features=%d max_meas=%ld meas_overflow=%ld\n",
                 core::db_full_refusals, vio->db.count, vio->state.num_slam_features,
                 core::dbg_max_meas, core::feat_meas_overflow);
    // Every one of these is data the pipeline threw away. All zero is the
    // expected result on a healthy bag; nonzero says which bound is binding.
    std::fprintf(stderr,
                 "[DROP] imu_evictions=%ld imu_stale=%ld imu_window_trunc=%ld unpropagated_frames=%ld\n",
                 msckf::imu_buffer_evictions, msckf::imu_stale_drops,
                 msckf::imu_window_truncations, msckf::frame_unpropagated_drops);
    std::fprintf(stderr, "[EPI] computed=%ld no_baseline=%ld hist(0.00-1.00,20 bins): ",
                 core::epi_computed, core::epi_no_baseline);
    for (int i = 0; i < 20; ++i) std::fprintf(stderr, "%ld ", core::epi_score_hist[i]);
    std::fprintf(stderr, "\n");
    return vio->is_initialized ? 0 : 8;
}
