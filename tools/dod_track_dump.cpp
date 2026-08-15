// Feeds EuRoC ASL stereo images through DOD's tracker ONLY (no estimator, no
// IMU, no initialization) and dumps the surviving cam0 track ids each frame.
// Track lifetime is a pure frontend property, so this isolates it from every
// estimator difference and needs nothing but OpenCV + Eigen.
//
// Pair with tools/ov_track_dump.cpp, which does the same through official
// OpenVINS's TrackKLT. Analyse both with scripts/track_lifetime.py.
//
//   VIO_TRACK_DUMP=dod_tracks.txt ./dod_track_dump <mav0_dir> [max_frames]

#include "../core/tracker.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int kImageWidth = 752;
constexpr int kImageHeight = 480;

// mav0/camN/data.csv is "#timestamp [ns],filename".
std::vector<std::pair<double, std::string>> read_index(const std::string& cam_dir) {
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

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MAV0_DIR [MAX_FRAMES]  (set VIO_TRACK_DUMP)\n", argv[0]);
        return 2;
    }
    const std::string mav0 = argv[1];
    const long max_frames = argc > 2 ? std::atol(argv[2]) : -1;
    if (std::getenv("VIO_TRACK_DUMP") == nullptr) {
        std::fprintf(stderr, "VIO_TRACK_DUMP is not set; nothing would be written\n");
        return 2;
    }

    // Same intrinsics the EuRoC runner uses (mav0/camN/sensor.yaml).
    const double cam0_calib[8] = {458.654, 457.296, 367.215, 248.375,
                                  -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};
    const double cam1_calib[8] = {457.587, 456.134, 379.999, 255.238,
                                  -0.28368365, 0.07451284, -0.00010473, -3.55590700e-05};
    core::CameraModel cameras[2];
    core::init_camera(cameras[0], core::CameraModelType::RADTAN, kImageWidth, kImageHeight, cam0_calib);
    core::init_camera(cameras[1], core::CameraModelType::RADTAN, kImageWidth, kImageHeight, cam1_calib);

    core::TrackerData tracker;
    core::TrackerOptions tracker_options; // defaults, exactly as the EuRoC runner leaves them
    const std::size_t storage_bytes = core::tracker_image_storage_bytes(kImageWidth, kImageHeight);
    std::vector<std::uint8_t> image_storage(storage_bytes);
    if (!core::init_tracker(tracker, tracker_options, cameras, 2, image_storage.data(), storage_bytes)) {
        std::fprintf(stderr, "init_tracker failed\n");
        return 3;
    }
    std::vector<core::Feature> observations(std::size_t(core::TRACKER_MAX_FEATURES) * 2U);

    const auto left_index = read_index(mav0 + "/cam0");
    const auto right_index = read_index(mav0 + "/cam1");
    if (left_index.empty() || left_index.size() != right_index.size()) {
        std::fprintf(stderr, "cam0/cam1 index mismatch (%zu vs %zu)\n", left_index.size(), right_index.size());
        return 4;
    }

    cv::setNumThreads(4);
    cv::setRNGSeed(0);
    long frames = 0;
    double track_seconds = 0.0;
    for (std::size_t i = 0; i < left_index.size(); ++i) {
        if (max_frames > 0 && frames >= max_frames) break;
        const cv::Mat left = cv::imread(left_index[i].second, cv::IMREAD_GRAYSCALE);
        const cv::Mat right = cv::imread(right_index[i].second, cv::IMREAD_GRAYSCALE);
        if (left.empty() || right.empty()) {
            std::fprintf(stderr, "failed to read frame %zu\n", i);
            return 5;
        }
        const auto track_start = std::chrono::steady_clock::now();
        core::track_stereo_frame(tracker, left_index[i].first, left, right,
                                 observations.data(), int(observations.size()));
        track_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - track_start).count();
        ++frames;
    }
    std::fprintf(stderr, "dod: %ld frames, tracker mean %.3f ms/frame\n",
                 frames, frames ? 1000.0 * track_seconds / double(frames) : 0.0);
    return 0;
}
