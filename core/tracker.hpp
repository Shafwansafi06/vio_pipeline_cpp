#pragma once

#include "cam.hpp"
#include "feature.hpp"

#include <opencv2/core.hpp>

#include <cstdint>

namespace core {

constexpr int TRACKER_MAX_CAMERAS = 2;
constexpr int TRACKER_MAX_FEATURES = 256;

struct TrackerOptions {
    int num_features = 200;
    int fast_threshold = 30;
    int grid_x = 5;
    int grid_y = 5;
    int min_px_dist = 15;
    int histogram_method = 1; // 0 none, 1 equalize, 2 CLAHE
    // Feature ids must start above the ArUco tag reserve. Official does
    // `currid = 4 * numaruco + 1` in TrackBase, and StateHelper::marginalize_slam
    // refuses to marginalize any landmark whose id is <= 4 * max_aruco_features
    // (the low ids belong to tags, which are never dropped). Starting at 1
    // meant DOD's first ~4096 features were permanently unmarginalizable: the
    // first 50 SLAM landmarks got promoted early with low ids and could never
    // be retired, so the landmark slots were occupied forever by tracks that
    // had long since died.
    int num_aruco = 1024;
    int pyramid_levels = 5;
    int window_size = 15;
    bool stereo = true;
};

struct TrackPoint {
    float x = 0.0F;
    float y = 0.0F;
    std::int32_t id = -1;
};

// Persistent tracker state is bounded and contiguous. Images are supplied by the
// transport adapter and copied into fixed-capacity frame storage at initialization.
struct TrackerData {
    TrackerOptions options;
    CameraModel cameras[TRACKER_MAX_CAMERAS];
    TrackPoint previous[TRACKER_MAX_CAMERAS][TRACKER_MAX_FEATURES];
    int previous_count[TRACKER_MAX_CAMERAS] = {0, 0};
    std::uint32_t next_feature_id = 1;
    int width = 0;
    int height = 0;
    std::uint8_t* previous_images[TRACKER_MAX_CAMERAS] = {nullptr, nullptr};
    std::uint8_t* current_images[TRACKER_MAX_CAMERAS] = {nullptr, nullptr};
};

std::size_t tracker_image_storage_bytes(int width, int height);
bool init_tracker(TrackerData& tracker, const TrackerOptions& options,
                  const CameraModel* cameras, int num_cameras,
                  void* image_storage, std::size_t image_storage_size);

// Produces one Feature record per observation (one or two records can share an
// ID for stereo). The caller feeds these records directly to the estimator DB.
int track_stereo_frame(TrackerData& tracker, double timestamp,
                       const cv::Mat& left, const cv::Mat& right,
                       Feature* observations, int observation_capacity);

} // namespace core
