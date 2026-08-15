#include "tracker.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace core {
namespace {

// Two-view epipolar consistency check on a temporal KLT track (old->new
// within one camera). Missing from the initial port; without it, KLT tracks
// that converge onto the wrong structure (drifted, slid along an edge) still
// report status=1 and feed a plausible-but-geometrically-wrong correspondence
// straight into every MSCKF update -- this was corrupting the estimate on
// real bag data even though every synthetic/unit test passed. Matches the
// reference tracker's use of cv2.findFundamentalMat(FM_RANSAC) on undistorted
// points; skipped (accept-all) below the reference's own minimum sample size.
void ransac_inliers(const cv::Point2f* old_undist, const cv::Point2f* new_undist, int count,
                    std::uint8_t* inlier_mask, double focal_length) {
    std::fill(inlier_mask, inlier_mask + count, std::uint8_t{1});
    if (count < 10) return; // too few points for a reliable fundamental-matrix fit
    std::vector<cv::Point2f> old_pts(old_undist, old_undist + count);
    std::vector<cv::Point2f> new_pts(new_undist, new_undist + count);
    cv::Mat mask;
    // Matches official OpenVINS's perform_matching threshold exactly:
    // 2.0 / max_focallength (not 1.0 / fx-only, which over-rejects).
    const double threshold = focal_length > 0.0 ? 2.0 / focal_length : 2.0 / 300.0;
    cv::findFundamentalMat(old_pts, new_pts, cv::FM_RANSAC, threshold, 0.999, mask);
    if (mask.empty() || mask.rows != count) return; // fit failed; fall back to accept-all
    for (int i = 0; i < count; ++i) inlier_mask[i] = mask.at<std::uint8_t>(i);
}

using Point2f = cv::Point2f;

constexpr int kCloseGridMax = 128;

int detect_grid_fast(const cv::Mat& image, const TrackerOptions& options,
                     TrackPoint* tracks, int count, std::uint32_t& next_id) {
    const int num_features = options.num_features;
    const int grid_x = std::max(1, options.grid_x);
    const int grid_y = std::max(1, options.grid_y);
    const int min_px_dist = std::max(1, options.min_px_dist);

    static std::uint8_t grid_count[64];
    std::memset(grid_count, 0, std::size_t(grid_x * grid_y));
    const float size_x = float(image.cols) / float(grid_x);
    const float size_y = float(image.rows) / float(grid_y);

    const int close_w = std::min(kCloseGridMax, std::max(1, image.cols / min_px_dist));
    const int close_h = std::min(kCloseGridMax, std::max(1, image.rows / min_px_dist));
    static std::uint8_t grid_close[kCloseGridMax][kCloseGridMax];
    for (int y = 0; y < close_h; ++y) std::memset(grid_close[y], 0, std::size_t(close_w));

    for (int i = 0; i < count; ++i) {
        const int xg = std::min(grid_x - 1, std::max(0, int(tracks[i].x / size_x)));
        const int yg = std::min(grid_y - 1, std::max(0, int(tracks[i].y / size_y)));
        if (grid_count[yg * grid_x + xg] < 255) ++grid_count[yg * grid_x + xg];
        const int xc = std::min(close_w - 1, std::max(0, int(tracks[i].x) / min_px_dist));
        const int yc = std::min(close_h - 1, std::max(0, int(tracks[i].y) / min_px_dist));
        grid_close[yc][xc] = 255;
    }

    const int needed = num_features - count;
    if (needed < std::min(20, num_features / 2)) return count;

    const int num_features_grid = int(double(num_features) / double(grid_x * grid_y)) + 1;
    const int num_features_grid_req = std::max(1, int(0.5 * double(num_features_grid)));
    const int cell_w = std::max(1, image.cols / grid_x);
    const int cell_h = std::max(1, image.rows / grid_y);

    static Point2f new_pts[TRACKER_MAX_FEATURES];
    int new_count = 0;
    std::vector<cv::KeyPoint> cell_kps;
    for (int gy = 0; gy < grid_y && count < num_features; ++gy) {
        for (int gx = 0; gx < grid_x && count < num_features; ++gx) {
            if (grid_count[gy * grid_x + gx] >= num_features_grid_req) continue;
            const int x0 = gx * cell_w;
            const int y0 = gy * cell_h;
            if (x0 + cell_w > image.cols || y0 + cell_h > image.rows) continue;
            cell_kps.clear();
            cv::FAST(image(cv::Rect(x0, y0, cell_w, cell_h)), cell_kps, options.fast_threshold, true);
            std::sort(cell_kps.begin(), cell_kps.end(),
                      [](const cv::KeyPoint& a, const cv::KeyPoint& b) { return a.response > b.response; });
            int accepted = 0;
            for (std::size_t i = 0; i < cell_kps.size() && accepted < num_features_grid && count < num_features; ++i) {
                const float px = cell_kps[i].pt.x + float(x0);
                const float py = cell_kps[i].pt.y + float(y0);
                if (px < 0.0F || px >= float(image.cols) || py < 0.0F || py >= float(image.rows)) continue;
                const int xc = std::min(close_w - 1, std::max(0, int(px) / min_px_dist));
                const int yc = std::min(close_h - 1, std::max(0, int(py) / min_px_dist));
                if (grid_close[yc][xc] > 127) continue;
                grid_close[yc][xc] = 255;
                new_pts[new_count++] = Point2f(px, py);
                ++accepted;
                ++count;
            }
        }
    }

    if (new_count > 0) {
        std::vector<cv::Point2f> refine(new_pts, new_pts + new_count);
        cv::cornerSubPix(image, refine, cv::Size(5, 5), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001));
        const int base = count - new_count;
        for (int i = 0; i < new_count; ++i) {
            tracks[base + i] = {refine[i].x, refine[i].y, std::int32_t(++next_id)};
        }
    }
    return count;
}

// STATUS (2026-07-23): the real-FAST detector above (matching official's
// Grider_FAST/cornerSubPix + two-level occupancy grid) is ACTIVE, combined
// with official's exact triangulation thresholds. On infinity.bag this gives
// ATE 0.24 m (vs 1.11 m with the old hand-rolled detector) -- DOD now traces
// the figure-8 nearly matching official (~3x from official's 0.070 m). On
// circle.bag it still diverges: DOD's port produces only ~3050 tracked frames
// vs official's ~4758 (feature starvation -- the port loses tracking on
// circle's slow motion where official does not). The remaining gap is in this
// detector's feature-management (redetection scheduling / stereo pairing of
// new features), not the estimator math (audited equivalent to official).
void preprocess(const cv::Mat& input, cv::Mat& output, int method) {
    if (method == 1) {
        cv::equalizeHist(input, output);
    } else if (method == 2) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(10.0, cv::Size(8, 8));
        clahe->apply(input, output);
    } else if (input.data != output.data) {
        input.copyTo(output);
    }
}

void tracks_to_points(const TrackPoint* tracks, int count, Point2f* points) {
    for (int i = 0; i < count; ++i) points[i] = Point2f(tracks[i].x, tracks[i].y);
}

void run_lk(const cv::Mat& from, const cv::Mat& to, const Point2f* old_points,
            Point2f* new_points, std::uint8_t* status, float* error, int count,
            const TrackerOptions& options) {
    if (count == 0) return;
    std::memcpy(new_points, old_points, std::size_t(count) * sizeof(Point2f));
    cv::Mat old_mat(count, 1, CV_32FC2, const_cast<Point2f*>(old_points));
    cv::Mat new_mat(count, 1, CV_32FC2, new_points);
    cv::Mat status_mat(count, 1, CV_8UC1, status);
    cv::Mat error_mat(count, 1, CV_32FC1, error);
    cv::calcOpticalFlowPyrLK(from, to, old_mat, new_mat, status_mat, error_mat,
                             cv::Size(options.window_size, options.window_size),
                             options.pyramid_levels,
                             cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);
}

bool in_bounds(const Point2f& point, int width, int height) {
    return point.x >= 0.0F && point.y >= 0.0F && point.x < float(width) && point.y < float(height);
}

void make_observation(Feature& feature, int id, int cam_id, double timestamp,
                      const Point2f& point, const CameraModel& camera) {
    feature = Feature{};
    feature.featid = id;
    feature.num_measurements = 1;
    FeatureMeasurement& measurement = feature.measurements[0];
    measurement.cam_id = cam_id;
    measurement.timestamp = timestamp;
    measurement.uv = Eigen::Vector2d(point.x, point.y);
    measurement.uv_norm = undistort(camera, measurement.uv);
}

} // namespace

std::size_t tracker_image_storage_bytes(int width, int height) {
    return std::size_t(width) * std::size_t(height) * TRACKER_MAX_CAMERAS * 2U;
}

bool init_tracker(TrackerData& tracker, const TrackerOptions& options,
                  const CameraModel* cameras, int num_cameras,
                  void* image_storage, std::size_t image_storage_size) {
    if (!image_storage || num_cameras != 2 || options.num_features <= 0 ||
        options.num_features > TRACKER_MAX_FEATURES || cameras[0].width != cameras[1].width ||
        cameras[0].height != cameras[1].height) return false;
    const std::size_t frame_bytes = std::size_t(cameras[0].width) * std::size_t(cameras[0].height);
    if (image_storage_size < frame_bytes * 4U) return false;
    tracker = TrackerData{};
    tracker.options = options;
    tracker.cameras[0] = cameras[0];
    tracker.cameras[1] = cameras[1];
    tracker.width = cameras[0].width;
    tracker.height = cameras[0].height;
    // Match official's TrackBase: currid = 4 * numaruco + 1, so every tracked
    // feature id sits above the ArUco reserve and stays marginalizable.
    tracker.next_feature_id = 4U * (std::uint32_t)std::max(0, options.num_aruco);
    auto* bytes = static_cast<std::uint8_t*>(image_storage);
    for (int camera = 0; camera < 2; ++camera) {
        tracker.previous_images[camera] = bytes + frame_bytes * std::size_t(camera);
        tracker.current_images[camera] = bytes + frame_bytes * std::size_t(2 + camera);
    }
    return true;
}

// Architectural rewrite (2026-07-22): tracks per camera are now fully
// INDEPENDENT lists (own count, own survivors), matching official OpenVINS's
// TrackKLT.cpp design (`pts_last[cam_id]`, matched only by feature ID) instead
// of the original DOD design where both cameras' lists were forced to the
// same size and paired by array index.
//
// The paired-index design was the root cause of the "stale-seed" bug: when
// one camera's temporal KLT failed for a track, the OLD code had to invent
// *something* to write into that camera's slot to keep both arrays the same
// length, so it carried forward last frame's coordinate -- which then got
// seeded against a brand-new image next frame, silently corrupting that
// camera's observations for the rest of the track's life. A same-frame
// stereo cross-camera "recovery" match was added as a patch for this, but it
// only papered over the symptom: recovering right-from-left was safe, but
// recovering left-from-right caused a reproducible catastrophic divergence
// on circle.bag (45 m ATE) whose root cause was never isolated -- most
// likely because a bad recovered match, forced into existence purely to keep
// array lengths equal, is a fundamentally made-up measurement that shouldn't
// exist at all under official's actual design.
//
// With independent lists, that entire class of bug is structurally
// impossible: a camera that fails to track a point simply drops that id from
// its own list. No stale coordinate is ever carried forward, and no
// invented cross-camera "recovery" is needed -- if a track survives in only
// one camera, it lives on as a genuine mono observation (matching official's
// tolerance for mono-only tracks) until a fresh grid detection + stereo pairing
// re-establishes a (new-ID) stereo pair for that physical point, exactly as
// official does.
int track_stereo_frame(TrackerData& tracker, double timestamp,
                       const cv::Mat& left, const cv::Mat& right,
                       Feature* observations, int observation_capacity) {
    if (!observations || left.type() != CV_8UC1 || right.type() != CV_8UC1 ||
        left.cols != tracker.width || right.cols != tracker.width ||
        left.rows != tracker.height || right.rows != tracker.height) return 0;

    cv::Mat current[2] = {
        cv::Mat(tracker.height, tracker.width, CV_8UC1, tracker.current_images[0]),
        cv::Mat(tracker.height, tracker.width, CV_8UC1, tracker.current_images[1])
    };
    cv::Mat previous[2] = {
        cv::Mat(tracker.height, tracker.width, CV_8UC1, tracker.previous_images[0]),
        cv::Mat(tracker.height, tracker.width, CV_8UC1, tracker.previous_images[1])
    };
    preprocess(left, current[0], tracker.options.histogram_method);
    preprocess(right, current[1], tracker.options.histogram_method);

    int observation_count = 0;
    static TrackPoint new_previous[2][TRACKER_MAX_FEATURES];
    int new_count[2] = {0, 0};

    // Independent per-camera temporal KLT + epipolar RANSAC + survival.
    for (int cam = 0; cam < 2; ++cam) {
        const int old_count = tracker.previous_count[cam];
        if (old_count == 0) continue;

        static Point2f old_pts[TRACKER_MAX_FEATURES], new_pts[TRACKER_MAX_FEATURES];
        static std::uint8_t status[TRACKER_MAX_FEATURES];
        static float error[TRACKER_MAX_FEATURES];
        tracks_to_points(tracker.previous[cam], old_count, old_pts);
        run_lk(previous[cam], current[cam], old_pts, new_pts, status, error, old_count, tracker.options);

        static Point2f old_norm[TRACKER_MAX_FEATURES], new_norm[TRACKER_MAX_FEATURES];
        for (int i = 0; i < old_count; ++i) {
            const Eigen::Vector2d o = undistort(tracker.cameras[cam], Eigen::Vector2d(old_pts[i].x, old_pts[i].y));
            const Eigen::Vector2d n = undistort(tracker.cameras[cam], Eigen::Vector2d(new_pts[i].x, new_pts[i].y));
            old_norm[i] = Point2f(float(o.x()), float(o.y()));
            new_norm[i] = Point2f(float(n.x()), float(n.y()));
        }
        static std::uint8_t ransac_ok[TRACKER_MAX_FEATURES];
        const double focal = std::max(tracker.cameras[cam].values[0], tracker.cameras[cam].values[1]);
        ransac_inliers(old_norm, new_norm, old_count, ransac_ok, focal);

        for (int i = 0; i < old_count; ++i) {
            if (!status[i] || !ransac_ok[i] || !in_bounds(new_pts[i], tracker.width, tracker.height)) continue;
            const int id = tracker.previous[cam][i].id;
            if (new_count[cam] < TRACKER_MAX_FEATURES) {
                new_previous[cam][new_count[cam]++] = {new_pts[i].x, new_pts[i].y, id};
            }
            if (observation_count < observation_capacity) {
                make_observation(observations[observation_count++], id, cam, timestamp, new_pts[i], tracker.cameras[cam]);
            }
        }
    }

    // New-feature detection on the left (reference) camera, then an
    // immediate stereo match to seed a right-camera counterpart when
    // possible. A left detection is kept as a valid mono track even when the
    // stereo match fails or lands out of bounds -- matching official's
    // tolerance for mono-only tracks from birth, not just after a later
    // camera drop.
    const int before_detection = new_count[0];
    new_count[0] = detect_grid_fast(current[0], tracker.options, new_previous[0], new_count[0], tracker.next_feature_id);
    if (new_count[0] > before_detection) {
        const int added = new_count[0] - before_detection;
        static Point2f detected_left[TRACKER_MAX_FEATURES], detected_right[TRACKER_MAX_FEATURES];
        static std::uint8_t status_stereo[TRACKER_MAX_FEATURES];
        static float error_stereo[TRACKER_MAX_FEATURES];
        tracks_to_points(new_previous[0] + before_detection, added, detected_left);
        run_lk(current[0], current[1], detected_left, detected_right,
               status_stereo, error_stereo, added, tracker.options);
        for (int i = 0; i < added; ++i) {
            if (status_stereo[i] && in_bounds(detected_right[i], tracker.width, tracker.height) &&
                new_count[1] < TRACKER_MAX_FEATURES) {
                new_previous[1][new_count[1]++] = {detected_right[i].x, detected_right[i].y, new_previous[0][before_detection + i].id};
            }
        }
    }

    for (int cam = 0; cam < 2; ++cam) {
        tracker.previous_count[cam] = new_count[cam];
        std::memcpy(tracker.previous[cam], new_previous[cam], std::size_t(new_count[cam]) * sizeof(TrackPoint));
    }
    current[0].copyTo(previous[0]);
    current[1].copyTo(previous[1]);

    // Track instrumentation. Per frame: "F <ts>", then one "<cam> <id> <u> <v>"
    // line per surviving track -- the exact analogue of official TrackKLT's
    // `pts_last`/`ids_last` after feed_stereo. Raw distorted pixels are dumped
    // and everything derived (lifetime, parallax, epipolar residual) is computed
    // offline by scripts/track_lifetime.py, so DOD and official are measured by
    // identical code with identical undistortion. Off unless VIO_TRACK_DUMP is set.
    if (const char* dump_path = std::getenv("VIO_TRACK_DUMP")) {
        static std::FILE* dump = std::fopen(dump_path, "w");
        if (dump != nullptr) {
            std::fprintf(dump, "F %.9f\n", timestamp);
            for (int cam = 0; cam < 2; ++cam) {
                for (int i = 0; i < tracker.previous_count[cam]; ++i) {
                    const TrackPoint& point = tracker.previous[cam][i];
                    std::fprintf(dump, "%d %d %.4f %.4f\n", cam, point.id, point.x, point.y);
                }
            }
            std::fflush(dump);
        }
    }
    return observation_count;
}

} // namespace core
