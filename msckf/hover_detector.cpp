#include "hover_detector.hpp"

namespace msckf {

void init_hover_detector(HoverDetectorData& detector, const HoverDetectorOptions& options) {
    detector.options = options;
    detector.is_hovering = false;
    detector.pending_raw = false;
    detector.consecutive_count = 0;
}

bool update_hover_detector(HoverDetectorData& detector, const core::FeatureDatabase& db,
                          double previous_timestamp, double current_timestamp,
                          const Eigen::Matrix3d& R_curr_from_prev) {
    int matched = 0;
    double sum_residual = 0.0;

    for (std::size_t i = 0; i < db.count; ++i) {
        const core::Feature& feat = db.features[i];
        const core::FeatureMeasurement* prev_meas = nullptr;
        const core::FeatureMeasurement* curr_meas = nullptr;
        for (int m = 0; m < feat.num_measurements; ++m) {
            const core::FeatureMeasurement& meas = feat.measurements[m];
            if (meas.cam_id != 0) continue;
            if (meas.timestamp == previous_timestamp) prev_meas = &meas;
            else if (meas.timestamp == current_timestamp) curr_meas = &meas;
        }
        if (!prev_meas || !curr_meas) continue;

        Eigen::Vector3d b_prev(prev_meas->uv_norm.x(), prev_meas->uv_norm.y(), 1.0);
        Eigen::Vector3d b_curr(curr_meas->uv_norm.x(), curr_meas->uv_norm.y(), 1.0);
        b_prev.normalize();
        b_curr.normalize();

        const Eigen::Vector3d predicted = R_curr_from_prev * b_prev;
        sum_residual += (b_curr - predicted).norm();
        ++matched;
    }

    bool raw_decision = detector.is_hovering; // hold last confirmed mode if not enough data
    if (matched >= detector.options.min_features) {
        const double d_k = sum_residual / double(matched);
        raw_decision = d_k < detector.options.epsilon;
    }

    if (raw_decision == detector.pending_raw) {
        ++detector.consecutive_count;
    } else {
        detector.pending_raw = raw_decision;
        detector.consecutive_count = 1;
    }

    if (detector.consecutive_count >= detector.options.hysteresis_frames) {
        detector.is_hovering = raw_decision;
    }
    return detector.is_hovering;
}

} // namespace msckf
