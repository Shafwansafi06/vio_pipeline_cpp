#pragma once
#include <Eigen/Core>
#include "../core/feature.hpp"

namespace msckf {

// Motion classifier from Kottas, Wu & Roumeliotis, "Detecting and Dealing
// with Hovering Maneuvers in Vision-aided Inertial Navigation Systems"
// (Sect. III-C). Classifies hovering (zero translation) vs generic motion
// using only the rotation-compensated bearing-vector residual between the
// two most recent camera poses -- no IMU velocity/disparity heuristics.
// For each feature tracked at both poses, with unit bearing b_k = z_k/||z_k||
// (Eq. 26-28 use the "0-pt RANSAC" case: under zero translation,
// b_{k+1} ~= C(q_{k+1<-k}) b_k, so the mean residual
//   d_k = (1/M) sum_i || b_{k+1}^i - C(q_{k+1<-k}) b_k^i ||_2
// is small only when the camera did not translate between the two poses,
// regardless of rotation. Thresholding d_k < epsilon (Eq. 29) with a
// consecutive-decision hysteresis (paper, end of Sect. III-C) gives a
// hovering/non-hovering classification per frame.
struct HoverDetectorOptions {
    double epsilon = 0.005;   // threshold on the mean bearing residual d_k
    int min_features = 5;     // minimum correspondences to trust d_k at all
    int hysteresis_frames = 3; // consecutive agreeing raw decisions required to flip mode
};

struct HoverDetectorData {
    HoverDetectorOptions options;
    bool is_hovering = false; // confirmed (hysteresis-applied) mode
    bool pending_raw = false;
    int consecutive_count = 0;
};

void init_hover_detector(HoverDetectorData& detector, const HoverDetectorOptions& options);

// Updates the classifier using feature correspondences (cam_id 0 only,
// matching the paper's monocular formulation) between `previous_timestamp`
// and `current_timestamp`, and the IMU-propagated relative rotation
// R_curr_from_prev (v_in_current = R_curr_from_prev * v_in_previous).
// Returns the confirmed mode after applying hysteresis; also updates
// detector.is_hovering in place.
bool update_hover_detector(HoverDetectorData& detector, const core::FeatureDatabase& db,
                          double previous_timestamp, double current_timestamp,
                          const Eigen::Matrix3d& R_curr_from_prev);

} // namespace msckf
