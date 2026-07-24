/**
 * ov_compat_prefix.hpp
 *
 * Forced-include prefix for Open_VINS vendor sources compiled against OpenCV5.
 * Injected via -include flag. Must be a single word (no spaces in path).
 *
 * OpenCV5 reorganized:
 *   cv::undistortPoints         -> opencv2/geometry/3d.hpp  (cv:: namespace, same)
 *   cv::fisheye::undistortPoints -> opencv2/geometry/3d.hpp (cv::fisheye:: namespace, same)
 *
 * The symbols are still in cv:: / cv::fisheye:: — they just require a different
 * header inclusion. We pre-include it so OV code works without modification.
 */
#pragma once
#include <cassert>
#include <opencv2/geometry/3d.hpp>
