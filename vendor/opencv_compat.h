/**
 * opencv_compat.h
 *
 * OpenCV 5 compatibility shim for Open_VINS vendor code.
 * In OpenCV 5, cv::undistortPoints and cv::fisheye::undistortPoints moved from
 * <opencv2/calib3d.hpp> to <opencv2/geometry/3d.hpp>. This header re-exports
 * them so vendor code that includes <opencv2/opencv.hpp> can still find them
 * in the cv:: and cv::fisheye:: namespaces as expected.
 */
#pragma once

// Include the OpenCV5 header that has the actual implementations
#include <opencv2/geometry/3d.hpp>
