// Undistortion, kept apart from core/cam.cpp on purpose.
//
// Official Open_VINS does not undistort analytically -- CamRadtan::undistort_f
// and CamEqui::undistort_f hand the point to cv::undistortPoints /
// cv::fisheye::undistortPoints in float32 (CamRadtan.h:99, CamEqui.h). Those
// routines have their own fixed iteration count and their own arithmetic, so
// no hand-rolled Newton loop reproduces them bit-for-bit; the only way to match
// is to make the same call.
//
// That call needs OpenCV, and the estimator library is deliberately OpenCV-free,
// so this translation unit belongs to the `vio_frontend` target instead. Every
// caller of undistort() is in core/tracker.cpp, which is already in that target.
//
// NOTE: this is exposed as undistort_official(), separate from the legacy
// core::undistort() the frontend still calls. See core/cam.cpp for why the swap
// is deferred to the frontend parity work.
#include "cam.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace core {

Eigen::Vector2d undistort_official(const CameraModel& cam, const Eigen::Vector2d& uv_dist) {
    // Official rounds to float before undistorting (CamBase::undistort_d).
    const Eigen::Vector2f in = uv_dist.cast<float>();

    cv::Matx33d camK(cam.values[0], 0.0, cam.values[2],
                     0.0, cam.values[1], cam.values[3],
                     0.0, 0.0, 1.0);
    cv::Vec4d camD(cam.values[4], cam.values[5], cam.values[6], cam.values[7]);

    cv::Mat mat(1, 2, CV_32F);
    mat.at<float>(0, 0) = in(0);
    mat.at<float>(0, 1) = in(1);
    mat = mat.reshape(2);  // Nx1, 2-channel

    if (cam.type == CameraModelType::RADTAN) {
        cv::undistortPoints(mat, mat, camK, camD);
    } else {
        cv::fisheye::undistortPoints(mat, mat, camK, camD);
    }

    mat = mat.reshape(1);  // Nx2, 1-channel
    return Eigen::Vector2d(mat.at<float>(0, 0), mat.at<float>(0, 1));
}

}  // namespace core
