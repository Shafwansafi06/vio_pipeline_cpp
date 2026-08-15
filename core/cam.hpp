#pragma once
#include <Eigen/Core>

namespace core {

enum class CameraModelType {
    RADTAN,
    EQUIDISTANT
};

struct CameraModel {
    CameraModelType type;
    int width;
    int height;
    double values[8]; // fx, fy, cx, cy, d0, d1, d2, d3
    Eigen::Matrix3d K;
    Eigen::Vector4d D;
};

void init_camera(CameraModel& cam, CameraModelType type, int width, int height, const double values[8]);
void set_camera_values(CameraModel& cam, const double values[8]);

// Legacy analytic undistortion (8 fixed-point iterations, double). Used by the
// frontend today; NOT bit-parity with official.
Eigen::Vector2d undistort(const CameraModel& cam, const Eigen::Vector2d& uv_dist);

// Parity-exact undistortion: cv::undistortPoints in float32, matching
// official's CamRadtan/CamEqui::undistort_f. Lives in the OpenCV frontend
// target. Proven ULP-0 by tests/bitdiff_cam.cpp.
Eigen::Vector2d undistort_official(const CameraModel& cam, const Eigen::Vector2d& uv_dist);
Eigen::Vector2d distort(const CameraModel& cam, const Eigen::Vector2d& uv_norm);

void compute_distort_jacobian(const CameraModel& cam, const Eigen::Vector2d& uv_norm,
                              Eigen::Matrix2d& H_dz_dzn, Eigen::Matrix<double, 2, 8>& H_dz_dzeta);

} // namespace core
