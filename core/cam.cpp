#include "cam.hpp"
#include <cmath>
#include <cassert>

namespace core {

void init_camera(CameraModel& cam, CameraModelType type, int width, int height, const double values[8]) {
    cam.type = type;
    cam.width = width;
    cam.height = height;
    set_camera_values(cam, values);
}

void set_camera_values(CameraModel& cam, const double values[8]) {
    for (int i = 0; i < 8; ++i) {
        cam.values[i] = values[i];
    }
    
    cam.K << values[0], 0.0,       values[2],
             0.0,       values[1], values[3],
             0.0,       0.0,       1.0;
             
    cam.D << values[4], values[5], values[6], values[7];
}

// LEGACY undistortion -- deliberately NOT bit-parity with official.
//
// core::undistort_official() (core/undistort_cv.cpp) is the parity-exact
// version and is proven ULP-0 against the oracle by bitdiff_cam. It is not
// wired into the frontend yet, because swapping it in ALONE regresses EuRoC
// MH_01 from 0.407 m to 6.28 m ATE. That was bisected: reverting only the
// undistort change restores 0.4074 m, while reverting only the distort and
// Jacobian changes leaves 6.28 m.
//
// The reason is that parity is a property of the WHOLE pipeline. DOD's tracker
// is not yet matched to official's TrackKLT, so pairing official's undistortion
// with DOD's feature selection produces a hybrid that is faithful to neither.
// The swap belongs with the frontend parity work, not before it.
Eigen::Vector2d undistort(const CameraModel& cam, const Eigen::Vector2d& uv_dist) {
    if (cam.type == CameraModelType::RADTAN) {
        double fx = cam.values[0];
        double fy = cam.values[1];
        double cx = cam.values[2];
        double cy = cam.values[3];
        double k1 = cam.values[4];
        double k2 = cam.values[5];
        double p1 = cam.values[6];
        double p2 = cam.values[7];
        
        double x_d = (uv_dist[0] - cx) / fx;
        double y_d = (uv_dist[1] - cy) / fy;
        
        double x = x_d;
        double y = y_d;
        
        for (int i = 0; i < 8; ++i) {
            double r2 = x*x + y*y;
            double r4 = r2 * r2;
            double radial = 1.0 + k1 * r2 + k2 * r4;
            double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
            x = (x_d - dx) / radial;
            y = (y_d - dy) / radial;
        }
        return Eigen::Vector2d(x, y);
    } else {
        double fx = cam.values[0];
        double fy = cam.values[1];
        double cx = cam.values[2];
        double cy = cam.values[3];
        double k1 = cam.values[4];
        double k2 = cam.values[5];
        double k3 = cam.values[6];
        double k4 = cam.values[7];
        
        double x_d = (uv_dist[0] - cx) / fx;
        double y_d = (uv_dist[1] - cy) / fy;
        
        double r_d = std::sqrt(x_d * x_d + y_d * y_d);
        
        if (r_d < 1e-10) {
            return Eigen::Vector2d(x_d, y_d);
        }
        
        double theta = r_d;
        for (int i = 0; i < 10; ++i) {
            double theta2 = theta * theta;
            double theta4 = theta2 * theta2;
            double theta6 = theta4 * theta2;
            double theta8 = theta4 * theta4;
            
            double f = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8) - r_d;
            double df = 1.0 + 3.0 * k1 * theta2 + 5.0 * k2 * theta4 + 7.0 * k3 * theta6 + 9.0 * k4 * theta8;
            theta = theta - f / df;
        }
        
        double r = std::tan(theta);
        double scale = r / r_d;
        return Eigen::Vector2d(x_d * scale, y_d * scale);
    }
}

// ---------------------------------------------------------------------------
// Bit-parity with official Open_VINS.
//
// Two things below look like they could be simplified, and must not be:
//
//   1. Official's distort_d/distort_f round through float32 (CamBase.h casts
//      to Vector2f, calls distort_f, casts back). The estimator's predicted
//      pixel -- and therefore every residual -- carries that rounding. Keeping
//      full double precision here would be "more accurate" and would not match.
//
//   2. Official computes r = sqrt(x^2+y^2) and then r_2 = r*r, rather than
//      using x^2+y^2 directly. The sqrt/square round trip is lossy, so the two
//      differ in the last bits. Same for `2*p2*x + 4*p2*x` where 6*p2*x would
//      be the obvious tidy-up, and for `theta_d * inv_r` where theta_d / r
//      would read better.
//
// `undistort` is deliberately absent from this file: official implements it
// with cv::undistortPoints, so it lives in the OpenCV frontend
// (core/undistort_cv.cpp) and this library stays free of OpenCV.
// ---------------------------------------------------------------------------

Eigen::Vector2d distort(const CameraModel& cam, const Eigen::Vector2d& uv_norm) {
    // Official rounds the input to float before distorting (CamBase::distort_d).
    const Eigen::Vector2f in = uv_norm.cast<float>();
    Eigen::Vector2f out;

    if (cam.type == CameraModelType::RADTAN) {
        const double* cam_d = cam.values;
        double r = std::sqrt(in(0) * in(0) + in(1) * in(1));
        double r_2 = r * r;
        double r_4 = r_2 * r_2;
        double x1 = in(0) * (1 + cam_d[4] * r_2 + cam_d[5] * r_4) + 2 * cam_d[6] * in(0) * in(1) +
                    cam_d[7] * (r_2 + 2 * in(0) * in(0));
        double y1 = in(1) * (1 + cam_d[4] * r_2 + cam_d[5] * r_4) + cam_d[6] * (r_2 + 2 * in(1) * in(1)) +
                    2 * cam_d[7] * in(0) * in(1);
        out(0) = (float)(cam_d[0] * x1 + cam_d[2]);
        out(1) = (float)(cam_d[1] * y1 + cam_d[3]);
    } else {
        const double* cam_d = cam.values;
        double r = std::sqrt(in(0) * in(0) + in(1) * in(1));
        double theta = std::atan(r);
        double theta_d = theta + cam_d[4] * std::pow(theta, 3) + cam_d[5] * std::pow(theta, 5) +
                         cam_d[6] * std::pow(theta, 7) + cam_d[7] * std::pow(theta, 9);
        double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;
        double cdist = (r > 1e-8) ? theta_d * inv_r : 1.0;
        double x1 = in(0) * cdist;
        double y1 = in(1) * cdist;
        out(0) = (float)(cam_d[0] * x1 + cam_d[2]);
        out(1) = (float)(cam_d[1] * y1 + cam_d[3]);
    }
    return out.cast<double>();
}

void compute_distort_jacobian(const CameraModel& cam, const Eigen::Vector2d& uv_norm,
                              Eigen::Matrix2d& H_dz_dzn, Eigen::Matrix<double, 2, 8>& H_dz_dzeta) {
    // Official takes Vector2d here and stays in double throughout -- no float
    // round trip, unlike distort above.
    const double* cam_d = cam.values;

    if (cam.type == CameraModelType::RADTAN) {
        double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
        double r_2 = r * r;
        double r_4 = r_2 * r_2;

        H_dz_dzn.setZero();
        double x = uv_norm(0);
        double y = uv_norm(1);
        double x_2 = uv_norm(0) * uv_norm(0);
        double y_2 = uv_norm(1) * uv_norm(1);
        double x_y = uv_norm(0) * uv_norm(1);
        H_dz_dzn(0, 0) = cam_d[0] * ((1 + cam_d[4] * r_2 + cam_d[5] * r_4) + (2 * cam_d[4] * x_2 + 4 * cam_d[5] * x_2 * r_2) +
                                     2 * cam_d[6] * y + (2 * cam_d[7] * x + 4 * cam_d[7] * x));
        H_dz_dzn(0, 1) = cam_d[0] * (2 * cam_d[4] * x_y + 4 * cam_d[5] * x_y * r_2 + 2 * cam_d[6] * x + 2 * cam_d[7] * y);
        H_dz_dzn(1, 0) = cam_d[1] * (2 * cam_d[4] * x_y + 4 * cam_d[5] * x_y * r_2 + 2 * cam_d[6] * x + 2 * cam_d[7] * y);
        H_dz_dzn(1, 1) = cam_d[1] * ((1 + cam_d[4] * r_2 + cam_d[5] * r_4) + (2 * cam_d[4] * y_2 + 4 * cam_d[5] * y_2 * r_2) +
                                     2 * cam_d[7] * x + (2 * cam_d[6] * y + 4 * cam_d[6] * y));

        double x1 = uv_norm(0) * (1 + cam_d[4] * r_2 + cam_d[5] * r_4) + 2 * cam_d[6] * uv_norm(0) * uv_norm(1) +
                    cam_d[7] * (r_2 + 2 * uv_norm(0) * uv_norm(0));
        double y1 = uv_norm(1) * (1 + cam_d[4] * r_2 + cam_d[5] * r_4) + cam_d[6] * (r_2 + 2 * uv_norm(1) * uv_norm(1)) +
                    2 * cam_d[7] * uv_norm(0) * uv_norm(1);

        H_dz_dzeta.setZero();
        H_dz_dzeta(0, 0) = x1;
        H_dz_dzeta(0, 2) = 1;
        H_dz_dzeta(0, 4) = cam_d[0] * uv_norm(0) * r_2;
        H_dz_dzeta(0, 5) = cam_d[0] * uv_norm(0) * r_4;
        H_dz_dzeta(0, 6) = 2 * cam_d[0] * uv_norm(0) * uv_norm(1);
        H_dz_dzeta(0, 7) = cam_d[0] * (r_2 + 2 * uv_norm(0) * uv_norm(0));
        H_dz_dzeta(1, 1) = y1;
        H_dz_dzeta(1, 3) = 1;
        H_dz_dzeta(1, 4) = cam_d[1] * uv_norm(1) * r_2;
        H_dz_dzeta(1, 5) = cam_d[1] * uv_norm(1) * r_4;
        H_dz_dzeta(1, 6) = cam_d[1] * (r_2 + 2 * uv_norm(1) * uv_norm(1));
        H_dz_dzeta(1, 7) = 2 * cam_d[1] * uv_norm(0) * uv_norm(1);
    } else {
        double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
        double theta = std::atan(r);
        double theta_d = theta + cam_d[4] * std::pow(theta, 3) + cam_d[5] * std::pow(theta, 5) +
                         cam_d[6] * std::pow(theta, 7) + cam_d[7] * std::pow(theta, 9);

        double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;
        double cdist = (r > 1e-8) ? theta_d * inv_r : 1.0;

        // Official assembles this from six named blocks and one matrix product.
        // Collapsing it into scalars changes the summation order.
        Eigen::Matrix<double, 2, 2> duv_dxy = Eigen::Matrix<double, 2, 2>::Zero();
        duv_dxy << cam_d[0], 0, 0, cam_d[1];

        Eigen::Matrix<double, 2, 2> dxy_dxyn = Eigen::Matrix<double, 2, 2>::Zero();
        dxy_dxyn << theta_d * inv_r, 0, 0, theta_d * inv_r;

        Eigen::Matrix<double, 2, 1> dxy_dr = Eigen::Matrix<double, 2, 1>::Zero();
        dxy_dr << -uv_norm(0) * theta_d * inv_r * inv_r, -uv_norm(1) * theta_d * inv_r * inv_r;

        Eigen::Matrix<double, 1, 2> dr_dxyn = Eigen::Matrix<double, 1, 2>::Zero();
        dr_dxyn << uv_norm(0) * inv_r, uv_norm(1) * inv_r;

        Eigen::Matrix<double, 2, 1> dxy_dthd = Eigen::Matrix<double, 2, 1>::Zero();
        dxy_dthd << uv_norm(0) * inv_r, uv_norm(1) * inv_r;

        double dthd_dth = 1 + 3 * cam_d[4] * std::pow(theta, 2) + 5 * cam_d[5] * std::pow(theta, 4) +
                          7 * cam_d[6] * std::pow(theta, 6) + 9 * cam_d[7] * std::pow(theta, 8);
        double dth_dr = 1 / (r * r + 1);

        H_dz_dzn = duv_dxy * (dxy_dxyn + (dxy_dr + dxy_dthd * dthd_dth * dth_dr) * dr_dxyn);

        double x1 = uv_norm(0) * cdist;
        double y1 = uv_norm(1) * cdist;

        H_dz_dzeta.setZero();
        H_dz_dzeta(0, 0) = x1;
        H_dz_dzeta(0, 2) = 1;
        H_dz_dzeta(0, 4) = cam_d[0] * uv_norm(0) * inv_r * std::pow(theta, 3);
        H_dz_dzeta(0, 5) = cam_d[0] * uv_norm(0) * inv_r * std::pow(theta, 5);
        H_dz_dzeta(0, 6) = cam_d[0] * uv_norm(0) * inv_r * std::pow(theta, 7);
        H_dz_dzeta(0, 7) = cam_d[0] * uv_norm(0) * inv_r * std::pow(theta, 9);
        H_dz_dzeta(1, 1) = y1;
        H_dz_dzeta(1, 3) = 1;
        H_dz_dzeta(1, 4) = cam_d[1] * uv_norm(1) * inv_r * std::pow(theta, 3);
        H_dz_dzeta(1, 5) = cam_d[1] * uv_norm(1) * inv_r * std::pow(theta, 5);
        H_dz_dzeta(1, 6) = cam_d[1] * uv_norm(1) * inv_r * std::pow(theta, 7);
        H_dz_dzeta(1, 7) = cam_d[1] * uv_norm(1) * inv_r * std::pow(theta, 9);
    }
}

} // namespace core
