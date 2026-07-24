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

Eigen::Vector2d distort(const CameraModel& cam, const Eigen::Vector2d& uv_norm) {
    if (cam.type == CameraModelType::RADTAN) {
        double fx = cam.values[0];
        double fy = cam.values[1];
        double cx = cam.values[2];
        double cy = cam.values[3];
        double k1 = cam.values[4];
        double k2 = cam.values[5];
        double p1 = cam.values[6];
        double p2 = cam.values[7];
        
        double x = uv_norm[0];
        double y = uv_norm[1];
        
        double r2 = x*x + y*y;
        double r4 = r2 * r2;
        double radial = 1.0 + k1 * r2 + k2 * r4;
        double x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        double y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        
        return Eigen::Vector2d(fx * x_dist + cx, fy * y_dist + cy);
    } else {
        double fx = cam.values[0];
        double fy = cam.values[1];
        double cx = cam.values[2];
        double cy = cam.values[3];
        double k1 = cam.values[4];
        double k2 = cam.values[5];
        double k3 = cam.values[6];
        double k4 = cam.values[7];
        
        double x = uv_norm[0];
        double y = uv_norm[1];
        
        double r = std::sqrt(x*x + y*y);
        double theta = std::atan(r);
        
        double theta2 = theta * theta;
        double theta4 = theta2 * theta2;
        double theta6 = theta4 * theta2;
        double theta8 = theta4 * theta4;
        
        double scaling = 1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
        double theta_d = theta * scaling;
        
        double cdist = (r > 1e-8) ? (theta_d / r) : 1.0;
        
        double x_dist = x * cdist;
        double y_dist = y * cdist;
        
        return Eigen::Vector2d(fx * x_dist + cx, fy * y_dist + cy);
    }
}

void compute_distort_jacobian(const CameraModel& cam, const Eigen::Vector2d& uv_norm,
                              Eigen::Matrix2d& H_dz_dzn, Eigen::Matrix<double, 2, 8>& H_dz_dzeta) {
    if (cam.type == CameraModelType::RADTAN) {
        double fx = cam.values[0];
        double fy = cam.values[1];
        double k1 = cam.values[4];
        double k2 = cam.values[5];
        double p1 = cam.values[6];
        double p2 = cam.values[7];
        
        double x = uv_norm[0];
        double y = uv_norm[1];
        double x2 = x * x;
        double y2 = y * y;
        double xy = x * y;
        double r2 = x2 + y2;
        double r4 = r2 * r2;
        
        H_dz_dzn.setZero();
        double rad_factor = 1.0 + k1 * r2 + k2 * r4;
        
        double dxd_dx = rad_factor + (2.0 * k1 * x2 + 4.0 * k2 * x2 * r2) + 2.0 * p1 * y + 6.0 * p2 * x;
        H_dz_dzn(0, 0) = fx * dxd_dx;
        
        double dxd_dy = (2.0 * k1 * xy + 4.0 * k2 * xy * r2) + 2.0 * p1 * x + 2.0 * p2 * y;
        H_dz_dzn(0, 1) = fx * dxd_dy;
        
        double dyd_dx = (2.0 * k1 * xy + 4.0 * k2 * xy * r2) + 2.0 * p1 * x + 2.0 * p2 * y;
        H_dz_dzn(1, 0) = fy * dyd_dx;
        
        double dyd_dy = rad_factor + (2.0 * k1 * y2 + 4.0 * k2 * y2 * r2) + 6.0 * p1 * y + 2.0 * p2 * x;
        H_dz_dzn(1, 1) = fy * dyd_dy;
        
        H_dz_dzeta.setZero();
        double x_dist_k = x * rad_factor + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2);
        double y_dist_k = y * rad_factor + p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy;
        
        H_dz_dzeta(0, 0) = x_dist_k;
        H_dz_dzeta(1, 1) = y_dist_k;
        H_dz_dzeta(0, 2) = 1.0;
        H_dz_dzeta(1, 3) = 1.0;
        
        H_dz_dzeta(0, 4) = fx * x * r2;
        H_dz_dzeta(1, 4) = fy * y * r2;
        
        H_dz_dzeta(0, 5) = fx * x * r4;
        H_dz_dzeta(1, 5) = fy * y * r4;
        
        H_dz_dzeta(0, 6) = fx * (2.0 * x * y);
        H_dz_dzeta(1, 6) = fy * (r2 + 2.0 * y2);
        
        H_dz_dzeta(0, 7) = fx * (r2 + 2.0 * x2);
        H_dz_dzeta(1, 7) = fy * (2.0 * x * y);
    } else {
        double fx = cam.values[0];
        double fy = cam.values[1];
        double k1 = cam.values[4];
        double k2 = cam.values[5];
        double k3 = cam.values[6];
        double k4 = cam.values[7];
        
        double x = uv_norm[0];
        double y = uv_norm[1];
        double r = std::sqrt(x * x + y * y);
        double theta = std::atan(r);
        
        double theta2 = theta * theta;
        double theta3 = theta * theta2;
        double theta4 = theta2 * theta2;
        double theta5 = theta * theta4;
        double theta6 = theta4 * theta2;
        double theta7 = theta * theta6;
        double theta8 = theta4 * theta4;
        double theta9 = theta * theta8;
        
        double theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
        
        double inv_r = (r > 1e-8) ? (1.0 / r) : 1.0;
        double inv_r2 = inv_r * inv_r;
        double cdist = (r > 1e-8) ? (theta_d * inv_r) : 1.0;
        
        Eigen::Matrix2d duv_dxy;
        duv_dxy << fx, 0.0, 0.0, fy;
        
        double dthd_dth = 1.0 + 3.0 * k1 * theta2 + 5.0 * k2 * theta4 + 7.0 * k3 * theta6 + 9.0 * k4 * theta8;
        double dth_dr = 1.0 / (1.0 + r * r);
        
        Eigen::RowVector2d dr_dxyn(x * inv_r, y * inv_r);
        
        Eigen::Matrix2d M1 = Eigen::Matrix2d::Identity() * cdist;
        
        double dthd_dr = dthd_dth * dth_dr;
        double d_scale_dr = (r * dthd_dr - theta_d) * inv_r2;
        
        Eigen::Vector2d dxy_dr(x * d_scale_dr, y * d_scale_dr);
        
        Eigen::Matrix2d dxy_dist_dxyn = M1 + dxy_dr * dr_dxyn;
        H_dz_dzn = duv_dxy * dxy_dist_dxyn;
        
        H_dz_dzeta.setZero();
        H_dz_dzeta(0, 0) = x * cdist;
        H_dz_dzeta(1, 1) = y * cdist;
        H_dz_dzeta(0, 2) = 1.0;
        H_dz_dzeta(1, 3) = 1.0;
        
        double term_x = fx * x * inv_r;
        double term_y = fy * y * inv_r;
        
        H_dz_dzeta(0, 4) = term_x * theta3;
        H_dz_dzeta(1, 4) = term_y * theta3;
        
        H_dz_dzeta(0, 5) = term_x * theta5;
        H_dz_dzeta(1, 5) = term_y * theta5;
        
        H_dz_dzeta(0, 6) = term_x * theta7;
        H_dz_dzeta(1, 6) = term_y * theta7;
        
        H_dz_dzeta(0, 7) = term_x * theta9;
        H_dz_dzeta(1, 7) = term_y * theta9;
    }
}

} // namespace core
