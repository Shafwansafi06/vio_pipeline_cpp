#include "quat_ops.hpp"
#include <iostream>

namespace type {

Eigen::Vector4d rot_2_quat(const Eigen::Matrix3d& rot) {
    Eigen::Vector4d q = Eigen::Vector4d::Zero();
    double T = rot.trace();

    if ((rot(0, 0) >= T) && (rot(0, 0) >= rot(1, 1)) && (rot(0, 0) >= rot(2, 2))) {
        q[0] = std::sqrt((1.0 + (2.0 * rot(0, 0)) - T) / 4.0);
        q[1] = (1.0 / (4.0 * q[0])) * (rot(0, 1) + rot(1, 0));
        q[2] = (1.0 / (4.0 * q[0])) * (rot(0, 2) + rot(2, 0));
        q[3] = (1.0 / (4.0 * q[0])) * (rot(1, 2) - rot(2, 1));
    }
    else if ((rot(1, 1) >= T) && (rot(1, 1) >= rot(0, 0)) && (rot(1, 1) >= rot(2, 2))) {
        q[1] = std::sqrt((1.0 + (2.0 * rot(1, 1)) - T) / 4.0);
        q[0] = (1.0 / (4.0 * q[1])) * (rot(0, 1) + rot(1, 0));
        q[2] = (1.0 / (4.0 * q[1])) * (rot(1, 2) + rot(2, 1));
        q[3] = (1.0 / (4.0 * q[1])) * (rot(2, 0) - rot(0, 2));
    }
    else if ((rot(2, 2) >= T) && (rot(2, 2) >= rot(0, 0)) && (rot(2, 2) >= rot(1, 1))) {
        q[2] = std::sqrt((1.0 + (2.0 * rot(2, 2)) - T) / 4.0);
        q[0] = (1.0 / (4.0 * q[2])) * (rot(0, 2) + rot(2, 0));
        q[1] = (1.0 / (4.0 * q[2])) * (rot(1, 2) + rot(2, 1));
        q[3] = (1.0 / (4.0 * q[2])) * (rot(0, 1) - rot(1, 0));
    }
    else {
        q[3] = std::sqrt((1.0 + T) / 4.0);
        q[0] = (1.0 / (4.0 * q[3])) * (rot(1, 2) - rot(2, 1));
        q[1] = (1.0 / (4.0 * q[3])) * (rot(2, 0) - rot(0, 2));
        q[2] = (1.0 / (4.0 * q[3])) * (rot(0, 1) - rot(1, 0));
    }

    if (q[3] < 0) {
        q = -q;
    }

    return q / q.norm();
}

Eigen::Matrix3d skew_x(const Eigen::Vector3d& w) {
    Eigen::Matrix3d skew;
    skew << 0.0, -w[2],  w[1],
            w[2],  0.0, -w[0],
           -w[1],  w[0],  0.0;
    return skew;
}

Eigen::Matrix3d quat_2_Rot(const Eigen::Vector4d& q) {
    Eigen::Vector3d q_vec = q.head<3>();
    double q_w = q[3];
    Eigen::Matrix3d q_x = skew_x(q_vec);
    Eigen::Matrix3d Rot = ((2.0 * (q_w * q_w) - 1.0) * Eigen::Matrix3d::Identity()) 
                          - (2.0 * q_w * q_x) + (2.0 * q_vec * q_vec.transpose());
    return Rot;
}

Eigen::Vector4d quat_multiply(const Eigen::Vector4d& q, const Eigen::Vector4d& p) {
    Eigen::Vector3d q_vec = q.head<3>();
    double q_w = q[3];

    Eigen::Matrix4d Qm = Eigen::Matrix4d::Zero();
    Qm.block<3, 3>(0, 0) = q_w * Eigen::Matrix3d::Identity() - skew_x(q_vec);
    Qm.block<3, 1>(0, 3) = q_vec;
    Qm.block<1, 3>(3, 0) = -q_vec.transpose();
    Qm(3, 3) = q_w;

    Eigen::Vector4d q_t = Qm * p;

    if (q_t[3] < 0) {
        q_t = -q_t;
    }

    return q_t / q_t.norm();
}

Eigen::Vector3d vee(const Eigen::Matrix3d& w_x) {
    Eigen::Vector3d w;
    w << w_x(2, 1), w_x(0, 2), w_x(1, 0);
    return w;
}

Eigen::Matrix3d exp_so3(const Eigen::Vector3d& w) {
    double theta = w.norm();
    Eigen::Matrix3d w_x = skew_x(w);
    double A, B;

    if (theta < 1e-7) {
        A = 1.0;
        B = 0.5;
    } else {
        A = std::sin(theta) / theta;
        B = (1.0 - std::cos(theta)) / (theta * theta);
    }

    if (theta == 0.0) {
        return Eigen::Matrix3d::Identity();
    } else {
        return Eigen::Matrix3d::Identity() + A * w_x + B * w_x * w_x;
    }
}

Eigen::Vector3d log_so3(const Eigen::Matrix3d& R) {
    double tr = R.trace();
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();

    if (tr + 1.0 < 1e-10) {
        if (std::abs(R(2, 2) + 1.0) > 1e-5) {
            omega = (M_PI / std::sqrt(2.0 + 2.0 * R(2, 2))) * Eigen::Vector3d(R(0, 2), R(1, 2), 1.0 + R(2, 2));
        } else if (std::abs(R(1, 1) + 1.0) > 1e-5) {
            omega = (M_PI / std::sqrt(2.0 + 2.0 * R(1, 1))) * Eigen::Vector3d(R(0, 1), 1.0 + R(1, 1), R(2, 1));
        } else {
            omega = (M_PI / std::sqrt(2.0 + 2.0 * R(0, 0))) * Eigen::Vector3d(1.0 + R(0, 0), R(1, 0), R(2, 0));
        }
    } else {
        double magnitude = 0.0;
        double tr_3 = tr - 3.0;

        if (tr_3 < -1e-7) {
            double theta = std::acos((tr - 1.0) / 2.0);
            magnitude = theta / (2.0 * std::sin(theta));
        } else {
            magnitude = 0.5 - tr_3 / 12.0;
        }

        omega = magnitude * Eigen::Vector3d(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
    }

    return omega;
}

Eigen::Matrix4d exp_se3(const Eigen::Matrix<double, 6, 1>& vec) {
    Eigen::Vector3d w = vec.head<3>();
    Eigen::Vector3d u = vec.tail<3>();
    double theta = w.norm();
    Eigen::Matrix3d wskew = skew_x(w);
    double A, B, C;

    if (theta < 1e-7) {
        A = 1.0;
        B = 0.5;
        C = 1.0 / 6.0;
    } else {
        A = std::sin(theta) / theta;
        B = (1.0 - std::cos(theta)) / (theta * theta);
        C = (1.0 - A) / (theta * theta);
    }

    Eigen::Matrix3d V = Eigen::Matrix3d::Identity() + B * wskew + C * wskew * wskew;

    Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
    mat.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() + A * wskew + B * wskew * wskew;
    mat.block<3, 1>(0, 3) = V * u;
    mat(3, 3) = 1.0;

    return mat;
}

Eigen::Matrix<double, 6, 1> log_se3(const Eigen::Matrix4d& mat) {
    Eigen::Matrix3d R = mat.block<3, 3>(0, 0);
    Eigen::Vector3d T = mat.block<3, 1>(0, 3);
    Eigen::Vector3d w = log_so3(R);
    double t = w.norm();

    Eigen::Matrix<double, 6, 1> vec;
    if (t < 1e-10) {
        vec.head<3>() = w;
        vec.tail<3>() = T;
    } else {
        Eigen::Matrix3d W = skew_x(w / t);
        double Tan = std::tan(0.5 * t);
        Eigen::Vector3d u = T - (0.5 * t) * (W * T) + (1.0 - t / (2.0 * Tan)) * (W * W * T);
        vec.head<3>() = w;
        vec.tail<3>() = u;
    }
    return vec;
}

Eigen::Matrix4d hat_se3(const Eigen::Matrix<double, 6, 1>& vec) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
    mat.block<3, 3>(0, 0) = skew_x(vec.head<3>());
    mat.block<3, 1>(0, 3) = vec.tail<3>();
    return mat;
}

Eigen::Matrix4d Inv_se3(const Eigen::Matrix4d& T) {
    Eigen::Matrix4d Tinv = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d R_t = T.block<3, 3>(0, 0).transpose();
    Tinv.block<3, 3>(0, 0) = R_t;
    Tinv.block<3, 1>(0, 3) = -R_t * T.block<3, 1>(0, 3);
    return Tinv;
}

Eigen::Vector4d Inv(const Eigen::Vector4d& q) {
    Eigen::Vector4d qinv = q;
    qinv.head<3>() = -q.head<3>();
    return qinv;
}

Eigen::Matrix4d Omega(const Eigen::Vector3d& w) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
    mat.block<3, 3>(0, 0) = -skew_x(w);
    mat.block<3, 1>(0, 3) = w;
    mat.block<1, 3>(3, 0) = -w.transpose();
    return mat;
}

Eigen::Vector4d quatnorm(const Eigen::Vector4d& q_t) {
    Eigen::Vector4d q = q_t;
    if (q[3] < 0) {
        q = -q;
    }
    return q / q.norm();
}

Eigen::Matrix3d Jl_so3(const Eigen::Vector3d& w) {
    double theta = w.norm();
    if (theta < 1e-6) {
        return Eigen::Matrix3d::Identity();
    } else {
        Eigen::Vector3d a = w / theta;
        Eigen::Matrix3d J = (std::sin(theta) / theta) * Eigen::Matrix3d::Identity() +
                            (1.0 - std::sin(theta) / theta) * (a * a.transpose()) +
                            ((1.0 - std::cos(theta)) / theta) * skew_x(a);
        return J;
    }
}

Eigen::Matrix3d Jr_so3(const Eigen::Vector3d& w) {
    return Jl_so3(-w);
}

Eigen::Vector3d rot2rpy(const Eigen::Matrix3d& rot) {
    Eigen::Vector3d rpy = Eigen::Vector3d::Zero();
    rpy[1] = std::atan2(-rot(2, 0), std::sqrt(rot(0, 0)*rot(0, 0) + rot(1, 0)*rot(1, 0)));

    if (std::abs(std::cos(rpy[1])) > 1.0e-12) {
        rpy[2] = std::atan2(rot(1, 0) / std::cos(rpy[1]), rot(0, 0) / std::cos(rpy[1]));
        rpy[0] = std::atan2(rot(2, 1) / std::cos(rpy[1]), rot(2, 2) / std::cos(rpy[1]));
    } else {
        rpy[2] = 0.0;
        rpy[0] = std::atan2(rot(0, 1), rot(1, 1));
    }
    return rpy;
}

Eigen::Matrix3d rot_x(double t) {
    double ct = std::cos(t);
    double st = std::sin(t);
    Eigen::Matrix3d R;
    R << 1.0, 0.0, 0.0,
         0.0,  ct, -st,
         0.0,  st,  ct;
    return R;
}

Eigen::Matrix3d rot_y(double t) {
    double ct = std::cos(t);
    double st = std::sin(t);
    Eigen::Matrix3d R;
    R <<  ct, 0.0,  st,
         0.0, 1.0, 0.0,
         -st, 0.0,  ct;
    return R;
}

Eigen::Matrix3d rot_z(double t) {
    double ct = std::cos(t);
    double st = std::sin(t);
    Eigen::Matrix3d R;
    R <<  ct, -st, 0.0,
          st,  ct, 0.0,
         0.0, 0.0, 1.0;
    return R;
}

} // namespace type
