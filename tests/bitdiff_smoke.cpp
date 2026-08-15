// Parity stage 2.1 -- Lie / quaternion math, DOD vs official Open_VINS.
//
// Doubles as the harness smoke test: if this links and runs, the oracle is
// wired up correctly and every later stage can be built the same way.
#include "bitdiff.hpp"

#include "../type/quat_ops.hpp"
#include <utils/quat_ops.h>  // ov_core

int main() {
    bitdiff::Report rep;
    bitdiff::Rng rng(20260814);

    // Angle magnitudes spanning the small-angle branches both sides take.
    const double scales[] = {1e-12, 1e-8, 1e-4, 1e-2, 0.5, 1.5, 3.0, 6.0};

    for (int trial = 0; trial < 200; ++trial) {
        const double s = scales[trial % 8];
        Eigen::Vector3d w = rng.vec(3, -1.0, 1.0).normalized() * s * rng.uniform(0.5, 1.0);
        Eigen::Vector3d t = rng.vec(3, -5.0, 5.0);

        rep.expect("skew_x", type::skew_x(w), ov_core::skew_x(w));
        rep.expect("exp_so3", type::exp_so3(w), ov_core::exp_so3(w));
        rep.expect("Jl_so3", type::Jl_so3(w), ov_core::Jl_so3(w));
        rep.expect("Jr_so3", type::Jr_so3(w), ov_core::Jr_so3(w));
        rep.expect("Omega", type::Omega(w), ov_core::Omega(w));

        // Feed log_so3 / rot_2_quat a genuine rotation, not noise.
        Eigen::Matrix3d R = ov_core::exp_so3(w);
        rep.expect("log_so3", type::log_so3(R), ov_core::log_so3(R));
        rep.expect("rot_2_quat", type::rot_2_quat(R), ov_core::rot_2_quat(R));
        rep.expect("rot2rpy", type::rot2rpy(R), ov_core::rot2rpy(R));
        rep.expect("vee", type::vee(ov_core::skew_x(w)), ov_core::vee(ov_core::skew_x(w)));

        Eigen::Vector4d q = ov_core::rot_2_quat(R);
        Eigen::Vector4d q2 = ov_core::rot_2_quat(ov_core::exp_so3(t * 0.1));
        rep.expect("quat_2_Rot", type::quat_2_Rot(q), ov_core::quat_2_Rot(q));
        rep.expect("quat_multiply", type::quat_multiply(q, q2), ov_core::quat_multiply(q, q2));
        rep.expect("Inv(quat)", type::Inv(q), ov_core::Inv(q));

        // quatnorm's job is sign/normalisation fixup, so hand it an unnormalised one.
        Eigen::Vector4d qraw = q * rng.uniform(-2.0, 2.0);
        rep.expect("quatnorm", type::quatnorm(qraw), ov_core::quatnorm(qraw));

        Eigen::Matrix<double, 6, 1> xi;
        xi << w, t;
        rep.expect("hat_se3", type::hat_se3(xi), ov_core::hat_se3(xi));
        rep.expect("exp_se3", type::exp_se3(xi), ov_core::exp_se3(xi));

        Eigen::Matrix4d T = ov_core::exp_se3(xi);
        rep.expect("log_se3", type::log_se3(T), ov_core::log_se3(T));
        rep.expect("Inv_se3", type::Inv_se3(T), ov_core::Inv_se3(T));

        double a = rng.uniform(-3.2, 3.2);
        rep.expect("rot_x", type::rot_x(a), ov_core::rot_x(a));
        rep.expect("rot_y", type::rot_y(a), ov_core::rot_y(a));
        rep.expect("rot_z", type::rot_z(a), ov_core::rot_z(a));
    }

    return rep.finish("stage 2.1 quat_ops");
}
