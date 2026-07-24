// Self-check for the ported linear-MLE dynamic initializer
// (initialize::dynamic_initialize). Forward-simulates a rotating+translating
// IMU trajectory with fine Euler steps, generates perfectly-consistent IMU
// readings and feature bearings, then asserts the initializer recovers the
// correct gravity direction and speed. Gauge-free checks (invariant to the
// yaw/position gauge the initializer is free to choose):
//   - speed |v| at the newest pose matches truth,
//   - gravity direction in the body frame matches truth.
// If the CPI preintegration, linear-system assembly, or Dongsi constrained
// solve is broken, gravity will not converge (init returns false) or the
// direction/speed asserts fail.
#include <cassert>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

#include "../initialize/initialization.hpp"
#include "../core/feature.hpp"
#include "../core/sensor_data.hpp"
#include "../type/quat_ops.hpp"

static Eigen::Matrix3d Exp_so3(const Eigen::Vector3d& w) {
    double th = w.norm();
    if (th < 1e-12) return Eigen::Matrix3d::Identity();
    Eigen::Vector3d a = w / th;
    Eigen::Matrix3d K = type::skew_x(a);
    return Eigen::Matrix3d::Identity() + std::sin(th) * K + (1.0 - std::cos(th)) * K * K;
}

int main() {
    std::cout << "Starting dynamic-initializer verification...\n";

    const double g_mag = 9.81;
    const Eigen::Vector3d grav_G(0.0, 0.0, g_mag); // OpenVINS "up" gravity vector

    // True motion (world/G frame)
    const Eigen::Vector3d omega(0.25, -0.15, 0.30);       // body angular velocity [rad/s]
    const Eigen::Vector3d a_G(0.4, -0.3, 0.2);            // world linear accel [m/s^2]
    const Eigen::Vector3d v0_G(0.5, 0.1, -0.2);           // initial world velocity
    // Initial orientation: a real tilt so gravity is not axis-aligned.
    Eigen::Matrix3d R_GtoI0 = Exp_so3(Eigen::Vector3d(0.2, -0.3, 0.1));

    // JPL kinematics: dR_GtoI/dt = -[w]x R_GtoI  =>  R_GtoI(t) = Exp(-w t) R_GtoI0
    auto R_GtoI = [&](double t) { return Exp_so3(-omega * t) * R_GtoI0; };
    auto v_G = [&](double t) { return (v0_G + a_G * t).eval(); };
    auto p_G = [&](double t) { return (v0_G * t + 0.5 * a_G * t * t).eval(); }; // p0 = 0

    // ---- Fine IMU stream over [-0.10, 1.10] s at 400 Hz ----
    static core::ImuData imu[2048];
    int n_imu = 0;
    for (double t = -0.10; t <= 1.1005; t += 0.0025) {
        core::ImuData d;
        d.timestamp = t;
        d.wm = omega;                                  // gyro = body angular velocity
        d.am = R_GtoI(t) * (a_G + grav_G);             // specific force in body frame
        imu[n_imu++] = d;
    }

    // ---- Features: 12 points ahead, projected into the camera at each frame ----
    // Camera == IMU (identity extrinsics), so p_FinC = R_GtoIk*(p_FinG - p_IkinG).
    // Close features (depth ~2-4 m) so velocity/depth are well observable via
    // parallax over the 1 s arc -- distant features leave metric velocity
    // near-unobservable and noise-dominated.
    Eigen::Vector3d feats_G[16];
    {
        int k = 0;
        for (int ix = -1; ix <= 1; ++ix)
            for (int iy = -1; iy <= 1; ++iy)
                feats_G[k++] = R_GtoI0.transpose() * Eigen::Vector3d(0.8 * ix, 0.8 * iy, 3.0);
        feats_G[9]  = R_GtoI0.transpose() * Eigen::Vector3d(0.4, 0.5, 2.2);
        feats_G[10] = R_GtoI0.transpose() * Eigen::Vector3d(-0.5, 0.6, 4.0);
        feats_G[11] = R_GtoI0.transpose() * Eigen::Vector3d(0.6, -0.5, 2.6);
        feats_G[12] = R_GtoI0.transpose() * Eigen::Vector3d(-0.3, -0.4, 3.5);
        feats_G[13] = R_GtoI0.transpose() * Eigen::Vector3d(0.9, 0.1, 2.0);
        feats_G[14] = R_GtoI0.transpose() * Eigen::Vector3d(-0.8, 0.3, 3.2);
        feats_G[15] = R_GtoI0.transpose() * Eigen::Vector3d(0.2, -0.7, 2.8);
    }
    const int NFEAT = 16;

    // Tiny deterministic noise: a perfectly noise-free system is rank-
    // degenerate in the constrained solve (the companion matrix drops rank);
    // real trackers always carry sub-pixel noise, which regularizes it.
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    auto noise = [&]() { rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
                         return ((double)(rng >> 40) / (double)(1ULL << 24) - 0.5) * 6e-4; };

    static core::FeatureDatabase db;
    db.count = 0;
    for (double tc = 0.0; tc <= 1.0005; tc += 0.05) {
        Eigen::Matrix3d Rk = R_GtoI(tc);
        Eigen::Vector3d pk = p_G(tc);
        for (int f = 0; f < NFEAT; ++f) {
            Eigen::Vector3d p_FinC = Rk * (feats_G[f] - pk);
            if (p_FinC.z() <= 0.1) continue;
            double un = p_FinC.x() / p_FinC.z() + noise();
            double vn = p_FinC.y() / p_FinC.z() + noise();
            core::update_feature(db, f, tc, 0, un, vn, un, vn);
        }
    }

    // ---- Configure + run the dynamic initializer ----
    initialize::InitializerOptions cfg;
    cfg.init_window_time = 1.0;
    cfg.init_max_features = 8;
    cfg.gravity_mag = g_mag;
    cfg.init_calib_camImu_dt = 0.0;
    cfg.init_dyn_use = true;
    cfg.init_dyn_num_pose = 5;
    cfg.init_dyn_min_deg = 1.0;

    double extr[2][7] = {{0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0}};

    type::Variable t_imu;
    Eigen::Matrix<double, 15, 15> cov;
    double ts = 0.0;
    bool ok = initialize::dynamic_initialize(cfg, db, imu, n_imu, extr, 1, t_imu, cov, ts);
    if (!ok) { std::cout << "FAIL: dynamic_initialize did not converge\n"; return 1; }

    // Newest-pose time it initialized at (= 1.0 here)
    if (std::abs(ts - 1.0) > 1e-6) { std::cout << "FAIL: init timestamp " << ts << " != 1.0\n"; return 1; }

    // Extract the seeded state: q_G'toIk, v_IkinG'  (G' = its gravity-aligned frame)
    const double* val = t_imu.value;
    Eigen::Vector4d q_est(val[0], val[1], val[2], val[3]);
    Eigen::Vector3d v_est(val[7], val[8], val[9]);
    Eigen::Matrix3d R_est = type::quat_2_Rot(q_est); // G' -> Ik

    // Gauge-free check 1: speed at newest pose
    double speed_true = v_G(1.0).norm();
    double speed_err = std::abs(v_est.norm() - speed_true);
    std::cout << "  speed: est=" << v_est.norm() << " true=" << speed_true
              << " err=" << speed_err << "\n";
    if (speed_err >= 0.05) { std::cout << "FAIL: recovered speed off by " << speed_err << "\n"; return 1; }

    // Gauge-free check 2: gravity direction in the body frame
    Eigen::Vector3d g_body_est = R_est * grav_G;                 // gravity dir in Ik (from G')
    Eigen::Vector3d g_body_true = R_GtoI(1.0) * grav_G;          // true gravity dir in Ik
    double cos_ang = g_body_est.normalized().dot(g_body_true.normalized());
    double ang_deg = std::acos(std::min(1.0, std::max(-1.0, cos_ang))) * 180.0 / M_PI;
    std::cout << "  gravity-direction error: " << ang_deg << " deg\n";
    if (ang_deg >= 1.0) { std::cout << "FAIL: gravity direction off by " << ang_deg << " deg\n"; return 1; }

    std::cout << "All dynamic-initializer checks passed.\n";
    return 0;
}
