// Parity stages 2.3-2.5 -- IMU propagation, DOD vs official Open_VINS.
//
// Covers reading selection/interpolation, the three mean predictors, and the
// F/G transition and noise Jacobians. These run at IMU rate, so any divergence
// here compounds into every downstream update.
#include "bitdiff.hpp"

#include "../msckf/propagator.hpp"
#include "../msckf/state.hpp"

#include <state/Propagator.h>
#include <state/State.h>
#include <state/StateOptions.h>

// predict_mean_* and compute_F_and_G_* are protected in official. Subclassing
// is the least invasive way to reach them -- the oracle's sources stay
// untouched, exactly as upstream ships them.
class OracleProp : public ov_msckf::Propagator {
public:
    OracleProp(ov_msckf::NoiseManager n, double g) : ov_msckf::Propagator(n, g) {}
    using ov_msckf::Propagator::compute_F_and_G_analytic;
    using ov_msckf::Propagator::compute_F_and_G_discrete;
    using ov_msckf::Propagator::compute_Xi_sum;
    using ov_msckf::Propagator::predict_mean_analytic;
    using ov_msckf::Propagator::predict_mean_discrete;
    using ov_msckf::Propagator::predict_mean_rk4;
};

namespace {

// The two states must start from identical numbers or nothing below means
// anything. Returns the shared IMU value vector actually used.
Eigen::Matrix<double, 16, 1> make_states(msckf::State& dod, std::shared_ptr<ov_msckf::State>& ov,
                                         bool do_fej, msckf::IntegrationMethod method) {
    msckf::StateOptions dod_opt;
    dod_opt.do_fej = do_fej;
    dod_opt.integration_method = method;
    dod_opt.num_cameras = 1;
    msckf::init_state(dod, dod_opt);

    ov_msckf::StateOptions ov_opt;
    ov_opt.do_fej = do_fej;
    ov_opt.integration_method = (method == msckf::IntegrationMethod::RK4)
                                    ? ov_msckf::StateOptions::IntegrationMethod::RK4
                                    : ((method == msckf::IntegrationMethod::DISCRETE)
                                           ? ov_msckf::StateOptions::IntegrationMethod::DISCRETE
                                           : ov_msckf::StateOptions::IntegrationMethod::ANALYTICAL);
    ov_opt.num_cameras = 1;
    ov_opt.do_calib_camera_pose = false;
    ov_opt.do_calib_camera_intrinsics = false;
    ov_opt.do_calib_camera_timeoffset = false;
    ov_opt.do_calib_imu_intrinsics = false;
    ov_opt.do_calib_imu_g_sensitivity = false;
    ov = std::make_shared<ov_msckf::State>(ov_opt);

    // A non-trivial attitude, velocity and bias -- an identity state hides
    // whole classes of ordering bugs.
    Eigen::Matrix<double, 16, 1> v;
    v << 0.02, -0.03, 0.05, 0.99805, 1.5, -2.5, 0.75, 0.4, -0.15, 0.05, 0.001, -0.002, 0.0015, 0.02, -0.03, 0.01;
    v.head<4>() = v.head<4>() / v.head<4>().norm();

    for (int i = 0; i < 16; ++i) dod.imu.value[i] = v(i);
    for (int i = 0; i < 16; ++i) dod.imu.fej[i] = v(i);
    ov->_imu->set_value(v);
    ov->_imu->set_fej(v);
    dod.timestamp = 1.0;
    ov->_timestamp = 1.0;
    return v;
}

}  // namespace

int main() {
    bitdiff::Report rep;
    bitdiff::Rng rng(0xBEEF01);

    const double gravity_mag = 9.81;
    msckf::PropagatorNoises dod_noises;
    ov_msckf::NoiseManager ov_noises;
    ov_noises.sigma_w = dod_noises.sigma_w;
    ov_noises.sigma_a = dod_noises.sigma_a;
    ov_noises.sigma_wb = dod_noises.sigma_wb;
    ov_noises.sigma_ab = dod_noises.sigma_ab;
    ov_noises.sigma_w_2 = dod_noises.sigma_w_2;
    ov_noises.sigma_a_2 = dod_noises.sigma_a_2;
    ov_noises.sigma_wb_2 = dod_noises.sigma_wb_2;
    ov_noises.sigma_ab_2 = dod_noises.sigma_ab_2;

    msckf::PropagatorData dod_prop;
    msckf::init_propagator(dod_prop, dod_noises, gravity_mag);
    OracleProp ov_prop(ov_noises, gravity_mag);

    struct Mode {
        const char* name;
        msckf::IntegrationMethod m;
    };
    const Mode modes[] = {{"discrete", msckf::IntegrationMethod::DISCRETE},
                          {"rk4", msckf::IntegrationMethod::RK4},
                          {"analytic", msckf::IntegrationMethod::ANALYTICAL}};

    for (const Mode& mode : modes) {
        for (int fej = 0; fej < 2; ++fej) {
            msckf::State dod;
            std::shared_ptr<ov_msckf::State> ov;
            make_states(dod, ov, fej == 1, mode.m);

            char name[96];
            for (int trial = 0; trial < 60; ++trial) {
                double dt = (trial % 3 == 0) ? 0.005 : rng.uniform(1e-4, 0.02);
                Eigen::Vector3d w_hat = rng.vec(3, -1.5, 1.5);
                Eigen::Vector3d a_hat = rng.vec(3, -3.0, 3.0);
                Eigen::Vector3d w_hat2 = w_hat + rng.vec(3, -0.05, 0.05);
                Eigen::Vector3d a_hat2 = a_hat + rng.vec(3, -0.05, 0.05);
                Eigen::Vector3d w_unc = w_hat + rng.vec(3, -0.001, 0.001);
                Eigen::Vector3d a_unc = a_hat + rng.vec(3, -0.001, 0.001);

                // --- 2.4 mean prediction -------------------------------------
                Eigen::Vector4d dq, oq;
                Eigen::Vector3d dv, dp, ov_v, ov_p;

                msckf::predict_mean_discrete(dod_prop, dod, dt, w_hat, a_hat, dq, dv, dp);
                ov_prop.predict_mean_discrete(ov, dt, w_hat, a_hat, oq, ov_v, ov_p);
                std::snprintf(name, sizeof(name), "%s/fej%d predict_mean_discrete q", mode.name, fej);
                rep.expect(name, dq, oq);
                std::snprintf(name, sizeof(name), "%s/fej%d predict_mean_discrete v", mode.name, fej);
                rep.expect(name, dv, ov_v);
                std::snprintf(name, sizeof(name), "%s/fej%d predict_mean_discrete p", mode.name, fej);
                rep.expect(name, dp, ov_p);

                msckf::predict_mean_rk4(dod_prop, dod, dt, w_hat, a_hat, w_hat2, a_hat2, dq, dv, dp);
                ov_prop.predict_mean_rk4(ov, dt, w_hat, a_hat, w_hat2, a_hat2, oq, ov_v, ov_p);
                std::snprintf(name, sizeof(name), "%s/fej%d predict_mean_rk4 q", mode.name, fej);
                rep.expect(name, dq, oq);
                std::snprintf(name, sizeof(name), "%s/fej%d predict_mean_rk4 v", mode.name, fej);
                rep.expect(name, dv, ov_v);
                std::snprintf(name, sizeof(name), "%s/fej%d predict_mean_rk4 p", mode.name, fej);
                rep.expect(name, dp, ov_p);

                // --- 2.4b Xi_sum + analytic mean -----------------------------
                Eigen::Matrix<double, 3, 18> dXi, oXi;
                msckf::compute_Xi_sum(dod, dt, w_hat, a_hat, dXi);
                ov_prop.compute_Xi_sum(ov, dt, w_hat, a_hat, oXi);
                std::snprintf(name, sizeof(name), "%s/fej%d Xi_sum", mode.name, fej);
                rep.expect(name, dXi, oXi);

                msckf::predict_mean_analytic(dod_prop, dod, dt, w_hat, a_hat, dq, dv, dp, dXi);
                ov_prop.predict_mean_analytic(ov, dt, w_hat, a_hat, oq, ov_v, ov_p, oXi);
                std::snprintf(name, sizeof(name), "%s/fej%d predict_mean_analytic q", mode.name, fej);
                rep.expect(name, dq, oq);
                std::snprintf(name, sizeof(name), "%s/fej%d predict_mean_analytic v", mode.name, fej);
                rep.expect(name, dv, ov_v);
                std::snprintf(name, sizeof(name), "%s/fej%d predict_mean_analytic p", mode.name, fej);
                rep.expect(name, dp, ov_p);

                // --- 2.5 transition and noise Jacobians ----------------------
                const int n = (int)ov->imu_intrinsic_size() + 15;
                msckf::ImuTransitionMatrix dF;
                msckf::ImuNoiseJacobian dG;
                // compute_F_and_G_discrete writes into F/G without resizing
                // them, so the caller owns the sizing (see Propagator.cpp's
                // predict_and_compute). Passing default-constructed MatrixXd
                // segfaults.
                Eigen::MatrixXd oF = Eigen::MatrixXd::Zero(n, n);
                Eigen::MatrixXd oG = Eigen::MatrixXd::Zero(n, 12);

                // DOD's compute_F_and_G_* fill only the non-zero blocks of a
                // caller-owned buffer; production zeroes it first
                // (propagator.cpp:302, predict_and_compute). The test has to
                // do the same or it compares stale values.
                dF.setZero();
                dG.setZero();
                msckf::compute_F_and_G_discrete(dod_prop, dod, dt, w_hat, a_hat, w_unc, a_unc, dq, dv, dp, dF, dG);
                ov_prop.compute_F_and_G_discrete(ov, dt, w_hat, a_hat, w_unc, a_unc, oq, ov_v, ov_p, oF, oG);
                std::snprintf(name, sizeof(name), "%s/fej%d F_discrete", mode.name, fej);
                rep.expect(name, dF.topLeftCorner(n, n).eval(), oF);
                std::snprintf(name, sizeof(name), "%s/fej%d G_discrete", mode.name, fej);
                rep.expect(name, dG.topLeftCorner(n, 12).eval(), oG);

                oF = Eigen::MatrixXd::Zero(n, n);
                oG = Eigen::MatrixXd::Zero(n, 12);
                dF.setZero();
                dG.setZero();
                msckf::compute_F_and_G_analytic(dod_prop, dod, dt, w_hat, a_hat, w_unc, a_unc, dq, dv, dp, dXi, dF, dG);
                ov_prop.compute_F_and_G_analytic(ov, dt, w_hat, a_hat, w_unc, a_unc, oq, ov_v, ov_p, oXi, oF, oG);
                std::snprintf(name, sizeof(name), "%s/fej%d F_analytic", mode.name, fej);
                rep.expect(name, dF.topLeftCorner(n, n).eval(), oF);
                std::snprintf(name, sizeof(name), "%s/fej%d G_analytic", mode.name, fej);
                rep.expect(name, dG.topLeftCorner(n, 12).eval(), oG);
            }
        }
    }

    // --- 2.3 reading selection and interpolation -----------------------------
    {
        std::vector<ov_core::ImuData> ov_imu;
        core::ImuData dod_imu[256];
        int count = 0;
        double t = 0.9;
        for (int i = 0; i < 200; ++i) {
            core::ImuData d;
            d.timestamp = t;
            d.wm = rng.vec(3, -1.0, 1.0);
            d.am = rng.vec(3, -2.0, 12.0);
            dod_imu[count++] = d;
            ov_core::ImuData o;
            o.timestamp = d.timestamp;
            o.wm = d.wm;
            o.am = d.am;
            ov_imu.push_back(o);
            t += 0.005;
        }

        // Windows that start and end mid-interval, which is the case that
        // actually exercises interpolation.
        const double t0s[] = {1.0, 1.0023, 1.10001, 0.95};
        const double t1s[] = {1.2, 1.3077, 1.45999, 1.05};
        for (int k = 0; k < 4; ++k) {
            core::ImuData out[256];
            int n = msckf::select_imu_readings(dod_imu, count, t0s[k], t1s[k], out, 256);
            std::vector<ov_core::ImuData> ovout =
                ov_msckf::Propagator::select_imu_readings(ov_imu, t0s[k], t1s[k], false);

            char name[96];
            std::snprintf(name, sizeof(name), "select_imu_readings[%d] count", k);
            rep.expect_scalar(name, (double)n, (double)ovout.size());
            if (n != (int)ovout.size()) continue;
            for (int i = 0; i < n; ++i) {
                std::snprintf(name, sizeof(name), "select_imu_readings[%d][%d] ts", k, i);
                rep.expect_scalar(name, out[i].timestamp, ovout[i].timestamp);
                std::snprintf(name, sizeof(name), "select_imu_readings[%d][%d] wm", k, i);
                rep.expect(name, out[i].wm, ovout[i].wm);
                std::snprintf(name, sizeof(name), "select_imu_readings[%d][%d] am", k, i);
                rep.expect(name, out[i].am, ovout[i].am);
            }
        }
    }

    return rep.finish("stages 2.3-2.5 propagation");
}
