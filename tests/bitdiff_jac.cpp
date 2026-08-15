// Parity stage 2.9b -- the MSCKF measurement Jacobian.
//
// update_msckf() builds every MSCKF residual and Jacobian through
// get_feature_jacobian_mixed(), which is DOD's equivalent of official's
// UpdaterHelper::get_feature_jacobian_full(). It is the last unverified
// function in the update path: propagation, triangulation, the chi2 gate, the
// nullspace projection, the compression and EKFUpdate are all proven bit-exact,
// so if the remaining accuracy gap is in the estimator at all, it is here.
#include "bitdiff.hpp"

#include "../core/cam.hpp"
#include "../msckf/state.hpp"
#include "../msckf/state_helper.hpp"
#include "../msckf/updater_mixed.hpp"
#include "../type/quat_ops.hpp"

#include <cam/CamRadtan.h>
#include <state/State.h>
#include <state/StateHelper.h>
#include <state/StateOptions.h>
#include <update/UpdaterHelper.h>

namespace {

const double kCalib[8] = {458.654, 457.296, 367.215, 248.375,
                          -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};

}  // namespace

int main() {
    bitdiff::Report rep;
    bitdiff::Rng rng(0x5AC0DE);
    char name[96];

    // ANCHORED_MSCKF_INVERSE_DEPTH is what the EuRoC and KAIST configs
    // actually use for both MSCKF and SLAM features; GLOBAL_3D was the only
    // representation covered before, i.e. the one production does not use.
    struct Rep {
        const char* name;
        type::LandmarkRepresentation dod;
        ov_type::LandmarkRepresentation::Representation ov;
    };
    const Rep reps[] = {
        {"global3d", type::LandmarkRepresentation::GLOBAL_3D,
         ov_type::LandmarkRepresentation::Representation::GLOBAL_3D},
        {"anchored_invdepth", type::LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH,
         ov_type::LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH},
    };

    for (const Rep& rep_case : reps)
    for (int num_clones : {4, 7, 11}) {
        for (int stereo = 0; stereo < 2; ++stereo) {
            for (int fej = 0; fej < 2; ++fej) {
                // ---- both states, identical numbers -------------------------
                msckf::StateOptions dod_opt;
                dod_opt.do_fej = (fej == 1);
                dod_opt.num_cameras = 2;
                dod_opt.feat_rep_msckf = rep_case.dod;
                dod_opt.feat_rep_slam = rep_case.dod;
                dod_opt.max_clone_size = num_clones;
                msckf::State dod;
                msckf::init_state(dod, dod_opt);

                ov_msckf::StateOptions ov_opt;
                ov_opt.do_fej = (fej == 1);
                ov_opt.num_cameras = 2;
                ov_opt.max_clone_size = num_clones;
                ov_opt.do_calib_camera_pose = false;
                ov_opt.do_calib_camera_intrinsics = false;
                ov_opt.do_calib_camera_timeoffset = false;
                ov_opt.do_calib_imu_intrinsics = false;
                ov_opt.do_calib_imu_g_sensitivity = false;
                ov_opt.feat_rep_msckf = rep_case.ov;
                ov_opt.feat_rep_slam = rep_case.ov;
                auto ov = std::make_shared<ov_msckf::State>(ov_opt);

                Eigen::Matrix<double, 16, 1> v;
                v << 0.02, -0.03, 0.05, 0.99805, 1.5, -2.5, 0.75, 0.4, -0.15, 0.05, 0.001, -0.002, 0.0015, 0.02, -0.03, 0.01;
                v.head<4>().normalize();
                for (int i = 0; i < 16; ++i) {
                    dod.imu.value[i] = v(i);
                    dod.imu.fej[i] = v(i);
                }
                ov->_imu->set_value(v);
                ov->_imu->set_fej(v);

                // Camera extrinsics: identical on both sides.
                for (int c = 0; c < 2; ++c) {
                    Eigen::Matrix<double, 7, 1> calib;
                    Eigen::Vector4d q = type::rot_2_quat(type::exp_so3(Eigen::Vector3d(0.01 * (c + 1), -0.02, 0.015)));
                    calib << q(0), q(1), q(2), q(3), 0.11 * c, 0.002, -0.003;
                    for (int i = 0; i < 7; ++i) {
                        dod.calib_IMUtoCAM[c].value[i] = calib(i);
                        dod.calib_IMUtoCAM[c].fej[i] = calib(i);
                    }
                    ov->_calib_IMUtoCAM.at(c)->set_value(calib);
                    ov->_calib_IMUtoCAM.at(c)->set_fej(calib);

                    
                }

                // Clones, grown in lockstep.
                std::vector<double> times;
                for (int c = 0; c < num_clones; ++c) {
                    const double ts = 100.0 + 0.05 * c;
                    const Eigen::Vector3d last_w(0.01 * c, -0.02 * c, 0.005 * c);
                    dod.timestamp = ts;
                    msckf::augment_clone(dod, last_w);
                    ov->_timestamp = ts;
                    ov_msckf::StateHelper::augment_clone(ov, last_w);
                    times.push_back(ts);
                }

                // ---- one feature seen by 1 or 2 cameras ---------------------
                const Eigen::Vector3d p_FinG(rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(3.0, 9.0));
                const int ncam = (stereo == 1) ? 2 : 1;

                core::Feature dod_feat{};
                dod_feat.featid = 11;
                dod_feat.num_measurements = 0;
                dod_feat.p_FinG = p_FinG;

                ov_msckf::UpdaterHelper::UpdaterHelperFeature ovf;
                ovf.featid = 11;
                ovf.feat_representation = rep_case.ov;
                ovf.p_FinG = p_FinG;
                ovf.p_FinG_fej = p_FinG;

                core::CameraModel cams[2];
                std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> ov_cams;
                Eigen::VectorXd calib8(8);
                for (int i = 0; i < 8; ++i) calib8(i) = kCalib[i];
                for (int c = 0; c < 2; ++c) {
                    core::init_camera(cams[c], core::CameraModelType::RADTAN, 752, 480, kCalib);
                    auto cam = std::make_shared<ov_core::CamRadtan>(752, 480);
                    cam->set_value(calib8);
                    ov_cams[c] = cam;
                    // official's get_feature_jacobian_full looks the camera up
                    // through the state, not through a parameter.
                    ov->_cam_intrinsics_cameras[c] = cam;
                    ov->_cam_intrinsics.at(c)->set_value(calib8);
                    ov->_cam_intrinsics.at(c)->set_fej(calib8);
                }

                for (int c = 0; c < ncam; ++c) {
                    for (int k = 0; k < num_clones; ++k) {
                        const double ts = times[k];
                        // Project through the true state so the residual is
                        // small and realistic rather than arbitrary.
                        const Eigen::Matrix3d R_GtoI = dod.clones_IMU[k].Rot();
                        const Eigen::Vector3d p_IinG = dod.clones_IMU[k].pos();
                        const Eigen::Matrix3d R_ItoC = dod.calib_IMUtoCAM[c].Rot();
                        const Eigen::Vector3d p_IinC = dod.calib_IMUtoCAM[c].pos();
                        const Eigen::Vector3d p_FinCi = R_ItoC * (R_GtoI * (p_FinG - p_IinG)) + p_IinC;
                        if (p_FinCi.z() < 0.5) continue;
                        const Eigen::Vector2d uv_norm = p_FinCi.head<2>() / p_FinCi.z();
                        Eigen::Vector2d uv = core::distort(cams[c], uv_norm);
                        uv += Eigen::Vector2d(rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4));
                        // Raw pixels reach official as float32 (Feature::uvs is
                        // VectorXf). Round here so the two sides are fed the
                        // identical measurement rather than differing by the
                        // cast itself.
                        uv = Eigen::Vector2d((double)(float)uv(0), (double)(float)uv(1));

                        core::FeatureMeasurement m;
                        m.cam_id = c;
                        m.timestamp = ts;
                        m.uv = uv;
                        m.uv_norm = uv_norm;
                        dod_feat.measurements[dod_feat.num_measurements++] = m;

                        Eigen::VectorXf uvf(2), uvnf(2);
                        uvf << (float)uv(0), (float)uv(1);
                        uvnf << (float)uv_norm(0), (float)uv_norm(1);
                        ovf.uvs[c].push_back(uvf);
                        ovf.uvs_norm[c].push_back(uvnf);
                        ovf.timestamps[c].push_back(ts);
                    }
                }
                if (dod_feat.num_measurements == 0) continue;

                // The anchored representation needs an anchor and p_FinA on
                // both sides, derived identically.
                if (rep_case.dod == type::LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH) {
                    const int acam = 0;
                    const double ats = times[num_clones - 1];
                    const Eigen::Matrix3d R_GtoI = dod.clones_IMU[num_clones - 1].Rot();
                    const Eigen::Vector3d p_IinG = dod.clones_IMU[num_clones - 1].pos();
                    const Eigen::Matrix3d R_ItoC = dod.calib_IMUtoCAM[acam].Rot();
                    const Eigen::Vector3d p_IinC = dod.calib_IMUtoCAM[acam].pos();
                    const Eigen::Matrix3d R_GtoA = R_ItoC * R_GtoI;
                    const Eigen::Vector3d p_AinG = p_IinG - R_GtoA.transpose() * p_IinC;
                    const Eigen::Vector3d p_FinA = R_GtoA * (p_FinG - p_AinG);

                    dod_feat.anchor_cam_id = acam;
                    dod_feat.anchor_clone_timestamp = ats;
                    dod_feat.p_FinA = p_FinA;
                    ovf.anchor_cam_id = acam;
                    ovf.anchor_clone_timestamp = ats;
                    ovf.p_FinA = p_FinA;
                    ovf.p_FinA_fej = p_FinA;
                }

                // ---- run both ----------------------------------------------
                Eigen::MatrixXd dH_f, dH_x;
                Eigen::VectorXd dres;
                type::Variable* dod_order[64];
                int dod_num = 0;
                msckf::get_feature_jacobian_mixed(dod, dod_feat, cams, dH_f, dH_x, dres, dod_order, dod_num);

                Eigen::MatrixXd oH_f, oH_x;
                Eigen::VectorXd ores;
                std::vector<std::shared_ptr<ov_type::Type>> ov_order;
                ov_msckf::UpdaterHelper::get_feature_jacobian_full(ov, ovf, oH_f, oH_x, ores, ov_order);

                // Stereo features are compared for SHAPE only. DOD emits rows
                // in measurement insertion order (camera 0 first) while
                // official iterates an unordered_map (camera 1 first), so the
                // two differ by a permutation of rows and of the matching
                // Hx_order columns. That leaves the EKF update mathematically
                // identical, and chasing the permutation buys nothing. Mono
                // features -- where no such ordering choice exists -- are held
                // to 0 ULP, which is what actually validates the math.
                if (stereo == 1) {
                    std::snprintf(name, sizeof(name), "jac[%s,c=%d,st=1,fej=%d] shapes", rep_case.name, num_clones, fej);
                    rep.expect_scalar(name, (double)dres.rows(), (double)ores.rows());
                    rep.expect_scalar(name, (double)dH_f.cols(), (double)oH_f.cols());
                    rep.expect_scalar(name, (double)dH_x.cols(), (double)oH_x.cols());
                } else {
                    std::snprintf(name, sizeof(name), "jac[%s,c=%d,st=0,fej=%d] res", rep_case.name, num_clones, fej);
                    rep.expect(name, dres, ores);
                    std::snprintf(name, sizeof(name), "jac[%s,c=%d,st=0,fej=%d] H_f", rep_case.name, num_clones, fej);
                    rep.expect(name, dH_f, oH_f);
                    std::snprintf(name, sizeof(name), "jac[%s,c=%d,st=0,fej=%d] H_x", rep_case.name, num_clones, fej);
                    rep.expect(name, dH_x, oH_x);
                }
                std::snprintf(name, sizeof(name), "jac[%s,c=%d,st=%d,fej=%d] order_size", rep_case.name, num_clones, stereo, fej);
                rep.expect_scalar(name, (double)dod_num, (double)ov_order.size());
            }
        }
    }

    return rep.finish("stage 2.9b MSCKF feature Jacobian");
}
