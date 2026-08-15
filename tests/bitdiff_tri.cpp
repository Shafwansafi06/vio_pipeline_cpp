// Parity stage 2.8 -- feature triangulation, DOD vs official Open_VINS.
//
// This is the stage behind the documented accept-rate gap (official accepts
// ~70% of MSCKF features on circle.bag, DOD ~9%), so both the accept/reject
// DECISION and the triangulated point are compared.
#include "bitdiff.hpp"

#include "../core/feature.hpp"
#include "../type/quat_ops.hpp"

#include <feat/Feature.h>
#include <feat/FeatureInitializer.h>
#include <feat/FeatureInitializerOptions.h>

// compute_error is protected in official; subclass to reach it without
// touching the oracle's sources.
class OracleInit : public ov_core::FeatureInitializer {
public:
    explicit OracleInit(ov_core::FeatureInitializerOptions o) : ov_core::FeatureInitializer(o) {}
    using ov_core::FeatureInitializer::compute_error;
};

namespace {

// Builds the same geometry into both representations.
struct Scene {
    core::Feature dod;
    std::shared_ptr<ov_core::Feature> ov = std::make_shared<ov_core::Feature>();
    core::ClonesCamera dod_clones;
    std::unordered_map<size_t, std::unordered_map<double, ov_core::FeatureInitializer::ClonePose>> ov_clones;
};

// n_cam0/n_cam1 let us hit the anchor tie-break (equal counts) explicitly.
Scene make_scene(bitdiff::Rng& rng, int n_cam0, int n_cam1, double depth) {
    Scene s;
    s.dod.featid = 7;
    s.ov->featid = 7;
    s.dod.num_measurements = 0;
    s.dod_clones.num_clones = 0;

    const Eigen::Vector3d p_FinG(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), depth);

    const int counts[2] = {n_cam0, n_cam1};
    int clone_idx = 0;
    for (int cam = 0; cam < 2; ++cam) {
        for (int k = 0; k < counts[cam]; ++k) {
            const double ts = 100.0 + 0.05 * k;

            // A gently moving camera -- enough parallax to triangulate.
            Eigen::Vector3d w(0.02 * k, -0.01 * k, 0.015 * k);
            Eigen::Matrix3d R_GtoCi = type::exp_so3(w);
            Eigen::Vector3d p_CiinG(0.1 * k + 0.03 * cam, -0.05 * k, 0.02 * k);

            s.dod_clones.poses[cam][k] = {R_GtoCi, p_CiinG};
            if (cam == 0) s.dod_clones.timestamps[k] = ts;
            s.ov_clones[cam].insert({ts, ov_core::FeatureInitializer::ClonePose(R_GtoCi, p_CiinG)});

            const Eigen::Vector3d p_FinCi = R_GtoCi * (p_FinG - p_CiinG);
            // Store float-representable values: official's uvs_norm are VectorXf,
            // so anything else would differ for reasons that are not the
            // algorithm's fault.
            const float un = (float)(p_FinCi(0) / p_FinCi(2));
            const float vn = (float)(p_FinCi(1) / p_FinCi(2));

            core::FeatureMeasurement m;
            m.cam_id = cam;
            m.timestamp = ts;
            m.uv = Eigen::Vector2d(un * 400.0 + 320.0, vn * 400.0 + 240.0);
            m.uv_norm = Eigen::Vector2d((double)un, (double)vn);
            s.dod.measurements[s.dod.num_measurements++] = m;

            Eigen::VectorXf uvn(2);
            uvn << un, vn;
            Eigen::VectorXf uv(2);
            uv << (float)m.uv(0), (float)m.uv(1);
            s.ov->uvs_norm[cam].push_back(uvn);
            s.ov->uvs[cam].push_back(uv);
            s.ov->timestamps[cam].push_back(ts);
            ++clone_idx;
        }
    }
    s.dod_clones.num_clones = std::max(n_cam0, n_cam1);
    return s;
}

}  // namespace

int main() {
    bitdiff::Report rep;
    bitdiff::Rng rng(0x71A2C0);

    core::FeatureInitializerOptions dod_opt;
    ov_core::FeatureInitializerOptions ov_opt;
    ov_opt.triangulate_1d = dod_opt.triangulate_1d;
    ov_opt.refine_features = dod_opt.refine_features;
    ov_opt.max_cond_number = dod_opt.max_cond_number;
    ov_opt.min_dist = dod_opt.min_dist;
    ov_opt.max_dist = dod_opt.max_dist;
    ov_opt.init_lamda = dod_opt.init_lamda;
    ov_opt.max_lamda = dod_opt.max_lamda;
    ov_opt.max_runs = dod_opt.max_runs;
    ov_opt.min_dx = dod_opt.min_dx;
    ov_opt.min_dcost = dod_opt.min_dcost;
    ov_opt.lam_mult = dod_opt.lam_mult;
    OracleInit ov_init(ov_opt);

    struct Shape {
        int n0, n1;
    };
    // Includes n0 == n1, which is where the anchor tie-break shows up.
    const Shape shapes[] = {{6, 0}, {6, 6}, {4, 6}, {6, 4}, {3, 3}, {8, 2}};

    char name[96];
    for (const Shape& sh : shapes) {
        for (int trial = 0; trial < 40; ++trial) {
            const double depth = rng.uniform(1.0, 12.0);
            Scene s = make_scene(rng, sh.n0, sh.n1, depth);

            const bool dod_ok = core::single_triangulation(s.dod, s.dod_clones, dod_opt);
            const bool ov_ok = ov_init.single_triangulation(s.ov, s.ov_clones);

            std::snprintf(name, sizeof(name), "tri[%d,%d] accepted", sh.n0, sh.n1);
            rep.expect_scalar(name, dod_ok ? 1.0 : 0.0, ov_ok ? 1.0 : 0.0);
            if (!dod_ok || !ov_ok) continue;

            std::snprintf(name, sizeof(name), "tri[%d,%d] anchor_cam", sh.n0, sh.n1);
            rep.expect_scalar(name, (double)s.dod.anchor_cam_id, (double)s.ov->anchor_cam_id);
            std::snprintf(name, sizeof(name), "tri[%d,%d] anchor_ts", sh.n0, sh.n1);
            rep.expect_scalar(name, s.dod.anchor_clone_timestamp, s.ov->anchor_clone_timestamp);
            std::snprintf(name, sizeof(name), "tri[%d,%d] p_FinA", sh.n0, sh.n1);
            rep.expect(name, s.dod.p_FinA, s.ov->p_FinA);
            std::snprintf(name, sizeof(name), "tri[%d,%d] p_FinG", sh.n0, sh.n1);
            rep.expect(name, s.dod.p_FinG, s.ov->p_FinG);

            // --- compute_error at the triangulated point ---------------------
            const double alpha = s.dod.p_FinA(0) / s.dod.p_FinA(2);
            const double beta = s.dod.p_FinA(1) / s.dod.p_FinA(2);
            const double rho = 1.0 / s.dod.p_FinA(2);
            std::snprintf(name, sizeof(name), "tri[%d,%d] compute_error", sh.n0, sh.n1);
            rep.expect_scalar(name, core::compute_error(s.dod_clones, s.dod, alpha, beta, rho),
                              ov_init.compute_error(s.ov_clones, s.ov, alpha, beta, rho));

            // --- Gauss-Newton refinement -------------------------------------
            const bool dod_gn = core::single_gaussnewton(s.dod, s.dod_clones, dod_opt);
            const bool ov_gn = ov_init.single_gaussnewton(s.ov, s.ov_clones);
            std::snprintf(name, sizeof(name), "gn[%d,%d] accepted", sh.n0, sh.n1);
            rep.expect_scalar(name, dod_gn ? 1.0 : 0.0, ov_gn ? 1.0 : 0.0);
            if (!dod_gn || !ov_gn) continue;
            std::snprintf(name, sizeof(name), "gn[%d,%d] p_FinA", sh.n0, sh.n1);
            rep.expect(name, s.dod.p_FinA, s.ov->p_FinA);
            std::snprintf(name, sizeof(name), "gn[%d,%d] p_FinG", sh.n0, sh.n1);
            rep.expect(name, s.dod.p_FinG, s.ov->p_FinG);
        }
    }

    return rep.finish("stage 2.8 triangulation");
}
