// Properties of core::parallax_noise_scale(), the parallax-limited measurement
// noise model. Trace-free: every check below is a property that must survive a
// reimplementation of the formula.
//
//   P1  lambda <= 0 returns exactly 1.0 (bit-identical no-op; this is what
//       keeps the ten recorded ATEs unchanged when the model is off).
//   P2  monotone non-decreasing in depth at fixed baseline.
//   P3  monotone non-increasing in baseline at fixed depth.
//   P4  clamped to max_scale, always finite, always >= 1.0.
//   P5  degenerate geometry (no depth, no baseline) saturates rather than
//       dividing by zero, and is counted.
#include <cmath>
#include <iostream>

#include "../core/feature.hpp"

namespace {

// One feature seen from two cam-0 clones separated by `baseline` metres, whose
// triangulation sits at depth `Z` in its anchor camera.
core::Feature make_feat(double Z) {
    core::Feature f;
    f.num_measurements = 2;
    f.measurements[0] = {0, 100.0, {}, {}};
    f.measurements[1] = {0, 101.0, {}, {}};
    f.p_FinA = Eigen::Vector3d(0.0, 0.0, Z);
    return f;
}

core::ClonesCamera make_clones(double baseline) {
    core::ClonesCamera c;
    c.num_clones = 2;
    c.timestamps[0] = 100.0;
    c.timestamps[1] = 101.0;
    for (int cam = 0; cam < 2; ++cam) {
        c.poses[cam][0] = {Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()};
        c.poses[cam][1] = {Eigen::Matrix3d::Identity(), Eigen::Vector3d(baseline, 0.0, 0.0)};
    }
    return c;
}

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::cerr << "FAIL: " << what << "\n"; ++failures; }
}

} // namespace

int main() {
    constexpr double kMax = 100.0;
    const core::ClonesCamera clones = make_clones(1.0);

    // P1: exact no-op. Not "close to 1.0" -- exactly 1.0, or the division at
    // the call site perturbs the trajectory.
    for (double Z : {1.0, 40.0, 100.0, 1000.0}) {
        const core::Feature f = make_feat(Z);
        check(core::parallax_noise_scale(f, clones, 0.0, kMax) == 1.0, "P1 lambda=0 not exactly 1.0");
        check(core::parallax_noise_scale(f, clones, -1.0, kMax) == 1.0, "P1 lambda<0 not exactly 1.0");
    }

    // P2: non-decreasing in depth. Uses a large cap so the clamp does not mask
    // the monotonicity.
    double prev = 0.0;
    for (double Z = 1.0; Z <= 200.0; Z += 1.0) {
        const double s = core::parallax_noise_scale(make_feat(Z), clones, 0.01, 1e9);
        check(s >= prev, "P2 not monotone in depth");
        check(s >= 1.0 && std::isfinite(s), "P4 out of range while sweeping depth");
        prev = s;
    }

    // P3: non-increasing in baseline at fixed depth.
    const core::Feature f100 = make_feat(100.0);
    prev = 1e300;
    for (double b = 0.05; b <= 5.0; b += 0.05) {
        const double s = core::parallax_noise_scale(f100, make_clones(b), 0.01, 1e9);
        check(s <= prev, "P3 not monotone in baseline");
        prev = s;
    }

    // P4: the clamp actually binds for a geometry that would otherwise blow up.
    check(core::parallax_noise_scale(make_feat(1.0e6), clones, 1.0, kMax) == kMax, "P4 clamp not applied");

    // P5: degenerate geometry saturates instead of producing inf/NaN, and each
    // early-out is counted separately.
    const long depth0 = core::pnw_no_depth, base0 = core::pnw_no_baseline;
    check(core::parallax_noise_scale(make_feat(0.0), clones, 1.0, kMax) == kMax, "P5 zero depth not saturated");
    check(core::pnw_no_depth == depth0 + 1, "P5 zero depth not counted");
    check(core::parallax_noise_scale(f100, make_clones(0.0), 1.0, kMax) == kMax, "P5 zero baseline not saturated");
    check(core::pnw_no_baseline == base0 + 1, "P5 zero baseline not counted");

    if (failures == 0) std::cout << "verify_parallax_noise: OK\n";
    return failures == 0 ? 0 : 1;
}
