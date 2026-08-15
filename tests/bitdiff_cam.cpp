// Parity stage 1.2 -- camera model, DOD vs official Open_VINS.
//
// Everything the estimator ever sees passes through these three functions, so
// a mismatch here contaminates triangulation, residuals and Jacobians alike.
//
// Note that official's undistort_d/distort_d round through float32 (CamBase.h:
// they cast to Vector2f, call the _f variant, cast back). That is a real part
// of the reference behaviour, not an accident, so DOD has to reproduce it.
#include "bitdiff.hpp"

#include "../core/cam.hpp"
#include <cam/CamRadtan.h>
#include <cam/CamEqui.h>

namespace {

struct CalibCase {
    const char* name;
    core::CameraModelType type;
    int w, h;
    double v[8];
};

// Real calibrations, because distortion magnitude is exactly what has bitten
// this port before: EuRoC's coefficients are ~40x KAIST's.
const CalibCase kCases[] = {
    {"kaist_rectified", core::CameraModelType::RADTAN, 640, 480,
     {385.06, 385.06, 315.06, 233.46, 0.0, 0.0, 0.0, 0.0}},
    {"euroc_cam0", core::CameraModelType::RADTAN, 752, 480,
     {458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05}},
    {"euroc_cam1", core::CameraModelType::RADTAN, 752, 480,
     {457.587, 456.134, 379.999, 255.238, -0.28368365, 0.07451284, -0.00010473, -3.55590700e-05}},
    {"synthetic_radtan", core::CameraModelType::RADTAN, 640, 480,
     {460.0, 460.0, 320.0, 240.0, 0.01, -0.02, 0.001, -0.002}},
    {"synthetic_equi", core::CameraModelType::EQUIDISTANT, 640, 480,
     {460.0, 460.0, 320.0, 240.0, -0.01, 0.005, -0.001, 0.0002}},
};

}  // namespace

int main() {
    bitdiff::Report rep;

    for (const CalibCase& cc : kCases) {
        core::CameraModel dod;
        core::init_camera(dod, cc.type, cc.w, cc.h, cc.v);

        Eigen::VectorXd calib(8);
        for (int i = 0; i < 8; ++i) calib(i) = cc.v[i];

        std::shared_ptr<ov_core::CamBase> ov;
        if (cc.type == core::CameraModelType::RADTAN)
            ov = std::make_shared<ov_core::CamRadtan>(cc.w, cc.h);
        else
            ov = std::make_shared<ov_core::CamEqui>(cc.w, cc.h);
        ov->set_value(calib);

        bitdiff::Rng rng(0xC0FFEE);
        char name[96];

        // Sweep the image plane, including well outside it -- tracked points do
        // land near and past the borders.
        for (int i = 0; i < 400; ++i) {
            Eigen::Vector2d uv(rng.uniform(-40.0, cc.w + 40.0), rng.uniform(-40.0, cc.h + 40.0));
            std::snprintf(name, sizeof(name), "%s undistort", cc.name);
            rep.expect(name, core::undistort_official(dod, uv), ov->undistort_d(uv));

            Eigen::Vector2d nrm(rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2));
            std::snprintf(name, sizeof(name), "%s distort", cc.name);
            rep.expect(name, core::distort(dod, nrm), ov->distort_d(nrm));

            Eigen::Matrix2d dHn;
            Eigen::Matrix<double, 2, 8> dHz;
            core::compute_distort_jacobian(dod, nrm, dHn, dHz);
            Eigen::MatrixXd ov_Hn, ov_Hz;
            ov->compute_distort_jacobian(nrm, ov_Hn, ov_Hz);
            std::snprintf(name, sizeof(name), "%s dz_dzn", cc.name);
            rep.expect(name, dHn, ov_Hn);
            std::snprintf(name, sizeof(name), "%s dz_dzeta", cc.name);
            rep.expect(name, dHz, ov_Hz);
        }
    }

    return rep.finish("stage 1.2 camera model");
}
