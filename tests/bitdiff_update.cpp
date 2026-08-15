// Parity stage 2.9 -- MSCKF nullspace projection and measurement compression.
//
// Both were HouseholderQR-with-explicit-Q in DOD and in-place Givens in
// official. Those give different orthonormal bases, not just different last
// bits, so this compares the actual projected/compressed systems.
#include "bitdiff.hpp"

#include "../msckf/updaters.hpp"

#include <update/UpdaterHelper.h>

int main() {
    bitdiff::Report rep;
    bitdiff::Rng rng(0x09DA7E);

    char name[96];

    // Shapes mirroring real MSCKF blocks: 2*M rows for M measurements, and a
    // state block of a few clones. The tall/thin and short/wide cases matter
    // because measurement_compress_inplace early-returns on the latter.
    struct Shape {
        int rows, cols;
    };
    const Shape shapes[] = {{20, 12}, {8, 6}, {44, 30}, {6, 12}, {12, 12}, {30, 7}};

    for (const Shape& sh : shapes) {
        for (int trial = 0; trial < 30; ++trial) {
            // --- nullspace projection (H_f always has 3 columns) -------------
            Eigen::MatrixXd H_f(sh.rows, 3), H_x(sh.rows, sh.cols);
            Eigen::VectorXd res(sh.rows);
            for (int r = 0; r < sh.rows; ++r) {
                for (int c = 0; c < 3; ++c) H_f(r, c) = rng.uniform(-5.0, 5.0);
                for (int c = 0; c < sh.cols; ++c) H_x(r, c) = rng.uniform(-5.0, 5.0);
                res(r) = rng.uniform(-2.0, 2.0);
            }

            Eigen::MatrixXd dH_f = H_f, dH_x = H_x;
            Eigen::VectorXd dres = res;
            Eigen::MatrixXd oH_f = H_f, oH_x = H_x;
            Eigen::VectorXd ores = res;

            msckf::nullspace_project_inplace(dH_f, dH_x, dres);
            ov_msckf::UpdaterHelper::nullspace_project_inplace(oH_f, oH_x, ores);

            std::snprintf(name, sizeof(name), "nullspace[%dx%d] H_x", sh.rows, sh.cols);
            rep.expect(name, dH_x, oH_x);
            std::snprintf(name, sizeof(name), "nullspace[%dx%d] res", sh.rows, sh.cols);
            rep.expect(name, dres, ores);

            // --- measurement compression, fed the projected system -----------
            Eigen::MatrixXd cH = dH_x, oc_H = oH_x;
            Eigen::VectorXd cr = dres, oc_r = ores;
            msckf::measurement_compress_inplace(cH, cr);
            ov_msckf::UpdaterHelper::measurement_compress_inplace(oc_H, oc_r);

            std::snprintf(name, sizeof(name), "compress[%dx%d] H_x", sh.rows, sh.cols);
            rep.expect(name, cH, oc_H);
            std::snprintf(name, sizeof(name), "compress[%dx%d] res", sh.rows, sh.cols);
            rep.expect(name, cr, oc_r);
        }
    }

    return rep.finish("stage 2.9 nullspace + compression");
}
