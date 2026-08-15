// Parity stage 1.1 -- the chi-squared 0.95 gate.
//
// Official builds its table at runtime with boost; we embed it. This test is
// what keeps the two from drifting: it recomputes boost's value for every dof
// and demands the embedded literal match bit-for-bit.
#include "bitdiff.hpp"
#include "../msckf/chi2_table.hpp"

#include <boost/math/distributions/chi_squared.hpp>

int main() {
    bitdiff::Report rep;

    for (int dof = 1; dof < msckf::CHI2_TABLE_SIZE; ++dof) {
        boost::math::chi_squared dist(dof);
        double ov = boost::math::quantile(dist, 0.95);
        char name[32];
        std::snprintf(name, sizeof(name), "chi2_95[dof=%d]", dof);
        rep.expect_scalar(name, msckf::CHI2_TABLE_95[dof], ov);
    }

    return rep.finish("stage 1.1 chi2 table");
}
