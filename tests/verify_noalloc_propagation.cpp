#include <Eigen/Core>
#include <cmath>
#include <iostream>

#include "../msckf/propagator.hpp"
#include "../msckf/state.hpp"

int main() {
    msckf::StateOptions options;
    options.do_fej = true;
    options.integration_method = msckf::IntegrationMethod::RK4;

    msckf::State state;
    msckf::init_state(state, options);
    state.timestamp = 1.00;

    msckf::PropagatorNoises noises;
    msckf::PropagatorData propagator;
    msckf::init_propagator(propagator, noises, 9.81);

    core::ImuData first;
    first.timestamp = 1.00;
    first.wm << 0.01, -0.02, 0.03;
    first.am << 0.1, -0.2, 9.75;

    core::ImuData second;
    second.timestamp = 1.01;
    second.wm << 0.011, -0.018, 0.029;
    second.am << 0.11, -0.19, 9.76;

    const core::ImuData readings[2] = {first, second};

    Eigen::internal::set_is_malloc_allowed(false);
    const bool propagated =
        msckf::propagate_only(propagator, state, second.timestamp, readings, 2);
    Eigen::internal::set_is_malloc_allowed(true);

    constexpr int dim = 15;
    if (!propagated || !state.Cov.topLeftCorner(dim, dim).allFinite()) {
        std::cerr << "non-finite propagation output\n";
        return 1;
    }
    if (!state.Cov.topLeftCorner(dim, dim)
             .isApprox(state.Cov.topLeftCorner(dim, dim).transpose(), 1e-12)) {
        std::cerr << "propagated covariance lost symmetry\n";
        return 1;
    }

    std::cout << "allocation-free IMU state and covariance propagation passed\n";
    return 0;
}
