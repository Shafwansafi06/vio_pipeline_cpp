// Regression test: after many augment_clone()/marginalize_old_clone() cycles,
// (a) every state.variables[] pointer that aliases a clones_IMU[] slot must
//     point at the CURRENT address of that slot (state_helper.cpp's ring-shift
//     in marginalize_old_clone rewrites these pointers by hand), and
// (b) no two active variables may share a covariance offset, and every
//     offset must fit inside [0, cov_size).
#include <iostream>

#include "../msckf/state.hpp"
#include "../msckf/state_helper.hpp"

namespace {

bool check_invariants(const msckf::State& state, int round) {
    // (a) every clones_IMU[i] with i < num_clones must be referenced by
    // exactly one entry in variables[], and that entry must be &clones_IMU[i]
    // itself (not a stale address from before a ring-shift).
    for (int i = 0; i < state.num_clones; ++i) {
        const type::Variable* expected = &state.clones_IMU[i];
        bool found = false;
        for (int v = 0; v < state.num_variables; ++v) {
            if (state.variables[v] == expected) {
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "round " << round << ": clones_IMU[" << i
                      << "] has no matching entry in variables[] (stale pointer after shift)\n";
            return false;
        }
    }

    // (b) no duplicate/out-of-range covariance offsets among active variables.
    bool used[msckf::STATE_COV_CAPACITY] = {false};
    for (int v = 0; v < state.num_variables; ++v) {
        const type::Variable* var = state.variables[v];
        if (var->id < 0 || var->id + var->size > state.cov_size) {
            std::cerr << "round " << round << ": variable id=" << var->id
                      << " size=" << var->size << " out of bounds for cov_size="
                      << state.cov_size << "\n";
            return false;
        }
        for (int k = var->id; k < var->id + var->size; ++k) {
            if (used[k]) {
                std::cerr << "round " << round << ": covariance offset " << k
                          << " claimed by two variables\n";
                return false;
            }
            used[k] = true;
        }
    }

    // clones_IMU[] must stay timestamp-sorted ascending (append-only, only the
    // front is ever marginalized).
    for (int i = 1; i < state.num_clones; ++i) {
        if (state.clones_IMU[i].timestamp <= state.clones_IMU[i - 1].timestamp) {
            std::cerr << "round " << round << ": clones_IMU[] timestamps not strictly increasing\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    msckf::StateOptions options;
    options.max_clone_size = 5;
    msckf::State state;
    msckf::init_state(state, options);

    const Eigen::Vector3d last_w = Eigen::Vector3d::Zero();
    constexpr int kRounds = 40; // several full ring-buffer wraps at max_clone_size=5
    for (int round = 0; round < kRounds; ++round) {
        state.timestamp = 1.0 + 0.1 * round;
        msckf::augment_clone(state, last_w);
        msckf::marginalize_old_clone(state);
        if (!check_invariants(state, round)) return 1;
    }

    if (state.num_clones != options.max_clone_size) {
        std::cerr << "expected steady-state num_clones=" << options.max_clone_size
                  << ", got " << state.num_clones << "\n";
        return 1;
    }

    std::cout << "clone pointer + covariance-offset invariants held across "
              << kRounds << " augment/marginalize cycles\n";
    return 0;
}
