#include "state_helper.hpp"

#include <chrono>
#include "../type/quat_ops.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <cmath>

namespace msckf {

double ekf_ms_ma = 0.0, ekf_ms_s = 0.0, ekf_ms_k = 0.0, ekf_ms_cov = 0.0;

// Number of times EKFUpdate saw a negative covariance diagonal -- the condition
// official Open_VINS treats as fatal. Reported by the runners at shutdown.
long ekf_negative_diagonal_count = 0;

type::Variable get_imu_pose(const type::Variable& imu) {
    type::Variable pose;
    type::init_posejpl(pose);
    for (int i = 0; i < 7; ++i) {
        pose.value[i] = imu.value[i];
        pose.fej[i] = imu.fej[i];
    }
    pose.id = imu.id;
    return pose;
}

// Wilson-Hilferty approximation for Chi-Square 95% quantile
static double chi2_ppf_95(int dof) {
    if (dof <= 0) return 0.0;
    if (dof == 1) return 3.841;
    if (dof == 2) return 5.991;
    if (dof == 3) return 7.815;
    if (dof == 4) return 9.488;
    if (dof == 5) return 11.070;
    double k = static_cast<double>(dof);
    double term = 1.0 - 2.0 / (9.0 * k);
    double sd = std::sqrt(2.0 / (9.0 * k));
    double z = 1.64485362695;
    return k * std::pow(term + z * sd, 3.0);
}

void EKFPropagation(State& state, type::Variable** order_NEW, int num_NEW,
                                 type::Variable** order_OLD, int num_OLD,
                                 const Eigen::Ref<const Eigen::MatrixXd>& Phi,
                                 const Eigen::Ref<const Eigen::MatrixXd>& Q) {
    if (num_NEW == 0 || num_OLD == 0) {
        std::cerr << "EKFPropagation() - Called with empty variables!\n";
        std::exit(1);
    }
    
    int size_order_NEW = 0;
    for (int i = 0; i < num_NEW; ++i) {
        if (i < num_NEW - 1) {
            assert(order_NEW[i]->id + order_NEW[i]->size == order_NEW[i+1]->id);
        }
        size_order_NEW += order_NEW[i]->size;
    }
    
    int size_order_OLD = 0;
    for (int i = 0; i < num_OLD; ++i) {
        size_order_OLD += order_OLD[i]->size;
    }
    
    assert(size_order_NEW == Phi.rows());
    assert(size_order_OLD == Phi.cols());
    assert(size_order_NEW == Q.rows());
    assert(size_order_NEW == Q.cols());
    
    int total_size = state.cov_size;
    assert(total_size <= STATE_COV_CAPACITY);
    assert(size_order_NEW <= IMU_ERROR_STATE_CAPACITY);
    alignas(64) static thread_local double cov_phi_t_storage[
        STATE_COV_CAPACITY * IMU_ERROR_STATE_CAPACITY];
    Eigen::Map<Eigen::MatrixXd> cov_phi_t(
        cov_phi_t_storage, total_size, size_order_NEW);
    cov_phi_t.setZero();
    
    int current_it = 0;
    for (int i = 0; i < num_OLD; ++i) {
        type::Variable* var = order_OLD[i];
        int var_id = var->id;
        int var_sz = var->size;
        
        const auto phi_block = Phi.block(0, current_it, size_order_NEW, var_sz);
        const auto cov_block = state.Cov.block(0, var_id, total_size, var_sz);
        
        cov_phi_t.noalias() +=
            cov_block * phi_block.transpose();
        current_it += var_sz;
    }
    
    alignas(64) static thread_local double phi_cov_phi_t_storage[
        IMU_ERROR_STATE_CAPACITY * IMU_ERROR_STATE_CAPACITY];
    Eigen::Map<Eigen::MatrixXd> phi_cov_phi_t(
        phi_cov_phi_t_storage, size_order_NEW, size_order_NEW);
    // Official: `Phi_Cov_PhiT = Q.selfadjointView<Eigen::Upper>()`. What was
    // here averaged Q with its own transpose instead, which agrees only when Q
    // is already exactly symmetric.
    phi_cov_phi_t = Q.selfadjointView<Eigen::Upper>();
    
    current_it = 0;
    for (int i = 0; i < num_OLD; ++i) {
        type::Variable* var = order_OLD[i];
        int var_id = var->id;
        int var_sz = var->size;
        
        const auto phi_block = Phi.block(0, current_it, size_order_NEW, var_sz);
        const auto cov_phit_block = cov_phi_t.block(var_id, 0, var_sz, size_order_NEW);
        
        phi_cov_phi_t.noalias() +=
            phi_block * cov_phit_block;
        current_it += var_sz;
    }
    
    int start_id = order_NEW[0]->id;
    int phi_size = Phi.rows();
    
    state.Cov.block(start_id, 0, phi_size, total_size) =
        cov_phi_t.transpose();
    state.Cov.block(0, start_id, total_size, phi_size) =
        cov_phi_t;
    state.Cov.block(start_id, start_id, phi_size, phi_size) =
        phi_cov_phi_t;
    
    // No full-matrix symmetrisation here: official has none, and the three
    // block writes above already leave the covariance as symmetric as official
    // leaves it (the off-diagonal blocks are literal transposes of each other,
    // the new diagonal block is symmetric by construction, and the rest of the
    // matrix is untouched). The loop that used to be here was an invention that
    // touched all N^2 entries with a cache-hostile stride every propagation --
    // 0.309 -> 0.106 ms/frame on EuRoC MH_01 with 50 SLAM landmarks.
    for (int i = 0; i < total_size; ++i) {
        if (state.Cov(i, i) < 0.0) {
            std::cerr << "EKFPropagation() - diagonal at " << i << " is " << state.Cov(i, i) << "\n";
            std::exit(1);
        }
    }
}

void EKFUpdate(State& state, type::Variable** H_order, int num_H,
                            const Eigen::MatrixXd& H, const Eigen::VectorXd& res,
                            const Eigen::MatrixXd& R) {
    assert(res.size() == R.rows());
    assert(H.rows() == res.size());

    // Transcribed from official StateHelper::EKFUpdate. What used to be here
    // differed in three ways, all of them inventions:
    //
    //   1. It formed the full S, symmetrised it as 0.5*(S + S^T), and solved
    //      with LDLT plus a CompleteOrthogonalDecomposition pseudo-inverse
    //      fallback. Official builds S in the upper triangle only and solves
    //      with an LLT on selfadjointView<Upper>.
    //   2. It added 1e-9 to every diagonal of S as jitter. Official adds none.
    //   3. It floored any covariance diagonal below 1e-12 up to 1e-12.
    //      Official treats a negative diagonal as a corrupt filter: it prints
    //      the index and exits. Flooring hides exactly the condition that
    //      signals divergence, letting the filter keep running (and drifting)
    //      instead of failing. The floor is gone; see the counter below.

    int total_size = state.cov_size;
    const auto ekf_t0 = std::chrono::steady_clock::now();
    Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(total_size, res.size());

    int current_it = 0;
    int H_id[100];
    for (int i = 0; i < num_H; ++i) {
        H_id[i] = current_it;
        current_it += H_order[i]->size;
    }

    // M_a = P * H^T, accumulated one MEASUREMENT variable at a time over the
    // full column of the covariance. Official loops over every state variable
    // and, inside that, over every measurement variable -- mathematically the
    // same sum, but ~65 x 13 tiny GEMMs (each with its own heap-allocated M_i)
    // instead of ~13 tall ones. Same arithmetic, same result, far fewer calls.
    for (int k = 0; k < num_H; ++k) {
        type::Variable* meas_var = H_order[k];
        M_a.noalias() += state.Cov.block(0, meas_var->id, total_size, meas_var->size) *
                         H.block(0, H_id[k], H.rows(), meas_var->size).transpose();
    }

    const auto ekf_t1 = std::chrono::steady_clock::now();
    // S = H P H^T + R. The rows of M_a belonging to the measurement variables
    // ARE P_small * H^T, so S is just H times those -- no need to gather an
    // n_small x n_small marginal covariance again (that gather allocates and
    // copies the whole block matrix a second time per update).
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(R.rows(), R.rows());
    for (int k = 0; k < num_H; ++k) {
        type::Variable* meas_var = H_order[k];
        S.noalias() += H.block(0, H_id[k], H.rows(), meas_var->size) *
                       M_a.block(meas_var->id, 0, meas_var->size, res.size());
    }
    S.triangularView<Eigen::Upper>() += R;
    Eigen::MatrixXd Sinv = Eigen::MatrixXd::Identity(R.rows(), R.rows());
    S.selfadjointView<Eigen::Upper>().llt().solveInPlace(Sinv);
    const auto ekf_t2 = std::chrono::steady_clock::now();
    Eigen::MatrixXd K = M_a * Sinv.selfadjointView<Eigen::Upper>();
    const auto ekf_t3 = std::chrono::steady_clock::now();

    state.Cov.block(0, 0, total_size, total_size).triangularView<Eigen::Upper>() -= K * M_a.transpose();
    // Mirror the upper triangle into the lower one column by column. The
    // previous `Cov.block(...) = Cov.block(...).selfadjointView<Upper>()` is a
    // self-assignment Eigen has to evaluate through a full N x N temporary,
    // which it allocates on every update.
    //
    // Do NOT "simplify" this into one dense `Cov -= K * M_a.transpose()`.
    // K*M_a^T is symmetric in exact arithmetic but not in floating point, and
    // the resulting asymmetry drives covariance diagonals negative within a few
    // hundred frames -- measured: EKFPropagation aborts on MH_01 at ~t+16 s.
    for (int c = 0; c + 1 < total_size; ++c) {
        state.Cov.block(c + 1, c, total_size - c - 1, 1) =
            state.Cov.block(c, c + 1, 1, total_size - c - 1).transpose();
    }

    const auto ekf_t4 = std::chrono::steady_clock::now();
    ekf_ms_ma += std::chrono::duration<double, std::milli>(ekf_t1 - ekf_t0).count();
    ekf_ms_s += std::chrono::duration<double, std::milli>(ekf_t2 - ekf_t1).count();
    ekf_ms_k += std::chrono::duration<double, std::milli>(ekf_t3 - ekf_t2).count();
    ekf_ms_cov += std::chrono::duration<double, std::milli>(ekf_t4 - ekf_t3).count();

    // Official exits the process here. We count and report instead, so a run
    // that would have died still produces a trajectory to diagnose -- but the
    // count must be reported, never silently absorbed.
    for (int i = 0; i < total_size; ++i) {
        if (state.Cov(i, i) < 0.0) {
            ++ekf_negative_diagonal_count;
            if (ekf_negative_diagonal_count <= 10) {
                std::fprintf(stderr, "[EKFUpdate]: negative covariance diagonal at %d is %.6g\n",
                             i, state.Cov(i, i));
            }
        }
    }

    Eigen::VectorXd dx = K * res;

    for (int i = 0; i < state.num_variables; ++i) {
        type::Variable* var = state.variables[i];
        Eigen::VectorXd update_vec = dx.segment(var->id, var->size);
        type::update_variable(*var, update_vec);
    }
}

void set_initial_covariance(State& state, const Eigen::MatrixXd& covariance,
                                         type::Variable** order, int num_order) {
    int i_index = 0;
    for (int i = 0; i < num_order; ++i) {
        int k_index = 0;
        for (int k = 0; k < num_order; ++k) {
            int block_rows = order[i]->size;
            int block_cols = order[k]->size;
            
            Eigen::MatrixXd src_block = covariance.block(i_index, k_index, block_rows, block_cols);
            state.Cov.block(order[i]->id, order[k]->id, block_rows, block_cols) = src_block;
            
            k_index += block_cols;
        }
        i_index += order[i]->size;
    }
    
    int total_size = state.cov_size;
    state.Cov.block(0, 0, total_size, total_size) = 0.5 * (state.Cov.block(0, 0, total_size, total_size) + state.Cov.block(0, 0, total_size, total_size).transpose());
}

void get_marginal_covariance_into(const State& state, type::Variable** small_variables, int num_small,
                                  Eigen::MatrixXd& out) {
    int cov_size = 0;
    for (int i = 0; i < num_small; ++i) {
        cov_size += small_variables[i]->size;
    }
    // resize() is a no-op when the size already matches, so a caller passing a
    // reused buffer allocates at most once. The chi2 gate calls this per
    // feature per frame, where the allocation was the dominant cost.
    out.resize(cov_size, cov_size);
    int i_index = 0;
    for (int i = 0; i < num_small; ++i) {
        int rows = small_variables[i]->size;
        int k_index = 0;
        for (int k = 0; k < num_small; ++k) {
            int cols = small_variables[k]->size;
            out.block(i_index, k_index, rows, cols) =
                state.Cov.block(small_variables[i]->id, small_variables[k]->id, rows, cols);
            k_index += cols;
        }
        i_index += rows;
    }
}

Eigen::MatrixXd get_marginal_covariance(const State& state, type::Variable** small_variables, int num_small) {
    int cov_size = 0;
    for (int i = 0; i < num_small; ++i) {
        cov_size += small_variables[i]->size;
    }
    
    // No Zero() -- every block below is written, so the fill is wasted -- and no
    // per-block temporary. Copying through `MatrixXd src_block = Cov.block(...)`
    // heap-allocates once per (i,k) pair, which is ~169 allocations per call at
    // 13 variables, and this is called once per feature in the chi2 gate.
    Eigen::MatrixXd Small_cov(cov_size, cov_size);
    int i_index = 0;
    for (int i = 0; i < num_small; ++i) {
        int rows = small_variables[i]->size;
        int k_index = 0;
        for (int k = 0; k < num_small; ++k) {
            int cols = small_variables[k]->size;
            Small_cov.block(i_index, k_index, rows, cols) =
                state.Cov.block(small_variables[i]->id, small_variables[k]->id, rows, cols);
            k_index += cols;
        }
        i_index += rows;
    }
    return Small_cov;
}

Eigen::MatrixXd get_full_covariance(const State& state) {
    return state.Cov.block(0, 0, state.cov_size, state.cov_size);
}

void marginalize(State& state, type::Variable* marg) {
    bool found = false;
    for (int i = 0; i < state.num_variables; ++i) {
        if (state.variables[i] == marg) {
            found = true;
            break;
        }
    }
    if (!found) {
        std::cerr << "marginalize() - Called on variable not in state!\n";
        std::exit(1);
    }
    
    int marg_size = marg->size;
    int marg_id = marg->id;
    int old_dim = state.cov_size;
    int x2_size = old_dim - marg_id - marg_size;
    int new_dim = old_dim - marg_size;
    
    Eigen::MatrixXd Cov_new = Eigen::MatrixXd::Zero(new_dim, new_dim);
    
    if (marg_id > 0) {
        Cov_new.block(0, 0, marg_id, marg_id) = state.Cov.block(0, 0, marg_id, marg_id);
    }
    if (marg_id > 0 && x2_size > 0) {
        Cov_new.block(0, marg_id, marg_id, x2_size) = state.Cov.block(0, marg_id + marg_size, marg_id, x2_size);
        Cov_new.block(marg_id, 0, x2_size, marg_id) = Cov_new.block(0, marg_id, marg_id, x2_size).transpose();
    }
    if (x2_size > 0) {
        Cov_new.block(marg_id, marg_id, x2_size, x2_size) = state.Cov.block(marg_id + marg_size, marg_id + marg_size, x2_size, x2_size);
    }
    
    state.Cov.block(0, 0, new_dim, new_dim) = Cov_new;
    state.cov_size = new_dim;
    
    int write_idx = 0;
    for (int i = 0; i < state.num_variables; ++i) {
        type::Variable* var = state.variables[i];
        if (var != marg) {
            if (var->id > marg_id) {
                var->id -= marg_size;
            }
            state.variables[write_idx++] = var;
        }
    }
    state.num_variables = write_idx;
    marg->id = -1;
}

type::Variable* clone_var(State& state, type::Variable* variable_to_clone) {
    int total_size = variable_to_clone->size;
    int old_size = state.cov_size;
    int new_loc = old_size;
    
    // Grow covariance
    state.Cov.block(old_size, 0, total_size, old_size + total_size).setZero();
    state.Cov.block(0, old_size, old_size + total_size, total_size).setZero();
    
    type::Variable* new_clone = nullptr;
    
    for (int k = 0; k < state.num_variables; ++k) {
        type::Variable* var = state.variables[k];
        if (var->id == variable_to_clone->id) {
            int old_loc = var->id;
            
            state.Cov.block(new_loc, new_loc, total_size, total_size) = state.Cov.block(old_loc, old_loc, total_size, total_size);
            state.Cov.block(0, new_loc, old_size, total_size) = state.Cov.block(0, old_loc, old_size, total_size);
            state.Cov.block(new_loc, 0, total_size, old_size) = state.Cov.block(old_loc, 0, total_size, old_size);
            
            if (variable_to_clone->type == type::VariableType::POSEJPL) {
                if (state.num_clones < 20) {
                    new_clone = &state.clones_IMU[state.num_clones++];
                    *new_clone = type::clone_variable(*variable_to_clone);
                    new_clone->id = new_loc;
                }
            }
            break;
        }
    }
    
    if (new_clone) {
        state.variables[state.num_variables++] = new_clone;
        state.cov_size += total_size;
    }
    return new_clone;
}

bool initialize(State& state, type::Variable* new_variable,
                             type::Variable** H_order, int num_H,
                             const Eigen::MatrixXd& H_R, const Eigen::MatrixXd& H_L,
                             const Eigen::MatrixXd& R, const Eigen::VectorXd& res,
                             double chi_2_mult) {
    for (int i = 0; i < state.num_variables; ++i) {
        if (state.variables[i] == new_variable) {
            std::cerr << "initialize() - Variable already in state!\n";
            std::exit(1);
        }
    }
    
    int new_var_size = new_variable->size;
    assert(new_var_size == H_L.cols());
    
    Eigen::MatrixXd H_L_wk = H_L;
    Eigen::MatrixXd H_R_wk = H_R;
    Eigen::VectorXd res_wk = res;
    
    // Official triangularises H_L with in-place GIVENS rotations, applying the
    // same rotations to res and H_R. This used a HouseholderQR with an
    // explicitly formed Q and then multiplied Q^T through -- the identical
    // mistake already fixed once in nullspace_project_inplace (parity manual,
    // divergence #22). The two produce DIFFERENT orthonormal bases, so
    // Hxinit / H_finit / resinit / Hup / resup all come out different, which
    // means every SLAM landmark entered the state with a different initial
    // estimate and covariance than official would have given it.
    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n = 0; n < H_L_wk.cols(); ++n) {
        for (int m = (int)H_L_wk.rows() - 1; m > n; m--) {
            tempHo_GR.makeGivens(H_L_wk(m - 1, n), H_L_wk(m, n));
            (H_L_wk.block(m - 1, n, 2, H_L_wk.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (res_wk.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (H_R_wk.block(m - 1, 0, 2, H_R_wk.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
        }
    }

    Eigen::MatrixXd Hxinit = H_R_wk.block(0, 0, new_var_size, H_R_wk.cols());
    Eigen::MatrixXd H_finit = H_L_wk.block(0, 0, new_var_size, new_var_size);
    Eigen::VectorXd resinit = res_wk.block(0, 0, new_var_size, 1);
    Eigen::MatrixXd Rinit = R.block(0, 0, new_var_size, new_var_size);

    Eigen::MatrixXd Hup = H_R_wk.block(new_var_size, 0, H_R_wk.rows() - new_var_size, H_R_wk.cols());
    Eigen::VectorXd resup = res_wk.block(new_var_size, 0, res_wk.rows() - new_var_size, 1);
    Eigen::MatrixXd Rup = R.block(new_var_size, new_var_size, R.rows() - new_var_size, R.rows() - new_var_size);

    // Official: no symmetrisation, no jitter, and an LLT rather than an LDLT.
    Eigen::MatrixXd P_up = get_marginal_covariance(state, H_order, num_H);
    Eigen::MatrixXd S = Hup * P_up * Hup.transpose() + Rup;
    double chi2_val = resup.dot(S.llt().solve(resup));

    const double chi2_check = chi2_ppf_95((int)res.size());

    if (chi2_val > chi_2_mult * chi2_check) {
        return false;
    }
    
    initialize_invertible(state, new_variable, H_order, num_H, Hxinit, H_finit, Rinit, resinit);
    
    if (Hup.rows() > 0) {
        EKFUpdate(state, H_order, num_H, Hup, resup, Rup);
    }
    return true;
}

void initialize_invertible(State& state, type::Variable* new_variable,
                                       type::Variable** H_order, int num_H,
                                       const Eigen::MatrixXd& H_R, const Eigen::MatrixXd& H_L,
                                       const Eigen::MatrixXd& R, const Eigen::VectorXd& res) {
    for (int i = 0; i < state.num_variables; ++i) {
        if (state.variables[i] == new_variable) {
            std::exit(1);
        }
    }
    
    int oldSize = state.cov_size;
    int new_var_sz = new_variable->size;
    int res_sz = res.size();
    
    Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(oldSize, res_sz);
    int current_it = 0;
    int H_id[100];
    for (int i = 0; i < num_H; ++i) {
        H_id[i] = current_it;
        current_it += H_order[i]->size;
    }
    
    // Same shape as EKFUpdate: accumulate M_a = P * H_R^T over MEASUREMENT
    // variables against full-height covariance blocks. The previous nested loop
    // allocated three heap temporaries per (state variable, measurement
    // variable) pair -- `M_i`, `cov_block` and `H_R_block` -- which DHAT
    // measured as 463k allocations over 12 s of EuRoC, the single largest
    // allocation source in the pipeline.
    for (int k = 0; k < num_H; ++k) {
        type::Variable* meas_var = H_order[k];
        M_a.noalias() += state.Cov.block(0, meas_var->id, oldSize, meas_var->size) *
                         H_R.block(0, H_id[k], H_R.rows(), meas_var->size).transpose();
    }

    // M = H_R P H_R^T + R, built from the rows of M_a that belong to the
    // measurement variables (those rows already are P_small * H_R^T), so the
    // marginal covariance never has to be gathered.
    Eigen::MatrixXd M = R;
    for (int k = 0; k < num_H; ++k) {
        type::Variable* meas_var = H_order[k];
        M.noalias() += H_R.block(0, H_id[k], H_R.rows(), meas_var->size) *
                       M_a.block(meas_var->id, 0, meas_var->size, res_sz);
    }
    
    Eigen::MatrixXd H_Linv = H_L.inverse();
    Eigen::MatrixXd P_LL = H_Linv * M * H_Linv.transpose();
    
    int newSize = oldSize + new_var_sz;
    
    state.Cov.block(oldSize, 0, new_var_sz, oldSize + new_var_sz).setZero();
    state.Cov.block(0, oldSize, oldSize + new_var_sz, new_var_sz).setZero();
    
    Eigen::MatrixXd cross_term = -M_a * H_Linv.transpose();
    state.Cov.block(0, oldSize, oldSize, new_var_sz) = cross_term;
    state.Cov.block(oldSize, 0, new_var_sz, oldSize) = cross_term.transpose();
    state.Cov.block(oldSize, oldSize, new_var_sz, new_var_sz) = P_LL;
    
    Eigen::VectorXd dx = H_Linv * res;
    type::update_variable(*new_variable, dx);
    
    new_variable->id = oldSize;
    state.variables[state.num_variables++] = new_variable;
    state.cov_size = newSize;
}

void augment_clone(State& state, const Eigen::Vector3d& last_w) {
    // Check timestamp collision
    for (int i = 0; i < state.num_clones; ++i) {
        if (state.clones_IMU[i].timestamp == state.timestamp) {
            std::cerr << "augment_clone() - Clone already exists at timestamp!\n";
            std::exit(1);
        }
    }
    
    type::Variable imu_pose = get_imu_pose(state.imu);
    type::Variable* pose = clone_var(state, &imu_pose);
    if (!pose) return;
    
    pose->timestamp = state.timestamp;
    
    if (state.options.do_calib_camera_timeoffset) {
        Eigen::Matrix<double, 6, 1> dnc_dt = Eigen::Matrix<double, 6, 1>::Zero();
        dnc_dt.head<3>() = last_w;
        // IMU velocity: elements [7..10] of imu value
        Eigen::Vector3d vel(state.imu.value[7], state.imu.value[8], state.imu.value[9]);
        dnc_dt.tail<3>() = vel;
        
        int pose_id = pose->id;
        int dt_id = state.calib_dt_CAMtoIMU.id;
        
        // Col update: Cov[:, pose_id:pose_id+6] += Cov[:, dt_id] * dnc_dt.transpose()
        Eigen::MatrixXd col_update = state.Cov.block(0, dt_id, state.cov_size, 1) * dnc_dt.transpose();
        state.Cov.block(0, pose_id, state.cov_size, 6) += col_update;
        
        // Row update: Cov[pose_id:pose_id+6, :] += dnc_dt * Cov[dt_id, :]
        Eigen::MatrixXd row_update = dnc_dt * state.Cov.block(dt_id, 0, 1, state.cov_size);
        state.Cov.block(pose_id, 0, 6, state.cov_size) += row_update;
    }
}

void marginalize_old_clone(State& state) {
    if (state.num_clones > state.options.max_clone_size) {
        double marginal_time = margtimestep(state);
        if (marginal_time == -1.0) return;
        
        int idx = -1;
        for (int i = 0; i < state.num_clones; ++i) {
            if (state.clones_IMU[i].timestamp == marginal_time) {
                idx = i;
                break;
            }
        }
        
        if (idx != -1) {
            marginalize(state, &state.clones_IMU[idx]);
            // Shift remaining clones in clones_IMU array
            for (int i = idx; i < state.num_clones - 1; ++i) {
                type::Variable* old_address = &state.clones_IMU[i + 1];
                state.clones_IMU[i] = state.clones_IMU[i + 1];
                for (int variable = 0; variable < state.num_variables; ++variable) {
                    if (state.variables[variable] == old_address) {
                        state.variables[variable] = &state.clones_IMU[i];
                        break;
                    }
                }
            }
            state.num_clones--;
        }
    }
}

void marginalize_lifo_clone(State& state) {
    // LIFO scheme (Kottas, Wu & Roumeliotis, Sect. III-B): while hovering,
    // replace the most-recently-added clone with the new one instead of
    // dropping the oldest, so the generic-motion baseline already in the
    // window is preserved and the system does not lose observability.
    // `augment_clone` has already appended the new clone as the last
    // element; the one to drop is the previous "current" clone, now second
    // to last.
    if (state.num_clones > state.options.max_clone_size && state.num_clones >= 2) {
        const int drop_idx = state.num_clones - 2;
        marginalize(state, &state.clones_IMU[drop_idx]);
        type::Variable* old_address = &state.clones_IMU[state.num_clones - 1];
        state.clones_IMU[drop_idx] = state.clones_IMU[state.num_clones - 1];
        for (int variable = 0; variable < state.num_variables; ++variable) {
            if (state.variables[variable] == old_address) {
                state.variables[variable] = &state.clones_IMU[drop_idx];
                break;
            }
        }
        state.num_clones--;
    }
}

void marginalize_slam(State& state) {
    int i = 0;
    while (i < state.num_slam_features) {
        type::Variable& feat = state.features_SLAM[i];
        if (feat.should_marg && feat.feat_id > 4 * state.options.max_aruco_features) {
            marginalize(state, &feat);
            // Shift remaining features
            for (int k = i; k < state.num_slam_features - 1; ++k) {
                type::Variable* old_address = &state.features_SLAM[k + 1];
                state.features_SLAM[k] = state.features_SLAM[k + 1];
                for (int variable = 0; variable < state.num_variables; ++variable) {
                    if (state.variables[variable] == old_address) {
                        state.variables[variable] = &state.features_SLAM[k];
                        break;
                    }
                }
            }
            state.num_slam_features--;
            // do not increment i since we shifted
        } else {
            i++;
        }
    }
}

} // namespace msckf
