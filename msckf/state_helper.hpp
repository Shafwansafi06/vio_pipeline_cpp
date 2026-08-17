#pragma once
#include <Eigen/Core>
#include "state.hpp"
#include "../type/type.hpp"

namespace msckf {

extern double ekf_ms_ma, ekf_ms_s, ekf_ms_k, ekf_ms_cov;

// Count of negative covariance diagonals seen in EKFUpdate (official exits on
// the first one). Non-zero means the filter is running in a state official
// would call corrupt.
extern long ekf_negative_diagonal_count;

void EKFPropagation(State& state, type::Variable** order_NEW, int num_NEW,
                               type::Variable** order_OLD, int num_OLD,
                               const Eigen::Ref<const Eigen::MatrixXd>& Phi,
                               const Eigen::Ref<const Eigen::MatrixXd>& Q);
                               
void EKFUpdate(State& state, type::Variable** H_order, int num_H,
                          const Eigen::MatrixXd& H, const Eigen::VectorXd& res,
                          const Eigen::MatrixXd& R, Eigen::VectorXd* dx_out = nullptr);
                          
void set_initial_covariance(State& state, const Eigen::MatrixXd& covariance,
                                       type::Variable** order, int num_order);
                                       
void get_marginal_covariance_into(const State& state, type::Variable** small_variables, int num_small,
                                  Eigen::MatrixXd& out);
Eigen::MatrixXd get_marginal_covariance(const State& state, type::Variable** small_variables, int num_small);
    
Eigen::MatrixXd get_full_covariance(const State& state);
    
void marginalize(State& state, type::Variable* marg);
    
type::Variable* clone_var(State& state, type::Variable* variable_to_clone);
    
bool initialize(State& state, type::Variable* new_variable,
                           type::Variable** H_order, int num_H,
                           const Eigen::MatrixXd& H_R, const Eigen::MatrixXd& H_L,
                           const Eigen::MatrixXd& R, const Eigen::VectorXd& res,
                           double chi_2_mult);
                           
void initialize_invertible(State& state, type::Variable* new_variable,
                                      type::Variable** H_order, int num_H,
                                      const Eigen::MatrixXd& H_R, const Eigen::MatrixXd& H_L,
                                      const Eigen::MatrixXd& R, const Eigen::VectorXd& res);
                                      
void augment_clone(State& state, const Eigen::Vector3d& last_w);
    
void marginalize_old_clone(State& state);
void marginalize_lifo_clone(State& state);
    
void marginalize_slam(State& state);

// Helper to extract IMU pose as a PoseJPL variable for cloning
type::Variable get_imu_pose(const type::Variable& imu);

} // namespace msckf
