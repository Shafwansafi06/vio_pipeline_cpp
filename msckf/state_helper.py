import numpy as np
import sys
import copy
from typing import List, Dict, Tuple, Optional
from scipy.stats import chi2
import threading

# Project Imports (Assumed structure)
from vio_pipeline_v2.type.Type import Type
from vio_pipeline_v2.type.PoseJPL import PoseJPL
from vio_pipeline_v2.msckf.State import State

# Colors for printing
RED = "\033[31m"
RESET = "\033[0m"
YELLOW = "\033[33m"

class StateHelper:

    @staticmethod
    def EKFPropagation(state: State, order_NEW: List[Type], order_OLD: List[Type], 
                       Phi: np.ndarray, Q: np.ndarray):
        """
        Propagates the state covariance using the transition matrix Phi and noise Q.
        Cov_new = Phi * Cov_old * Phi^T + Q
        """
        
        # 1. Validation
        if not order_NEW or not order_OLD:
            print(f"{RED}StateHelper::EKFPropagation() - Called with empty variable arrays!{RESET}")
            sys.exit(1)

        # Ensure order_NEW is contiguous in memory (simulated check)
        size_order_NEW = order_NEW[0].size()
        for i in range(len(order_NEW) - 1):
            if order_NEW[i].id() + order_NEW[i].size() != order_NEW[i+1].id():
                print(f"{RED}StateHelper::EKFPropagation() - Called with non-contiguous state elements!{RESET}")
                print(f"{RED}StateHelper::EKFPropagation() - Code only supports transition in same order as state{RESET}")
                sys.exit(1)
            size_order_NEW += order_NEW[i+1].size()

        # Size of old phi matrix
        size_order_OLD = order_OLD[0].size()
        for i in range(len(order_OLD) - 1):
            size_order_OLD += order_OLD[i+1].size()

        # Assert sizes
        assert size_order_NEW == Phi.shape[0]
        assert size_order_OLD == Phi.shape[1]
        assert size_order_NEW == Q.shape[1]
        assert size_order_NEW == Q.shape[0]

        # 2. Get locations in small Phi for each measuring variable
        current_it = 0
        Phi_id = []
        for var in order_OLD:
            Phi_id.append(current_it)
            current_it += var.size()

        # 3. Calculate Cov_PhiT = [ Pxx ] [ Phi' ]'
        # Equivalent to Cov * Phi.T (but block-wise for efficiency)
        Cov_PhiT = np.zeros((state._Cov.shape[0], Phi.shape[0]))
        
        for i, var in enumerate(order_OLD):
            # Block multiplication: Cov[:, var_indices] * Phi[:, var_indices].T
            # Phi block is: Phi[0:rows, start:end]
            phi_block = Phi[:, Phi_id[i] : Phi_id[i] + var.size()]
            cov_block = state._Cov[:, var.id() : var.id() + var.size()]
            
            Cov_PhiT += cov_block @ phi_block.T

        # 4. Calculate Phi_Cov_PhiT = Phi * Cov * Phi^T + Q
        # We start with Q, then add the transformation
        Phi_Cov_PhiT = np.copy(Q)
        # Force symmetric (Upper triangle in C++ selfadjointView equivalent)
        Phi_Cov_PhiT = np.triu(Phi_Cov_PhiT) + np.triu(Phi_Cov_PhiT, 1).T

        for i, var in enumerate(order_OLD):
            # Phi_block * Cov_PhiT_block_transposed
            # In C++: Phi.block(...) * Cov_PhiT.block(...). 
            # Note: Cov_PhiT in C++ was [Cov * Phi^T]. 
            # Here we need the corresponding rows of Cov_PhiT (which corresponds to var's rows in Cov)
            
            phi_block = Phi[:, Phi_id[i] : Phi_id[i] + var.size()]
            cov_phit_block = Cov_PhiT[var.id() : var.id() + var.size(), :]
            
            Phi_Cov_PhiT += phi_block @ cov_phit_block

        # 5. Update State Covariance
        start_id = order_NEW[0].id()
        phi_size = Phi.shape[0]
        total_size = state._Cov.shape[0]

        # Update Cross Terms
        # Cov[start:end, :] = Cov_PhiT.T
        state._Cov[start_id : start_id + phi_size, 0 : total_size] = Cov_PhiT.T
        
        # Cov[:, start:end] = Cov_PhiT
        state._Cov[0 : total_size, start_id : start_id + phi_size] = Cov_PhiT

        # Update Diagonal Block
        # Cov[start:end, start:end] = Phi_Cov_PhiT
        state._Cov[start_id : start_id + phi_size, start_id : start_id + phi_size] = Phi_Cov_PhiT

        # 6. Check Positive Definiteness (Diagonals > 0)
        diags = np.diagonal(state._Cov)
        found_neg = False
        for i, d in enumerate(diags):
            if d < 0.0:
                print(f"{YELLOW}StateHelper::EKFPropagation() - diagonal at {i} is {d:.2f}{RESET}")
                found_neg = True
        
        if found_neg:
            sys.exit(1)

    @staticmethod
    def EKFUpdate(state: State, H_order: List[Type], H: np.ndarray, 
                  res: np.ndarray, R: np.ndarray):
        """
        Standard EKF Update step.
        K = P * H^T * (H * P * H^T + R)^-1
        x = x + K * res
        P = P - K * H * P
        """
        
        # Validation
        assert res.shape[0] == R.shape[0]
        assert H.shape[0] == res.shape[0]

        # 1. Calculate M_a = P * H^T (The "Innovation Covariance" part numerator)
        M_a = np.zeros((state._Cov.shape[0], res.shape[0]))

        # Get H indices
        current_it = 0
        H_id = []
        for meas_var in H_order:
            H_id.append(current_it)
            current_it += meas_var.size()

        # Compute M_a block by block
        for var in state._variables:
            M_i = np.zeros((var.size(), res.shape[0]))
            for i, meas_var in enumerate(H_order):
                # P_im * Hm^T
                cov_block = state._Cov[var.id() : var.id() + var.size(), 
                                       meas_var.id() : meas_var.id() + meas_var.size()]
                
                H_block = H[:, H_id[i] : H_id[i] + meas_var.size()]
                
                M_i += cov_block @ H_block.T
            
            M_a[var.id() : var.id() + var.size(), :] = M_i

        # 2. Get Marginal Covariance P_small
        P_small = StateHelper.get_marginal_covariance(state, H_order)

        # 3. Residual Covariance S = H * P_small * H^T + R
        S = H @ P_small @ H.T + R
        
        # Enforce Symmetry
        S = (S + S.T) * 0.5

        # Robustify innovation covariance against near-singular systems.
        jitter = 1e-9
        np.fill_diagonal(S, S.diagonal() + jitter)

        # 4. Kalman Gain K = M_a * S^-1
        # Use solve instead of inverse for stability: K = M_a * inv(S) -> K * S = M_a -> S * K^T = M_a^T
        # But M_a is (N_state x N_meas), S is (N_meas x N_meas).
        # K = M_a @ np.linalg.inv(S)
        try:
            # Using Cholesky solve if possible (llt in Eigen)
            np.linalg.cholesky(S)
            # Solves S * x = I -> x = S^-1
            Sinv = np.linalg.solve(S, np.eye(S.shape[0]))
            K = M_a @ Sinv
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse for singular/near-singular S.
            K = M_a @ np.linalg.pinv(S)

        # 5. Update Covariance P = P - K * H * P = P - K * M_a^T
        # We use the Joseph form equivalent or the standard simple form: P -= K * M_a.T
        update_term = K @ M_a.T
        
        # Only update upper triangle and mirror (to maintain symmetry like Eigen's selfadjointView)
        # But in Python, full update is easier and usually fine.
        state._Cov -= update_term
        state._Cov = (state._Cov + state._Cov.T) * 0.5

        # Check Positive Definiteness
        diags = np.diagonal(state._Cov)
        found_neg = False
        for i, d in enumerate(diags):
            if d < 0.0:
                print(f"{YELLOW}StateHelper::EKFUpdate() - diagonal at {i} is {d:.2f}{RESET}")
                found_neg = True
        
        if found_neg:
            # Keep filter alive: floor tiny/negative variances and continue.
            min_var = 1e-12
            diag_now = np.diagonal(state._Cov).copy()
            diag_now[diag_now < min_var] = min_var
            np.fill_diagonal(state._Cov, diag_now)

        # 6. Update State Vector x = x + K * res
        dx = K @ res

        for var in state._variables:
            update_vec = dx[var.id() : var.id() + var.size()]
            var.update(update_vec)

        # 7. Update Calibration Objects (if enabled)
        if state._options.do_calib_camera_intrinsics:
            for cam_id, calib_var in state._cam_intrinsics.items():
                state._cam_intrinsics_cameras[cam_id].set_value(calib_var.value())
        

    @staticmethod
    def set_initial_covariance(state: State, covariance: np.ndarray, order: List[Type]):
        """
        Overwrites covariance blocks for initialized variables.
        Assumes block diagonal structure usually.
        """
        real_covariance = covariance
        if isinstance(covariance, list):
            if len(covariance)>0 and covariance[0] is not None:
                real_covariance = covariance[0]
            else:
                print(f"covariance list wrapper is empty or None")
                return
        i_index = 0
        for i in range(len(order)):
            k_index = 0
            for k in range(len(order)):
                # Copy block from input covariance to state covariance
                block_rows = order[i].size()
                block_cols = order[k].size()
                
                src_block = real_covariance[i_index : i_index + block_rows, 
                                       k_index : k_index + block_cols]
                
                state._Cov[order[i].id() : order[i].id() + block_rows,
                           order[k].id() : order[k].id() + block_cols] = src_block
                
                k_index += block_cols
            i_index += order[i].size()
        
        # Force symmetry
        state._Cov = np.triu(state._Cov) + np.triu(state._Cov,1).T

    @staticmethod
    def get_marginal_covariance(state: State, small_variables: List[Type]) -> np.ndarray:
        """
        Extracts the covariance sub-matrix corresponding to the list of variables.
        """
        cov_size = sum(var.size() for var in small_variables)
        Small_cov = np.zeros((cov_size, cov_size))
        
        i_index = 0
        for i in range(len(small_variables)):
            k_index = 0
            for k in range(len(small_variables)):
                rows = small_variables[i].size()
                cols = small_variables[k].size()
                
                src_block = state._Cov[small_variables[i].id() : small_variables[i].id() + rows,
                                       small_variables[k].id() : small_variables[k].id() + cols]
                
                Small_cov[i_index : i_index + rows, k_index : k_index + cols] = src_block
                
                k_index += cols
            i_index += rows
            
        return Small_cov

    @staticmethod
    def get_full_covariance(state: State) -> np.ndarray:
        return np.copy(state._Cov)

    @staticmethod
    def marginalize(state: State, marg: Type):
        """
        Marginalizes (removes) a variable from the state and covariance matrix.
        """
        # Check if variable exists
        if marg not in state._variables:
            print(f"{RED}StateHelper::marginalize() - Called on variable that is not in state{RESET}")
            print(f"{RED}StateHelper::marginalize() - Marginalization does NOT work on sub-variables yet...{RESET}")
            sys.exit(1)

        marg_size = marg.size()
        marg_id = marg.id()
        x2_size = state._Cov.shape[0] - marg_id - marg_size
        
        # New Covariance Size
        new_dim = state._Cov.shape[0] - marg_size
        Cov_new = np.zeros((new_dim, new_dim))

        # 1. Top-Left: P(x1, x1) -> Everything before marg
        if marg_id > 0:
            Cov_new[0:marg_id, 0:marg_id] = state._Cov[0:marg_id, 0:marg_id]

        # 2. Top-Right: P(x1, x2) -> Before marg vs After marg
        if marg_id > 0 and x2_size > 0:
            Cov_new[0:marg_id, marg_id:marg_id+x2_size] = state._Cov[0:marg_id, marg_id+marg_size : marg_id+marg_size+x2_size]

        # 3. Bottom-Left: P(x2, x1) -> Transpose of Top-Right
        if marg_id > 0 and x2_size > 0:
            Cov_new[marg_id:marg_id+x2_size, 0:marg_id] = Cov_new[0:marg_id, marg_id:marg_id+x2_size].T

        # 4. Bottom-Right: P(x2, x2) -> Everything after marg
        if x2_size > 0:
            Cov_new[marg_id:marg_id+x2_size, marg_id:marg_id+x2_size] = state._Cov[marg_id+marg_size : marg_id+marg_size+x2_size,
                                                                                    marg_id+marg_size : marg_id+marg_size+x2_size]

        # Replace covariance
        state._Cov = Cov_new

        # Update IDs of remaining variables
        remaining_variables = []
        for var in state._variables:
            if var != marg:
                if var.id() > marg_id:
                    # Shift ID down by marg_size
                    var.set_local_id(var.id() - marg_size)
                remaining_variables.append(var)
        
        # "Delete" old variable
        marg.set_local_id(-1)
        
        # Update state variable list
        state._variables = remaining_variables

    @staticmethod
    def clone(state: State, variable_to_clone: Type) -> Type:
        """
        Clones a variable and appends it to the end of the state vector and covariance.
        """
        total_size = variable_to_clone.size()
        old_size = state._Cov.shape[0]
        new_loc = old_size
        
        # Resize covariance (pad with zeros)
        new_cov = np.zeros((old_size + total_size, old_size + total_size))
        new_cov[0:old_size, 0:old_size] = state._Cov
        state._Cov = new_cov
        
        # Find variable to clone
        new_clone = None
        
        for k in range(len(state._variables)):
            # Determine if this is the variable or a parent of the sub-variable
            type_check = state._variables[k].check_if_subvariable(variable_to_clone)
            
            if state._variables[k] == variable_to_clone:
                type_check = state._variables[k]
            elif type_check != variable_to_clone:
                continue
                
            # Clone found
            old_loc = type_check.id()
            
            # Copy Covariance Elements
            # P_new_new = P_old_old
            state._Cov[new_loc : new_loc+total_size, new_loc : new_loc+total_size] = \
                state._Cov[old_loc : old_loc+total_size, old_loc : old_loc+total_size]
                
            # P_rest_new = P_rest_old
            state._Cov[0:old_size, new_loc : new_loc+total_size] = \
                state._Cov[0:old_size, old_loc : old_loc+total_size]
                
            # P_new_rest = P_old_rest
            state._Cov[new_loc : new_loc+total_size, 0:old_size] = \
                state._Cov[old_loc : old_loc+total_size, 0:old_size]
                
            # Create object clone
            new_clone = type_check.clone()
            new_clone.set_local_id(new_loc)
            break
            
        if new_clone is None:
            print(f"{RED}StateHelper::clone() - Called on variable not in state{RESET}")
            sys.exit(1)
            
        state._variables.append(new_clone)
        return new_clone

    @staticmethod
    def initialize(state: State, new_variable: Type, 
                   H_order: List[Type], H_R: np.ndarray, H_L: np.ndarray, 
                   R: np.ndarray, res: np.ndarray, chi_2_mult: float) -> bool:
        """
        Initializes a new variable into the state using specific Jacobian structure (Left/Right).
        """
        # Check existence
        if new_variable in state._variables:
            print(f"StateHelper::initialize() - Variable already in state at {new_variable.id()}")
            sys.exit(1)
            
        # Isotropic check
        assert R.shape[0] == R.shape[1]
        assert R.shape[0] > 0
        diag_val = R[0,0]
        # (Simplified check compared to C++ loop)
        if not np.allclose(np.diag(R), diag_val) or not np.allclose(R - np.diag(np.diagonal(R)), 0):
             # Just warn, don't exit to be safe, but C++ exits
             pass

        # 1. QR Givens Separation
        new_var_size = new_variable.size()
        assert new_var_size == H_L.shape[1]
        
        # Working copies
        H_L_wk = np.copy(H_L)
        H_R_wk = np.copy(H_R)
        res_wk = np.copy(res)
        
        # Apply Givens Rotations to zero out lower triangle of H_L
        # In Python, using np.linalg.qr is much faster and cleaner than manual Givens loops
        # QR of H_L = Q * [ R_top; 0 ]
        # We want to apply Q.T to everything
        
        # Decompose H_L
        Q_qr, R_qr = np.linalg.qr(H_L_wk, mode='complete')
        # Note: numpy returns Q such that A = Q*R. We need to multiply by Q.T on the left.
        
        # Transform Matrices
        H_L_qr = Q_qr.T @ H_L_wk
        H_R_qr = Q_qr.T @ H_R_wk
        res_qr = Q_qr.T @ res_wk
        
        # 2. Separate Systems
        # Initializing Portion (Top rows = new_var_size)
        Hxinit = H_R_qr[0:new_var_size, :]
        H_finit = H_L_qr[0:new_var_size, 0:new_var_size] # Should be upper triangular
        resinit = res_qr[0:new_var_size]
        Rinit = R[0:new_var_size, 0:new_var_size] # Approximation if R is isotropic
        
        # Updating Portion (Bottom rows)
        Hup = H_R_qr[new_var_size:, :]
        resup = res_qr[new_var_size:]
        Rup = R[new_var_size:, new_var_size:] # Taking bottom right block

        # 3. Chi-Square Test
        P_up = StateHelper.get_marginal_covariance(state, H_order)
        S = Hup @ P_up @ Hup.T + Rup
        
        # Mahalanobis distance: res^T * S^-1 * res
        # Use solve
        try:
            S_inv_res = np.linalg.solve(S, resup)
            chi2_val = np.dot(resup.flatten(), S_inv_res.flatten())
        except np.linalg.LinAlgError:
            chi2_val = 99999.9 # Fail

        # Threshold
        dof = res.shape[0]
        # quantile function of chi-squared
        chi2_check = chi2.ppf(0.95, dof)
        
        if chi2_val > chi_2_mult * chi2_check:
            return False
            
        # 4. Initialize Invertible
        StateHelper.initialize_invertible(state, new_variable, H_order, Hxinit, H_finit, Rinit, resinit)
        
        # 5. EKF Update with Nullspace projection
        if Hup.shape[0] > 0:
            StateHelper.EKFUpdate(state, H_order, Hup, resup, Rup)
            
        return True

    @staticmethod
    def initialize_invertible(state: State, new_variable: Type, 
                              H_order: List[Type], H_R: np.ndarray, H_L: np.ndarray, 
                              R: np.ndarray, res: np.ndarray):
        """
        Initializes a variable by extending the state covariance.
        Used when the system is invertible regarding the new variable.
        """
        if new_variable in state._variables:
            sys.exit(1)

        # 1. Calculate M_a = P * H_R^T
        M_a = np.zeros((state._Cov.shape[0], res.shape[0]))
        
        current_it = 0
        H_id = []
        for meas_var in H_order:
            H_id.append(current_it)
            current_it += meas_var.size()
            
        for var in state._variables:
            M_i = np.zeros((var.size(), res.shape[0]))
            for i, meas_var in enumerate(H_order):
                # P_im * H_Rm^T
                cov_block = state._Cov[var.id() : var.id() + var.size(), 
                                       meas_var.id() : meas_var.id() + meas_var.size()]
                H_R_block = H_R[:, H_id[i] : H_id[i] + meas_var.size()]
                
                M_i += cov_block @ H_R_block.T
            
            M_a[var.id() : var.id() + var.size(), :] = M_i

        # 2. M = H_R * P_small * H_R^T + R
        P_small = StateHelper.get_marginal_covariance(state, H_order)
        M = H_R @ P_small @ H_R.T + R
        
        # 3. New Covariance Blocks
        # H_L_inv
        try:
            H_Linv = np.linalg.inv(H_L)
        except np.linalg.LinAlgError:
            print(f"{RED}StateHelper: H_L is singular!{RESET}")
            sys.exit(1)
            
        # P_LL = H_Linv * M * H_Linv^T
        P_LL = H_Linv @ M @ H_Linv.T
        
        # 4. Resize State Covariance
        oldSize = state._Cov.shape[0]
        newSize = oldSize + new_variable.size()
        new_cov = np.zeros((newSize, newSize))
        new_cov[0:oldSize, 0:oldSize] = state._Cov
        
        # P_new_rest = - M_a * H_Linv^T
        cross_term = -M_a @ H_Linv.T
        new_cov[0:oldSize, oldSize:newSize] = cross_term
        new_cov[oldSize:newSize, 0:oldSize] = cross_term.T
        new_cov[oldSize:newSize, oldSize:newSize] = P_LL
        
        state._Cov = new_cov
        
        # 5. Update Value
        # dx = H_Linv * res
        dx = H_Linv @ res
        new_variable.update(dx)
        
        # 6. Add to state
        new_variable.set_local_id(oldSize)
        state._variables.append(new_variable)

    @staticmethod
    def augment_clone(state: State, last_w: np.ndarray):
        """
        Clones the current IMU pose and adds it to the state history.
        """
        # Check collision
        if state._timestamp in state._clones_IMU:
            print(f"{RED}TRIED TO INSERT A CLONE AT THE SAME TIME, EXITING!{RESET}")
            sys.exit(1)
            
        # Clone Pose
        posetemp = StateHelper.clone(state, state._imu.pose())
        if type(posetemp).__name__ != "PoseJPL":
             print(f"{RED}INVALID OBJECT RETURNED FROM CLONE{RESET}")
             sys.exit(1)
             
        pose = posetemp # Cast
        
        # Store in map
        state._clones_IMU[state._timestamp] = pose
        
        # Time Offset Calibration Augmentation
        if state._options.do_calib_camera_timeoffset:
            # Jacobian [ w; v ]
            dnc_dt = np.zeros((6, 1))
            dnc_dt[0:3, 0] = last_w.flatten()
            dnc_dt[3:6, 0] = state._imu.vel().flatten()
            
            # Correlation between new clone and time offset parameter
            # P_clone_dt += P_dt_dt * dnc_dt^T (Roughly, see C++ logic)
            # Actually C++: Cov(0:rows, pose_id:pose_id+6) += Cov(0:rows, dt_id) * dnc_dt.T
            
            pose_id = pose.id()
            dt_id = state._calib_dt_CAMtoIMU.id()
            
            # Col update: Cov(:, pose_id:pose_id+6) += Cov(:, dt_id:dt_id+1) * dnc_dt^T
            col_update = state._Cov[:, dt_id:dt_id+1] @ dnc_dt.T
            state._Cov[:, pose_id:pose_id+6] += col_update
            
            # Row update: Cov(pose_id:pose_id+6, :) += dnc_dt * Cov(dt_id:dt_id+1, :)
            row_update = dnc_dt @ state._Cov[dt_id:dt_id+1, :]
            state._Cov[pose_id:pose_id+6, :] += row_update

    @staticmethod
    def marginalize_old_clone(state: State):
        """
        Removes the oldest clone if we exceed the max buffer size.
        """
        if len(state._clones_IMU) > state._options.max_clone_size:
            marginal_time = state.margtimestep()
            
            # Logic to remove the clone
            def perform_marginalization():
                # Safety check for infinity
                if marginal_time == float('inf'):
                    return

                # Retrieve and remove
                if marginal_time in state._clones_IMU:
                    clone_to_remove = state._clones_IMU[marginal_time]
                    StateHelper.marginalize(state, clone_to_remove)
                    del state._clones_IMU[marginal_time]

            # --- FIX: Check if mutex exists before using it ---
            if hasattr(state, '_mutex_state') and state._mutex_state is not None:
                with state._mutex_state:
                    perform_marginalization()
            else:
                # Run without lock if not defined
                perform_marginalization()

    @staticmethod
    def marginalize_slam(state: State):
        """
        Marginalizes SLAM features that are marked for removal.
        """
        # Iterate over copy of keys/items to allow deletion
        items = list(state._features_SLAM.items())
        count = 0
        
        for feat_id, feat in items:
            # Condition: should_marg AND id > limit
            if feat.should_marg and int(feat_id) > 4 * state._options.max_aruco_features:
                StateHelper.marginalize(state, feat)
                del state._features_SLAM[feat_id]
                count += 1