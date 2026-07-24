import numpy as np
import scipy.stats
import time
from typing import List, Dict, Optional, Any

# Imports
from vio_pipeline_v2.core import Feature
from vio_pipeline_v2.core import FeatureInitializer
from vio_pipeline_v2.msckf.State import State
from vio_pipeline_v2.msckf.state_helper import StateHelper
from vio_pipeline_v2.msckf.updater_helper import UpdaterHelper, UpdaterHelperFeature
from vio_pipeline_v2.type import Landmark, LandmarkRepresentation
# from utils.print import print_all, print_warning # Assuming print utils exist

class ClonePose:
    """Helper struct for Camera Poses (R_GtoC, p_CinG)"""
    def __init__(self, R, p):
        self.R = R
        self.p = p
    
    def Rot(self): return self.R
    def pos(self): return self.p

class UpdaterMSCKF:
    def __init__(self, options, feat_init_options):
        self._options = options
        
        # Save raw pixel noise squared
        self._options.sigma_pix_sq = self._options.sigma_pix ** 2

        # Initialize Feature Initializer
        self.initializer_feat = FeatureInitializer(feat_init_options)

        # Precompute Chi-Squared Table (Confidence 0.95)
        # We compute up to 500 degrees of freedom
        self.chi_squared_table = {}
        for i in range(1, 501):
            # ppf is the inverse cdf (quantile function)
            self.chi_squared_table[i] = scipy.stats.chi2.ppf(0.95, df=i)

    def update(self, state: State, feature_vec: List[Feature]):
        """
        Main MSCKF Update function.
        :param state: The current system state.
        :param feature_vec: List of features to process (will be modified/cleared).
        """
        
        # Return if no features
        if not feature_vec:
            return

        # Start timing
        rT0 = time.time()

        # 0. Get all timestamps our clones are at
        # state._clones_IMU is Dict[timestamp, PoseJPL]
        clonetimes = list(state._clones_IMU.keys())

        # 1. Clean feature measurements
        # Iterate backwards to allow safe removal
        ct_before_clean = len(feature_vec)
        
        i = len(feature_vec) - 1
        while i >= 0:
            feat = feature_vec[i]
            
            # Remove measurements not in current clone window
            feat.clean_old_measurements(clonetimes)
            
            # Count measurements
            ct_meas = sum(len(times) for times in feat.timestamps.values())
            
            # Remove if insufficient
            if ct_meas < 2:
                feat.to_delete = True
                del feature_vec[i]
            
            i -= 1

        rT1 = time.time()

        # 2. Create vector of cloned *CAMERA* poses
        # Dict[cam_id, Dict[timestamp, ClonePose]]
        clones_cam = {}
        
        for cam_id, clone_calib in state._calib_IMUtoCAM.items():
            clones_cami = {}
            
            for ts, clone_imu in state._clones_IMU.items():
                # R_GtoCi = R_ItoC * R_GtoI
                # Note: clone_calib is R_ItoC (IMU to Cam)
                R_GtoCi = clone_calib.Rot() @ clone_imu.Rot()
                
                # p_CioinG = p_IinG - R_GtoCi^T * p_CinI
                # p_CinI is clone_calib.pos()
                p_CioinG = clone_imu.pos() - R_GtoCi.T @ clone_calib.pos()
                
                clones_cami[ts] = ClonePose(R_GtoCi, p_CioinG)
            
            clones_cam[cam_id] = clones_cami

        # 3. Triangulate features
        ct_before_tri = len(feature_vec)
        i = len(feature_vec) - 1
        while i >= 0:
            feat = feature_vec[i]
            
            # Triangulate
            if self.initializer_feat._options.triangulate_1d:
                success_tri = self.initializer_feat.single_triangulation_1d(feat, clones_cam)
            else:
                success_tri = self.initializer_feat.single_triangulation(feat, clones_cam)
            
            # Gauss-Newton Refine
            success_refine = True
            if self.initializer_feat._options.refine_features:
                success_refine = self.initializer_feat.single_gaussnewton(feat, clones_cam)
            
            if not success_tri or not success_refine:
                feat.to_delete = True
                del feature_vec[i]
            
            i -= 1

        rT2 = time.time()
        ct_after_tri = len(feature_vec)

        # Skip numerically weak batches early (typical when tracker temporarily loses scene texture).
        if ct_after_tri < max(1, int(self._options.min_features_for_update)):
            for f in feature_vec:
                f.to_delete = True
            print(f"[MSCKF-UP][SKIP] too_few_features tri={ct_after_tri} "
                  f"min={self._options.min_features_for_update}")
            return

        # Calculate max sizes for memory reservation
        max_meas_size = 0
        for feat in feature_vec:
            for times in feat.timestamps.values():
                max_meas_size += 2 * len(times)

        # Max state size (Total covariance - SLAM features which aren't used here)
        # This logic assumes MSCKF doesn't update SLAM landmarks in the covariance directly in this function
        max_hx_size = state.max_covariance_size() 
        for landmark in state._features_SLAM.values():
            max_hx_size -= landmark.size()

        # Large Jacobian (Hx_big) and Residual (res_big)
        # Note: In Python we usually build list of blocks and stack them, rather than pre-allocating zeros
        # but following C++ logic for structure.
        
        hx_blocks = [] # Will store (H_block, variable) tuples
        res_blocks = []
        
        # Mapping to keep track of unique state variables involved
        # Dict[VariableID, ColumnIndex]
        hx_mapping = {}
        hx_order_big = []
        ct_jacob = 0
        ct_chi2_reject = 0
        
        # 4. Compute Linear System, Nullspace, Reject
        i = 0
        while i < len(feature_vec):
            feat_ptr = feature_vec[i]
            
            # Convert to Helper Feature Struct
            feat = UpdaterHelperFeature()
            feat.featid = feat_ptr.featid
            feat.uvs = feat_ptr.uvs
            feat.uvs_norm = feat_ptr.uvs_norm
            feat.timestamps = feat_ptr.timestamps
            
            # Representation Logic
            feat.feat_representation = state._options.feat_rep_msckf
            if state._options.feat_rep_msckf == LandmarkRepresentation.Representation.ANCHORED_INVERSE_DEPTH_SINGLE:
                feat.feat_representation = LandmarkRepresentation.Representation.ANCHORED_MSCKF_INVERSE_DEPTH

            # Copy State (Standard vs FEJ)
            # Python assignment is ref, so we might need copy if modifying? 
            # Usually get_feature_jacobian just reads.
            if LandmarkRepresentation.is_relative_representation(feat.feat_representation):
                feat.anchor_cam_id = feat_ptr.anchor_cam_id
                feat.anchor_clone_timestamp = feat_ptr.anchor_clone_timestamp
                feat.p_FinA = feat_ptr.p_FinA
                feat.p_FinA_fej = feat_ptr.p_FinA # Assuming no separate FEJ storage for features in this object
            else:
                feat.p_FinG = feat_ptr.p_FinG
                feat.p_FinG_fej = feat_ptr.p_FinG

            # Get Jacobian 
            # H_f (Feature Jacobian), H_x (State Jacobian), res (Residual), Hx_order (Variables)
            H_f, H_x, res, Hx_order = UpdaterHelper.get_feature_jacobian_full(state, feat)

            # Nullspace Project (Removes H_f)
            # H_x and res are modified in-place to become H_o and r_o
            H_x, res = UpdaterHelper.nullspace_project_inplace(H_f, H_x, res)

            # Degenerate projection can produce empty or non-finite constraints.
            if H_x.shape[0] == 0 or res.shape[0] == 0:
                feat_ptr.to_delete = True
                del feature_vec[i]
                continue
            if not np.all(np.isfinite(H_x)) or not np.all(np.isfinite(res)):
                feat_ptr.to_delete = True
                del feature_vec[i]
                continue

            # Chi2 Check
            # S = H * P * H^T + R
            P_marg = StateHelper.get_marginal_covariance(state, Hx_order)
            
            S = H_x @ P_marg @ H_x.T
            # Add measurement noise to diagonal
            np.fill_diagonal(S, S.diagonal() + self._options.sigma_pix_sq)
            S = (S + S.T) * 0.5

            # Reject numerically unstable feature systems before chi2 test.
            try:
                cond_S = np.linalg.cond(S)
            except np.linalg.LinAlgError:
                cond_S = float('inf')
            if (not np.isfinite(cond_S)) or cond_S > float(self._options.max_innovation_condition):
                ct_chi2_reject += 1
                feat_ptr.to_delete = True
                del feature_vec[i]
                continue
            
            # Mahalanobis distance: res^T * S^-1 * res
            try:
                # Solve S * x = res, then dot res * x
                chi2 = np.dot(res.T, np.linalg.solve(S, res))
                if isinstance(chi2, np.ndarray): chi2 = chi2.item()
            except np.linalg.LinAlgError:
                chi2 = float('inf')

            # Threshold
            dof = res.shape[0]
            if dof < 500:
                chi2_check = self.chi_squared_table.get(dof, 1000.0)
            else:
                chi2_check = scipy.stats.chi2.ppf(0.95, df=dof)
                print(f"[WARN]: chi2_check over limit {dof}")

            if chi2 > self._options.chi2_multipler * chi2_check:
                ct_chi2_reject += 1
                feat_ptr.to_delete = True
                del feature_vec[i]
                continue
            
            # Valid Feature -> Append to system
            
            # Align H_x columns to Global Hx_big structure
            # We construct a sparse-like entry for this feature row
            
            # Map local variables to global columns
            row_start = len(res_blocks)
            # H_x is (rows, cols). 
            
            # For this feature, we create a list of blocks mapped to specific global variables
            # But simpler approach: Map everything to a growing dense matrix or list of blocks
            
            # Let's map Hx_order variables to global indices
            feature_H_blocks = [] # List of (col_idx, matrix_block)
            
            col_offset = 0
            for var in Hx_order:
                if var not in hx_mapping:
                    hx_mapping[var] = ct_jacob
                    hx_order_big.append(var)
                    ct_jacob += var.size()
                
                global_col = hx_mapping[var]
                block = H_x[:, col_offset : col_offset + var.size()]
                feature_H_blocks.append((global_col, block))
                col_offset += var.size()

            hx_blocks.append(feature_H_blocks)
            res_blocks.append(res)
            
            i += 1

        rT3 = time.time()

        # Mark all as processed
        for f in feature_vec:
            f.to_delete = True

        if not res_blocks:
            return

        # Stack the Big System
        total_rows = sum(r.shape[0] for r in res_blocks)
        total_cols = ct_jacob
        
        Hx_big = np.zeros((total_rows, total_cols))
        res_big = np.zeros(total_rows)
        
        current_row = 0
        for idx, res in enumerate(res_blocks):
            rows = res.shape[0]
            res_big[current_row : current_row+rows] = res if res.ndim==1 else res.flatten()
            
            # Fill H matrix
            for global_col, block in hx_blocks[idx]:
                cols = block.shape[1]
                Hx_big[current_row : current_row+rows, global_col : global_col+cols] = block
            
            current_row += rows

        # 5. Measurement Compression (QR)
        # Decomposes Hx_big into Q*R, updates res_big
        Hx_big, res_big = UpdaterHelper.measurement_compress_inplace(Hx_big, res_big)
        
        if Hx_big.shape[0] < 1:
            return

        if Hx_big.shape[0] < max(1, int(self._options.min_update_rows)):
            print(f"[MSCKF-UP][SKIP] too_few_rows rows={Hx_big.shape[0]} "
                  f"min={self._options.min_update_rows} feats={len(res_blocks)}")
            return

        if not np.all(np.isfinite(Hx_big)) or not np.all(np.isfinite(res_big)):
            print("[MSCKF-UP][SKIP] non_finite_system")
            return

        rT4 = time.time()

        # Noise Matrix (Isotropic after compression)
        R_big = self._options.sigma_pix_sq * np.eye(res_big.shape[0])

        print(f"[MSCKF-UP] feats={len(res_blocks)} tri={ct_after_tri} chi2_rej={ct_chi2_reject} "
              f"Hx={Hx_big.shape}")

        # 6. EKF Update
        StateHelper.EKFUpdate(state, hx_order_big, Hx_big, res_big, R_big)

        rT5 = time.time()

        # Timing stats (Printing)
        # (verbose timing removed for performance)