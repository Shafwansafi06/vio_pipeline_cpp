import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional

# Imports
from vio_pipeline_v2.msckf.State import State
from vio_pipeline_v2.type import LandmarkRepresentation
from vio_pipeline_v2.type import skew_x, quat_2_Rot
# Assuming PoseJPL and other types have .size(), .Rot(), .pos() methods

class UpdaterHelperFeature:
    """Helper struct to hold feature data during update."""
    def __init__(self):
        self.featid = -1
        self.uvs = {}        # Dict[cam_id, List[np.ndarray]]
        self.uvs_norm = {}   # Dict[cam_id, List[np.ndarray]]
        self.timestamps = {} # Dict[cam_id, List[float]]
        
        self.feat_representation = LandmarkRepresentation.Representation.GLOBAL_3D
        
        self.anchor_cam_id = -1
        self.anchor_clone_timestamp = 0.0
        
        self.p_FinA = np.zeros(3)
        self.p_FinA_fej = np.zeros(3)
        
        self.p_FinG = np.zeros(3)
        self.p_FinG_fej = np.zeros(3)

class UpdaterHelper:
    
    @staticmethod
    def get_feature_jacobian_representation(state: State, feature: UpdaterHelperFeature) -> Tuple[np.ndarray, List[np.ndarray], List[Any]]:
        """
        Computes the Jacobian of the feature position in the Global frame (p_FinG) 
        with respect to the feature representation parameters (lambda) and the state (x).
        
        Returns:
            H_f (np.ndarray): Jacobian w.r.t feature parameters (3xN)
            H_x (List[np.ndarray]): List of Jacobians w.r.t state variables
            x_order (List[Any]): List of state variable objects corresponding to H_x
        """
        H_x = []
        x_order = []

        # 1. Global XYZ
        if feature.feat_representation == LandmarkRepresentation.Representation.GLOBAL_3D:
            H_f = np.eye(3)
            return H_f, H_x, x_order

        # 2. Global Inverse Depth
        if feature.feat_representation == LandmarkRepresentation.Representation.GLOBAL_FULL_INVERSE_DEPTH:
            p_FinG = feature.p_FinG_fej if state._options.do_fej else feature.p_FinG
            
            # Spherical to Cartesian Jacobian
            # Param: [theta, phi, rho]
            # p = 1/rho * [cos(phi)cos(theta), cos(phi)sin(theta), sin(phi)]
            # Actually C++ implementation:
            # g_theta = atan2(y, x), g_phi = acos(rho * z), g_rho = 1/norm
            # Standard Spherical conversion derivatives
            
            # Reconstruct params
            norm = np.linalg.norm(p_FinG)
            rho = 1.0 / norm
            # Note: C++ uses acos(rho*z) for phi? Usually phi is elevation. 
            # Following C++ Jacobian structure:
            
            p_inv = np.zeros(3)
            p_inv[0] = math.atan2(p_FinG[1], p_FinG[0]) # theta
            p_inv[1] = math.acos(rho * p_FinG[2])       # phi
            p_inv[2] = rho

            sin_th = math.sin(p_inv[0]); cos_th = math.cos(p_inv[0])
            sin_phi = math.sin(p_inv[1]); cos_phi = math.cos(p_inv[1])
            rho = p_inv[2]

            H_f = np.zeros((3, 3))
            # d(x)/d(theta, phi, rho)
            H_f[0,0] = -(1.0/rho)*sin_th*sin_phi
            H_f[0,1] = (1.0/rho)*cos_th*cos_phi
            H_f[0,2] = -(1.0/(rho**2))*cos_th*sin_phi
            
            H_f[1,0] = (1.0/rho)*cos_th*sin_phi
            H_f[1,1] = (1.0/rho)*sin_th*cos_phi
            H_f[1,2] = -(1.0/(rho**2))*sin_th*sin_phi
            
            H_f[2,0] = 0.0
            H_f[2,1] = -(1.0/rho)*sin_phi
            H_f[2,2] = -(1.0/(rho**2))*cos_phi
            
            return H_f, H_x, x_order

        # --- Anchored Representations ---
        
        assert feature.anchor_cam_id != -1
        
        # Get Anchor Data
        clone_anchor = state._clones_IMU[feature.anchor_clone_timestamp]
        calib_anchor = state._calib_IMUtoCAM[feature.anchor_cam_id]
        
        R_ItoC = calib_anchor.Rot()
        p_IinC = calib_anchor.pos()
        R_GtoI = clone_anchor.Rot()
        p_IinG = clone_anchor.pos()
        p_FinA = feature.p_FinA

        # FEJ Logic for Anchor
        if state._options.do_fej:
            # Best Estimate Feature in Global
            # p_FinG_best = R_GtoI^T * R_ItoC^T * (p_FinA - p_IinC) + p_IinG
            R_GtoI_best = clone_anchor.Rot()
            p_IinG_best = clone_anchor.pos()
            p_FinG_best = R_GtoI_best.T @ R_ItoC.T @ (feature.p_FinA - p_IinC) + p_IinG_best
            
            # Transform back using FEJ states
            R_GtoI = clone_anchor.Rot_fej()
            p_IinG = clone_anchor.pos_fej()
            # p_FinA = (R_GtoI^T * R_ItoC^T)^T * (p_FinG - p_IinG) + p_IinC
            R_CtoG = R_GtoI.T @ R_ItoC.T
            p_FinA = R_CtoG.T @ (p_FinG_best - p_IinG) + p_IinC

        # R_CtoG = R_ItoC * R_GtoI (Transposed logic in C++: R_GtoI^T * R_ItoC^T)
        # C++: R_CtoG = R_GtoI.transpose() * R_ItoC.transpose(); -> This is Rotation from Camera to Global
        R_CtoG = R_GtoI.T @ R_ItoC.T

        # 1. Jacobian w.r.t Anchor Clone (IMU Pose)
        # H_anc = [ -R_GtoI^T * skew(R_ItoC^T * (p_FinA - p_IinC))  |  I ]
        term = R_ItoC.T @ (p_FinA - p_IinC)
        H_anc = np.zeros((3, 6))
        H_anc[:, 0:3] = -R_GtoI.T @ skew_x(term)
        H_anc[:, 3:6] = np.eye(3)
        
        x_order.append(clone_anchor)
        H_x.append(H_anc)

        # 2. Jacobian w.r.t Anchor Calibration
        if state._options.do_calib_camera_pose:
            # H_calib = [ -R_CtoG * skew(p_FinA - p_IinC) | -R_CtoG ]
            H_calib = np.zeros((3, 6))
            H_calib[:, 0:3] = -R_CtoG @ skew_x(p_FinA - p_IinC)
            H_calib[:, 3:6] = -R_CtoG
            
            x_order.append(calib_anchor)
            H_x.append(H_calib)

        # 3. Jacobian w.r.t Feature Representation (H_f)
        
        # Case: Anchored 3D
        if feature.feat_representation == LandmarkRepresentation.Representation.ANCHORED_3D:
            H_f = R_CtoG
            return H_f, H_x, x_order

        # Case: Anchored Full Inverse Depth (Spherical)
        if feature.feat_representation == LandmarkRepresentation.Representation.ANCHORED_FULL_INVERSE_DEPTH:
            norm = np.linalg.norm(p_FinA)
            rho = 1.0 / norm
            theta = math.atan2(p_FinA[1], p_FinA[0])
            phi = math.acos(rho * p_FinA[2])
            
            sin_th = math.sin(theta); cos_th = math.cos(theta)
            sin_phi = math.sin(phi); cos_phi = math.cos(phi)
            
            d_pfinA_dpinv = np.zeros((3, 3))
            d_pfinA_dpinv[0,0] = -(1/rho)*sin_th*sin_phi
            d_pfinA_dpinv[0,1] = (1/rho)*cos_th*cos_phi
            d_pfinA_dpinv[0,2] = -(1/rho**2)*cos_th*sin_phi
            
            d_pfinA_dpinv[1,0] = (1/rho)*cos_th*sin_phi
            d_pfinA_dpinv[1,1] = (1/rho)*sin_th*cos_phi
            d_pfinA_dpinv[1,2] = -(1/rho**2)*sin_th*sin_phi
            
            d_pfinA_dpinv[2,0] = 0.0
            d_pfinA_dpinv[2,1] = -(1/rho)*sin_phi
            d_pfinA_dpinv[2,2] = -(1/rho**2)*cos_phi
            
            H_f = R_CtoG @ d_pfinA_dpinv
            return H_f, H_x, x_order

        # Case: Anchored MSCKF Inverse Depth (alpha, beta, rho)
        # u = x/z, v = y/z, rho = 1/z
        if feature.feat_representation == LandmarkRepresentation.Representation.ANCHORED_MSCKF_INVERSE_DEPTH:
            alpha = p_FinA[0] / p_FinA[2]
            beta = p_FinA[1] / p_FinA[2]
            rho = 1.0 / p_FinA[2]
            
            d_pfinA_dpinv = np.zeros((3, 3))
            # d(x)/d(alpha) = 1/rho
            # d(x)/d(rho) = -alpha/rho^2
            d_pfinA_dpinv[0,0] = 1.0/rho
            d_pfinA_dpinv[0,2] = -(1.0/(rho**2)) * alpha
            
            d_pfinA_dpinv[1,1] = 1.0/rho
            d_pfinA_dpinv[1,2] = -(1.0/(rho**2)) * beta
            
            d_pfinA_dpinv[2,2] = -(1.0/(rho**2))
            
            H_f = R_CtoG @ d_pfinA_dpinv
            return H_f, H_x, x_order

        # Case: Single Depth (1D)
        # p = (1/rho) * bearing
        if feature.feat_representation == LandmarkRepresentation.Representation.ANCHORED_INVERSE_DEPTH_SINGLE:
            rho = 1.0 / p_FinA[2]
            bearing = rho * p_FinA # Normalized direction
            
            # Jacobian of p w.r.t rho
            d_pfinA_drho = -(1.0 / (rho**2)) * bearing
            
            H_f = R_CtoG @ d_pfinA_drho.reshape(3, 1) # Result is 3x1
            return H_f, H_x, x_order

        raise RuntimeError(f"Invalid representation: {feature.feat_representation}")

    @staticmethod
    def get_feature_jacobian_full(state: State, feature: UpdaterHelperFeature) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
        """
        Computes the full Jacobian chain: residual -> feature -> state.
        
        Returns:
            H_f: Jacobian w.r.t feature (2*M x K)
            H_x: Jacobian w.r.t state (2*M x L)
            res: Residual vector (2*M)
            x_order: List of state variables in H_x
        """
        
        # 1. Collect State Variables involved
        map_hx = {} # obj -> col_index
        x_order = []
        total_hx = 0
        
        # Helper to add variable
        def add_var(var):
            nonlocal total_hx
            if var not in map_hx:
                map_hx[var] = total_hx
                x_order.append(var)
                total_hx += var.size()

        # Calibration
        for cam_id in feature.timestamps.keys():
            if state._options.do_calib_camera_pose:
                add_var(state._calib_IMUtoCAM[cam_id])
            if state._options.do_calib_camera_intrinsics:
                add_var(state._cam_intrinsics[cam_id])
            
            # Clones
            for ts in feature.timestamps[cam_id]:
                add_var(state._clones_IMU[ts])

        # Anchor
        if LandmarkRepresentation.is_relative_representation(feature.feat_representation):
            add_var(state._clones_IMU[feature.anchor_clone_timestamp])
            if state._options.do_calib_camera_pose:
                add_var(state._calib_IMUtoCAM[feature.anchor_cam_id])

        # 2. Calculate Feature Position in Global (p_FinG)
        p_FinG = feature.p_FinG
        p_FinG_fej = feature.p_FinG_fej
        
        if LandmarkRepresentation.is_relative_representation(feature.feat_representation):
            R_ItoC = state._calib_IMUtoCAM[feature.anchor_cam_id].Rot()
            p_IinC = state._calib_IMUtoCAM[feature.anchor_cam_id].pos()
            R_GtoI = state._clones_IMU[feature.anchor_clone_timestamp].Rot()
            p_IinG = state._clones_IMU[feature.anchor_clone_timestamp].pos()
            
            # p_FinG = R_GtoI^T * R_ItoC^T * (p_FinA - p_IinC) + p_IinG
            p_FinG = R_GtoI.T @ R_ItoC.T @ (feature.p_FinA - p_IinC) + p_IinG
            p_FinG_fej = p_FinG # Approximation or recalculate with FEJ values if strictly needed

        # 3. Compute H_f base (dp_FinG / d_lambda) and d_state (dp_FinG / d_state)
        # This part comes from the representation Jacobian helper
        dpfg_dlambda, dpfg_dx, dpfg_dx_order = UpdaterHelper.get_feature_jacobian_representation(state, feature)

        # 4. Allocate Matrices
        total_meas = sum(len(times) for times in feature.timestamps.values())
        jacob_size = 1 if feature.feat_representation == LandmarkRepresentation.Representation.ANCHORED_INVERSE_DEPTH_SINGLE else 3
        
        H_f = np.zeros((2 * total_meas, jacob_size))
        H_x = np.zeros((2 * total_meas, total_hx))
        res = np.zeros(2 * total_meas)

        # 5. Iterate Measurements
        c = 0
        for cam_id, timestamps in feature.timestamps.items():
            
            calibration = state._calib_IMUtoCAM[cam_id]
            distortion = state._cam_intrinsics[cam_id]
            R_ItoC = calibration.Rot()
            p_IinC = calibration.pos()

            for m, ts in enumerate(timestamps):
                
                clone_Ii = state._clones_IMU[ts]
                R_GtoIi = clone_Ii.Rot()
                p_IiinG = clone_Ii.pos()

                # Current Feature in IMU Frame
                p_FinIi = R_GtoIi @ (p_FinG - p_IiinG)
                
                # Current Feature in Camera Frame
                p_FinCi = R_ItoC @ p_FinIi + p_IinC
                
                # Normalize
                uv_norm = np.array([p_FinCi[0]/p_FinCi[2], p_FinCi[1]/p_FinCi[2]])
                
                # Distort
                uv_dist = state._cam_intrinsics_cameras[cam_id].distort_d(uv_norm)
                
                # Residual
                uv_m = feature.uvs[cam_id][m]
                res[2*c : 2*c+2] = uv_m - uv_dist

                # --- FEJ Logic ---
                if state._options.do_fej:
                    R_GtoIi = clone_Ii.Rot_fej()
                    p_IiinG = clone_Ii.pos_fej()
                    # Recalculate p_FinCi based on FEJ
                    p_FinIi = R_GtoIi @ (p_FinG_fej - p_IiinG)
                    p_FinCi = R_ItoC @ p_FinIi + p_IinC

                # Jacobian: d(res)/d(normalized) * d(normalized)/d(camera_pt)
                
                # 1. d(distorted)/d(normalized) & d(dist)/d(intrinsics)
                dz_dzn, dz_dzeta = state._cam_intrinsics_cameras[cam_id].compute_distort_jacobian(uv_norm)
                
                # 2. d(normalized)/d(p_FinCi)
                z = p_FinCi[2]
                dzn_dpfc = np.array([
                    [1/z, 0, -p_FinCi[0]/(z*z)],
                    [0, 1/z, -p_FinCi[1]/(z*z)]
                ])

                # 3. d(p_FinCi)/d(p_FinG)
                # p_FinCi = R_ItoC * R_GtoIi * (p_FinG - p_IiinG) + p_IinC
                dpfc_dpfg = R_ItoC @ R_GtoIi

                # 4. d(p_FinCi)/d(Clone)
                # d/dtheta = R_ItoC * skew(R_GtoIi * (p_FinG - p_IiinG))
                #          = R_ItoC * skew(p_FinIi)
                # d/dp = -R_ItoC * R_GtoIi
                dpfc_dclone = np.zeros((3, 6))
                dpfc_dclone[:, 0:3] = R_ItoC @ skew_x(p_FinIi)
                dpfc_dclone[:, 3:6] = -dpfc_dpfg

                # --- Chain Rule Assembly ---
                
                dz_dpfc = dz_dzn @ dzn_dpfc
                dz_dpfg = dz_dpfc @ dpfc_dpfg # 2x3

                # H_f (Feature) = dz/dp_FinG * dp_FinG/d_lambda
                H_f[2*c:2*c+2, :] = dz_dpfg @ dpfg_dlambda

                # H_x (Clone)
                col_idx = map_hx[clone_Ii]
                H_x[2*c:2*c+2, col_idx:col_idx+6] = dz_dpfc @ dpfc_dclone

                # H_x (Anchor components via dp_FinG/d_state)
                for i_dx, var_dx in enumerate(dpfg_dx_order):
                    col_idx = map_hx[var_dx]
                    H_x[2*c:2*c+2, col_idx:col_idx+var_dx.size()] += dz_dpfg @ dpfg_dx[i_dx]

                # H_x (Calibration Extrinsics)
                if state._options.do_calib_camera_pose:
                    # d(p_FinCi)/d(theta_cal) = skew(p_FinCi - p_IinC) ?? 
                    # Logic: p_c = R * p_i + p
                    # d/dth = -skew(p_c - p) ? Check logic
                    # C++: skew(p_FinCi - p_IinC)
                    dpfc_dcalib = np.zeros((3, 6))
                    dpfc_dcalib[:, 0:3] = skew_x(p_FinCi - p_IinC)
                    dpfc_dcalib[:, 3:6] = np.eye(3)
                    
                    col_idx = map_hx[calibration]
                    H_x[2*c:2*c+2, col_idx:col_idx+6] += dz_dpfc @ dpfc_dcalib

                # H_x (Calibration Intrinsics)
                if state._options.do_calib_camera_intrinsics:
                    col_idx = map_hx[distortion]
                    H_x[2*c:2*c+2, col_idx:col_idx+distortion.size()] = dz_dzeta

                c += 1

        return H_f, H_x, res, x_order

    @staticmethod
    def nullspace_project_inplace(H_f: np.ndarray, H_x: np.ndarray, res: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects the linear system onto the left nullspace of H_f.
        Uses full QR: H_f = Q * [R; 0].  Q2 (cols K..M) spans the left nullspace.
        Apply Q2.T to H_x and res to eliminate the feature Jacobian.
        Returns (H_x_proj, res_proj).
        """
        rows, cols = H_f.shape
        if rows <= cols:
            return np.zeros((0, H_x.shape[1])), np.zeros(0)

        # Full QR: Q is (rows x rows), columns cols..rows span the left nullspace of H_f
        Q, _ = np.linalg.qr(H_f, mode='complete')
        Q_null = Q[:, cols:]   # shape (rows, rows-cols)

        H_x_proj = Q_null.T @ H_x
        res_proj  = Q_null.T @ res

        return H_x_proj, res_proj

    @staticmethod
    def measurement_compress_inplace(H_x: np.ndarray, res: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compresses the measurement matrix using Givens rotations (QR).
        Mirrors C++ implementation exactly:
          - Givens rotations zero out rows below the diagonal (Golub & Van Loan, Alg 5.2.4)
          - Crops to r = min(rows, cols) rows after triangularization
        If rows <= cols (fat matrix), returns unchanged (no compression needed).
        """
        rows, cols = H_x.shape
        if rows <= cols:
            return H_x, res

        # Apply Givens rotations column by column, bottom-up (matches C++ loops exactly).
        # For column n, eliminate entries in rows m = rows-1 down to n+1
        # by applying a 2x2 Givens rotation to rows (m-1, m).
        H = H_x.copy()
        r = res.copy()

        for n in range(cols):
            for m in range(rows - 1, n, -1):
                # Compute Givens rotation G that zeros H[m, n] using H[m-1, n]
                a = H[m - 1, n]
                b = H[m, n]
                if b == 0.0:
                    continue
                norm_ab = math.hypot(a, b)
                # C++ makeGivens(a, b) gives G s.t. G * [a; b] = [r; 0]
                # with c = a/r, s = b/r.  applyOnTheLeft with adjoint applies [c s; -s c].
                c = a / norm_ab
                s = b / norm_ab

                # Apply to rows (m-1) and m of H, columns n..cols-1
                H_m1 = H[m - 1, n:].copy()
                H_m  = H[m,     n:].copy()
                H[m - 1, n:] =  c * H_m1 + s * H_m
                H[m,     n:] = -s * H_m1 + c * H_m

                # Apply to res
                r_m1 = r[m - 1]
                r_m  = r[m]
                r[m - 1] =  c * r_m1 + s * r_m
                r[m]     = -s * r_m1 + c * r_m

        # Crop to r = min(rows, cols) rows — the triangular part
        keep = min(rows, cols)
        return H[:keep, :], r[:keep]