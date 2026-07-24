import numpy as np
import math
from typing import Dict, Any

# Imports
from vio_pipeline_v2.core.feature import Feature
from vio_pipeline_v2.type.quat_ops import skew_x  # <--- Imported from your existing library

class FeatureInitializer:
    """
    Class for initializing features (triangulation).
    Contains methods for Linear Triangulation, 1D Triangulation, and Gauss-Newton optimization.
    """
    def __init__(self, options):
        self._options = options

    def single_triangulation(self, feat: Feature, clonesCAM: Dict[int, Dict[float, Any]]) -> bool:
        """
        Standard linear triangulation (DLT-like approach) for a single feature.
        Solves A*x = b for the feature position in the anchor frame.
        """
        
        # 1. Determine Anchor Frame (Camera with most measurements)
        anchor_most_meas = -1
        most_meas = 0
        total_meas = 0
        
        for cam_id, times in feat.timestamps.items():
            count = len(times)
            total_meas += count
            if count > most_meas:
                anchor_most_meas = cam_id
                most_meas = count
        
        feat.anchor_cam_id = anchor_most_meas
        feat.anchor_clone_timestamp = feat.timestamps[feat.anchor_cam_id][-1]

        # 2. Setup Linear System
        A = np.zeros((3, 3))
        b = np.zeros(3)

        # Get Anchor Pose
        anchor_clone = clonesCAM[feat.anchor_cam_id][feat.anchor_clone_timestamp]
        R_GtoA = anchor_clone.Rot()
        p_AinG = anchor_clone.pos()

        # 3. Loop through all measurements to build system
        for cam_id, times in feat.timestamps.items():
            for m, timestamp in enumerate(times):
                
                # Get Clone Pose
                curr_clone = clonesCAM[cam_id][timestamp]
                R_GtoCi = curr_clone.Rot()
                p_CiinG = curr_clone.pos()

                # Calculate Relative Pose (Anchor -> Current)
                R_AtoCi = R_GtoCi @ R_GtoA.T
                p_CiinA = R_GtoA @ (p_CiinG - p_AinG)

                # Get Normalized UV measurement
                uv_norm = feat.uvs_norm[cam_id][m]
                b_i = np.array([uv_norm[0], uv_norm[1], 1.0])
                
                # Rotate bearing into Anchor frame
                b_i = R_AtoCi.T @ b_i
                b_i = b_i / np.linalg.norm(b_i)
                
                Bperp = skew_x(b_i)

                # Append to linear system (Normal Equations: A^T A x = A^T b)
                # Here Ai = Bperp^T * Bperp
                Ai = Bperp.T @ Bperp
                A += Ai
                b += Ai @ p_CiinA

        # 4. Solve Linear System
        try:
            p_f = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return False

        # 5. Check Condition Number
        try:
            U, S, Vh = np.linalg.svd(A)
            if S[-1] == 0:
                condA = float('inf')
            else:
                condA = S[0] / S[-1]
        except:
            return False

        # Validity Checks
        if (abs(condA) > self._options.max_cond_number or 
            p_f[2] < self._options.min_dist or 
            p_f[2] > self._options.max_dist or 
            np.isnan(np.linalg.norm(p_f))):
            return False

        # 6. Store Result
        feat.p_FinA = p_f
        feat.p_FinG = R_GtoA.T @ feat.p_FinA + p_AinG
        
        return True

    def single_triangulation_1d(self, feat: Feature, clonesCAM: Dict[int, Dict[float, Any]]) -> bool:
        """
        1D Triangulation (Depth estimation along a known bearing).
        """
        
        # 1. Determine Anchor Frame
        anchor_most_meas = -1
        most_meas = 0
        for cam_id, times in feat.timestamps.items():
            count = len(times)
            if count > most_meas:
                anchor_most_meas = cam_id
                most_meas = count
        
        feat.anchor_cam_id = anchor_most_meas
        feat.anchor_clone_timestamp = feat.timestamps[feat.anchor_cam_id][-1]
        idx_anchor_bearing = len(feat.timestamps[feat.anchor_cam_id]) - 1

        # 2. Setup Linear System (Scalar)
        A = 0.0
        b = 0.0

        # Anchor Pose
        anchor_clone = clonesCAM[feat.anchor_cam_id][feat.anchor_clone_timestamp]
        R_GtoA = anchor_clone.Rot()
        p_AinG = anchor_clone.pos()

        # Bearing vector in Anchor Frame
        uv_anchor = feat.uvs_norm[feat.anchor_cam_id][idx_anchor_bearing]
        bearing_inA = np.array([uv_anchor[0], uv_anchor[1], 1.0])
        bearing_inA = bearing_inA / np.linalg.norm(bearing_inA)

        # 3. Loop through measurements
        for cam_id, times in feat.timestamps.items():
            for m, timestamp in enumerate(times):
                
                # Skip the anchor measurement itself
                if cam_id == feat.anchor_cam_id and m == idx_anchor_bearing:
                    continue

                # Relative Pose
                curr_clone = clonesCAM[cam_id][timestamp]
                R_GtoCi = curr_clone.Rot()
                p_CiinG = curr_clone.pos()

                R_AtoCi = R_GtoCi @ R_GtoA.T
                p_CiinA = R_GtoA @ (p_CiinG - p_AinG)

                # Measurement vector
                uv_norm = feat.uvs_norm[cam_id][m]
                b_i = np.array([uv_norm[0], uv_norm[1], 1.0])
                b_i = R_AtoCi.T @ b_i
                b_i = b_i / np.linalg.norm(b_i)
                
                Bperp = skew_x(b_i)

                # Add to system
                BperpBanchor = Bperp @ bearing_inA
                
                A += np.dot(BperpBanchor, BperpBanchor)
                b += np.dot(BperpBanchor, Bperp @ p_CiinA)

        if A == 0:
            return False

        # 4. Solve for Depth
        depth = b / A
        p_f = depth * bearing_inA

        # 5. Validity Checks
        if (p_f[2] < self._options.min_dist or 
            p_f[2] > self._options.max_dist or 
            np.isnan(np.linalg.norm(p_f))):
            return False

        # 6. Store Result
        feat.p_FinA = p_f
        feat.p_FinG = R_GtoA.T @ feat.p_FinA + p_AinG
        
        return True

    def single_gaussnewton(self, feat: Feature, clonesCAM: Dict[int, Dict[float, Any]]) -> bool:
        """
        Gauss-Newton optimization for feature position (Inverse Depth Parameterization).
        """
        
        # 1. Initialize Inverse Depth Parameters
        if feat.p_FinA[2] == 0:
            return False
            
        rho = 1.0 / feat.p_FinA[2]
        alpha = feat.p_FinA[0] * rho
        beta = feat.p_FinA[1] * rho

        # Optimization Params
        lam = self._options.init_lamda
        eps = 10000.0
        runs = 0
        max_runs = self._options.max_runs
        
        recompute = True
        Hess = np.zeros((3, 3))
        grad = np.zeros(3)

        # Calculate initial cost
        cost_old = self.compute_error(clonesCAM, feat, alpha, beta, rho)

        # Get Anchor Pose
        anchor_clone = clonesCAM[feat.anchor_cam_id][feat.anchor_clone_timestamp]
        R_GtoA = anchor_clone.Rot()
        p_AinG = anchor_clone.pos()

        # 2. Optimization Loop
        while runs < max_runs and lam < self._options.max_lamda and eps > self._options.min_dx:
            
            if recompute:
                Hess.fill(0.0)
                grad.fill(0.0)
                
                # Compute Jacobian and Residuals
                for cam_id, times in feat.timestamps.items():
                    for m, timestamp in enumerate(times):
                        
                        # Relative Pose
                        curr_clone = clonesCAM[cam_id][timestamp]
                        R_GtoCi = curr_clone.Rot()
                        p_CiinG = curr_clone.pos()

                        R_AtoCi = R_GtoCi @ R_GtoA.T
                        p_CiinA = R_GtoA @ (p_CiinG - p_AinG)
                        p_AinCi = -R_AtoCi @ p_CiinA

                        # Intermediate Variables
                        hi1 = R_AtoCi[0, 0]*alpha + R_AtoCi[0, 1]*beta + R_AtoCi[0, 2] + rho*p_AinCi[0]
                        hi2 = R_AtoCi[1, 0]*alpha + R_AtoCi[1, 1]*beta + R_AtoCi[1, 2] + rho*p_AinCi[1]
                        hi3 = R_AtoCi[2, 0]*alpha + R_AtoCi[2, 1]*beta + R_AtoCi[2, 2] + rho*p_AinCi[2]
                        
                        if hi3 == 0: continue
                        hi3_sq = hi3 * hi3

                        # Jacobian
                        d_z1_d_alpha = (R_AtoCi[0, 0]*hi3 - hi1*R_AtoCi[2, 0]) / hi3_sq
                        d_z1_d_beta  = (R_AtoCi[0, 1]*hi3 - hi1*R_AtoCi[2, 1]) / hi3_sq
                        d_z1_d_rho   = (p_AinCi[0]*hi3 - hi1*p_AinCi[2]) / hi3_sq
                        
                        d_z2_d_alpha = (R_AtoCi[1, 0]*hi3 - hi2*R_AtoCi[2, 0]) / hi3_sq
                        d_z2_d_beta  = (R_AtoCi[1, 1]*hi3 - hi2*R_AtoCi[2, 1]) / hi3_sq
                        d_z2_d_rho   = (p_AinCi[1]*hi3 - hi2*p_AinCi[2]) / hi3_sq

                        H = np.array([
                            [d_z1_d_alpha, d_z1_d_beta, d_z1_d_rho],
                            [d_z2_d_alpha, d_z2_d_beta, d_z2_d_rho]
                        ])

                        # Residual
                        z_pred = np.array([hi1/hi3, hi2/hi3])
                        res = feat.uvs_norm[cam_id][m] - z_pred

                        # Summation
                        grad += H.T @ res
                        Hess += H.T @ H

            # Levenberg-Marquardt Step
            Hess_l = Hess.copy()
            for r in range(3):
                Hess_l[r, r] *= (1.0 + lam)
            
            try:
                dx = np.linalg.solve(Hess_l, grad)
            except np.linalg.LinAlgError:
                recompute = False
                lam = lam * self._options.lam_mult
                continue

            # Check Error
            cost = self.compute_error(clonesCAM, feat, alpha + dx[0], beta + dx[1], rho + dx[2])

            # Check Convergence
            if cost <= cost_old and (cost_old - cost) / cost_old < self._options.min_dcost:
                alpha += dx[0]
                beta += dx[1]
                rho += dx[2]
                break

            # Update Logic
            if cost <= cost_old:
                recompute = True
                cost_old = cost
                alpha += dx[0]
                beta += dx[1]
                rho += dx[2]
                runs += 1
                lam = lam / self._options.lam_mult
                eps = np.linalg.norm(dx)
            else:
                recompute = False
                lam = lam * self._options.lam_mult
                continue

        # 3. Finalize
        if rho == 0: return False
        
        feat.p_FinA[0] = alpha / rho
        feat.p_FinA[1] = beta / rho
        feat.p_FinA[2] = 1.0 / rho

        # 4. Baseline Check
        p_vec = feat.p_FinA.reshape(3, 1)
        Q, _ = np.linalg.qr(p_vec, mode='complete')
        Q_tangent = Q[:, 1:3] 

        base_line_max = 0.0

        for cam_id, times in feat.timestamps.items():
            for m, timestamp in enumerate(times):
                curr_clone = clonesCAM[cam_id][timestamp]
                p_CiinG = curr_clone.pos()
                p_CiinA = R_GtoA @ (p_CiinG - p_AinG)
                
                base_line = np.linalg.norm(Q_tangent.T @ p_CiinA)
                if base_line > base_line_max:
                    base_line_max = base_line

        # 5. Final Validity Checks
        dist = feat.p_FinA[2]
        norm = np.linalg.norm(feat.p_FinA)
        
        if (dist < self._options.min_dist or 
            dist > self._options.max_dist or 
            base_line_max == 0 or 
            (norm / base_line_max) > self._options.max_baseline or 
            np.isnan(norm)):
            return False

        # Set Global Position
        feat.p_FinG = R_GtoA.T @ feat.p_FinA + p_AinG
        return True

    def compute_error(self, clonesCAM, feat, alpha, beta, rho) -> float:
        """
        Helper to compute total reprojection error for specific parameters.
        """
        err = 0.0

        anchor_clone = clonesCAM[feat.anchor_cam_id][feat.anchor_clone_timestamp]
        R_GtoA = anchor_clone.Rot()
        p_AinG = anchor_clone.pos()

        for cam_id, times in feat.timestamps.items():
            for m, timestamp in enumerate(times):
                
                curr_clone = clonesCAM[cam_id][timestamp]
                R_GtoCi = curr_clone.Rot()
                p_CiinG = curr_clone.pos()

                R_AtoCi = R_GtoCi @ R_GtoA.T
                p_CiinA = R_GtoA @ (p_CiinG - p_AinG)
                p_AinCi = -R_AtoCi @ p_CiinA

                hi1 = R_AtoCi[0, 0]*alpha + R_AtoCi[0, 1]*beta + R_AtoCi[0, 2] + rho*p_AinCi[0]
                hi2 = R_AtoCi[1, 0]*alpha + R_AtoCi[1, 1]*beta + R_AtoCi[1, 2] + rho*p_AinCi[1]
                hi3 = R_AtoCi[2, 0]*alpha + R_AtoCi[2, 1]*beta + R_AtoCi[2, 2] + rho*p_AinCi[2]

                if hi3 == 0: 
                    err += 1e9
                    continue

                z_pred = np.array([hi1/hi3, hi2/hi3])
                res = feat.uvs_norm[cam_id][m] - z_pred
                
                err += np.sum(res**2)

        return err