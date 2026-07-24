import numpy as np
import cv2
from vio_pipeline_v2.core.cam_base import CamBase

class CamEqui(CamBase):
    """
    Fisheye / Equidistant model pinhole camera class.
    
    Implements the projection model used in OpenCV fisheye calibration.
    Mapping: [u, v] = f(normalized_xyz, params)
    """
    
    def __init__(self, width: int, height: int):
        super().__init__(width, height)

    def undistort_f(self, uv_dist: np.ndarray) -> np.ndarray:
        """
        Given a raw uv point, undistort it to normalized coordinates.
        Uses OpenCV's optimized undistortPoints function.
        """
        # OpenCV needs float32
        pts = np.array([[[uv_dist[0], uv_dist[1]]]], dtype=np.float32)
        
        # Camera Matrix K (3x3) and Distortion D (4x1)
        K = self.camera_k_OPENCV
        D = self.camera_d_OPENCV
        
        # undistortPoints(src, K, D) -> returns normalized coordinates if P is not provided
        # Result is (N, 1, 2)
        undistorted = cv2.fisheye.undistortPoints(pts, K, D)
        
        return undistorted[0, 0]

    def distort_f(self, uv_norm: np.ndarray) -> np.ndarray:
        """
        Given normalized coordinates (x, y, 1), distort to raw image pixels.
        
        Formulas:
        r = sqrt(x^2 + y^2)
        theta = atan(r)
        theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
        x_dist = (theta_d / r) * x
        y_dist = (theta_d / r) * y
        u = fx * x_dist + cx
        v = fy * y_dist + cy
        """
        # Parameters: [fx, fy, cx, cy, k1, k2, k3, k4]
        cam_d = self.camera_values
        
        # 1. Calculate r and theta
        x, y = uv_norm[0], uv_norm[1]
        r = np.sqrt(x*x + y*y)
        theta = np.arctan(r)
        
        # 2. Calculate theta_d (distorted angle)
        # k1=4, k2=5, k3=6, k4=7 indices in camera_values
        k1, k2, k3, k4 = cam_d[4], cam_d[5], cam_d[6], cam_d[7]
        
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        
        # theta_d = theta * (1 + ...)
        scaling = (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)
        theta_d = theta * scaling
        
        # 3. Handle small r (limit as r->0, theta_d/r -> 1)
        # If r is very small, scaling factor (theta_d / r) approx 1
        cdist = 1.0
        if r > 1e-8:
            cdist = theta_d / r
            
        x_dist = x * cdist
        y_dist = y * cdist
        
        # 4. Project to pixels
        fx, fy, cx, cy = cam_d[0], cam_d[1], cam_d[2], cam_d[3]
        
        u = fx * x_dist + cx
        v = fy * y_dist + cy
        
        return np.array([u, v])

    def compute_distort_jacobian(self, uv_norm: np.ndarray) -> tuple:
        """
        Computes the Jacobian of the distortion function w.r.t normalized coords and intrinsics.
        
        Returns:
            H_dz_dzn: (2x2) Jacobian of pixel coords w.r.t normalized coords
            H_dz_dzeta: (2x8) Jacobian of pixel coords w.r.t intrinsic parameters
        """
        cam_d = self.camera_values
        fx, fy = cam_d[0], cam_d[1]
        k1, k2, k3, k4 = cam_d[4], cam_d[5], cam_d[6], cam_d[7]
        
        x, y = uv_norm[0], uv_norm[1]
        r = np.sqrt(x*x + y*y)
        theta = np.arctan(r)
        
        # Powers of theta
        theta2 = theta * theta
        theta3 = theta * theta2
        theta4 = theta2 * theta2
        theta5 = theta * theta4
        theta6 = theta4 * theta2
        theta7 = theta * theta6
        theta8 = theta4 * theta4
        theta9 = theta * theta8
        
        # Theta distorted
        theta_d = theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)
        
        # Terms for derivatives
        inv_r = 1.0 / r if r > 1e-8 else 1.0
        inv_r2 = inv_r * inv_r
        
        # cdist = theta_d / r
        cdist = theta_d * inv_r if r > 1e-8 else 1.0
        
        # --- 1. Jacobian w.r.t Normalized Coordinates (H_dz_dzn) ---
        
        # Chain rule components:
        # duv / dxy_dist = [fx 0; 0 fy]
        duv_dxy = np.array([[fx, 0], [0, fy]])
        
        # dxy_dist / dxyn
        # x_dist = x * (theta_d/r)
        # product rule on x * (theta_d/r)
        
        # Partial: d(theta_d)/d(theta)
        dthd_dth = 1 + 3*k1*theta2 + 5*k2*theta4 + 7*k3*theta6 + 9*k4*theta8
        
        # Partial: d(theta)/d(r) = 1 / (1 + r^2)
        dth_dr = 1.0 / (1.0 + r*r)
        
        # Partial: d(r)/d(normalized)
        dr_dxyn = np.array([[x * inv_r, y * inv_r]]) # 1x2
        
        # Matrix M1: d(xy_dist)/d(xyn) considering theta_d/r as constant
        # Diag(theta_d/r)
        M1 = np.eye(2) * cdist
        
        # Matrix M2: x * d(theta_d/r)/d(xyn)
        # d(theta_d/r)/dr = (r * dthd_dr - theta_d) / r^2
        # dthd_dr = dthd_dth * dth_dr
        dthd_dr = dthd_dth * dth_dr
        d_scale_dr = (r * dthd_dr - theta_d) * inv_r2
        
        # d(xy_dist)/dr = [x; y] * d_scale_dr
        dxy_dr = np.array([[x], [y]]) * d_scale_dr
        
        # Combine: d(xy_dist)/d(xyn) = M1 + dxy_dr * dr_dxyn
        dxy_dist_dxyn = M1 + dxy_dr @ dr_dxyn
        
        H_dz_dzn = duv_dxy @ dxy_dist_dxyn
        
        # --- 2. Jacobian w.r.t Intrinsics (H_dz_dzeta) ---
        # Params: [fx, fy, cx, cy, k1, k2, k3, k4]
        
        H_dz_dzeta = np.zeros((2, 8))
        
        x_dist = x * cdist
        y_dist = y * cdist
        
        # fx, fy, cx, cy
        H_dz_dzeta[0, 0] = x_dist
        H_dz_dzeta[1, 1] = y_dist
        H_dz_dzeta[0, 2] = 1.0
        H_dz_dzeta[1, 3] = 1.0
        
        # Distortion k1, k2, k3, k4
        # u = fx * x * (theta_d / r) + cx
        # du/dk1 = fx * (x/r) * d(theta_d)/dk1
        # d(theta_d)/dk1 = theta^3
        
        term_x = fx * x * inv_r
        term_y = fy * y * inv_r
        
        # k1
        H_dz_dzeta[0, 4] = term_x * theta3
        H_dz_dzeta[1, 4] = term_y * theta3
        
        # k2
        H_dz_dzeta[0, 5] = term_x * theta5
        H_dz_dzeta[1, 5] = term_y * theta5
        
        # k3
        H_dz_dzeta[0, 6] = term_x * theta7
        H_dz_dzeta[1, 6] = term_y * theta7
        
        # k4
        H_dz_dzeta[0, 7] = term_x * theta9
        H_dz_dzeta[1, 7] = term_y * theta9
        
        return H_dz_dzn, H_dz_dzeta