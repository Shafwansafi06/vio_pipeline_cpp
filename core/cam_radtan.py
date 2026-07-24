import numpy as np
import cv2
from vio_pipeline_v2.core.cam_base import CamBase

class CamRadtan(CamBase):
    """
    Radial-tangential / Brown-Conrady model pinhole camera model class.
    
    Standard OpenCV model:
    x_dist = x_n * (1 + k1*r^2 + k2*r^4) + 2*p1*x_n*y_n + p2*(r^2 + 2*x_n^2)
    y_dist = y_n * (1 + k1*r^2 + k2*r^4) + p1*(r^2 + 2*y_n^2) + 2*p2*x_n*y_n
    u = fx * x_dist + cx
    v = fy * y_dist + cy
    """
    
    def __init__(self, width: int, height: int):
        super().__init__(width, height)

    def undistort_f(self, uv_dist: np.ndarray) -> np.ndarray:
        """
        Given a raw uv point, undistort it to normalized coordinates.
        """
        # Prepare for OpenCV (N, 1, 2)
        pts = np.array([[[uv_dist[0], uv_dist[1]]]], dtype=np.float32)
        
        K = self.camera_k_OPENCV
        D = self.camera_d_OPENCV
        
        # cv2.undistortPoints returns normalized coordinates if P is empty
        undistorted = cv2.undistortPoints(pts, K, D)
        
        return undistorted[0, 0]

    def distort_f(self, uv_norm: np.ndarray) -> np.ndarray:
        """
        Given normalized coordinates (x, y), distort to raw image pixels.
        """
        cam_d = self.camera_values
        # Params: fx, fy, cx, cy, k1, k2, p1, p2
        fx, fy, cx, cy = cam_d[0], cam_d[1], cam_d[2], cam_d[3]
        k1, k2, p1, p2 = cam_d[4], cam_d[5], cam_d[6], cam_d[7]
        
        x = uv_norm[0]
        y = uv_norm[1]
        
        r2 = x*x + y*y
        r4 = r2 * r2
        
        # Radial term: (1 + k1*r^2 + k2*r^4)
        radial = (1.0 + k1 * r2 + k2 * r4)
        
        # Tangential terms
        # x_dist = x * radial + 2*p1*x*y + p2*(r^2 + 2*x^2)
        x_dist = x * radial + (2.0 * p1 * x * y) + p2 * (r2 + 2.0 * x * x)
        
        # y_dist = y * radial + p1*(r^2 + 2*y^2) + 2*p2*x*y
        y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + (2.0 * p2 * x * y)
        
        # Project to pixel
        u = fx * x_dist + cx
        v = fy * y_dist + cy
        
        return np.array([u, v])

    def compute_distort_jacobian(self, uv_norm: np.ndarray) -> tuple:
        """
        Computes the Jacobian of the distortion function w.r.t normalized coords and intrinsics.
        """
        cam_d = self.camera_values
        fx, fy = cam_d[0], cam_d[1]
        k1, k2, p1, p2 = cam_d[4], cam_d[5], cam_d[6], cam_d[7]
        
        x = uv_norm[0]
        y = uv_norm[1]
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        r4 = r2 * r2
        
        # --- 1. Jacobian w.r.t Normalized Coordinates (H_dz_dzn) ---
        H_dz_dzn = np.zeros((2, 2))
        
        # Radial factor and derivative parts
        rad_factor = 1.0 + k1*r2 + k2*r4
        
        # du/dx = fx * d(x_dist)/dx
        # d(x_dist)/dx = rad_factor + x*(2*k1*x + 4*k2*r2*x) + 2*p1*y + 2*p2*x + 4*p2*x
        #              = rad_factor + 2*k1*x^2 + 4*k2*x^2*r^2 + 2*p1*y + 6*p2*x
        dxd_dx = rad_factor + (2*k1*x2 + 4*k2*x2*r2) + 2*p1*y + 6*p2*x
        H_dz_dzn[0, 0] = fx * dxd_dx
        
        # du/dy = fx * d(x_dist)/dy
        # d(x_dist)/dy = x*(2*k1*y + 4*k2*r2*y) + 2*p1*x + p2*(2*y)
        dxd_dy = (2*k1*xy + 4*k2*xy*r2) + 2*p1*x + 2*p2*y
        H_dz_dzn[0, 1] = fx * dxd_dy
        
        # dv/dx = fy * d(y_dist)/dx
        # d(y_dist)/dx = y*(2*k1*x + 4*k2*r2*x) + p1*(2*x) + 2*p2*y
        dyd_dx = (2*k1*xy + 4*k2*xy*r2) + 2*p1*x + 2*p2*y
        H_dz_dzn[1, 0] = fy * dyd_dx
        
        # dv/dy = fy * d(y_dist)/dy
        # d(y_dist)/dy = rad_factor + y*(2*k1*y + 4*k2*r2*y) + p1*(2*y + 4*y) + 2*p2*x
        dyd_dy = rad_factor + (2*k1*y2 + 4*k2*y2*r2) + 6*p1*y + 2*p2*x
        H_dz_dzn[1, 1] = fy * dyd_dy
        
        # --- 2. Jacobian w.r.t Intrinsics (H_dz_dzeta) ---
        # Order: fx, fy, cx, cy, k1, k2, p1, p2
        H_dz_dzeta = np.zeros((2, 8))
        
        # Recompute x_dist, y_dist components for efficiency
        x_dist_k = x * rad_factor + 2*p1*xy + p2*(r2 + 2*x2)
        y_dist_k = y * rad_factor + p1*(r2 + 2*y2) + 2*p2*xy
        
        # Note: In C++ code, x1/y1 calculated there does NOT include fx/fy scaling yet
        # x_dist = x1, y_dist = y1 in C++ notation
        
        # fx
        H_dz_dzeta[0, 0] = x_dist_k
        # fy
        H_dz_dzeta[1, 1] = y_dist_k
        # cx
        H_dz_dzeta[0, 2] = 1.0
        # cy
        H_dz_dzeta[1, 3] = 1.0
        
        # k1 -> u depends on fx * x * (r^2)
        H_dz_dzeta[0, 4] = fx * x * r2
        H_dz_dzeta[1, 4] = fy * y * r2
        
        # k2 -> u depends on fx * x * (r^4)
        H_dz_dzeta[0, 5] = fx * x * r4
        H_dz_dzeta[1, 5] = fy * y * r4
        
        # p1
        # u: fx * (2*x*y)
        # v: fy * (r^2 + 2*y^2)
        H_dz_dzeta[0, 6] = fx * (2 * x * y)
        H_dz_dzeta[1, 6] = fy * (r2 + 2 * y2)
        
        # p2
        # u: fx * (r^2 + 2*x^2)
        # v: fy * (2*x*y)
        H_dz_dzeta[0, 7] = fx * (r2 + 2 * x2)
        H_dz_dzeta[1, 7] = fy * (2 * x * y)
        
        return H_dz_dzn, H_dz_dzeta