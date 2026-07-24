import numpy as np
import cv2
from abc import ABC, abstractmethod

class CamBase(ABC):
    """
    Base pinhole camera model class.
    
    This is the base class for all camera models.
    All these models are pinhole cameras, thus just have standard reprojection logic.
    """
    def __init__(self, width: int, height: int):
        """
        Default constructor.
        :param width: Width of the camera (raw pixels)
        :param height: Height of the camera (raw pixels)
        """
        self._width = width
        self._height = height
        
        # Raw set of camera intrinsic values
        self.camera_values = np.zeros(8)
        
        # Camera intrinsics in OpenCV format (3x3)
        self.camera_k_OPENCV = np.eye(3)
        
        # Camera distortion in OpenCV format (4,)
        self.camera_d_OPENCV = np.zeros(4)

    def set_value(self, calib: np.ndarray):
        """
        Set and update the camera calibration values.
        Should be called on startup and after update!
        :param calib: 8-element array (fx, fy, cx, cy, k1, k2, k3, k4)
        """
        # Assert we are of size eight
        assert calib.shape[0] == 8
        self.camera_values = calib.copy()

        # Camera matrix K
        # [fx  0 cx]
        # [ 0 fy cy]
        # [ 0  0  1]
        self.camera_k_OPENCV = np.array([
            [calib[0], 0.0,      calib[2]],
            [0.0,      calib[1], calib[3]],
            [0.0,      0.0,      1.0]
        ], dtype=np.float64)

        # Distortion parameters D
        # [k1, k2, k3, k4]
        self.camera_d_OPENCV = np.array([
            calib[4], calib[5], calib[6], calib[7]
        ], dtype=np.float64)

    @abstractmethod
    def undistort_f(self, uv_dist: np.ndarray) -> np.ndarray:
        """
        Given a raw uv point, undistort it into normalized camera coords.
        :param uv_dist: Raw uv coordinate (2,)
        :return: Normalized coordinates (2,)
        """
        pass

    def undistort_d(self, uv_dist: np.ndarray) -> np.ndarray:
        """
        Wrapper for double precision (Python uses float64 by default).
        """
        return self.undistort_f(uv_dist)

    def undistort_cv(self, uv_dist: tuple) -> tuple:
        """
        Wrapper for OpenCV points.
        :param uv_dist: tuple (x, y)
        :return: tuple (x, y) normalized
        """
        pt_in = np.array([uv_dist[0], uv_dist[1]], dtype=np.float64)
        pt_out = self.undistort_f(pt_in)
        return (pt_out[0], pt_out[1])

    @abstractmethod
    def distort_f(self, uv_norm: np.ndarray) -> np.ndarray:
        """
        Given a normalized uv coordinate, distort it to the raw image plane.
        :param uv_norm: Normalized coordinates (2,)
        :return: Raw uv coordinate (2,)
        """
        pass

    def distort_d(self, uv_norm: np.ndarray) -> np.ndarray:
        """Wrapper for double precision."""
        return self.distort_f(uv_norm)

    def distort_cv(self, uv_norm: tuple) -> tuple:
        """
        Wrapper for OpenCV points.
        :param uv_norm: tuple (x, y) normalized
        :return: tuple (x, y) raw
        """
        pt_in = np.array([uv_norm[0], uv_norm[1]], dtype=np.float64)
        pt_out = self.distort_f(pt_in)
        return (pt_out[0], pt_out[1])

    @abstractmethod
    def compute_distort_jacobian(self, uv_norm: np.ndarray) -> tuple:
        """
        Computes the derivative of raw distorted to normalized coordinate.
        :param uv_norm: Normalized coordinates
        :return: Tuple (H_dz_dzn, H_dz_dzeta)
                 H_dz_dzn: Derivative of measurement z w.r.t normalized (2x2)
                 H_dz_dzeta: Derivative of measurement z w.r.t intrinsics (2x8)
        """
        pass

    # --- Getters ---

    def get_value(self) -> np.ndarray:
        return self.camera_values

    def get_K(self) -> np.ndarray:
        return self.camera_k_OPENCV

    def get_D(self) -> np.ndarray:
        return self.camera_d_OPENCV

    def w(self) -> int:
        return self._width

    def h(self) -> int:
        return self._height