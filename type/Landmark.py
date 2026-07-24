import numpy as np
import math
from enum import Enum, auto

# Assuming Type is imported from your module
from vio_pipeline_v2.type.Type import Type
from vio_pipeline_v2.type.LandmarkRepresentation import LandmarkRepresentation


class Landmark(Type):
    """
    Derived Type class that implements a landmark state.
    
    Can support multiple representations:
    - Global 3D (XYZ)
    - Global Inverse Depth (Theta, Phi, Rho)
    - Anchored 3D (Relative XYZ)
    - Anchored MSCKF Inverse Depth (u, v, 1/z)
    - Anchored Single Inverse Depth (1/z)
    """

    def __init__(self, size, rep=LandmarkRepresentation.Representation.GLOBAL_3D):
        """
        Default constructor
        :param size: Degrees of freedom (error state size)
        :param rep: The representation type
        """
        super().__init__(size)
        self._feat_representation = rep
        self._featid = -1
        self._unique_camera_id = -1
        self.should_marg = False
        self.update_fail_count = 0
        
        # Used for ANCHORED_INVERSE_DEPTH_SINGLE
        self.uv_norm_zero = np.zeros(3)
        self.uv_norm_zero_fej = np.zeros(3)

        # Default values
        self._value = np.zeros(size)
        self._fej = np.zeros(size)

    def set_local_id(self, new_id):
        """Sets id used to track location of variable in the filter covariance"""
        self._id = new_id

    def update(self, dx):
        """
        Update variable due to perturbation of error state.
        Simple addition for all landmark types implemented here.
        """
        assert dx.shape[0] == self._size
        self.set_value(self._value + dx)

    def clone(self):
        """Create a deep copy of this landmark"""
        Clone = Landmark(self._size, self._feat_representation)
        Clone.set_value(self.value().copy())
        Clone.set_fej(self.fej().copy())
        Clone._featid = self._featid
        Clone._unique_camera_id = self._unique_camera_id
        Clone.should_marg = self.should_marg
        Clone.update_fail_count = self.update_fail_count
        Clone.uv_norm_zero = self.uv_norm_zero.copy()
        Clone.uv_norm_zero_fej = self.uv_norm_zero_fej.copy()
        return Clone

    def get_xyz(self, getfej=False):
        """
        Convert the internal representation to standard 3D XYZ coordinates.
        
        :param getfej: If true, uses the first estimate (FEJ) values.
        :return: 3x1 numpy array (XYZ)
        """
        # Select current value or FEJ
        current_val = self.fej() if getfej else self.value()

        # CASE: Global 3D or Anchored 3D
        if (self._feat_representation == LandmarkRepresentation.Representation.GLOBAL_3D or 
            self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_3D):
            return current_val

        # CASE: Global/Anchored Full Inverse Depth
        # Standard Spherical: (theta, phi, rho) -> (X, Y, Z)
        if (self._feat_representation == LandmarkRepresentation.Representation.GLOBAL_FULL_INVERSE_DEPTH or 
            self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_FULL_INVERSE_DEPTH):
            
            p_invFinG = current_val
            theta = p_invFinG[0]
            phi = p_invFinG[1]
            rho = p_invFinG[2] # 1/distance
            
            # Avoid division by zero
            inv_rho = 1.0 / rho if rho != 0 else 0.0
            
            p_FinG = np.zeros(3)
            # x = (1/rho) * cos(theta) * sin(phi)
            p_FinG[0] = inv_rho * math.cos(theta) * math.sin(phi)
            # y = (1/rho) * sin(theta) * sin(phi)
            p_FinG[1] = inv_rho * math.sin(theta) * math.sin(phi)
            # z = (1/rho) * cos(phi)
            p_FinG[2] = inv_rho * math.cos(phi)
            
            return p_FinG

        # CASE: Anchored MSCKF Inverse Depth
        # Representation: (u/z, v/z, 1/z) -> (u, v, 1) * z
        if self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_MSCKF_INVERSE_DEPTH:
            p_invFinA = current_val
            inv_z = p_invFinA[2]
            
            z = 1.0 / inv_z if inv_z != 0 else 0.0
            
            p_FinA = np.zeros(3)
            p_FinA[0] = z * p_invFinA[0]
            p_FinA[1] = z * p_invFinA[1]
            p_FinA[2] = z
            
            return p_FinA

        # CASE: Anchored Single Inverse Depth
        # Representation: (rho) + stored bearing vector
        if self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_INVERSE_DEPTH_SINGLE:
            rho = current_val[0]
            
            inv_rho = 1.0 / rho if rho != 0 else 0.0
            
            bearing = self.uv_norm_zero_fej if getfej else self.uv_norm_zero
            
            return inv_rho * bearing

        # Failure
        assert False, f"Unknown representation: {self._feat_representation}"
        return np.zeros(3)

    def set_from_xyz(self, p_FinG, isfej=False):
        """
        Set the internal representation from a standard 3D XYZ vector.
        
        :param p_FinG: 3x1 numpy array (XYZ)
        :param isfej: If true, updates the FEJ value.
        """
        assert p_FinG.shape[0] == 3

        # CASE: Global 3D or Anchored 3D
        if (self._feat_representation == LandmarkRepresentation.Representation.GLOBAL_3D or 
            self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_3D):
            if isfej:
                self.set_fej(p_FinG)
            else:
                self.set_value(p_FinG)
            return

        # CASE: Global/Anchored Full Inverse Depth
        if (self._feat_representation == LandmarkRepresentation.Representation.GLOBAL_FULL_INVERSE_DEPTH or 
            self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_FULL_INVERSE_DEPTH):
            
            norm = np.linalg.norm(p_FinG)
            g_rho = 1.0 / norm if norm != 0 else 0.0
            
            # phi = acos(z / r)
            g_phi = math.acos(g_rho * p_FinG[2]) if abs(g_rho * p_FinG[2]) <= 1.0 else 0.0
            
            # theta = atan2(y, x)
            g_theta = math.atan2(p_FinG[1], p_FinG[0])
            
            p_invFinG = np.array([g_theta, g_phi, g_rho])

            if isfej:
                self.set_fej(p_invFinG)
            else:
                self.set_value(p_invFinG)
            return

        # CASE: Anchored MSCKF Inverse Depth
        if self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_MSCKF_INVERSE_DEPTH:
            z = p_FinG[2]
            
            # Avoid division by zero
            inv_z = 1.0 / z if z != 0 else 0.0
            
            # state = [x/z, y/z, 1/z]
            p_invFinA_MSCKF = np.array([
                p_FinG[0] * inv_z,
                p_FinG[1] * inv_z,
                inv_z
            ])

            if isfej:
                self.set_fej(p_invFinA_MSCKF)
            else:
                self.set_value(p_invFinA_MSCKF)
            return

        # CASE: Anchored Single Inverse Depth
        if self._feat_representation == LandmarkRepresentation.Representation.ANCHORED_INVERSE_DEPTH_SINGLE:
            z = p_FinG[2]
            
            # State is just size 1: [1/z]
            temp = np.array([1.0 / z]) if z != 0 else np.zeros(1)

            # Store the normalized bearing vector
            bearing = (1.0 / z) * p_FinG if z != 0 else np.zeros(3)

            if not isfej:
                self.uv_norm_zero = bearing
                self.set_value(temp)
            else:
                self.uv_norm_zero_fej = bearing
                self.set_fej(temp)
            return

        # Failure
        assert False, f"Unknown representation: {self._feat_representation}"