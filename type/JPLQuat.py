import numpy as np
import copy

# Assuming standard imports
from vio_pipeline_v2.type.Type import Type
from vio_pipeline_v2.type.quat_ops import skew_x, quat_multiply, quat_2_Rot, quatnorm

class JPLQuat(Type):
    """
    Derived Type class that implements JPL quaternion.
    
    This quaternion uses a left-multiplicative error state and follows the "JPL convention".
    State Size: 4 (x,y,z,w)
    Error Size: 3 (rotation vector)
    """

    def __init__(self):
        # Initialize Type with error state size of 3 (degrees of freedom for rotation)
        super().__init__(3)

        # Initialize internal storage for cached Rotation Matrices
        self._R = np.eye(3)
        self._Rfej = np.eye(3)

        # Set default state value to Identity Quaternion [0, 0, 0, 1]
        q0 = np.zeros(4)
        q0[3] = 1.0
        
        # This will set _value, _fej AND compute _R, _Rfej
        self.set_value_internal(q0)
        self.set_fej_internal(q0)

    def update(self, dx):
        """
        Implements update operation by left-multiplying the current
        quaternion with a quaternion built from a small axis-angle perturbation.
        
        q_new = quatnorm([0.5*dx, 1]) * q_old
        
        :param dx: Axis-angle representation of the perturbing quaternion (Size 3)
        """
        assert dx.shape[0] == self._size  # Should be 3

        # Build perturbing quaternion dq
        dq = np.zeros(4)
        dq[0:3] = 0.5 * dx
        dq[3] = 1.0
        
        # Normalize dq
        dq = quatnorm(dq)

        # Update estimate (Left Multiply) and recompute R
        # Note: quat_multiply handles the math: dq * _value
        new_q = quat_multiply(dq, self._value)
        self.set_value(new_q)

    def set_value(self, new_value):
        """Sets the value of the estimate and recomputes internal R"""
        self.set_value_internal(new_value)

    def set_fej(self, new_value):
        """Sets the fej value and recomputes internal R_fej"""
        self.set_fej_internal(new_value)

    def clone(self):
        """Deep copy of the object"""
        Clone = JPLQuat()
        Clone.set_value(self.value()) # This handles copying _R internally
        Clone.set_fej(self.fej())     # This handles copying _Rfej internally
        return Clone

    # --- Accessors ---

    def Rot(self):
        """Rotation Matrix access"""
        return self._R

    def Rot_fej(self):
        """FEJ Rotation Matrix access"""
        return self._Rfej

    # --- Protected / Internal Helper Functions ---

    def set_value_internal(self, new_value):
        """
        Sets the value of the estimate and recomputes the internal rotation matrix
        """
        assert new_value.shape[0] == 4, f"JPLQuat must be size 4, got {new_value.shape[0]}"
        
        # Store the quaternion
        self._value = new_value

        # Compute associated rotation matrix immediately (Cache it)
        # This avoids re-computing it every time .Rot() is called
        self._R = quat_2_Rot(new_value)

    def set_fej_internal(self, new_value):
        """
        Sets the fej value and recomputes the fej rotation matrix
        """
        assert new_value.shape[0] == 4
        
        # Store the quaternion
        self._fej = new_value

        # Compute associated rotation matrix immediately
        self._Rfej = quat_2_Rot(new_value)