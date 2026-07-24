import numpy as np
import copy

# Assuming these exist in your environment
from vio_pipeline_v2.type.Type import Type
from vio_pipeline_v2.type.Vec import Vec
from vio_pipeline_v2.type.JPLQuat import JPLQuat
from vio_pipeline_v2.type.quat_ops import quatnorm, quat_multiply

class PoseJPL(Type):
    """
    Derived Type class that implements a 6 d.o.f pose.
    Internally uses a JPLQuat representation for orientation and 3D Vec position.
    """

    def __init__(self):
        # Initialize Type with error state size of 6 (3 rot + 3 pos)
        super().__init__(6)

        # Initialize subvariables
        self._q = JPLQuat()
        self._p = Vec(3)

        # Set our default state value (Identity Pose)
        # 7 Elements: 4 for Quat, 3 for Position
        pose0 = np.zeros(7)
        pose0[3] = 1.0  # Identity quaternion (0,0,0,1)
        
        self.set_value_internal(pose0)
        self.set_fej_internal(pose0)

    def set_local_id(self, new_id):
        """
        Sets id used to track location of variable in the filter covariance.
        Updates sub-variables as well.
        """
        self._id = new_id
        
        # Update Quaternion ID
        self._q.set_local_id(new_id)
        
        # Update Position ID
        # Offset is size of Q error state (usually 3) if valid
        offset = self._q.size() if new_id != -1 else 0
        self._p.set_local_id(new_id + offset)

    def update(self, dx):
        """
        Update q and p using the JPLQuat update for orientation and vector update for position.
        dx: 6 DOF vector (3 orientation error, 3 position error)
        """
        assert dx.shape[0] == self._size  # Should be 6

        newX = self._value.copy()

        # --- Orientation Update ---
        # Create small angle quaternion dq from the first 3 elements of dx
        dq = np.zeros(4)
        dq[0:3] = 0.5 * dx[0:3]
        dq[3] = 1.0
        
        # Normalize dq
        dq = quatnorm(dq)

        # Multiply: New_Quat = dq * Old_Quat
        # block(0,0,4,1) corresponds to indices 0:4
        newX[0:4] = quat_multiply(dq, self.quat())

        # --- Position Update ---
        # Simple addition for position (indices 3:6 of dx added to indices 4:7 of state)
        newX[4:7] += dx[3:6]

        self.set_value(newX)

    def set_value(self, new_value):
        """Sets the value of the estimate"""
        self.set_value_internal(new_value)

    def set_fej(self, new_value):
        """Sets the value of the first estimate"""
        self.set_fej_internal(new_value)

    def clone(self):
        """Deep copy of the object"""
        Clone = PoseJPL()
        Clone.set_value(self.value())
        Clone.set_fej(self.fej())
        return Clone

    def check_if_subvariable(self, check):
        """Checks if a given pointer is one of the sub-variables"""
        if check == self._q:
            return self._q
        elif check == self._p:
            return self._p
        return None

    # --- Accessors ---

    def Rot(self):
        """Rotation Matrix access"""
        return self._q.Rot()

    def Rot_fej(self):
        """FEJ Rotation Matrix access"""
        return self._q.Rot_fej()

    def quat(self):
        """Quaternion access (vector 4x1)"""
        return self._q.value()

    def quat_fej(self):
        """FEJ Quaternion access"""
        return self._q.fej()

    def pos(self):
        """Position access (vector 3x1)"""
        return self._p.value()

    def pos_fej(self):
        """FEJ Position access"""
        return self._p.fej()

    # --- Sub-variable Accessors ---

    def q(self):
        return self._q

    def p(self):
        return self._p

    # --- Protected / Internal Helper Functions ---

    def set_value_internal(self, new_value):
        assert new_value.shape[0] == 7
        
        # Set orientation value (first 4 elements)
        self._q.set_value(new_value[0:4])
        
        # Set position value (next 3 elements)
        self._p.set_value(new_value[4:7])

        # Store in base class
        self._value = new_value

    def set_fej_internal(self, new_value):
        assert new_value.shape[0] == 7
        
        # Set orientation fej value
        self._q.set_fej(new_value[0:4])
        
        # Set position fej value
        self._p.set_fej(new_value[4:7])

        # Store in base class
        self._fej = new_value