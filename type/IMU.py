import numpy as np
import copy

# Assuming these exist as requested
from vio_pipeline_v2.type.PoseJPL import PoseJPL
from vio_pipeline_v2.type.Type import Type
from vio_pipeline_v2.type.Vec import Vec
from vio_pipeline_v2.type.quat_ops import quatnorm, quat_multiply

class IMU(Type):
    """
    Derived Type class that implements an IMU state.
    Contains a PoseJPL, Vec velocity, Vec gyro bias, and Vec accel bias.
    """

    def __init__(self):
        # Initialize base class with size 15 (Error state size)
        super().__init__(15)

        # Create all the sub-variables
        self._pose = PoseJPL()
        self._v = Vec(3)
        self._bg = Vec(3)
        self._ba = Vec(3)

        # Set our default state value
        # Eigen::VectorXd::Zero(16, 1) -> numpy array of zeros
        imu0 = np.zeros(16)
        imu0[3] = 1.0 # Identity quaternion (x,y,z,w) -> w=1 at index 3
        
        self.set_value_internal(imu0)
        self.set_fej_internal(imu0)

    def set_local_id(self, new_id):
        """
        Sets id used to track location of variable in the filter covariance
        Note that we update the sub-variables also.
        """
        self._id = new_id
        
        self._pose.set_local_id(new_id)
        
        # Calculate offsets based on sizes
        # syntax: condition ? val_true : val_false
        pose_offset = self._pose.size() if new_id != -1 else 0
        self._v.set_local_id(self._pose.id() + pose_offset)
        
        v_offset = self._v.size() if new_id != -1 else 0
        self._bg.set_local_id(self._v.id() + v_offset)
        
        bg_offset = self._bg.size() if new_id != -1 else 0
        self._ba.set_local_id(self._bg.id() + bg_offset)

    def update(self, dx):
        """
        Performs update operation using JPLQuat update for orientation, then vector updates.
        dx: 15 DOF vector (q, p, v, bg, ba)
        """
        assert dx.shape[0] == self._size

        newX = self._value.copy()

        # Orientation Update
        # dq << .5 * dx.block(0, 0, 3, 1), 1.0
        dq = np.zeros(4)
        dq[0:3] = 0.5 * dx[0:3]
        dq[3] = 1.0
        
        dq = quatnorm(dq)

        # Quaternion multiplication
        # newX.block(0, 0, 4, 1) = ov_core::quat_multiply(dq, quat())
        newX[0:4] = quat_multiply(dq, self.quat())

        # Position Update
        # newX.block(4, 0, 3, 1) += dx.block(3, 0, 3, 1)
        newX[4:7] += dx[3:6]

        # Velocity Update
        newX[7:10] += dx[6:9]

        # Gyro Bias Update
        newX[10:13] += dx[9:12]

        # Accel Bias Update
        newX[13:16] += dx[12:15]

        self.set_value(newX)

    def set_value(self, new_value):
        """Sets the value of the estimate"""
        self.set_value_internal(new_value)

    def set_fej(self, new_value):
        """Sets the value of the first estimate"""
        self.set_fej_internal(new_value)

    def clone(self):
        """Deep copy of the object"""
        # In C++ this creates a new shared_ptr. In Python we create a new instance
        # and manually copy the values to mimic the behavior.
        Clone = IMU()
        Clone.set_value(self.value())
        Clone.set_fej(self.fej())
        return Clone

    def check_if_subvariable(self, check):
        """Checks if a given pointer is one of the sub-variables"""
        if check == self._pose:
            return self._pose
        # Recursively check pose sub-variables
        elif self._pose.check_if_subvariable(check) == check:
            return self._pose.check_if_subvariable(check)
        elif check == self._v:
            return self._v
        elif check == self._bg:
            return self._bg
        elif check == self._ba:
            return self._ba
        return None

    # --- Accessors ---

    def Rot(self):
        return self._pose.Rot()

    def Rot_fej(self):
        return self._pose.Rot_fej()

    def quat(self):
        return self._pose.quat()

    def quat_fej(self):
        return self._pose.quat_fej()

    def pos(self):
        return self._pose.pos()

    def pos_fej(self):
        return self._pose.pos_fej()

    def vel(self):
        return self._v.value()

    def vel_fej(self):
        return self._v.fej()

    def bias_g(self):
        return self._bg.value()

    def bias_g_fej(self):
        return self._bg.fej()

    def bias_a(self):
        return self._ba.value()

    def bias_a_fej(self):
        return self._ba.fej()

    # --- Sub-variable Accessors ---

    def pose(self):
        return self._pose

    def q(self):
        return self._pose.q()

    def p(self):
        return self._pose.p()

    def v(self):
        return self._v

    def bg(self):
        return self._bg

    def ba(self):
        return self._ba

    # --- Protected / Internal Helper Functions ---

    def set_value_internal(self, new_value):
        assert new_value.shape[0] == 16
        
        # Block assignments
        self._pose.set_value(new_value[0:7])
        self._v.set_value(new_value[7:10])
        self._bg.set_value(new_value[10:13])
        self._ba.set_value(new_value[13:16])

        # Store in base class member (assuming _value exists in Type)
        self._value = new_value

    def set_fej_internal(self, new_value):
        assert new_value.shape[0] == 16
        
        # Block assignments
        self._pose.set_fej(new_value[0:7])
        self._v.set_fej(new_value[7:10])
        self._bg.set_fej(new_value[10:13])
        self._ba.set_fej(new_value[13:16])

        # Store in base class member (assuming _fej exists in Type)
        self._fej = new_value