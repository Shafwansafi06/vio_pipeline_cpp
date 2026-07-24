import numpy as np
import copy

# Assuming Type is imported from your module
from vio_pipeline_v2.type.Type import Type

class Vec(Type):
    """
    Derived Type class that implements vector variables.
    Used for Velocity, Biases, or any standard Euclidean vector.
    """

    def __init__(self, dim):
        """
        Default constructor for Vec
        :param dim: Size of the vector (will be same as error state)
        """
        # Initialize base Type with the specific dimension
        super().__init__(dim)
        
        # Initialize value and fej to Zero vectors
        self._value = np.zeros(dim)
        self._fej = np.zeros(dim)

    def update(self, dx):
        """
        Implements the update operation through standard vector addition.
        v = v + dx
        
        :param dx: Additive error state correction
        """
        # Ensure the correction vector matches the variable size
        assert dx.shape[0] == self._size, f"Update dimension mismatch: {dx.shape[0]} vs {self._size}"
        
        # Standard vector addition
        # We use set_value to safely update the internal state
        self.set_value(self._value + dx)

    def clone(self):
        """
        Performs all the cloning (Deep Copy).
        Returns a new Vec object with the same current values.
        """
        # Create a new instance with the same dimension
        Clone = Vec(self._size)
        
        # Copy the current estimates
        # Note: We use copy() implicitly if set_value handles it, 
        # or explicit copy here to be safe.
        Clone.set_value(self.value().copy())
        Clone.set_fej(self.fej().copy())
        
        return Clone