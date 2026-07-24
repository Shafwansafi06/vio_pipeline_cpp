import numpy as np
from abc import ABC, abstractmethod

class Type(ABC):
    """
    Base class for estimated variables.
    
    This class defines how variables are represented or updated.
    Each variable is defined by its error state size and its location in the covariance matrix.
    """

    def __init__(self, size_):
        """
        Default constructor for our Type
        :param size_: degrees of freedom of variable (i.e., the size of the error state)
        """
        self._size = size_
        self._id = -1
        
        # Internal storage for value and first-estimate (FEJ)
        # These are initialized to None here, but derived classes 
        # should initialize them (e.g., to zero vectors) in their constructors.
        self._value = None
        self._fej = None

    def set_local_id(self, new_id):
        """
        Sets id used to track location of variable in the filter covariance
        Minimum ID is -1 (not in covariance).
        If ID > -1, this is the index location in the covariance matrix.
        """
        self._id = new_id

    def id(self):
        """Access to variable id"""
        return self._id

    def size(self):
        """Access to variable size (error state size)"""
        return self._size

    @abstractmethod
    def update(self, dx):
        """
        Update variable due to perturbation of error state.
        Must be implemented by derived classes.
        :param dx: Perturbation used to update the variable
        """
        pass

    def value(self):
        """Access variable's estimate"""
        return self._value

    def fej(self):
        """Access variable's first-estimate"""
        return self._fej

    def set_value(self, new_value):
        """
        Overwrite value of state's estimate
        """
        # In Python, we ensure dimensions match if _value is already initialized
        if self._value is not None:
            assert self._value.shape == new_value.shape, \
                f"Dimension mismatch in set_value: {self._value.shape} vs {new_value.shape}"
        self._value = new_value

    def set_fej(self, new_value):
        """
        Overwrite value of first-estimate
        """
        if self._fej is not None:
            assert self._fej.shape == new_value.shape, \
                f"Dimension mismatch in set_fej: {self._fej.shape} vs {new_value.shape}"
        self._fej = new_value

    @abstractmethod
    def clone(self):
        """
        Create a clone of this variable.
        Must be implemented by derived classes.
        """
        pass

    def check_if_subvariable(self, check):
        """
        Determine if passed variable is a sub-variable.
        If the passed variable is a sub-variable or the current variable this will return it.
        Otherwise it will return None.
        """
        # Default implementation returns None (nullptr in C++)
        return None