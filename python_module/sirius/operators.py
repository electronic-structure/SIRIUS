from .py_sirius import Hamiltonian0, Wave_functions
from .coefficient_array import PwCoeffs
import numpy as np

class S_operator:
    """
    Description: compute S|Psi>
    """
    def __init__(self, hamiltonian, kpointset):
        """

        """
        self.hamiltonian = hamiltonian
        self.kpointset = kpointset

    def apply(self, cn):
        """
        """
        raise NotImplementedError

    def __matmul__(self, cn):
        return self.apply(cn)
