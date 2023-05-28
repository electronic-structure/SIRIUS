import numpy as np
from scipy.constants import physical_constants
from ..coefficient_array import CoefficientArray
from .smearing import Smearing


def _s(x):
    """
    entropy term
    """
    x = np.array(x)
    out = np.zeros_like(x)
    ind = np.logical_or(np.isclose(x, 1), np.isclose(x, 0))
    z = x[~ind]
    out[~ind] = z**2 * np.log(z**2) + (1 - z**2) * np.log(1 - z**2)
    return out


def s(x):
    """
    entropy term
    """
    if isinstance(x, CoefficientArray):
        out = type(x)(dtype=x.dtype, ctype=np.array)
        for key, val in x._data.items():
            out[key] = _s(x[key])
        return out
    else:
        return _s(x)


class FreeEnergy:
    """
    copied from Baarman implementation
    """

    def __init__(self, E, T, smearing):
        """
        Keyword Arguments:
        energy      -- total energy object
        temperature -- temperature in Kelvin
        H           -- Hamiltonian
        smearing    --
        """
        self.energy = E
        self.T = T
        assert isinstance(smearing, Smearing)
        self.smearing = smearing
        if self.energy.kpointset.ctx().num_mag_dims() == 0:
            self.scale = 0.5
        else:
            self.scale = 1

    def __call__(self, cn, fn):
        """
        Keyword Arguments:
        cn   -- PW coefficients
        fn   -- occupations numbers
        """

        self.energy.kpointset.fn = fn
        E, HX = self.energy.compute(cn)
        entropy = self.smearing.entropy(fn)

        self.entropy = entropy
        self.ks_energy = E

        return E + entropy, HX
