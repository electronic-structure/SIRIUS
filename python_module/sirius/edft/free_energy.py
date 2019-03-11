import numpy as np
from scipy.constants import physical_constants
from ..coefficient_array import CoefficientArray


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
    def __init__(self, H, energy, T):
        """

        """
        self.H = H
        self.energy = energy
        self.kw = energy.kpointset.w
        self.T = T
        self.kb = (physical_constants['Boltzmann constant in eV/K'][0] /
                   physical_constants['Hartree energy in eV'][0])

    def entropy(self, fn):
        ns = 2 if self.energy.kpointset.ctx().num_mag_dims() == 0 else 1
        S = s(np.sqrt(fn/ns))
        return self.kb * self.T * np.real(np.sum(self.kw*S))

    def __call__(self, X, fn):
        """
        Keyword Arguments:
        X --
        f --
        """
        # self.energy.kpointset.sync_band_occupancies()
        self.energy.kpointset.fn = fn
        ns = 2 if self.energy.kpointset.ctx().num_mag_dims() == 0 else 1
        entropy = s(np.sqrt(fn/ns))
        E = self.energy(X)
        return E + self.kb * self.T * np.real(np.sum(self.kw*entropy))
