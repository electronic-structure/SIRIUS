from .helpers import *
from .coefficient_array import CoefficientArray, PwCoeffs
from .py_sirius import *
from .py_sirius import K_point_set, Density
from .logger import Logger
from .operators import S_operator
import numpy as np
from numpy import array, zeros
__all__ = ["ot", "baarman", "bands", "edft"]


class OccupancyDescriptor(object):
    """
    Accessor for occupation numbers
    """
    def __set__(self, instance, value):
        for key, v in value._data.items():
            k, ispn = key
            # append with zeros if necessary
            nb = instance.ctx().num_bands()
            f = zeros(nb)
            ll = list(array(v).flatten())
            f[:len(ll)] = ll
            instance[k].set_band_occupancy(ispn, f)
        instance.sync_band_occupancies()

    def __get__(self, instance, owner):

        out = CoefficientArray(dtype=np.double, ctype=np.array)

        for k in range(len(instance)):
            for ispn in range(instance.ctx().num_spins()):
                key = k, ispn
                out[key] = np.array(instance[k].band_occupancy(ispn))
        return out


class PWDescriptor(object):
    """
    Accessor for wave-function coefficients
    """
    def __set__(self, instance, value):
        from .helpers import store_pw_coeffs
        store_pw_coeffs(instance, value)

    def __get__(self, instance, owner):
        return PwCoeffs(instance)


class KPointWeightDescriptor(object):
    """
    Accessor for k-point weights
    """
    def __get__(self, instance, owner):

        out = CoefficientArray(dtype=np.double, ctype=np.array)

        for k in range(len(instance)):
            for ispn in range(instance.ctx().num_spins()):
                key = k, ispn
                out[key] = np.array(instance[k].weight())
        return out


class BandEnergiesDescriptor(object):
    """
    Accessor for band energies
    """
    def __get__(self, instance, owner):

        out = CoefficientArray(dtype=np.double, ctype=np.array)

        for k in range(len(instance)):
            for ispn in range(instance.ctx().num_spins()):
                key = k, ispn
                out[key] = np.array(instance[k].band_energies(ispn))
        return out


    def __set__(self, instance, value):
        for key, val in value._data.items():
            k, ispn = key
            for j, v in enumerate(val):
                instance[k].set_band_energy(j, ispn, v)
        instance.sync_band_energies()


class DensityDescriptor(object):
    def __init__(self, i):
        self.i = i

    def __get__(self, instance, owner):
        return np.array(instance.f_pw_local(self.i))

    def __set__(self, instance, value):
        instance.f_pw_local(self.i)[:] = value


class DensityMatrixDescriptor(object):
    def __get__(self, instance, owner):
        return np.array(instance.density_matrix())

    def __set__(self, instance, value):
        instance.density_matrix()[:] = value


K_point_set.fn = OccupancyDescriptor()
K_point_set.C = PWDescriptor()
K_point_set.w = KPointWeightDescriptor()
K_point_set.e = BandEnergiesDescriptor()

Density.rho = DensityDescriptor(0)
Density.mx = DensityDescriptor(1)
Density.my = DensityDescriptor(2)
Density.mz = DensityDescriptor(3)
Density.dm = DensityMatrixDescriptor()
