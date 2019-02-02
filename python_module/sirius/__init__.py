# from .bands import *
from .helpers import *
from .coefficient_array import CoefficientArray, PwCoeffs, inner, l2norm, diag
# from .baarman import *
from .py_sirius import *
from .py_sirius import K_point_set
import numpy as np
from numpy import array, zeros
# from . import ot
__all__ = ["ot", "baarman", "bands"]


class OccupancyDescriptor(object):
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
        import numpy as np

        out = CoefficientArray(dtype=np.double, ctype=np.array)

        for k in range(len(instance)):
            for ispn in range(instance.ctx().num_spins()):
                key = k, ispn
                out[key] = np.array(instance[k].band_occupancy(ispn))
        return out


class PWDescriptor(object):
    def __set__(self, instance, value):
        from .helpers import store_pw_coeffs
        store_pw_coeffs(instance, value)

    def __get__(self, instance, owner):
        return PwCoeffs(instance)


class KPointWeightDescriptor(object):
    def __get__(self, instance, owner):

        out = CoefficientArray(dtype=np.double, ctype=np.array)

        for k in range(len(instance)):
            for ispn in range(instance.ctx().num_spins()):
                key = k, ispn
                out[key] = np.array(instance[k].weight())
        return out


class BandEnergiesDescriptor(object):
    def __get__(self, instance, owner):
        from .coefficient_array import CoefficientArray

        out = CoefficientArray(dtype=np.double, ctype=np.array)

        for k in range(len(instance)):
            for ispn in range(instance.ctx().num_spins()):
                key = k, ispn
                out[key] = instance[k].band_energies(ispn)
        return out

    def __set__(self, instance, value):
        for key, val in value._data.items():
            k, ispn = key
            for j, v in enumerate(val):
                instance[k].set_band_energy(j, ispn, v)


K_point_set.fn = OccupancyDescriptor()
K_point_set.C = PWDescriptor()
K_point_set.w = KPointWeightDescriptor()
K_point_set.e = BandEnergiesDescriptor()
