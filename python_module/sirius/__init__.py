from .bands import *
from .helpers import *
from .baarman import *
from .py_sirius import *
from .coefficient_array import *
from . import ot

from .py_sirius import K_point_set


class OccupancyDescriptor(object):
    def __set__(self, instance, value):
        for key, v in value._data.items():
            k, ispn = key
            instance[k].set_band_occupancy(ispn, v)

    def __get__(self, instance, owner):
        from .coefficient_array import CoefficientArray
        import numpy as np

        out = CoefficientArray(dtype=np.double, ctype=np.array)

        for k in range(len(instance)):
            for ispn in range(instance.ctx().num_spins()):
                key = k, ispn
                out[key] = instance[k].band_occupancy(ispn)
        return out


K_point_set.fn = OccupancyDescriptor()
