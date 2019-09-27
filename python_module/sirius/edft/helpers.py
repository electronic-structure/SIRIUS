from sirius.coefficient_array import threaded
import numpy as np

@threaded
def has_enough_bands(fn):
    fn_sorted = np.sort(fn)
    return fn_sorted[0] < 1e-10
