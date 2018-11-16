def _fermi_entropy(fn, dd):
    import numpy as np
    fn = np.array(fn).flatten()
    return np.sum(-fn * np.log(fn + dd * (1 - fn)) +
                  (1 - fn) * np.log(1 - fn + dd * fn))


def fermi_entropy(fn, dd=1e-4):
    """
    Keyword Arguments:
    fn --  occupation numbers
    dd -- regularization parameter
    """
    from ..coefficient_array import CoefficientArray
    import numpy as np
    if isinstance(fn, CoefficientArray):
        out = CoefficientArray(dtype=np.double, ctype=np.array)
        for key, val in fn._data.items():
            out[key] = _fermi_entropy(val, dd)
        return out
    else:
        return _fermi_entropy(fn, dd)


def df_fermi_entropy(fn, dd=1e-4):
    """
    Keyword Arguments:
    fn --  occupation numbers
    dd -- regularization parameter
    """
    from ..coefficient_array import CoefficientArray
    import numpy as np

    if isinstance(fn, CoefficientArray):
        out = CoefficientArray(dtype=np.double, ctype=np.array)
        for key, val in fn._data.items():
            out[key] = _df_fermi_entropy(val, dd)
        return out
    else:
        return _df_fermi_entropy(fn, dd)


def _df_fermi_entropy(fn, dd):
    import numpy as np
    fn = np.array(fn).flatten()
    return -fn * (1 - dd) / (fn + dd * (1 - fn)) + (1 - fn) * (-1 + dd) / (
        1 - fn + dd * fn) - np.log(fn + dd *
                                   (1 - fn)) - np.log(1 - fn + dd * fn)


def _occupancy_admissible_ds(y, fn, mag):
    import numpy as np

    if mag:
        fmax = 1
    else:
        fmax = 2

    d1 = -fn / np.ma.masked_array(
        y, mask=np.logical_or(y >= 0,
                              np.abs(fn) < 1e-10)
        # mask=y>=0
    )
    d2 = (fmax - fn) / np.ma.masked_array(
        y,
        mask=np.logical_or(y <= 0,
                           np.abs(fmax - fn) < 1e-10)
        # mask = y <=0
    )
    both = np.ma.hstack((d1, d2))
    ds = np.min(both)
    if isinstance(ds, np.ma.core.MaskedConstant):
        ds = 0

    return ds


def occupancy_admissible_ds(y, fn, mag=False):
    """
    Computes maximal admissible step length

    Keyword Arguments:
    y  -- direction
    fn -- band occupancy
    mag -- (Default False) magnetization
    """
    from functools import reduce
    from ..coefficient_array import CoefficientArray
    from mpi4py import MPI
    import numpy as np

    if isinstance(fn, CoefficientArray):
        lmin = reduce(
            min,
            [_occupancy_admissible_ds(y[k], fn[k], mag) for k in y.keys()])
        loc = np.array(lmin, dtype=np.float64)
        rcvBuf = np.array(0.0, dtype=np.float64)

        MPI.COMM_WORLD.Allreduce([loc, MPI.DOUBLE], [rcvBuf, MPI.DOUBLE],
                                 op=MPI.MIN)
        return np.asscalar(rcvBuf)
    else:
        return _occupancy_admissible_ds(y, fn, mag)


def _constrain_occupancy_gradient(dfn, fn, mag):
    """
    """
    from scipy.optimize import minimize, Bounds
    import numpy as np

    if mag:
        fmax = 1
    else:
        fmax = 2

    s = 100
    lb = -s * np.ones_like(fn)
    ub = s * np.ones_like(fn)
    ub[np.isclose(fn, fmax)] = 0
    lb[np.isclose(fn, 0)] = 0

    bounds = Bounds(lb, ub)
    x0 = dfn
    res = minimize(
        lambda x: np.linalg.norm(x - dfn),
        x0,
        bounds=bounds,
        constraints={
            'fun': lambda y: np.sum(y),
            "type": "eq"
        })
    y = res['x']
    return y


def constrain_occupancy_gradient(dfn, fn, mag=False):
    from ..coefficient_array import CoefficientArray
    import numpy as np

    if isinstance(fn, CoefficientArray):
        fn = fn.flatten(ctype=np.array)
        out = CoefficientArray(dtype=np.double, ctype=np.array)
        for key, val in fn._data.items():
            out[key] = _constrain_occupancy_gradient(dfn[key], val, mag)
        return out
    else:
        return _constrain_occupancy_gradient(dfn, fn, mag)


class FreeEnergy:
    def __init__(self, energy, temperature, H):
        self.energy = energy
        self.temperature = temperature
        self.omega_k = self.energy.kpointset.w
        self.H = H
        # assert (isinstance(H, ApplyHamiltonian))

        if self.H.hamiltonian.ctx().num_mag_dims() == 0:
            self.scale = 0.5
        else:
            self.scale = 1

    def __call__(self, cn, fn):
        """
        Keyword Arguments:
        cn   -- Planewave coefficients
        fn   -- occupations numbers
        """
        import numpy as np
        from mpi4py import MPI

        self.energy.kpointset.fn = fn
        E = self.energy(cn)
        S = fermi_entropy(self.scale * fn)
        entropy_loc = self.temperature * np.sum(
            np.array(list((self.omega_k * S)._data.values())))

        loc = np.array(entropy_loc, dtype=np.float64)
        entropy = np.array(0.0, dtype=np.float64)
        MPI.COMM_WORLD.Allreduce([loc, MPI.DOUBLE], [entropy, MPI.DOUBLE],
                                 op=MPI.SUM)

        return E - np.asscalar(entropy)

    def grad(self, cn, fn):
        """
        Keyword Arguments:
        self --
        cn   -- planewave coefficients
        fn   -- occupation numbers

        Note: It does not update density, potential

        Returns:
        dAdC -- gradient with respect to pw coeffs
        dAdf -- gradient with respect to occupation numbers
        """
        from ..coefficient_array import einsum
        import numpy as np

        # Compute dAdC
        self.energy.kpointset.fn = fn
        dAdC = self.H(cn, scale=False) * self.omega_k

        # Compute dAdf
        # TODO: k-point weights missing?
        dAdfn = np.real(
            einsum('ij,ij->j', cn.conj(), dAdC)
        ) - self.temperature * self.omega_k * self.scale * df_fermi_entropy(
            self.scale * fn)
        return dAdC * fn, dAdfn.flatten(ctype=np.array)
