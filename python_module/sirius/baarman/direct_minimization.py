from scipy.constants import physical_constants
import numpy as np
from ..coefficient_array import CoefficientArray, einsum
from functools import reduce
from mpi4py import MPI
from copy import deepcopy


def _fermi_entropy(fn, dd):
    fn = np.array(fn).flatten()
    return np.sum(fn * np.log(fn + dd * (1 - fn)) +
                  (1 - fn) * np.log(1 - fn + dd * fn))


def fermi_entropy(fn, dd=1e-4):
    """
    Keyword Arguments:
    fn --  occupation numbers
    dd -- regularization parameter
    """
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

    if isinstance(fn, CoefficientArray):
        out = CoefficientArray(dtype=np.double, ctype=np.array)
        for key, val in fn._data.items():
            out[key] = _df_fermi_entropy(val, dd)
        return out
    else:
        return _df_fermi_entropy(fn, dd)


def _df_fermi_entropy(fn, dd):
    fn = np.array(fn).flatten()
    return fn * (1 - dd) / (fn + dd * (1 - fn)) + (1 - fn) * (-1 + dd) / (
        1 - fn + dd * fn) + np.log(fn + dd *
                                   (1 - fn)) - np.log(1 - fn + dd * fn)


def _occupancy_admissible_ds(y, fn, mag):
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
    if isinstance(fn, CoefficientArray):
        lmin = reduce(
            min,
            [_occupancy_admissible_ds(y[k], fn[k], mag) for k in y.keys()],
            np.finfo(np.float64).max)
        loc = np.array(lmin, dtype=np.float64)
        rcvBuf = np.array(0.0, dtype=np.float64)

        MPI.COMM_WORLD.Allreduce([loc, MPI.DOUBLE], [rcvBuf, MPI.DOUBLE],
                                 op=MPI.MIN)
        return np.asscalar(rcvBuf)
    else:
        return _occupancy_admissible_ds(y, fn, mag)


def _constrain_occupancy_gradient(dfn, fn, mag, kweights):
    """
    """
    from scipy.optimize import minimize, Bounds

    if mag:
        fmax = 1
    else:
        fmax = 2

    s = 100

    # reshape into 1d array
    shape = fn.shape
    fn = fn.reshape(-1)
    dfn = dfn.reshape(-1)
    kweights = kweights.reshape(-1)

    lb = -s * np.ones_like(fn)
    ub = s * np.ones_like(fn)
    ub[np.isclose(fn, fmax)] = 0
    lb[np.isclose(fn, 0)] = 0

    bounds = Bounds(lb, ub)
    x0 = dfn
    res = minimize(
        lambda x: np.linalg.norm(x + dfn),
        x0,
        bounds=bounds,
        constraints={
            'fun': lambda y: np.sum(kweights*y),
            "type": "eq"
        })
    assert res['success']
    # restore shape
    y = res['x'].reshape(shape)

    return y


def constrain_occupancy_gradient(dfn, fn, comm, kweights, mag):
    """
    Keyword Arguments:
    dfn      -- unconstrained gradient (times -1)
    fn       -- occupation numbers
    comm     -- mpi4py communicators
    kweights -- k-point weights
    mag      -- magnetization
    """
    if isinstance(fn, CoefficientArray):
        fn = fn.flatten(ctype=np.array)
        # allgather and stack to columns
        kw = deepcopy(fn)
        for k in kw._data.keys():
            kw[k] = np.ones_like(kw[k]) * kweights[k]

        vfn_tmp = comm.allgather(fn.to_array())
        # count number of elements per rank
        proc_sizes = list(map(np.size, vfn_tmp))
        # get offset per rank
        offsets = np.hstack([0, np.cumsum(proc_sizes[:-1])]).astype(np.int)
        # allgather occupation numbers and gradient (because constraint is global)
        vfn = np.hstack(vfn_tmp)
        vdfn = np.hstack(comm.allgather(dfn.to_array()))
        vkw = np.hstack(comm.allgather(kw.to_array()))
        y = _constrain_occupancy_gradient(vdfn, vfn, mag, vkw)
        # get local contribution
        offset = offsets[comm.rank]
        lsize = proc_sizes[comm.rank]
        y_loc = y[offset:offset+lsize]
        # set from y_loc
        constrained_gradient = deepcopy(fn)
        constrained_gradient.from_array(y_loc)
        return constrained_gradient
    else:
        return _constrain_occupancy_gradient(dfn, fn, mag)


class FreeEnergy:
    def __init__(self, energy, temperature, H, delta=1e-4):
        """
        Keyword Arguments:
        energy      -- total energy object
        temperature -- temperature in Kelvin
        H           -- Hamiltonian
        delta       -- smoothing parameter for entropy gradient
        """
        self.energy = energy
        self.temperature = temperature
        self.omega_k = self.energy.kpointset.w
        self.comm = self.energy.kpointset.ctx().comm_k()
        self.H = H
        self.kb = (physical_constants['Boltzmann constant in eV/K'][0] /
                   physical_constants['Hartree energy in eV'][0])
        self.delta = delta
        if self.H.hamiltonian.ctx().num_mag_dims() == 0:
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
        E = self.energy(cn)
        S = fermi_entropy(self.scale * fn, dd=self.delta)
        entropy_loc = self.kb * self.temperature * np.sum(
            np.array(list((self.omega_k * S)._data.values())))

        loc = np.array(entropy_loc, dtype=np.float64)
        entropy = np.array(0.0, dtype=np.float64)
        MPI.COMM_WORLD.Allreduce([loc, MPI.DOUBLE], [entropy, MPI.DOUBLE],
                                 op=MPI.SUM)
        return E + np.asscalar(entropy)

    def grad(self, cn, fn):
        """
        Keyword Arguments:
        cn   -- planewave coefficients
        fn   -- occupation numbers

        Note: density, potential are not update here

        Returns:
        dAdC -- gradient with respect to pw coeffs
        dAdf -- gradient with respect to occupation numbers
        """

        # Compute dAdC
        self.energy.kpointset.fn = fn
        dAdC = self.H(cn, scale=False) * self.omega_k

        # Compute dAdf
        dAdfn = np.real(
            einsum('ij,ij->j', cn.conj(), dAdC)
        ) + self.kb * self.temperature * self.omega_k * self.scale * df_fermi_entropy(
            self.scale * fn, dd=self.delta)
        return dAdC * fn, dAdfn.flatten(ctype=np.array)
