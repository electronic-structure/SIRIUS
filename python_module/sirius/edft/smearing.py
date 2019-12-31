from copy import deepcopy

import numpy as np
from mpi4py import MPI
from scipy import special
from scipy.constants import physical_constants

from sirius.coefficient_array import CoefficientArray, diag, inner, threaded

kb = (
    physical_constants["Boltzmann constant in eV/K"][0]
    / physical_constants["Hartree energy in eV"][0]
)


def find_chemical_potential(fun, mu0, tol=1e-10):
    """
    fun        -- ne - fn(mu)
    mu0        -- initial gues energies
    """
    mu = mu0
    de = 0.1
    sp = 1
    s = 1
    nmax = 1000
    counter = 0

    while np.abs(fun(mu)) > tol and counter < nmax:
        sp = s
        s = 1 if fun(mu) > 0 else -1
        if s == sp:
            de *= 1.25
        else:
            # decrease step size if we change direction
            de *= 0.25
        mu += s * de
        counter += 1
    assert np.abs(fun(mu)) < 1e-10
    return mu


@threaded
def df_fermi_entropy_reg(fn, dd):
    """
    Derivative of the regularized Fermi-Dirac entropy.

    Keyword Arguments:
    fn -- occupation numbers
    dd -- regularization parameter
    """
    fn = np.array(fn).flatten()
    return (
        fn * (1 - dd) / (fn + dd * (1 - fn))
        + (1 - fn) * (-1 + dd) / (1 - fn + dd * fn)
        + np.log(fn + dd * (1 - fn))
        - np.log(1 - fn + dd * fn)
    )


@threaded
def fermi_entropy_reg(fn, dd):
    """
    Regularized Fermi-Dirac entropy.

    Keyword Arguments:
    fn -- occupation numbers
    dd -- regularization parameter
    """

    fn = np.array(fn).flatten()
    return np.sum(fn * np.log(fn + dd * (1 - fn)) + (1 - fn) * np.log(1 - fn + dd * fn))


@threaded
def fermi_entropy(fn):
    """
    Fermi-Dirac entropy
    """
    idx = np.logical_or(np.isclose(fn, 0, atol=1e-20),
                        np.isclose(fn, 1, atol=1e-15))
    fi = fn[~idx]
    return np.sum(fi * np.log(fi) + (1 - fi) * np.log(1 - fi))


@threaded
def fermi_dirac(x):
    # np.exp(nlogm16) < 1e-16
    # nlogm16 = -37
    # is_one = x < nlogm16
    is_one = x < -50
    is_zero = x > 40
    out = np.zeros_like(x)
    out[is_one] = 1
    out[is_zero] = 0
    ii = np.logical_and(~is_one, ~is_zero)
    out[ii] = 1 / (1 + np.exp(x[ii]))
    return out


@threaded
def inverse_fermi_dirac(f):
    """
    """
    en = np.zeros_like(f, dtype=np.double)
    is_zero = np.isclose(f, 0, atol=1e-20)
    is_one = np.isclose(f, 1, rtol=1e-15)
    en[is_zero] = 50 + np.arange(0, np.sum(is_zero))
    # make sure we do not get degenerate band energies
    en[is_one] = -50 - np.arange(0, np.sum(is_one))
    ii = np.logical_not((np.logical_or(is_zero, is_one)))
    en[ii] = np.log(1 / f[ii] - 1)
    return en


@threaded
def fermi_dirac_reg(x, dd):
    """
    x = (ε-μ) / kT

    Keyword Arguments:
    x  --
    T  -- temperature in Kelvin
    dd -- regularization parameter
    """
    import scipy

    def fguess(x):
        return 1 / (1 + np.exp(x))

    rootf = lambda f: df_fermi_entropy_reg(f, dd=dd) + x
    res = scipy.optimize.root(rootf, x0=fguess(x))
    assert res["success"]
    return res["x"]


def chemical_potential(ek, T, nel, fermi_function, kw, comm):
    """
    find mu for Fermi-Dirac smearing

    Keyword Arguments:
    ek             -- band energies
    T              -- temperature
    nel            -- number of electrons
    fermi_function -- f((e-mu)/kT)
    kw             -- k-point weights
    comm           -- communicator
    """

    kT = kb * T
    if isinstance(ek, CoefficientArray):
        vek = np.hstack(comm.allgather(ek.to_array()))
        vkw = deepcopy(ek)
        for k in vkw._data.keys():
            vkw[k] = np.ones_like(vkw[k]) * kw[k]
        vkw = np.hstack(comm.allgather(vkw.to_array()))
    else:
        vek = ek
        vkw = kw
    # print('fermi_function.shape', fermi_function(vek).shape)
    # update occupation numbers
    mu = find_chemical_potential(
        lambda mu: nel - np.sum(vkw * fermi_function((vek - mu) / kT)), mu0=0
    )

    return mu


@threaded
def inverse_efermi_spline(fn):
    """
    inverts f(x)
    where x is (epsilon-mu) / kT

    Returns:
    x
    """

    ifu = fn > 0.5
    xi = np.zeros_like(fn)

    # remove numerical noise
    fn = np.where(fn < 0, 0, fn)

    ub = 8.0
    lb = -5.0

    if0 = efermi_spline(np.array(ub)) > fn
    if1 = efermi_spline(np.array(lb)) < fn

    ifb = np.logical_or(if0, if1)

    # and ~ifb => and is not at the boundary
    iifu = np.logical_and(ifu, ~ifb)
    iifl = np.logical_and(~ifu, ~ifb)
    xi[iifu] = (1 - np.sqrt(1 - 2 * np.log(2 - 2 * fn[iifu]))) / np.sqrt(2)
    xi[iifl] = (-1 + np.sqrt(1 - 2 * np.log(2 * fn[iifl]))) / np.sqrt(2)

    xi[if0] = ub
    xi[if1] = lb

    return xi


@threaded
def efermi_spline(x):
    """
    ...
    """
    x = np.array(x, copy=False)
    im = x < 0
    um = ~im

    out = np.empty_like(x)

    # # truncate
    # a = 5
    # x = np.where(x > a, a, x)
    # x = np.where(x < -a, -a, x)

    out[im] = 1 - 1 / 2 * np.exp(1 / 2 - (-1 / np.sqrt(2) + x[im]) ** 2)
    out[um] = 1 / 2 * np.exp(1 / 2 - (1 / np.sqrt(2) + x[um]) ** 2)

    return out


@threaded
def gaussian_spline_entropy(fn):
    """
    todo: add a docstring
    """
    from scipy.special import erfc

    S = np.empty_like(fn)
    ifu = fn > 0.5
    ifl = ~ifu

    dd = 1e-10

    S[ifu] = (
        np.sqrt(np.e * np.pi) * erfc(np.sqrt(0.5 - np.log(2 - 2 * fn[ifu] + dd)))
        - 2
        * np.sqrt(2)
        * (-1 + fn[ifu])
        * (-1 + np.sqrt(1 - 2 * np.log(2 - 2 * fn[ifu] + dd)))
    ) / 4.0
    S[ifl] = (
        np.sqrt(np.e * np.pi) * erfc(np.sqrt(0.5 - np.log(2 * fn[ifl] + dd)))
    ) / 4.0 + (fn[ifl] * (-1 + np.sqrt(1 - 2 * np.log(2 * fn[ifl] + dd)))) / np.sqrt(2)
    # feq0 = np.isclose(fn, 0, atol=1e-20)
    # feq1 = np.isclose(fn, 1, rtol=1e-14)
    # S[feq0] = 0
    # S[feq1] = 0

    return S


@threaded
def gaussian_spline_entropy_df(fn):
    """
    todo: add a docstring
    """

    dSdf = np.empty_like(fn)
    ifu = fn > 0.5
    ifl = ~ifu

    dd = 1e-10

    dSdf[ifu] = (
        1
        + (-1 - dd / (2 + dd - 2 * fn[ifu]) + 2 * np.log(2 + dd - 2 * fn[ifu]))
        / np.sqrt(1 - 2 * np.log(2 + dd - 2 * fn[ifu]))
    ) / np.sqrt(2)
    dSdf[ifl] = (
        -1
        + (1 + dd / (dd + 2 * fn[ifl]) - 2 * np.log(dd + 2 * fn[ifl]))
        / np.sqrt(1 - 2 * np.log(dd + 2 * fn[ifl]))
    ) / np.sqrt(2)

    return dSdf


@threaded
def gaussian_spline_entropy_x(x):
    """
    todo: add a docstring
    """
    z = np.abs(x)
    S = 0.25 * (2 * np.exp(-z * (np.sqrt(2) + z)) * z
                + np.sqrt(np.e * np.pi) * special.erfc(1 / np.sqrt(2) + z))

    return S


class Smearing:
    def __init__(self, T, nel, nspin, kw, comm=MPI.COMM_SELF):
        """
        Keyword Arguments:
        T    -- Temperature in Kelvin
        nel  -- number of electrons
        """
        self.kT = kb * T
        self.T = T
        self.nel = nel
        self.nspin = nspin
        self.kw = kw
        self.comm = comm
        # assert nspin == 2  # magnetic case

    @property
    def mo(self):
        """
        Returns:
        maximal occupancy, either 1 or 2
        """

        factor = {1: 2, 2: 1}
        return factor[self.nspin]


class GaussianSplineSmearingReg(Smearing):
    """
    Gaussian spline smearing
    """

    def __init__(self, T, nel, nspin, kw, comm=MPI.COMM_SELF):
        """
        Keyword Arguments:
        T    -- Temperature in Kelvin
        nel  -- number of electrons
        kw -- k-point weights
        """
        super().__init__(T=T, nel=nel, nspin=nspin, kw=kw, comm=comm)
        self.T = T

    def entropy(self, fn):
        # x = inverse_efermi_spline(fn)
        return -self.kT * np.sum(self.kw * gaussian_spline_entropy(fn))

    def ek(self, fn):
        x = inverse_efermi_spline(fn, a=10) * self.kT
        return x

    def fn(self, ek):
        """
        Keyword Arguments:
        ek -- band energies
        """
        mu = chemical_potential(
            ek,
            T=self.T,
            nel=self.nel,
            fermi_function=efermi_spline,
            kw=self.kw,
            comm=self.comm,
        )
        occ = efermi_spline((ek - mu) / self.kT)
        return occ * self.mo

    def dSdf(self, fn):
        """
        """
        return -self.kT * gaussian_spline_entropy_df(fn)


class GaussianSplineSmearing(Smearing):
    """
    Gaussian spline smearing
    """

    def __init__(self, T, nel, nspin, kw, comm=MPI.COMM_SELF):
        """
        Keyword Arguments:
        T    -- Temperature in Kelvin
        nel  -- number of electrons
        kw -- k-point weights
        """

        super().__init__(T=T, nel=nel, nspin=nspin, kw=kw, comm=comm)
        self.T = T

    def entropy(self, fn):
        x = inverse_efermi_spline(fn)
        return -self.kT * np.sum(self.kw * gaussian_spline_entropy_x(x))

    def ek(self, fn):
        x = inverse_efermi_spline(fn) * self.kT
        return x

    def fn(self, ek):
        """
        Keyword Arguments:
        ek -- band energies
        """
        factor = {1: 2, 2: 1}
        mu = chemical_potential(
            ek,
            T=self.T,
            nel=self.nel / factor[self.nspin],
            fermi_function=efermi_spline,
            kw=self.kw,
            comm=self.comm)
        occ = efermi_spline((ek - mu) / self.kT)
        return occ * factor[self.nspin], mu

    def dSdf(self, fn):
        """
        """
        x = inverse_efermi_spline(fn)
        return -self.kT * x


class FermiDiracSmearing(Smearing):
    """
    Fermi-Dirac smearing
    """

    def __init__(self, T, nel, nspin, kw, comm=MPI.COMM_SELF):
        """
        Keyword Arguments:
        T    -- Temperature in Kelvin
        nel  -- number of electrons
        kset -- k-point set
        """
        super().__init__(T, nel, nspin, kw, comm)

    def entropy(self, fn):
        factor = {1: 2, 2: 1}
        S = self.kT * np.sum(self.kw * fermi_entropy(fn / factor[self.nspin]))
        return S

    def fn(self, ek):
        """
        Keyword Arguments:
        ek  -- band energies
        """
        factor = {1: 2, 2: 1}
        mu = chemical_potential(
            ek,
            T=self.T,
            nel=self.nel / factor[self.nspin],
            fermi_function=fermi_dirac,
            kw=self.kw,
            comm=self.comm,
        )
        occ = fermi_dirac((ek - mu) / self.kT)
        return occ * factor[self.nspin], mu

    def ek(self, fn):
        """
        Keyword Arguments:
        fn  -- occupation numbers
        """

        fn_loc = fn / self.mo
        x = inverse_fermi_dirac(fn_loc)
        return x * self.kT


class FermiDiracSmearingReg(Smearing):
    """
    Fermi-Dirac smearing with regularization
    """

    def __init__(self, T, nel, nspin, kw, comm=MPI.COMM_SELF, dd=1e-10):
        """
        Keyword Arguments:
        T     -- Temperature in Kelvin
        nel   -- number of electrons
        nspin -- number of spins: 1 or 2
        kw    -- k-point weights
        comm  -- k-point communicator
        dd    -- regularization parameter
        """
        super().__init__(T=T, nel=nel, nspin=nspin, kw=kw, comm=comm)
        self.dd = dd
        if nspin == 1:
            self.scale = 0.5
        elif nspin == 2:
            self.scale = 1
        else:
            raise ValueError("npsin must be 1 or 2")

    def fn(self, ek):
        """
        occupation numbers
        TODO: this is not correct

        Keyword Arguments:
        ek -- band energies

        Returns:
        fn  -- occupation numbers
        mu  -- chemical potential
        """

        factor = {1: 2, 2: 1}
        mu = chemical_potential(
            ek,
            T=self.T,
            nel=self.nel / factor[self.nspin],
            kw=self.kw,
            comm=self.comm,
            fermi_function=lambda x: fermi_dirac(x),
        )
        fn = self.nspin * fermi_dirac((ek - mu) / self.kT)
        return fn, mu

    def entropy(self, fn):
        """
        computes the entropy

        Keyword Arguments:
        fn -- occupation numbers
        """

        S = fermi_entropy_reg(self.scale * fn, dd=self.dd)
        entropy_loc = self.kT * np.sum(np.array(list((self.kw * S)._data.values())))
        loc = np.array(entropy_loc, dtype=np.float64)
        entropy = np.array(0.0, dtype=np.float64)
        MPI.COMM_WORLD.Allreduce([loc, MPI.DOUBLE], [entropy, MPI.DOUBLE], op=MPI.SUM)
        return np.asscalar(entropy)

    def dSdf(self, fn):
        """
        computes the gradient wrt fn of the entropy

        Keyword Arguments:
        fn -- occupation numbers
        """
        from ..baarman.direct_minimization import df_fermi_entropy

        df = self.scale * df_fermi_entropy(self.scale * fn, dd=self.dd)
        return df


def make_gaussian_spline_smearing(T, ctx, kset):
    nel = ctx.unit_cell().num_valence_electrons
    mo = ctx.max_occupancy()
    nspin = {1: 2, 2: 1}
    # assert mo == 1
    smearing = GaussianSplineSmearing(T=T,
                                      nel=nel,
                                      nspin=nspin[mo],
                                      kw=kset.w,
                                      comm=kset.ctx().comm_k())
    return smearing


def make_fermi_dirac_smearing(T, ctx, kset):
    nel = ctx.unit_cell().num_valence_electrons
    mo = ctx.max_occupancy()
    # assert mo == 1
    nspin = {1: 2, 2: 1}
    smearing = FermiDiracSmearing(T=T,
                                  nel=nel,
                                  nspin=nspin[mo],
                                  kw=kset.w,
                                  comm=kset.ctx().comm_k())
    return smearing
