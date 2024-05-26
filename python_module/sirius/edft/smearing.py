from ..coefficient_array import CoefficientArray
from scipy.constants import physical_constants
from typing import Union
from types import ModuleType
from mpi4py import MPI
import numpy as np
from copy import deepcopy
from ..py_sirius import smearing as cxx_smearing
from ..py_sirius import Simulation_context, K_point_set

kb = (
    physical_constants["Boltzmann constant in eV/K"][0]
    / physical_constants["Hartree energy in eV"][0]
)


def make_smearing(label: str, kT: float, ctx: Simulation_context, kset: K_point_set):
    """
    smearing factory
    """

    if label == "fermi_dirac" or label == "fermi-dirac":
        func = cxx_smearing.fermi_dirac
    elif label == "gauss" or label == "gaussian":
        func = cxx_smearing.gaussian
    elif label == "methfessel_paxton" or label == "methfessel-paxton":
        func = cxx_smearing.methfessel_paxton
    elif label == "cold":
        func = cxx_smearing.cold
    else:
        raise NotImplementedError("invalid smearing: ", label)

    mo = ctx.max_occupancy()
    nel = ctx.unit_cell().num_valence_electrons

    return Smearing(
        func, kT=kT, mo=mo, num_electrons=nel, kw=kset.w, comm=kset.ctx().comm_k()
    )


def threaded_class(f):
    """Decorator for threaded application over CoefficientArray."""

    def _f(self, x, *args, **kwargs):
        if isinstance(x, CoefficientArray):
            out = type(x)()
            for k in x._data.keys():
                out[k] = f(self, x[k], *args, **kwargs)
            return out
        else:
            return f(self, x, *args, **kwargs)

    return _f


def find_chemical_potential(focc, ek, kT, nel, kw, comm):
    """
    find mu for Fermi-Dirac smearing

    Keyword Arguments:
    focc           -- occupation function, (x, kT) -> f_n
    ek             -- band energies
    kT             -- broadening [Ha]
    nel            -- number of electrons
    kw             -- k-point weights
    comm           -- communicator
    """

    def find_chemical_potential_(fun, mu0, tol=1e-11):
        """
        Arguments:
        fun        -- ne - fn(mu)
        mu0        -- initial guess
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
        assert np.abs(fun(mu)) < 1e-11
        return mu

    if isinstance(ek, CoefficientArray):
        vek = np.hstack(comm.allgather(ek.to_array()))
        vkw = deepcopy(ek)
        for k in vkw._data.keys():
            vkw[k] = np.ones_like(vkw[k]) * kw[k]
        vkw = np.hstack(comm.allgather(vkw.to_array()))
    else:
        vek = ek
        vkw = kw
    # update occupation numbers
    mu = find_chemical_potential_(
        lambda mu: nel - np.sum(vkw * focc(mu - vek, kT)), mu0=0
    )

    return mu


# SmearingType = Union[
#     ModuleType,
#     cxx_smearing.cold,
#     cxx_smearing.methfessel_paxton,
#     cxx_smearing.gaussian,
#     cxx_smearing.fermi_dirac,
# ]


class Smearing:
    def __init__(
        self,
        smearing_obj,
        kT,
        mo,
        kw,
        num_electrons,
        comm=MPI.COMM_SELF,
    ):
        """
        Arguments:
        smearing_obj  -- sirius.smearing.fermi_dirac/gaussian etc
        kT            -- broadening [Ha]
        mo            -- max occupancy, 1 or 2
        kw            -- k-point weights
        num_electrons -- number of electrons
        comm          -- k-point communicator
        """
        self.smearing_obj = smearing_obj
        self.kT = kT
        self.mo = mo
        self.kw = kw
        self.num_electrons = num_electrons
        self.comm = comm

    @threaded_class
    def _fd(self, x, kT):
        return self.mo * self.smearing_obj.occupancy(x, kT)

    @threaded_class
    def _delta(self, x, kT):
        return self.mo * self.smearing_obj.delta(x, kT)

    @threaded_class
    def _entropy(self, x, kT):
        return self.mo * self.smearing_obj.entropy(x, kT)

    def fn(self, ek):
        """
        Arguments:
        ek -- band energies [Ha]

        Returns:
        fn -- occupations
        mu -- chemical potential [Ha]

        """
        mu = find_chemical_potential(
            lambda x, kT: self._fd(x, kT),
            ek=ek,
            kT=self.kT,
            nel=self.num_electrons,
            kw=self.kw,
            comm=self.comm,
        )

        fi = self._fd(mu - ek, self.kT)
        return fi, mu

    def delta(self, x):
        """
        Arguments:
        x -- (μ - Ɛ)  [Ha]
        """

        return self._delta(x, self.kT)

    def entropy(self, x):
        """
        Arguments:
        x -- (μ - Ɛ)  [Ha]
        """
        entropy_loc = np.sum(self.kw * self._entropy(x, self.kT))
        loc = np.array(entropy_loc, dtype=np.float64)
        entropy = np.array(0.0, dtype=np.float64)
        self.comm.Allreduce([loc, MPI.DOUBLE], [entropy, MPI.DOUBLE], op=MPI.SUM)
        return np.ndarray.item(entropy)
