import json

import numpy as np
import pytest
from sirius import DFT_ground_state_find  # type: ignore
from sirius.coefficient_array import CoefficientArray  # type: ignore
from sirius.ot import (  # type: ignore
    ApplyHamiltonian,
    ConstrainedGradient,
    Energy,
    c,
    get_c0_x,
    make_kinetic_precond,
    minimize,
)


def assert_no_matrix(obj, where):
    """Fail if obj (or any block of a CoefficientArray) is a np.matrix."""
    blocks = obj._data.values() if isinstance(obj, CoefficientArray) else [obj]
    for b in blocks:
        assert not isinstance(b, np.matrix), f"np.matrix leaked into {where}"


class CheckedEnergy:
    """f(x) = E(c(x, c0)), checking the OT coefficients and the input to Energy."""

    def __init__(self, E, c0):
        self.E = E
        self.c0 = c0

    def __call__(self, x):
        assert_no_matrix(x, "minimize: x")
        cx = c(x, self.c0)
        assert_no_matrix(cx, "Energy input c(x, c0)")
        return self.E(cx)


class CheckedGradient:
    """Wraps ConstrainedGradient, checking its input and both outputs."""

    def __init__(self, df):
        self.df = df

    def __call__(self, x):
        assert_no_matrix(x, "ConstrainedGradient input")
        Gx, Hx = self.df(x)
        assert_no_matrix(Gx, "ConstrainedGradient Gx")
        assert_no_matrix(Hx, "ConstrainedGradient Hx")
        return Gx, Hx


class CheckedPreconditioner:
    """Wraps a preconditioner, checking the result of M @ g and M * g."""

    def __init__(self, M):
        self.M = M

    def __neg__(self):
        return CheckedPreconditioner(-self.M)

    def __matmul__(self, g):
        out = self.M @ g
        assert_no_matrix(out, "preconditioner M @ g")
        return out

    def __mul__(self, g):
        out = self.M * g
        assert_no_matrix(out, "preconditioner M * g")
        return out

    __rmul__ = __mul__


@pytest.fixture
def res():
    res = DFT_ground_state_find(
        num_dft_iter=2,
        config={
            **json.load(open("sirius.json", "r")),
            **{"control": {"processing_unit": "cpu", "verbosity": 2}},
        },
    )
    return res


def test_ot(res):
    kset = res["kpointset"]
    dft_obj = res["dft_gs"]
    potential = dft_obj.potential()
    density = dft_obj.density()
    # Hamiltonian, provides gradient H|Ψ>
    H = ApplyHamiltonian(potential, kset)
    # create object to compute the total energy
    E = Energy(kset, potential, density, H)

    c0, x = get_c0_x(kset)
    # kinetic preconditioner
    M = make_kinetic_precond(kset, c0, asPwCoeffs=True, eps=1e-3)
    assert_no_matrix(c0, "c0")
    assert_no_matrix(x, "initial x")
    # run OT method; f, df and M are wrapped to check that no np.matrix reaches
    # minimize / Energy
    _, niter, success, histE = minimize(
        x,
        f=CheckedEnergy(E, c0),
        df=CheckedGradient(ConstrainedGradient(H, c0)),
        maxiter=100,
        restart=10,
        mtype="PR",
        verbose=True,
        log=True,
        M=CheckedPreconditioner(M),
        tol=1e-10,
    )

    eref = -8.241561782322835
    assert histE[-1] == pytest.approx(eref, rel=1e-8)
