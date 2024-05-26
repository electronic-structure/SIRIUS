from sirius import DFT_ground_state_find  # type: ignore

from sirius.ot import (  # type: ignore
    minimize,
    Energy,
    ApplyHamiltonian,
    c,
    ConstrainedGradient,
    make_kinetic_precond,
    get_c0_x,
)
import json
import pytest


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
    # run OT method
    _, niter, success, histE = minimize(
        x,
        f=lambda x: E(c(x, c0)),
        df=ConstrainedGradient(H, c0),
        maxiter=100,
        restart=10,
        mtype="PR",
        verbose=True,
        log=True,
        M=M,
        tol=1e-10,
    )

    eref = -8.241561782322835
    assert histE[-1] == pytest.approx(eref, rel=1e-8)
