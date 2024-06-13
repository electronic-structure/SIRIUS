from sirius import DFT_ground_state, K_point_set, Simulation_context  # type: ignore
import json
import numpy as np
import pytest
from edft_gradients import check_gradient  # type: ignore


@pytest.fixture
def ctx():
    def make_new_ctx(pw_cutoff, gk_cutoff):
        # lattice vectors
        lat = np.array(
            [
                [5.397057814307557, 0.0, 0.0],
                [2.698528907153779, 4.673989172883661, 0.0],
                [2.698528907153779, 2.557996390961221, 4.406679252451386],
            ]
        )

        # basic input parameters
        inp = {
            "parameters": {
                "xc_functionals": ["XC_LDA_X", "XC_LDA_C_PZ"],
                "electronic_structure_method": "pseudopotential",
                "smearing_width": 0.025,
                "pw_cutoff": pw_cutoff,
                "gk_cutoff": gk_cutoff,
            },
            "control": {"verbosity": 0},
        }
        # create simulation context
        ctx = Simulation_context(json.dumps(inp))
        # set lattice vectors
        ctx.unit_cell().set_lattice_vectors(*lat)
        # add atom type
        ctx.unit_cell().add_atom_type("Al", "Al-sssp.json")
        # add atoms
        ctx.unit_cell().add_atom("Al", [0.0, 0.0, 0.0])
        # intialize and return simulation context
        ctx.initialize()
        return ctx

    pw_cutoff = 20  # in a.u.^-1
    gk_cutoff = 6  # in a.u.^-1
    return make_new_ctx(pw_cutoff, gk_cutoff)


@pytest.fixture
def kgrid(ctx):
    k = 2
    return K_point_set(ctx, [k, k, k], [0, 0, 0], True)


@pytest.fixture
def dft(kgrid):
    dft = DFT_ground_state(kgrid)
    dft.initial_state()
    tol = 1e-9
    dft.find(tol, tol, 1e-2, 1, False)
    return dft


def test_mvp2_gradient(dft):
    # extract wrappers from C++
    kT = 0.025  # Ha
    slope, slope_fd = check_gradient(dft, kT, fx=1, kappa=0)
    assert np.isclose(slope, slope_fd, atol=1e-5)

    slope, slope_fd = check_gradient(dft, kT, fx=0, kappa=1)
    assert np.isclose(slope, slope_fd, atol=1e-5)
