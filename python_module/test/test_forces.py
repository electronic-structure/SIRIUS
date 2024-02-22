import sirius
import json
import numpy as np
import pytest


@pytest.fixture
def ctx():
    def make_new_ctx(pw_cutoff, gk_cutoff):
        # lattice vectors
        a0 = 5.31148326730872
        lat = np.array([[0.0, a0, a0], [a0, 0.0, a0], [a0, a0, 0.0]])
        # basic input parameters
        inp = {
            "parameters": {
                "xc_functionals": ["XC_LDA_X", "XC_LDA_C_PZ"],
                "electronic_structure_method": "pseudopotential",
                "pw_cutoff": pw_cutoff,
                "gk_cutoff": gk_cutoff,
            },
            "control": {"verbosity": 0},
        }
        # create simulation context
        ctx = sirius.Simulation_context(json.dumps(inp))
        # set lattice vectors
        ctx.unit_cell().set_lattice_vectors(*lat)
        # add atom type
        ctx.unit_cell().add_atom_type("Si", "Si.json")
        # add atoms
        ctx.unit_cell().add_atom("Si", [0.0, 0.0, 0.0])
        ctx.unit_cell().add_atom("Si", [0.25, 0.5, 0.25])
        # intialize and return simulation context
        ctx.initialize()
        return ctx

    pw_cutoff = 20  # in a.u.^-1
    gk_cutoff = 8  # in a.u.^-1
    return make_new_ctx(pw_cutoff, gk_cutoff)


@pytest.fixture
def kgrid(ctx):
    k = 2
    return sirius.K_point_set(ctx, [k, k, k], [1, 1, 1], True)


@pytest.fixture
def dft(kgrid):
    dft = sirius.DFT_ground_state(kgrid)
    dft.initial_state()
    dft.find(1e-6, 1e-6, 1e-2, 100, False)
    return dft


stress_ref = np.array(
    [
        [1.29953431e-05, 1.27579700e-03, -7.79122237e-04],
        [1.27579700e-03, -2.24702544e-03, 1.27579700e-03],
        [-7.79122237e-04, 1.27579700e-03, 1.29953431e-05],
    ]
)

force_ref = np.array(
    [[0.1532408, -0.1532408],
     [-0.29043797, 0.29043797],
     [0.1532408, -0.1532408]]
)


def test_stress_and_force(dft):
    stress_obj = dft.stress()
    stress = np.array(stress_obj.calc_stress_total())

    force_obj = dft.forces()
    forces = np.array(force_obj.calc_forces_total(add_scf_corr=False))

    assert forces == pytest.approx(force_ref)
    assert stress == pytest.approx(stress_ref)
