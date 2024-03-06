from sirius import DFT_ground_state, K_point_set, Simulation_context
import json
import sirius.baarman as st
import sirius.ot as ot
import numpy as np
import pytest


def geodesic(X, Y, tau):
    """
    Keyword Arguments:
    X   --
    Y   --
    tau --
    """
    U, _ = st.stiefel_transport_operators(Y, X, tau)
    return U @ X


def p(X, Y, tau, E):
    """
    Keyword Arguments:
    X   --
    Y   --
    tau --
    """
    return E(geodesic(X, Y, tau))


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
                "smearing_width": 0.25,
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
        ctx.unit_cell().add_atom_type("Al", "Al.json")
        # add atoms
        ctx.unit_cell().add_atom("Al", [0.0, 0.0, 0.0])
        ctx.unit_cell().add_atom("Al", [0.0, 0.5, 0.0])
        # intialize and return simulation context
        ctx.initialize()
        return ctx

    pw_cutoff = 20  # in a.u.^-1
    gk_cutoff = 6  # in a.u.^-1
    return make_new_ctx(pw_cutoff, gk_cutoff)


@pytest.fixture
def kgrid(ctx):
    k = 3
    return K_point_set(ctx, [k, k, k], [0, 0, 0], True)


@pytest.fixture
def dft(kgrid):
    dft = DFT_ground_state(kgrid)
    dft.initial_state()
    tol = 1e-9
    dft.find(tol, tol, 1e-2, 100, False)
    return dft


def test_stiefel(dft):
    # extract wrappers from C++
    density = dft.density()
    potential = dft.potential()
    kset = dft.k_point_set()

    # create object to compute the total energy
    E = ot.Energy(kset, potential, density, ot.ApplyHamiltonian(potential, kset))
    # get PW coefficients from C++
    X = kset.C
    # get occupation numbers
    fn = kset.fn

    print("before E.compute(X)")
    _, HX = E.compute(X)
    print("after E.compute")
    dAdC = HX * fn * kset.w
    # project gradient of the free energy to the Stiefel manifold
    Y = st.stiefel_project_tangent(-dAdC, X)
    # print('evaluate energy along geodesic')
    ts = np.linspace(0, 1.5, 20)
    es = np.array([p(X, Y, t, lambda X: E(X)) for t in ts])
    es_ref = np.array(
        [
            -3.4647733910461684,
            -3.4647733910151826,
            -3.464773390984363,
            -3.4647733909534533,
            -3.464773390922475,
            -3.46477339089149,
            -3.4647733908605987,
            -3.464773390829641,
            -3.4647733907988147,
            -3.464773390767859,
            -3.464773390736947,
            -3.4647733907059735,
            -3.464773390675052,
            -3.4647733906441713,
            -3.4647733906132685,
            -3.4647733905822395,
            -3.464773390551416,
            -3.4647733905204747,
            -3.4647733904894618,
            -3.464773390458584,
        ]
    )
    assert es == pytest.approx(es_ref, abs=1e-8)
