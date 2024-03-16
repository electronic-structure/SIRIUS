from sirius import DFT_ground_state, K_point_set, Simulation_context
import json
import numpy as np
import pytest


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
    from sirius import S_operator, Sinv_operator
    from sirius.coefficient_array import diag, inner, l2norm
    from sirius.ot import ApplyHamiltonian, Energy
    from sirius.edft.neugebauer import _solve
    from sirius.edft.neugebauer import loewdin, grad_eta
    from sirius.edft import make_fermi_dirac_smearing
    from sirius.edft.free_energy import FreeEnergy

    # extract wrappers from C++
    density = dft.density()
    potential = dft.potential()
    kset = dft.k_point_set()
    ctx = kset.ctx()

    H = ApplyHamiltonian(potential, kset)
    E = Energy(kset, potential, density, H)
    sinv_op = Sinv_operator(ctx, potential, kset)
    s_op = S_operator(ctx, potential, kset)
    T = 100

    X = kset.C
    fn = kset.fn

    smearing = make_fermi_dirac_smearing(T, ctx, kset)

    M = FreeEnergy(E=E, T=T, smearing=smearing)
    kw = kset.w
    F0, _ = M(X, fn)
    print("initial energy:", F0)
    eta0 = kset.e
    print("eta0", eta0)
    w, U = diag(eta0).eigh()
    ek = w
    # rotate (e.g. in this case permute X)
    X = X @ U
    eta = diag(w)
    fn, mu = smearing.fn(ek)

    # evaluate total energy, gradient, overlap
    F0i, Hx = M(X, fn)

    # compute gradients
    HX = Hx * kw
    Hij = X.H @ HX

    g_eta = grad_eta(Hij, ek, fn, T, kw, mo=kset.ctx().max_occupancy())

    kappa = 1

    XhKHX = (X.H @ HX) * fn
    XhKSX = X.H @ (s_op @ X)

    # Lagrange multipliers
    LL = _solve(XhKSX, XhKHX)

    g_X = sinv_op @ HX * fn - X @ LL
    # delta_X = -K*(HX - X@LL) / kw
    delta_X = -g_X
    delta_eta = kappa * (Hij - kw * diag(ek)) / kw

    check_constraint = l2norm(X.H @ (s_op @ delta_X))
    if np.abs(check_constraint) > 1e-11:
        print(
            f"ERROR: 1st order orthogonality constraint FAILED, {check_constraint:.10g}"
        )
        raise Exception

    G_X = delta_X
    G_eta = delta_eta
    dts = np.linspace(0, 0.5, 15)
    fx = 1

    slope = np.real(2 * inner(g_X, fx * G_X) + inner(g_eta, G_eta))
    Fs = []
    Hs = []  # gradients along lines
    dt_slope = 1e-6
    dts = np.concatenate([np.array([0, dt_slope]), dts[1:]])
    for dt in dts:
        X_new = X + dt * fx * G_X
        eta_new = eta + dt * G_eta
        w, Ul = eta_new.eigh()
        Q_new = loewdin(X_new @ Ul, s_op)
        # update occupation numbers
        fn_new, mu = smearing.fn(w)
        print("orth err: %.3g, mu: %.5g" % (l2norm(X_new @ Ul - Q_new), mu))
        Floc, Hloc = M(Q_new, fn_new)
        Fs.append(Floc)
        Hs.append(Hloc)

    fit = np.polyfit(dts, Fs, deg=2)

    print("slope (fit): %.6g" % fit[1])
    print("slope      : %.6g" % slope)
    slope_fd = (Fs[1] - Fs[0]) / dt_slope
    print("slope (fd) : %.6g" % slope_fd)

    fid = 10
    F1 = Fs[fid]
    xi_trial = dts[fid]
    b = slope
    c = Fs[0]
    a = (F1 - b * xi_trial - c) / xi_trial**2
    fitl = np.array([a, b, c])
    err = np.linalg.norm(np.polyval(fitl, dts) - Fs)
    assert err < 1e-5
