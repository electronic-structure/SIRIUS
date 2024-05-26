from sirius.operators import S_operator, Sinv_operator  # type: ignore
from sirius.coefficient_array import diag, inner, l2norm  # type: ignore
from sirius.ot import ApplyHamiltonian, Energy  # type: ignore
from sirius.py_sirius import smearing as cxx_smearing  # type: ignore
from sirius.edft import smearing, loewdin  # type: ignore
from sirius.edft.neugebauer import grad_eta  # type: ignore
from sirius.edft.free_energy import FreeEnergy  # type: ignore
import numpy as np


def check_gradient(dft, kT, fx: float = 1, kappa: float = 0):
    """
    Arguments:
    kT -- smearing
    fx -- scalar weight for gradient wrt to PW coeffs
    kappa -- scalar weight for gradient wrt to pseudo-Hamiltonian

    Returns:
    slope    -- <G, g>
    slope_fd -- finite difference
    """
    density = dft.density()
    potential = dft.potential()
    kset = dft.k_point_set()
    ctx = kset.ctx()

    H = ApplyHamiltonian(potential, kset)
    E = Energy(kset, potential, density, H)
    sinv_op = Sinv_operator(ctx, potential, kset)
    s_op = S_operator(ctx, potential, kset)

    X = kset.C
    fn = kset.fn

    nel = ctx.unit_cell().num_valence_electrons
    mo = ctx.max_occupancy()

    broadening = smearing.Smearing(
        cxx_smearing.gaussian,
        kT=kT,
        mo=mo,
        num_electrons=nel,
        kw=kset.w,
        comm=kset.ctx().comm_k(),
    )

    M = FreeEnergy(E=E, smearing=broadening)
    kw = kset.w
    # ne = ctx.unit_cell().num_valence_electrons
    eta0 = kset.e
    # print("eta0", eta0)
    w, U = diag(eta0).eigh()
    ek = w
    # rotate (e.g. in this case permute X)
    X = X @ U
    eta = diag(w)
    fn, mu = M.smearing.fn(ek)

    # evaluate total energy, gradient, overlap
    F0i, Hx = M(X, fn=fn, mu=mu, ek=ek)

    print("Free energy after setting F_n:", F0i)

    # compute gradients
    HX = Hx * kw
    Hij = X.H @ HX

    # g_eta = grad_eta(Hij, ek, fn, T, kw, mo=kset.ctx().max_occupancy())
    g_eta = grad_eta(Hij, fn=fn, ek=ek, mu=mu, smearing=M.smearing, kw=kw)

    fx = 1  # X
    kappa = 0  # eta

    SX = s_op @ X

    # Hijf = (X.H @ HX) * fn
    Hijf = X.H @ HX
    LL = Hijf

    delta_X = sinv_op @ HX - X @ LL
    g_X = HX * fn - SX @ LL

    # preconditioned search direction
    delta_eta = kappa * (Hij - kw * diag(ek)) / kw

    G_X = -delta_X
    # G_eta = delta_eta
    # G_eta = -g_eta
    G_eta = delta_eta
    dts = np.linspace(0, 0.2, 15)

    check_constraint = l2norm(X.H @ (s_op @ G_X))
    if np.abs(check_constraint) > 1e-11:
        print(
            f"ERROR: 1st order orthogonality constraint FAILED, {check_constraint:.10g}"
        )
        raise Exception

    slope = np.real(2 * inner(g_X, fx * G_X) + inner(g_eta, G_eta))
    Fs = []
    Hs = []  # gradients along lines
    dt_slope = 1e-2
    dts = np.concatenate([np.array([0, dt_slope]), dts[1:]])
    for dt in dts:
        X_new = X + dt * fx * G_X
        eta_new = eta + dt * G_eta
        w, Ul = eta_new.eigh()
        Q_new = loewdin(X_new @ Ul, s_op)
        # update occupation numbers
        fn_new, mu = M.smearing.fn(w)
        # print(f'entropy {M.entropy:.4g}')
        print("orth err: %.3g, mu: %.5g" % (l2norm(X_new @ Ul - Q_new), mu))
        Floc, Hloc = M(Q_new, fn=fn_new, mu=mu, ek=w)
        Fs.append(Floc)
        Hs.append(Hloc)

    fit = np.polyfit(dts, Fs, deg=2)

    print("slope (fit): %.6g" % fit[1])
    print("slope      : %.6g" % slope)
    slope_fd = (Fs[1] - Fs[0]) / dt_slope
    print("slope (fd) : %.6g" % slope_fd)

    return slope, slope_fd
