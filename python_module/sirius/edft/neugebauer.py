"""Freysoldt, C., Boeck, S., & Neugebauer, J., Direct minimization technique
for metals in density functional theory.
http://dx.doi.org/10.1103/PhysRevB.79.241103
"""

import numpy as np
from scipy.constants import physical_constants

from ..coefficient_array import diag, inner, l2norm
from ..operators import US_Precond, Sinv_operator, S_operator
from ..logger import Logger
from ..py_sirius import sprint_magnetization
from .preconditioner import IdentityPreconditioner
from .free_energy import FreeEnergy
from .pseudo_hamiltonian import grad_eta, LineEvaluator
import time


logger = Logger()


kb = (
    physical_constants["Boltzmann constant in eV/K"][0]
    / physical_constants["Hartree energy in eV"][0]
)


class StepError(Exception):
    """StepError."""


class SlopeError(Exception):
    """Slope error. Non-descent direction."""


class CGRestart(Exception):
    """Slope error. Non-descent direction."""


def _solve(A, X):
    """
    returns A⁻¹ X
    """
    out = type(X)()
    for k in X.keys():
        out[k] = np.linalg.solve(A[k], X[k])
    return out


def btsearch(f: LineEvaluator, b, f0, maxiter=20, tau=0.5):
    """Backtracking search.
    Arguments:
    f  -- line search object (class F)
    b  -- initial step lenght
    f0 -- free energy at F(0)
    """

    x = b

    fref = f(0)
    if np.abs(fref.free_energy - f0) > 1e-13:
        logger("btsearch f(0) != f0, f(0): %.13f, f0: %.13f" % (fref.free_energy, f0))

    for i in range(maxiter):
        fx = f(x)
        if x < 1e-8:
            raise StepError("backtracking search could not find a new minimum")
        if fx.free_energy > f0:
            x *= tau
        else:
            logger("btsearch::F %.10f, F0=%.10f, x=%.3e" % (fx.free_energy, f0, x))
            return x, fx
    raise StepError("backtracking search could not find a new minimum")


class CGFailed(Exception):
    """"""

    pass


def polak_ribiere(**kwargs):
    g_X = kwargs["g_X"]
    gp_X = kwargs["gp_X"]
    g_eta = kwargs["g_eta"]
    gp_eta = kwargs["gp_eta"]
    delta_eta = kwargs["delta_eta"]
    deltaP_eta = kwargs["deltaP_eta"]
    delta_X = kwargs["delta_X"]
    deltaP_X = kwargs["deltaP_X"]
    gamma_eta = np.real(inner(delta_eta, g_eta - gp_eta))
    gamma_X = np.real(inner(delta_X, g_X - gp_X))
    gamma = max(
        0,
        (2 * gamma_X + gamma_eta)
        / (2 * np.real(inner(deltaP_X, gp_X)) + np.real(inner(deltaP_eta, gp_eta))),
    )
    return gamma


def fletcher_reeves(**kwargs):
    g_X = kwargs["g_X"]
    gp_X = kwargs["gp_X"]
    g_eta = kwargs["g_eta"]
    gp_eta = kwargs["gp_eta"]
    delta_X = kwargs["delta_X"]
    delta_eta = kwargs["delta_eta"]
    deltaP_X = kwargs["deltaP_X"]
    deltaP_eta = kwargs["deltaP_eta"]
    gamma_eta = np.real(inner(g_eta, delta_eta))
    gammaP_eta = np.real(inner(gp_eta, deltaP_eta))
    gamma_X = 2 * np.real(inner(g_X, delta_X))
    gammaP_X = 2 * np.real(inner(gp_X, deltaP_X))
    return (gamma_eta + gamma_X) / (gammaP_eta + gammaP_X)


def steepest_descent(**kwargs):
    return 0


class CG:
    def __init__(self, free_energy: FreeEnergy):
        """
        Arguments:
        free_energy -- Free Energy callable
        T -- temperature
        """
        self.M = free_energy
        self._save = False

        # ultrasoft
        potential = self.M.energy.potential
        kset = self.M.energy.kpointset
        ctx = kset.ctx()
        self.is_ultrasoft = kset.ctx().unit_cell().augmented
        if self.is_ultrasoft:
            self.Si = Sinv_operator(ctx, potential, kset)
            self.S = S_operator(ctx, potential, kset)
            self.K = US_Precond(ctx, potential, kset)
        else:
            self.S = None

    def qline_search(self, fline, xi_trial, F0, slope):
        """
        Keyword Arguments:
        fline     -- line evaluator
        xi_trial  -- trial point
        F0        -- free energy at t=0
        slope     --

        Returns:
        dict -- a SimpleNamespace dictionary
        """

        while True:
            # free energy at trial point
            F1 = fline(xi_trial).free_energy
            # find xi_min
            c = F0
            b = slope
            a = (F1 - b * xi_trial - c) / xi_trial**2
            xi_min = -b / (2 * a)
            if a < 0:
                # logger(" -- increasing xi_trial by factor 5")
                xi_trial *= 5
            else:
                break

        # predicted free energy
        Fpred = -(b**2) / 4 / a + c
        if Fpred > F0:
            logger(f"F0: {F0}")
            logger(f"Fpred: {Fpred} xi_min: {xi_min}")
            logger(f"F1: {F1} a: {a}")
            logger(f"slope: {slope}")
        if not Fpred < F0:
            # # reset Hamiltonian (side effects)
            # raise Exception(
            #     "Fatal Error: quadratic line search, predicted energy is higher than initial energy"
            # )
            res = fline(0)
            assert np.abs(res.free_energy - F0) < 1e-8
            raise StepError("quadratic line-search failed to find a new minima")

        # free energy at minimum
        # FE, Hx, X_n, f_n, ek, U = fline(xi_min)
        res_ximin = fline(xi_min)
        FE = res_ximin.free_energy
        # logger(
        #     f"qline prediction error, FE-Fpred: {FE-Fpred:.8e}, step-length {xi_min}"
        # )
        if not FE < F0:
            logger("quadratic line search failed.")
            # logger("F0:", F0)
            # logger(
            #     f"Fpred: {Fpred:.8e} xi_min: {xi_min:.4g}" f" xi_trial: {xi_trial:.4g}",
            # )
            # logger(f"F1: {F1:.8e}, a: {a:.13f}")
            # logger(f"slope: {slope:5g}")

            # reset Hamiltonian (side effects)
            res = fline(0)
            assert np.abs(res.free_energy - F0) < 1e-8
            raise StepError("quadratic line-search failed to find a new minima")

        return res_ximin

    def backtracking_search(self, fline, F0, tau=0.1):
        _, res = btsearch(fline, 1, F0, tau=tau, maxiter=9)
        return res

    def line_search(self, fline: LineEvaluator, xi_trial, F0, slope, error_callback):
        try:
            return self.qline_search(fline, xi_trial=xi_trial, F0=F0, slope=slope)
        except StepError:
            # try backtracking search, if this fails too, do cg_restart
            error_callback()
            try:
                return self.backtracking_search(fline, F0)
            except StepError:
                raise CGRestart

    def run(
        self,
        X,
        ek,
        maxiter=100,
        restart=20,
        tol=1e-10,
        kappa=0.3,
        tau=0.5,
        cgtype="FR",
        K=IdentityPreconditioner(),
        callback=lambda *args, **kwargs: None,
        error_callback=lambda *args, **kwargs: None,
    ):
        """
        Returns:
        X            -- pw coefficients
        fn           -- occupation numbers
        FE           -- free energy
        is_converged -- bool
        """

        if cgtype == "PR":
            cg_update = polak_ribiere
        elif cgtype == "FR":
            cg_update = fletcher_reeves
        elif cgtype == "SD":
            cg_update = steepest_descent
        else:
            raise ValueError("wrong type")

        if self.is_ultrasoft:
            K = self.K
            # TODO cleanup signature of run and don't pass the preconditioner anymore

        kset = self.M.energy.kpointset

        M = self.M
        kw = kset.w
        eta = diag(ek)
        w, U = eta.eigh()
        ek = w
        X = X @ U
        # set occupation numbers from band energies
        fn, mu = self.M.smearing.fn(ek)
        # compute initial free energy
        FE, Hx = M(X, fn=fn, mu=mu, ek=ek)
        logger("initial F: %.13f" % FE)

        HX = Hx * kw
        Hij = X.H @ HX
        g_eta = grad_eta(Hij, fn=fn, ek=ek, mu=mu, smearing=self.M.smearing, kw=kw)

        # Lagrange multipliers
        if self.is_ultrasoft:
            SX = self.S @ X
        else:
            SX = X
        XhKHX = SX.H @ (K @ HX)
        XhKX = SX.H @ (K @ SX)
        LL = _solve(XhKX, XhKHX)

        g_X = HX * fn - SX @ LL
        # check that constraints are fulfilled
        delta_X = -(K @ (HX - SX @ LL) / kw)

        g_X = HX * fn - SX @ LL
        G_X = delta_X

        G_eta = -kappa * g_eta
        delta_eta = G_eta

        fr = np.real(2 * inner(g_X, delta_X) + inner(g_eta, delta_eta))

        cg_restart_inprogress = False
        for ii in range(1, 1 + maxiter):
            tcgstart = time.time()
            slope = np.real(2 * inner(g_X, G_X) + inner(g_eta, G_eta))

            if self.is_ultrasoft:
                check_constraint = l2norm(X.H @ (self.S @ G_X))
            else:
                check_constraint = l2norm(X.H @ G_X)
            if np.abs(check_constraint) > 1e-11:
                raise Exception("Orthogonality constraint not satisfied.")

            if np.abs(slope) < tol:
                logger(f"step {ii} F: {FE:.10f} res: {slope:.4e}")
                return X, fn, FE, True

            if slope > 0:
                if cg_restart_inprogress:
                    raise SlopeError("Error: _ascent_ direction, slope %.4e" % slope)
                cg_restart_inprogress = True
            else:

                def error_cb():
                    return error_callback(
                        g_X=g_X,
                        G_X=G_X,
                        g_eta=g_eta,
                        G_eta=G_eta,
                        fn=fn,
                        X=X,
                        eta=eta,
                        FE=FE,
                        prefix="nlcg_dump%04d_" % ii,
                    )

                fline = LineEvaluator(X, eta, self.M, G_X, G_eta, overlap=self.S)
                try:
                    res = self.line_search(
                        fline, xi_trial=0.2, F0=FE, slope=slope, error_callback=error_cb
                    )

                    X = res.X
                    fn = res.fn
                    eta = diag(res.ek)
                    ek = res.ek
                    FE = res.free_energy
                    Hx = res.Hx
                    U = res.Ul
                    # line-search successful, reset restart
                    cg_restart_inprogress = False
                except CGRestart:
                    # line-search failure = both quadratic and backtracking search have failed
                    # attempt CG restart
                    if cg_restart_inprogress:
                        raise Exception("failed")
                    cg_restart_inprogress = True

            callback(
                g_X=g_X,
                G_X=G_X,
                g_eta=g_eta,
                G_eta=G_eta,
                fn=fn,
                X=X,
                eta=eta,
                FE=FE,
                it=ii,
            )

            logger(
                f"step {ii} F: {FE} res: X,eta {np.real(inner(g_X, G_X)):.5g}, {np.real(inner(g_eta, G_eta)):.5g}"
            )
            logger(f"entropy: {self.M.entropy:.8f}, ks-energy: {self.M.ks_energy:.8f}")

            mag_str = sprint_magnetization(
                self.M.energy.kpointset, self.M.energy.density
            )
            logger(mag_str)

            # keep previous search directions
            GP_X = G_X @ U
            GP_eta = U.H @ G_eta @ U
            # deltaP_X = delta_X @ U
            # deltaP_eta = U.H @ delta_eta @ U
            # compute new gradients
            HX = Hx * kw
            Hij = X.H @ HX
            # gp_eta = U.H @ g_eta @ U
            g_eta = grad_eta(Hij, fn=fn, ek=ek, mu=mu, smearing=self.M.smearing, kw=kw)
            # Lagrange multipliers
            if self.is_ultrasoft:
                SX = self.S @ X
            else:
                SX = X
            XhKHX = SX.H @ (K @ HX)
            XhKX = SX.H @ (K @ SX)
            LL = _solve(XhKX, XhKHX)

            # gp_X = g_X @ U
            g_X = HX * fn - SX @ LL
            # check that constraints are fulfilled
            delta_X = -(K @ (HX - SX @ LL) / kw)
            # check orthogonality constraint
            # assert l2norm(SX.H @ delta_X) < 1e-11
            delta_eta = kappa * (Hij - kw * diag(ek)) / kw
            # pseudo_energy_error = l2norm(diag(Hij)/kw-ek)
            # logger('pseudo energy error: %.6e' % pseudo_energy_error)
            # update kappa
            # dFdk = inner(g_eta, deltaP_eta) * tmin / kappa
            # dFdt = slope
            # # if |dFdk| >> |dFdt| => reduce kappa
            # # if |dFdk| << |dFdt| => increase kappa
            # logger('dFdk: %.4e + %.4e 1j' % (np.real(dFdk), np.imag(dFdk)))
            # logger('dFdt: %.4e + %4.e 1j' % (np.real(dFdt), np.imag(dFdk)))
            # logger('|dFdk|/|dFdt|: %.4e' % (np.abs(dFdk) / np.abs(dFdt)))

            # conjugated search directions
            if not ii % restart == 0 and not cg_restart_inprogress:

                fr_new = np.real(2 * inner(g_X, delta_X) + inner(g_eta, delta_eta))
                gamma = fr_new / fr
                fr = fr_new
                # gamma = cg_update(g_X=g_X, gp_X=gp_X, g_eta=g_eta, gp_eta=gp_eta,
                #                   deltaP_X=deltaP_X, deltaP_eta=deltaP_eta,
                #                   delta_X=delta_X, delta_eta=delta_eta)
            else:
                # logger("restart CG")
                gamma = 0
            # logger("gamma: ", gamma)
            if self.is_ultrasoft:
                gLL = _solve(X.H @ (self.S @ SX), X.H @ (self.S @ GP_X))
                G_X = delta_X + gamma * (GP_X - SX @ gLL)
            else:
                gLL = X.H @ GP_X
                G_X = delta_X + gamma * (GP_X - X @ gLL)

            G_eta = delta_eta + gamma * GP_eta
            tcgstop = time.time()
            # logger("\tcg step took: ", format(tcgstop - tcgstart, ".3f"), " seconds")

        return X, fn, FE, False
