"""Freysoldt, C., Boeck, S., & Neugebauer, J., Direct minimization technique
for metals in density functional theory.
http://dx.doi.org/10.1103/PhysRevB.79.241103
"""

import numpy as np
from scipy.constants import physical_constants

from ..coefficient_array import CoefficientArray as ca
from ..coefficient_array import diag, einsum, inner, l2norm
from ..operators import US_Precond, Sinv_operator, S_operator

from ..helpers import save_state
from ..logger import Logger
from ..py_sirius import sprint_magnetization
from .ortho import loewdin
from .preconditioner import IdentityPreconditioner
import time

logger = Logger()


kb = (physical_constants['Boltzmann constant in eV/K'][0] /
      physical_constants['Hartree energy in eV'][0])


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
    out = type(X)(dtype=X.dtype, ctype=X.ctype)
    for k in X.keys():
        out[k] = np.linalg.solve(A[k], X[k])
    return out


def grad_eta(Hij, ek, fn, T, kw, mo):
    """
    Computes ∂L/∂η

    Arguments:
    Hij -- subspace Hamiltonian
    ek  -- Fermi parameters ϵ_n
    fn  -- occupation numbers
    T   -- temperature
    kw  -- kpoint weights
    mo  -- max occupancy, mo=1 in magnetic case

    Returns:
    g_eta -- gradient wrt η of the free-energy Lagrangian
    """
    kT = kb * T
    g_eta_1 = -1/kT * diag(diag(Hij) - kw*ek) * fn * (mo-fn) / mo
    dFdmu = np.sum(np.real(
        1/kT * einsum('i,i', (diag(Hij) - kw*ek).asarray().flatten(), fn * (mo-fn))))
    sumfn = np.sum(kw*fn*(mo-fn) / mo)
    # g_eta_2 is zero if all f_i are either 0 or 1
    if np.abs(sumfn) < 1e-10:
        g_eta_2 = 0
    else:
        g_eta_2 = diag(kw * fn * (mo-fn) / mo / sumfn * dFdmu)
    # off-diagonal terms
    II = diag(ca.ones_like(fn))
    Eij = ek-ek.T + II
    Fij = fn-fn.T
    for k in Eij.keys():
        EEc = np.abs(Eij[k]) < 1e-10
        Eij[k] = np.where(EEc, 1, Eij[k])
        Fij[k] = np.where(EEc, 0, Fij[k])

    g_eta_3 = Fij / Eij * Hij * (1-II)
    g_eta = (g_eta_1 + g_eta_2 + g_eta_3)
    return g_eta


def btsearch(f, b, f0, maxiter=20, tau=0.5):
    """
    Backtracking search
    """

    x = b

    # TODO DEBUG
    fref=f(0)
    if np.abs(fref[0]-f0) > 1e-13:
        logger('btsearch f(0) != f0, f(0): %.13f, f0: %.13f' % (fref[0], f0))

    for i in range(maxiter):
        fx = f(x)
        if x < 1e-8:
            raise StepError('backtracking search could not find a new minimum')
        logger('btsearch::F %.10f, F0=%.10f, x=%.3e' % (fx[0], f0, x))
        if fx[0] > f0:
            x *= tau
        else:
            return x, fx
    raise StepError('backtracking search could not find a new minimum')


class F:
    """
    Evaluate free energy along a fixed direction.
    """

    def __init__(self, X, eta, M, G_X, G_eta, overlap):
        """
        Keyword Arguments:
        X     -- plane-wave coefficients
        eta   -- pseudo-Hamiltonian (must be diagonal)
        M     -- free energy evaluator
        G_X   -- direction X
        G_eta -- direction eta
        """

        self.X = X
        self.eta = eta
        self.M = M
        # search direction
        self.G_X = G_X
        self.G_eta = G_eta
        self.overlap = overlap

    def __call__(self, t):
        """
        Evaluate along line

        Returns:
        F  -- free energy
        Hx -- gradient (free of occupation, k-point-weight)
        X  -- plane-wave coefficients
        fn -- occupation numbers
        U  -- subspace rotation matrix
        """
        X_new = self.X + t * self.G_X
        eta_new = self.eta + t * self.G_eta
        ek, Ul = eta_new.eigh()
        X = loewdin(X_new, self.overlap) @ Ul
        fn, mu = self.M.smearing.fn(ek)

        # check fn

        FE, Hx = self.M(X, fn)
        return FE, Hx, X, fn, ek, Ul


class CGFailed(Exception):
    """
    """
    pass


def polak_ribiere(**kwargs):
    g_X = kwargs['g_X']
    gp_X = kwargs['gp_X']
    g_eta = kwargs['g_eta']
    gp_eta = kwargs['gp_eta']
    delta_eta = kwargs['delta_eta']
    deltaP_eta = kwargs['deltaP_eta']
    delta_X = kwargs['delta_X']
    deltaP_X = kwargs['deltaP_X']
    gamma_eta = np.real(inner(delta_eta, g_eta-gp_eta))
    gamma_X = np.real(inner(delta_X, g_X-gp_X))
    gamma = max(0,
                (2*gamma_X + gamma_eta)
                /
                (2*np.real(inner(deltaP_X, gp_X)) + np.real(inner(deltaP_eta, gp_eta))))
    return gamma


def fletcher_reeves(**kwargs):
    g_X = kwargs['g_X']
    gp_X = kwargs['gp_X']
    g_eta = kwargs['g_eta']
    gp_eta = kwargs['gp_eta']
    delta_X = kwargs['delta_X']
    delta_eta = kwargs['delta_eta']
    deltaP_X = kwargs['deltaP_X']
    deltaP_eta = kwargs['deltaP_eta']
    gamma_eta = np.real(inner(g_eta, delta_eta))
    gammaP_eta = np.real(inner(gp_eta, deltaP_eta))
    gamma_X = 2 * np.real(inner(g_X, delta_X))
    gammaP_X = 2 * np.real(inner(gp_X, deltaP_X))
    return (gamma_eta + gamma_X) / (gammaP_eta + gammaP_X)


def steepest_descent(**kwargs):
    return 0


class CG:
    def __init__(self, free_energy):
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
        self.is_ultrasoft = np.any([type.augment for type in kset.ctx().unit_cell().atom_types])
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
        X_n  --
        f_n  --
        ek   --
        F    -- free energy at minimum
        Hx   -- H*X_n at (X_n, f_n)
        U    -- subspace rotation matrix
        """

        while True:
            # free energy at trial point
            F1, _, _, _, _, _ = fline(xi_trial)
            # find xi_min
            c = F0
            b = slope
            a = (F1 - b*xi_trial - c) / xi_trial**2
            xi_min = -b/(2*a)
            if a < 0:
                logger(' -- increasing xi_trial by factor 5')
                xi_trial *= 5
            else:
                break

        # predicted free energy
        Fpred = -b**2/4/a + c
        if Fpred > F0:
            logger('F0:', F0)
            logger('Fpred:', Fpred, ' xi_min: ', xi_min)
            logger('F1: ', F1, ' a: ', a)
            logger('slope: ', slope)
        if not Fpred < F0:
            # reset Hamiltonian (side effects)
            fline(0)
            raise StepError('quadratic line-search failed to find a new minima')

        # free energy at minimum
        FE, Hx, X_n, f_n, ek, U = fline(xi_min)
        logger('qline prediction error, FE-Fpred: %.4e, step-length %.3e' % (FE-Fpred, xi_min))
        if not FE < F0:
            logger('==== failed step ====')
            logger('F0:', F0)
            logger('Fpred:', Fpred, ' xi_min: ', format(xi_min, '.4g'), 'xi_trial: ', format(xi_trial, '.4g'))
            logger('F1: ', F1, ' a: ', format(a, '.13f'))
            logger('slope: ', format(slope, '.5g'))

            # reset Hamiltonian (side effects)
            fline(0)
            raise StepError('quadratic line-search failed to find a new minima')

        return X_n, f_n, ek, FE, Hx, U, xi_min

    def backtracking_search(self, fline, F0, tau=0.1):
        t1, res = btsearch(fline, 1, F0, tau=tau, maxiter=9)
        F1, Hx1, X1, f1, ek1, Ul1 = res

        return X1, f1, ek1, F1, Hx1, Ul1, t1

    def line_search(self, fline, xi_trial, F0, slope, error_callback):
        try:
            return self.qline_search(fline, xi_trial, F0, slope)
        except StepError:
            # try backtracking search, if this fails too, do cg_restart
            error_callback()
            try:
                return self.backtracking_search(fline, F0)
            except StepError:
                raise CGRestart

    def run(self, X, ek,
            maxiter=100,
            restart=20,
            tol=1e-10,
            kappa=0.3,
            tau=0.5,
            cgtype='FR',
            K=IdentityPreconditioner(),
            callback=lambda *args, **kwargs: None,
            error_callback=lambda *args, **kwargs: None):
        """
        Returns:
        X            -- pw coefficients
        fn           -- occupation numbers
        FE           -- free energy
        is_converged -- bool
        """

        if cgtype == 'PR':
            cg_update = polak_ribiere
        elif cgtype == 'FR':
            cg_update = fletcher_reeves
        elif cgtype == 'SD':
            cg_update = steepest_descent
        else:
            raise ValueError('wrong type')

        if self.is_ultrasoft:
            K = self.K
            # TODO cleanup signature of run and don't pass the preconditioner anymore

        kset = self.M.energy.kpointset

        M = self.M
        T = self.M.T
        kw = kset.w
        m = kset.ctx().max_occupancy()
        eta = diag(ek)
        w, U = eta.eigh()
        ek = w
        X = X@U
        # set occupation numbers from band energies
        fn, _ = self.M.smearing.fn(ek)
        # compute initial free energy
        FE, Hx = M(X, fn)
        logger('initial F: %.13f' % FE)

        HX = Hx * kw
        Hij = X.H @ HX
        g_eta = grad_eta(Hij, ek, fn, T, kw, mo=m)

        # Lagrange multipliers
        if self.is_ultrasoft:
            SX = self.S @ X
        else:
            SX = X
        XhKHX = SX.H @ (K @ HX)
        XhKX = SX.H @ (K @ SX)
        LL = _solve(XhKX, XhKHX)

        g_X = (HX*fn - SX@LL)
        # check that constraints are fulfilled
        delta_X = -(K @ (HX - SX @ LL) / kw)

        g_X = (HX*fn - SX@LL)
        G_X = delta_X

        G_eta = -kappa*g_eta
        delta_eta = G_eta

        fr = np.real(2*inner(g_X, delta_X) + inner(g_eta, delta_eta))

        cg_restart_inprogress = False
        for ii in range(1, 1+maxiter):
            tcgstart = time.time()
            slope = np.real(2*inner(g_X, G_X) + inner(g_eta, G_eta))

            if np.abs(slope) < tol and ii > 1:
                return X, fn, FE, True

            if slope > 0:
                if cg_restart_inprogress:
                    raise SlopeError('Error: _ascent_ direction, slope %.4e' % slope)
                cg_restart_inprogress = True
            else:
                error_cb = lambda: error_callback(g_X=g_X, G_X=G_X, g_eta=g_eta,
                                                  G_eta=G_eta, fn=fn, X=X, eta=eta, FE=FE, prefix='nlcg_dump%04d_' % ii)

                fline = F(X, eta, self.M, G_X, G_eta, overlap=self.S)
                try:
                    X, fn, ek, FE, Hx, U, tmin = self.line_search(fline, xi_trial=0.2, F0=FE, slope=slope, error_callback=error_cb)
                    # line-search successful, reset restart
                    cg_restart_inprogress = False
                except CGRestart:
                    # line-search failure = both quadratic and backtracking search have failed
                    # attempt CG restart
                    if cg_restart_inprogress:
                        raise Exception('failed')
                    cg_restart_inprogress = True

            callback(g_X=g_X, G_X=G_X, g_eta=g_eta, G_eta=G_eta, fn=fn, X=X, eta=eta, FE=FE, it=ii)

            logger('step %5d' % ii, 'F: %.11f res: X,eta %+10.5e, %+10.5e' %
                   (FE, np.real(inner(g_X, G_X)), np.real(inner(g_eta, G_eta))))
            logger('\t entropy: %.13f, ks-energy: %.13f' % (self.M.entropy, self.M.ks_energy))

            mag_str = sprint_magnetization(self.M.energy.kpointset, self.M.energy.density)
            logger(mag_str)

            eta = diag(ek)

            # keep previous search directions
            GP_X = G_X@U
            GP_eta = U.H@G_eta@U
            deltaP_X = delta_X@U
            deltaP_eta = U.H@delta_eta@U
            # compute new gradients
            HX = Hx*kw
            Hij = X.H @ HX
            gp_eta = U.H @ g_eta @ U
            g_eta = grad_eta(Hij, ek, fn, T, kw, mo=m)
            # Lagrange multipliers
            if self.is_ultrasoft:
                SX = self.S @ X
            else:
                SX = X
            XhKHX = SX.H @ (K @ HX)
            XhKX = SX.H @ (K @ SX)
            LL = _solve(XhKX, XhKHX)

            gp_X = g_X@U
            g_X = (HX*fn - SX@LL)
            # check that constraints are fulfilled
            delta_X = -(K @ (HX - SX @ LL) / kw)
            # check orthogonality constraint
            # assert l2norm(SX.H @ delta_X) < 1e-11
            delta_eta = kappa * (Hij - kw*diag(ek)) / kw
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

                fr_new = np.real(2*inner(g_X, delta_X) + inner(g_eta, delta_eta))
                gamma = fr_new / fr
                fr = fr_new
                # gamma = cg_update(g_X=g_X, gp_X=gp_X, g_eta=g_eta, gp_eta=gp_eta,
                #                   deltaP_X=deltaP_X, deltaP_eta=deltaP_eta,
                #                   delta_X=delta_X, delta_eta=delta_eta)
            else:
                logger('restart CG')
                gamma = 0
            logger('gamma: ', gamma)
            if self.is_ultrasoft:
                gLL = _solve(X.H @ (self.S @ SX), X.H @ (self.S @ GP_X))
                G_X = delta_X + gamma * (GP_X - SX@gLL)
            else:
                gLL = X.H@GP_X
                G_X = delta_X + gamma * (GP_X - X@gLL)

            G_eta = delta_eta + gamma * GP_eta
            tcgstop = time.time()
            logger('\tcg step took: ', format(tcgstop-tcgstart, '.3f'), ' seconds')

        return X, fn, FE, False
