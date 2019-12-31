"""
Freysoldt, C., Boeck, S., & Neugebauer, J., Direct minimization technique
for metals in density functional theory.
http://dx.doi.org/10.1103/PhysRevB.79.241103
"""

import numpy as np
from scipy.constants import physical_constants

from ..coefficient_array import CoefficientArray as ca
from ..coefficient_array import diag, einsum, inner

from ..helpers import save_state
from ..logger import Logger
from ..py_sirius import magnetization
from .ortho import loewdin
from .preconditioner import IdentityPreconditioner
# from .helpers import has_enough_bands
# from ..utils.exceptions import NotEnoughBands

logger = Logger()


kb = (physical_constants['Boltzmann constant in eV/K'][0] /
      physical_constants['Hartree energy in eV'][0])


class StepError(Exception):
    """StepError."""


class SlopeError(Exception):
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
    g_eta_1 = -1/kT * diag(diag(Hij) - kw*ek) * fn * (mo-fn)
    dFdmu = np.sum(np.real(
        1/kT * einsum('i,i', (diag(Hij) - kw*ek).asarray().flatten(), fn * (mo-fn))))
    sumfn = np.sum(kw*fn*(mo-fn))
    # g_eta_2 is zero if all f_i are either 0 or 1
    if np.abs(sumfn) < 1e-10:
        g_eta_2 = 0
    else:
        g_eta_2 = diag(kw * fn * (mo-fn) / sumfn * dFdmu)
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

    for i in range(maxiter):
        fx = f(x)
        if fx[0] > f0:
            x *= tau
            logger('btsearch::F %.10f, x=%.4e' % (fx[0], x))
        else:
            return x, fx
    raise StepError('backtracking search could not find a new minimum')


def gss(f, a, b, tol=1e-3):
    """
    Golden section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.


    """

    (a, b) = (min(a, b), max(a, b))

    h = b - a
    if h <= tol:
        return (a, b)

    invphi = (np.sqrt(5) - 1) / 2  # 1/phi
    invphi2 = (3 - np.sqrt(5)) / 2  # 1/phi^2
    # required steps to achieve tolerance
    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for _ in range(n - 1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)

    if yc < yd:
        logger('gss: a=%.2g, d=%.2g' % (a, d))
        return (a, d)
    else:
        logger('gss: c=%.2g, b=%.2g' % (c, b))
        return (c, b)


class F:
    """
    Evaluate free energy along a fixed direction.
    """

    def __init__(self, X, eta, M, G_X, G_eta):
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
        X = loewdin(X_new) @ Ul
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


    def step(self, X, f, eta, G_X, G_eta, xi_trial, F0, slope, kwargs):
        """

        Keyword Arguments:
        X         --
        f         -- occupation numbers (just for debugging, not needed)
        eta       --
        G_X       --
        G_eta     --
        xi_trial  --
        F0        --
        slope     --

        Returns:
        X_n  --
        f_n  --
        ek   --
        F    -- free energy at minimum
        Hx   -- H*X_n at (X_n, f_n)
        U    -- subspace rotation matrix
        """

        # TODO: refactor
        fline = F(X, eta, self.M, G_X, G_eta)
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
            # save_state({'X': X, 'f': f,
            #             'eta': eta, 'G_X': G_X,
            #             'F0': F0, 'F1': F1,
            #             'a': a, 'b': b, 'c': c,
            #             'G_eta': G_eta, 'slope': slope, **kwargs}, self.M.energy.kpointset)
        if not Fpred < F0:
            # reset Hamiltonian (side effects)
            fline(0)
            raise StepError('quadratic line-search failed to find a new minima')

        # free energy at minimum
        FE, Hx, X_n, f_n, ek, U = fline(xi_min)
        logger('qline prediction error, FE-Fpred: %.10e, step-length %.4e' % (FE-Fpred, xi_min))
        if not FE < F0:
            logger('==== failed step ====')
            logger('F0:', F0)
            logger('Fpred:', Fpred, ' xi_min: ', xi_min, 'xi_trial: ', xi_trial)
            logger('F1: ', F1, ' a: ', a)
            logger('slope: ', slope)
            # save_state({'X': X, 'f': f,
            #             'F0': F0, 'F1': F1,
            #             'a': a, 'b': b, 'c': c,
            #             'eta': eta, 'G_X': G_X,
            #             'G_eta': G_eta, 'slope': slope, **kwargs}, self.M.energy.kpointset)

            # reset Hamiltonian (side effects)
            fline(0)
            raise StepError('quadratic line-search failed to find a new minima')

        return X_n, f_n, ek, FE, Hx, U, xi_min

    def step_golden_section_search(self, X, f, eta, Fline, F0):
        """
        Keyword Arguments:
        X         --
        eta       --
        Fline     -- g(t) = free_energy(Z + t * grad_Z), Z = [X, eta]
        xi_trial  --
        F0        --
        slope     --

        Returns:
        X_n  --
        f_n  --
        ek   --
        F    -- free energy at minimum
        U    -- subspace rotation matrix
        """

        t1, t2 = gss(Fline, a=0, b=0.5)
        F, Hx, Xn, fn, ek, Ul = Fline((t1+t2)/2)
        if not F < F0:
            logger('WARNING: gss has failed')
            logger('t1,t2 = %.5g, %.5g' % (t1, t2))
            logger('F0: %.8f' % F0)
            logger('F1: %.8f' % F)
            if self._save:
                save_state({'X': X, 'f': f,
                            'eta': eta, 'G_X': Fline.G_X,
                            'G_eta': Fline.G_eta}, Fline.M.energy.kpointset)

            raise StepError('GSS didn\'t find a new minimum')
        return Xn, fn, ek, F, Hx, Ul

    def backtracking_search(self, X, f, eta, Fline, F0, tau=0.5):
        t1, res = btsearch(Fline, 5, F0, tau=tau)
        F1, Hx1, X1, f1, ek1, Ul1 = res

        return X1, f1, ek1, F1, Hx1, Ul1, t1

    def run(self, X, fn,
            maxiter=100,
            restart=20,
            tol=1e-10,
            kappa=0.3,
            tau=0.5,
            cgtype='FR',
            K=IdentityPreconditioner(),
            callback=lambda *args, **kwargs: None):
        """
        Returns:
        X            -- pw coefficients
        fn           -- occupation numbers
        FE           -- free energy
        is_converged -- bool
        """

        use_g_eta=False
        if cgtype == 'PR':
            cg_update = polak_ribiere
        elif cgtype == 'FR':
            cg_update = fletcher_reeves
        elif cgtype == 'SD':
            cg_update = steepest_descent
        else:
            raise ValueError('wrong type')

        kset = self.M.energy.kpointset
        # occupation prec
        kappa0 = kappa

        M = self.M
        T = self.M.T
        kw = kset.w
        m = kset.ctx().max_occupancy()
        # set occupation numbers from band energies
        fn, _ = self.M.smearing.fn(kset.e)
        # ek = self.M.smearing.ek(fn)

        eta = diag(kset.e)
        w, U = eta.eigh()
        ek = w
        X = X@U
        # compute initial free energy
        FE, Hx = M(X, fn)
        logger('intial F: %.10g' % FE)

        HX = Hx * kw
        Hij = X.H @ HX
        g_eta = grad_eta(Hij, ek, fn, T, kw, mo=m)
        XhKHXF = X.H @ (K @ HX)
        XhKX = X.H @ (K @ X)
        LL = _solve(XhKX, XhKHXF)
        g_X = (HX*fn - X@LL)
        # check that constraints are fulfilled
        delta_X = -K * (HX - X @ LL) / kw

        g_X = (HX*fn - X@LL)

        G_X = delta_X
        G_eta = -g_eta
        delta_eta = G_eta

        cg_restart_inprogress = False
        for ii in range(1, 1+maxiter):
            slope = np.real(2*inner(g_X, G_X) + inner(g_eta, G_eta))

            if np.abs(slope) < tol and ii > 1:
                return X, fn, FE, True

            if slope > 0:
                if cg_restart_inprogress:
                    raise SlopeError('Error: _ascent_ direction, slope %.4e' % slope)
                cg_restart_inprogress = True
                # TODO: tmin is not set, but used later
            else:
                try:
                    X, fn, ek, FE, Hx, U, tmin = self.step(X, fn, eta, G_X, G_eta,
                                                           xi_trial=0.2, F0=FE, slope=slope,
                                                           kwargs={'gx': g_X, 'g_eta': g_eta})
                    # reset kappa
                    kappa = kappa0
                    cg_restart_inprogress = False
                except StepError:
                    # side effects
                    try:
                        Fline = F(X, eta, M, G_X, G_eta)
                        logger('btsearch')
                        X, fn, ek, FE, Hx, U, tmin = self.backtracking_search(X, fn, eta, Fline, FE, tau=tau)
                    except StepError:
                        # not even golden section search works
                        # restart CG and reduce kappa
                        cg_restart_inprogress = True
                        kappa = kappa/3
                        logger('kappa: ', kappa)
                        tmin = 0
            callback(g_X=g_X, G_X=G_X, g_eta=g_eta, G_eta=G_eta, fn=fn, X=X, eta=eta, FE=FE, it=ii)
            logger('step %5d' % ii, 'F: %.11f res: X,eta %+10.5e, %+10.5e' %
                   (FE, np.real(inner(g_X, G_X)), np.real(inner(g_eta, G_eta))))
            mag = magnetization(self.M.energy.density)
            logger('magnetization: %.5f %.5f %.5f, total: %.5f' % (mag[0], mag[1], mag[2], np.linalg.norm(mag)))
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
            XhKHXF = X.H @ (K @ HX)
            XhKX = X.H @ (K @ X)
            LL = _solve(XhKX, XhKHXF)
            gp_X = g_X@U
            g_X = (HX*fn - X@LL)
            # check that constraints are fulfilled
            delta_X = -K * (HX - X @ LL) / kw
            # assert l2norm(X.H @ delta_X) < 1e-11
            delta_eta = kappa * (Hij - kw*diag(ek)) / kw
            # update kappa
            dFdk = inner(g_eta, deltaP_eta) * tmin / kappa
            dFdt = slope
            # if |dFdk| >> |dFdt| => reduce kappa
            # if |dFdk| << |dFdt| => increase kappa
            logger('dFdk: %.4e + %.4e 1j' % (np.real(dFdk), np.imag(dFdk)))
            logger('dFdt: %.4e + %4.e 1j' % (np.real(dFdt), np.imag(dFdk)))
            logger('|dFdk|/|dFdt|: %.4e' % (np.abs(dFdk) / np.abs(dFdt)))

            # conjugated search directions
            if not ii % restart == 0 and not cg_restart_inprogress:
                gamma = cg_update(g_X=g_X, gp_X=gp_X, g_eta=g_eta, gp_eta=gp_eta,
                                  deltaP_X=deltaP_X, deltaP_eta=deltaP_eta,
                                  delta_X=delta_X, delta_eta=delta_eta)
            else:
                logger('restart CG')
                gamma = 0
            logger('gamma: ', gamma)
            G_X = delta_X + gamma * (GP_X - X@(X.H@GP_X))
            G_eta = delta_eta + gamma * GP_eta
            # ready for the next iteration ...
            # save_state({'G_X': G_X, 'gx': g_X, 'G_eta': G_eta,
            #             'f': fn,
            #             'g_eta': g_eta,
            #             'gamma': gamma,
            #             'slope': slope,
            #             'X': X, 'eta': eta}, M.energy.kpointset, prefix='iter%04d_' % ii)

        return X, fn, FE, False
