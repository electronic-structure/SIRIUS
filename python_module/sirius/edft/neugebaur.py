import numpy as np
from scipy.constants import physical_constants
from scipy.sparse import dia_matrix
from ..coefficient_array import CoefficientArray as ca
from ..coefficient_array import CoefficientArray, PwCoeffs
from ..coefficient_array import diag, einsum, inner, l2norm
from ..helpers import save_state
from ..logger import Logger
from .ortho import gram_schmidt, loewdin
from mpi4py import MPI
from copy import deepcopy

logger = Logger()


kb = (physical_constants['Boltzmann constant in eV/K'][0] /
      physical_constants['Hartree energy in eV'][0])


def _solve(A, X):
    """
    returns A⁻¹ X
    """
    out = type(X)(dtype=X.dtype, ctype=X.ctype)
    for k in X.keys():
        out[k] = np.linalg.solve(A[k], X[k])
    return out


def _fermi_function(x, T, mu, num_spins):
    """
    Keyword Arguments:
    x  --
    T  --
    mu --
    """
    exp_arg = (x - mu) / (kb*T)
    out = np.zeros_like(x, dtype=np.float64)
    oo = exp_arg < -50
    out[oo] = num_spins
    uo = exp_arg > 40
    out[uo] = 0
    re = np.logical_not(np.logical_or(oo, uo))
    out[re] = num_spins / (1 + np.exp(exp_arg[re]))
    return out


def fermi_function(x, T, mu, num_spins):
    """
    """
    assert T > 0

    if isinstance(x, CoefficientArray):
        out = type(x)(dtype=x.dtype, ctype=np.array)
        for key, _ in x._data.items():
            out[key] = _fermi_function(x[key], T, mu, num_spins)
        return out
    return _fermi_function(x, T, mu, num_spins)


def _inv_fermi_function(f, T, num_spins):
    """
    """
    f /= num_spins
    kT = kb*T
    en = np.zeros_like(f, dtype=np.double)
    is_zero = np.isclose(f, 0)
    is_one = np.isclose(f, 1)
    en[is_zero] = 50 + np.arange(0, np.sum(is_zero))
    # make sure we do not get degenerate band energies
    en[is_one] = -50 - np.arange(0, np.sum(is_one))
    ii = np.logical_not((np.logical_or(is_zero, is_one)))
    en[ii] = kT*np.log(1/f[ii] - 1)
    return en


def inv_fermi_function(f, T, num_spins):
    """
    given f compute ϵ
    """

    if isinstance(f, CoefficientArray):
        out = type(f)(dtype=f.dtype, ctype=np.array)
        for key, val in f._data.items():
            out[key] = _inv_fermi_function(f[key], T, num_spins)
        return out
    else:
        return _inv_fermi_function(f, T, num_spins)


def find_chemical_potential(fun, mu0, tol=1e-10):
    """
    fun        -- ne - fn(mu)
    mu0        -- initial gues energies
    """
    mu = mu0
    de = 0.1
    sp = 1
    s = 1
    nmax = 1000
    counter = 0

    while(np.abs(fun(mu)) > tol and counter < nmax):
        sp = s
        s = 1 if fun(mu) > 0 else -1
        if s == sp:
            de *= 1.25
        else:
            # decrease step size if we change direction
            de *= 0.25
        mu += s*de
        counter += 1
    return mu


def find_mu(kset, ek, T, tol=1e-10):
    """
    Wrapper for find_chemical_potential

    Arguments:
    kset -- kpointset
    ek   -- Fermi parameters
    T    -- temperature
    """
    ctx = kset.ctx()
    ne = ctx.unit_cell().num_valence_electrons()
    kw = kset.w
    m = ctx.max_occupancy()
    mu = find_chemical_potential(lambda mu: ne - np.sum(
        kw*fermi_function(ek, T, mu, m)),
        mu0=0, tol=tol)
    return mu


def grad_eta(Hij, ek, fn, T, kw):
    """
    Computes ∂L/∂η

    Arguments:
    Hij -- subspace Hamiltonian
    ek  -- Fermi parameters ϵ_n
    fn  -- occupation numbers
    T   -- temperature
    kw  -- kpoint weights

    Returns:
    g_eta -- gradient wrt η of the free-energy Lagrangian
    """
    kT = kb * T
    g_eta_1 = -1/kT * diag(diag(Hij) - kw*ek) * fn * (1-fn)
    dFdmu = np.sum(np.real(
        1/kT * einsum('i,i', (diag(Hij) - kw*ek).asarray().flatten(), fn * (1-fn))))
    sumfn = np.sum(kw*fn*(1-fn))
    # g_eta_2 is zero if all f_i are either 0 or 1
    if np.abs(sumfn) < 1e-9:
        g_eta_2 = 0
    else:
        g_eta_2 = diag(kw * fn * (1-fn) / sumfn * dFdmu)
    # off-diagonal terms
    II = diag(ca.ones_like(fn))
    Eij = ek-ek.T + II
    Fij = fn-fn.T
    for k in Eij.keys():
        EEc = np.abs(Eij[k]) < 1e-8
        Eij[k] = np.where(EEc, 1, Eij[k])
        Fij[k] = np.where(EEc, 0, Fij[k])

    g_eta_3 = Fij / Eij * Hij * (1-II)
    g_eta = (g_eta_1 + g_eta_2 + g_eta_3)
    return g_eta


def make_kinetic_precond(kpointset, eps=0.1):
    """
    Preconditioner
    P = 1 / (||k|| + ε)

    Keyword Arguments:
    kpointset --
    eps       -- ϵ
    """

    nk = len(kpointset)
    nc = kpointset.ctx().num_spins()
    P = PwCoeffs(dtype=np.float64, ctype=dia_matrix)
    for k in range(nk):
        kp = kpointset[k]
        gkvec = kp.gkvec()
        assert (gkvec.num_gvec() == gkvec.count())
        N = gkvec.count()
        d = np.array([
            1 / (np.sum(
                (np.array(gkvec.gkvec_cart(i)))**2) + eps)
            for i in range(N)
        ])
        for ispn in range(nc):
            P[k, ispn] = dia_matrix((d, 0), shape=(N, N))
    return DiagonalPreconditioner(P)


def btsearch(f, b, f0, maxiter=20, tau=0.5):
    """
    Backtracking search
    """

    x = b

    for i in range(maxiter):
        fx = f(x)
        if fx[0] > f0:
            x *= tau
        else:
            return x, fx
    raise ValueError('backtracking search could not find a new minimum')

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

    for k in range(n - 1):
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
        # print('gss: a=%.2g, d=%.2g' % (a, d))
        return (a, d)
    else:
        # print('gss: c=%.2g, b=%.2g' % (c, b))
        return (c, b)


class IdentityPreconditioner():
    """
    Identity preconditioner
    """
    def __init__(self, _f=1):
        self._f = _f

    def __matmul__(self, other):
        if self._f == -1:
            return -other
        elif self._f == 1:
            return other
        else:
            raise ValueError

    def __mul__(self, s):
        if self._f == -1:
            return -s
        elif self._f == 1:
            return s
        else:
            raise ValueError

    def __neg__(self):
        return IdentityPreconditioner(_f=-self._f)

    def __getitem__(self, key):
        return self._f

    __lmul__ = __mul__
    __rmul__ = __mul__


class DiagonalPreconditioner():
    """
    Apply diagonal preconditioner and project resulting gradient to satisfy the
    constraint.
    """

    def __init__(self, D):
        super().__init__()
        self.D = D

    def __matmul__(self, other):
        """
        """
        out = type(other)(dtype=other.dtype)
        if isinstance(other, CoefficientArray):
            for key, Dl in self.D.items():
                out[key] = Dl @ other[key]
        else:
            raise ValueError('wrong type given')
        return out

    def __mul__(self, s):
        """

        """
        if np.isscalar(s):
            out = type(self)(self.D)
            for key, Dl in self.D.items():
                out.D[key] = s*Dl
            return out
        elif isinstance(s, CoefficientArray):
            out = type(s)(dtype=s.dtype)
            for key in s.keys():
                out[key] = self.D[key] * s[key]
            return out

    __lmul__ = __mul__
    __rmul__ = __mul__

    def __neg__(self):
        """
        """
        if isinstance(self.D, CoefficientArray):
            out_data = type(self.D)(dtype=self.D.dtype, ctype=self.D.ctype)
            out = DiagonalPreconditioner(out_data)
            for k, v in self.D.items():
                out.D[k] = -v
            return out
        else:
            out = DiagonalPreconditioner(self.D)
            out.D = -self.D
            return out

    def __getitem__(self, key):
        return self.D[key]

    def inv(self):
        """
        inverse
        """
        D = type(self.D)(dtype=self.D.dtype, ctype=self.D.ctype)
        for k in self.D.keys():
            shape = self.D[k].shape
            D[k] = dia_matrix((1/self.D[k].data, 0), shape=shape)
        out = type(self)(D)
        return out


class F():
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

        T = self.M.T
        kset = self.M.energy.kpointset
        X_new = self.X + t * self.G_X
        eta_new = self.eta + t * self.G_eta
        ek, Ul = eta_new.eigh()
        X = loewdin(X_new) @ Ul
        kw = kset.w
        ne = kset.ctx().unit_cell().num_valence_electrons()
        m = kset.ctx().max_occupancy()

        # collect all band energies from every k-point rank
        comm = self.M.energy.kpointset.ctx().comm_k()
        vek = np.hstack(comm.allgather(ek.to_array()))
        vkw = deepcopy(ek)
        for k in vkw._data.keys():
            vkw[k] = np.ones_like(vkw[k]) * kw[k]
        vkw = np.hstack(comm.allgather(vkw.to_array()))

        # update occupation numbers
        mu = find_chemical_potential(lambda mu: ne - np.sum(vkw*fermi_function(vek, T, mu, m)),
                                     mu0=0)
        fn = fermi_function(ek, T, mu, m)
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
    gamma_eta = np.real(inner(g_eta, g_eta-gp_eta))
    gamma_X = np.real(inner(g_X, g_X-gp_X))
    gamma = max(0,
                (gamma_X + gamma_eta)
                /
                (l2norm(gp_X)**2 + l2norm(gp_eta)**2))
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
        kset = self.M.energy.kpointset
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
            raise ValueError('quadratic line-search failed to find a new minima')

        # free energy at minimum
        FE, Hx, X_n, f_n, ek, U = fline(xi_min)
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
            raise ValueError('quadratic line-search failed to find a new minima')

        return X_n, f_n, ek, FE, Hx, U

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

        t1, t2 = gss(Fline, a=0, b=5)
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

            raise ValueError('GSS didn\'t find a better value')
        return Xn, fn, ek, F, Hx, Ul

    def backtracking_search(self, X, f, eta, Fline, F0, tau=0.5):
        t1, res = btsearch(Fline, 5, F0, tau=tau)
        F1, Hx1, X1, f1, ek1, Ul1 = res

        return X1, f1, ek1, F1, Hx1, Ul1

    def run(self, X, fn, maxiter=100, restart=20, tol=1e-10,
            prec=False, kappa=0.3, eps=0.001, use_g_eta=False,
            tau=0.5, cgtype='FR'):

        if cgtype == 'PR':
            cg_update = polak_ribiere
        elif cgtype == 'FR':
            cg_update = fletcher_reeves
        elif cgtype == 'SD':
            cg_update = steepest_descent
        else:
            raise ValueError('wrong type')

        prec_direction = True

        kset = self.M.energy.kpointset
        if prec:
            K = make_kinetic_precond(kset, eps=eps)
        else:
            K = IdentityPreconditioner()
        # occupation prec
        kappa0 = kappa

        M = self.M
        H = self.M.H
        T = self.M.T
        kw = kset.w
        m = kset.ctx().max_occupancy()
        # set ek from fn
        ek = inv_fermi_function(fn, T, m)
        eta = diag(ek)
        w, U = eta.eigh()
        ek = w
        X = X@U
        # compute initial free energy
        FE, Hx = M(X, fn)
        logger('intial F: %.10g' % FE)
        # save_state({'fn': fn, 'ek': ek}, kset, 'init_occu')

        HX = Hx * kw
        Hij = X.H @ HX
        g_eta = grad_eta(Hij, ek, fn, T, kw)
        LL = Hij*fn
        g_X = (HX*fn - X@LL)

        G_X = -g_X
        G_eta = -g_eta
        delta_X = G_X
        delta_eta = G_eta

        cg_restart_inprogress = False
        for ii in range(1, 1+maxiter):
            slope = np.real(2*inner(g_X, G_X) + inner(g_eta, G_eta))

            if np.abs(slope) < tol:
                return X, fn

            if slope > 0:
                if cg_restart_inprogress:
                    # save_state({'X': X, 'f': fn,
                    #             'eta': eta, 'G_X': G_X,
                    #             'gx': g_X,
                    #             'g_eta': g_eta,
                    #             'F0': FE,
                    #             'slope': slope,
                    #             'G_eta': G_eta, 'slope': slope}, self.M.energy.kpointset)
                    raise ValueError('Error: _ascent_ direction, slope %.4e' % slope)
                else:
                    cg_restart_inprogress = True
            else:
                try:
                    X, fn, ek, FE, Hx, U = self.step(X, fn, eta, G_X, G_eta,
                                                     xi_trial=0.2, F0=FE, slope=slope,
                                                     kwargs={'gx': g_X, 'g_eta': g_eta})
                    # reset kappa
                    kappa = kappa0
                    cg_restart_inprogress = False
                except ValueError:
                    # side effects
                    try:
                        Fline = F(X, eta, M, G_X, G_eta)
                        # X, fn, ek, FE, U = self.step_golden_section_search(X, fn, eta, Fline, FE)
                        X, fn, ek, FE, Hx, U = self.backtracking_search(X, fn, eta, Fline, FE)
                    except ValueError:
                        # not even golden section search works
                        # restart CG and reduce kappa
                        cg_restart_inprogress = True
                        kappa = kappa/3
                        logger('kappa: ', kappa)

            logger('step %5d' % ii, 'F: %.11f res: X,eta %+10.5e, %+10.5e' %
                   (FE, np.real(inner(g_X, G_X)), np.real(inner(g_eta, G_eta))))
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
            g_eta = grad_eta(Hij, ek, fn, T, kw)
            # Lagrange multipliers
            XhKHXF = X.H @ (K @ HX)
            XhKX = X.H @ (K @ X)
            LL = _solve(XhKX, XhKHXF)
            gp_X = g_X@U
            g_X = (HX*fn - X@LL)
            # check that constraints are fulfilled
            delta_X = -K * (HX - X @ LL) / kw
            # assert l2norm(X.H @ delta_X) < 1e-11
            if not use_g_eta:
                delta_eta = kappa * (Hij - kw*diag(ek)) / kw
            else:
                delta_eta = -kappa * g_eta

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

        return X, fn
