import numpy as np
from sirius.baarman import stiefel_project_tangent, stiefel_transport_operators
import sirius.ot as ot
import h5py
from sirius import Logger

logger = Logger()


class GradientXError(Exception):
    pass


def save_state(kset, X, Y, f, y, tau_min, sigma_min, prefix='fail'):
    """
    dump current state to HDF5
    """
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    with h5py.File(prefix+'%d.h5' % rank, 'w') as fh5:
        _ = ot.save(fh5, 'X', X, kset)
        _ = ot.save(fh5, 'fn', f, kset)
        grpY = ot.save(fh5, 'Y', Y, kset)
        grpy = ot.save(fh5, 'y', y, kset)
        grpY.attrs['tau_min'] = tau_min
        grpy.attrs['sigma_min'] = sigma_min


def fermi_function(x, T, mu, num_spins):
    """
    Keyword Arguments:
    x  --
    T  --
    mu --
    """
    # if x < 5:
    #     return 1
    # else:
    #     return 1 / (1 + np.exp((x - mu) / T))

    return num_spins / (1 + np.exp((x - mu) / T))


def num_electrons(n, T, mu, num_spins):
    """

    """
    return sum(map(lambda x: fermi_function(x, T, mu, num_spins),
                   np.arange(n)))


def trim(X, fn, tol=1e-12):
    """
    Trim and sort.

    Keyword Arguments:
    X  -- pw coeffs
    fn -- occupation numbers
    """
    import numpy as np
    from sirius import CoefficientArray

    fnout = type(fn)(dtype=fn.dtype, ctype=fn.ctype)
    Xout = type(X)(dtype=X.dtype, ctype=X.ctype)
    sel = CoefficientArray(dtype=np.int, ctype=np.array)
    for k in fn._data.keys():
        floc = np.array(fn[k]).flatten()
        if (floc < tol).any():
            perm = np.argsort(floc)[::-1]
            floc = floc[perm]
            mask = floc > tol
            floc = floc[mask]
            fnout[k] = floc
            masked_perm = perm[mask]
            Xout[k] = X[k][:, masked_perm]
            sel[k] = masked_perm
        else:
            fnout[k] = floc
            Xout[k] = X[k]
            sel[k] = np.arange(len(floc))

    return Xout, fnout, sel


def quadratic_approximation(free_energy, dAdC, dAdf, Y, y, X, fn, te, se,
                            comm, kweights, mag):
    """
    Keyword Arguments:
    dAdC -- ∇ₓ A
    dAdf -- ∇f A
    Y    -- search direction X
    y    -- search direction f  (admissible)
    X    -- pw coefficients
    f    -- band occupancies
    te   -- interpolation offset X
    se   -- interpolation offset f

    Returns:
    coefficients of the 2d interpolating polynomial
    """

    # p(t,s) = c[0] t^2 + c[1] s^2 + c[2] t + c[3] s + c[4]

    import numpy as np
    from sirius.baarman import constrain_occupancy_gradient
    logger('\tquadratic approximation: te=%.4g, se=%.4g' % (te, se))

    U, _ = stiefel_transport_operators(Y, X, tau=te)
    at_trial_step = False

    coeffs = np.zeros(5)
    if se < 1e-10:
        # coeffs[4] = free_energy(X, fn)
        Asys = np.zeros((3, 3))
        rsys = np.zeros(3)

        Xnew = U @ X
        ynew = fn + se * y

        Asys[0, 1] = 1
        rsys[0] = 2 * np.real(inner(dAdC, Y))
        logger('slope:', rsys[0])
        Asys[1, [0, 1, 2]] = [te**2, te, 1]
        rsys[1] = free_energy(Xnew, ynew)
        Asys[2, 2] = 1
        eint1 = free_energy(X, fn)
        rsys[2] = eint1
        lcoeffs = np.linalg.solve(Asys, rsys)
        coeffs[[0, 2, 4]] = lcoeffs
    else:
        Asys = np.zeros((5, 5))
        rsys = np.zeros(5)
        # 1-th line: dτ p(0,0)
        Asys[1, 2] = 1
        rsys[1] = 2 * np.real(inner(dAdC, Y))
        if rsys[1] >= 0:
            raise GradientXError("in quadratic_approximation")
        # 2-nd line: dσ p(0,0)
        Asys[2, 3] = 1
        rsys[2] = np.real(inner(dAdf, y))
        # assert rsys[2] < 1e-7
        # 3-rd line: dτ p(te, se)
        # goto target point: U(te)X, fn+se*y
        Xnew = U @ X
        ynew = fn + se * y
        eint1 = free_energy(Xnew, ynew)
        # logger('\tfree energy at interpolation point: ', eint1)
        dAdCnew, dAdfnew = free_energy.grad(Xnew, ynew)
        ynew = constrain_occupancy_gradient(dAdfnew, ynew, comm, kweights, mag)
        Asys[3, [0, 2]] = [2 * te, 1]
        rsys[3] = np.real(inner(dAdCnew, Y))
        # 4-th line: dσ p(te, se)
        Asys[4, [1, 3]] = [2 * se, 1]
        rsys[4] = np.real(inner(dAdfnew, y))
        # 0-th line: p(0,0)
        # last: make sure that after return we are at X, fn
        if at_trial_step:
            Asys[0, :] = [te**2, se**2, te, se, 1]
            rsys[0] = eint1
            # reset state
            free_energy(X, fn)
        else:
            Asys[0, 4] = 1
            rsys[0] = free_energy(X, fn)
        coeffs = np.linalg.solve(Asys, rsys)
        # eint1_extrap = eval_quadratic_approx(coeffs, te, se)
        # logger('\tqapprox eint1: ', eint1)
        # logger('\tqapprox eint1_extrap: ', eint1_extrap)

    return coeffs, eint1


def eval_quadratic_approx(c, tau, sigma):
    return tau**2 * c[0] + sigma**2 * c[1] + tau * c[2] + sigma * c[3] + c[4]


class CG:
    def __init__(self, A, mag=True):
        """
        Keyword Arguments:
        A       -- free energy
        """
        import numpy as np
        self.A = A
        self.comm = A.comm
        # k-point weights (rank local)
        self.kweights = A.energy.kpointset.w
        # minimum step lengths
        self.min_tau = 0.1
        self.min_sigma = 0.1
        # last step length
        self.tau_l = self.min_tau
        self.sigma_l = self.min_sigma
        self._E = np.finfo('d').max
        self.mag = mag

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, value):
        if value <= self._E:
            self._E = value
            logger('%.8f accepted energy' % self._E)
        else:
            raise ValueError("CG step has failed")

    @staticmethod
    def polak_ribiere(Fnext, WF, F):
        from sirius import inner
        import numpy as np

        return np.real(inner(Fnext - WF, Fnext)) / np.real(inner(F, F))

    def run(self, X0, f0, A, maxiter=100, tol=1e-8):
        """
        Keyword Arguments:
        X0      -- initial guess PW
        f0      -- initial guess occupancy
        A       -- free energy
        maxiter -- (default 100)
        tol     -- (default 1e-8)
        """
        E = A(X0, f0)
        logger('entry E=', E)
        # X, fn, sel = trim(X0, f0, tol=1e-12)
        X, fn = deepcopy(X0), deepcopy(f0)
        Et = A(X, fn)
        assert (np.isclose(Et, E))
        dAdC, dAdf = A.grad(X, fn)

        Y = stiefel_project_tangent(-dAdC, X)

        for i in range(maxiter):
            X, Y, dAdC, fn, dAdf = self.cg_step(X, fn, dAdC, dAdf, Y)
            # logger('occupation numbers:', fn)
            # check for convergence
            res = np.real(inner(Y, dAdC))
            resf = np.real(inner(dAdf, dAdf))
            logger('iteration %03d, res %.5g f: %.5g' % (i, res, resf))
        return X, fn

    def cg_step(self, X, f, dAdC, dAdf, Y):
        """
        Keyword Arguments:
        X       -- current planewave coefficients
        f       -- next planewave coefficients
        Y       -- search direction
        sigma_k -- previous step length
        tau_k   -- previous step length
        """
        from sirius.baarman import constrain_occupancy_gradient
        from sirius.baarman import stiefel_transport_operators
        from sirius.baarman import occupancy_admissible_ds

        # current_energy = A(X, f)
        y = constrain_occupancy_gradient(dAdf, f, self.comm,
                                         self.kweights, self.mag)
        logger('\t||y||:', inner(y, y))
        sigma_max = occupancy_admissible_ds(y, f, self.mag)
        # logger('y:', y)

        # TODO: determine te, se
        te = max(self.min_tau, 0.1*self.tau_l)
        se = min(sigma_max, 0.1*max(self.min_sigma, self.sigma_l))
        # se = 0.1 * sigma_max
        logger('\tte=%.4g, se=%.4g' % (te, se))
        try:
            coeffs, Etrial = quadratic_approximation(
                self.A, dAdC, dAdf, Y, y, X=X, fn=f, te=te, se=se,
                comm=self.comm, kweights=self.kweights, mag=self.mag)
        except GradientXError:
            logger('!!!CG RESTART!!!')
            Y = -stiefel_project_tangent(dAdC, X)
            coeffs, Etrial = quadratic_approximation(
                self.A, dAdC, dAdf, Y, y, X=X, fn=f, te=te, se=se,
                comm=self.comm, kweights=self.kweights, mag=self.mag)

        if np.abs(coeffs[1]) < 1e-9:
            sigma_min = 0
        else:
            sigma_min = -coeffs[3] / (2 * coeffs[1])

        tau_min = -coeffs[2] / (2 * coeffs[0])

        if sigma_min < 0:
            sigma_min = sigma_max
            logger(
                '\tOCCUPATION APPROX FAILED! taking sigma_max'
            )
            save_state(self.A.energy.kpointset, X, Y, f, y, tau_min, sigma_min)
        if sigma_min > sigma_max:
            logger('\ttoo long step for occupation numbers, limit by max step, σ_min = %.5g, a= %.5g' % (sigma_min, coeffs[3]))
            sigma_min = sigma_max

        # store step lengths as estimate for next iteration
        self.tau_l = tau_min
        self.sigma_l = sigma_min

        U, W = stiefel_transport_operators(Y, X, tau=tau_min)
        # transport pw along geodesic
        Xnew = U @ X
        Yt = W @ Y
        assert np.isclose(inner(Yt, Yt), inner(Y, Y))
        # new occupation numbers
        ynew = f + sigma_min * y

        logger('\tstep f: ', sigma_min**2 * inner(y, y))

        E = self.A(Xnew, ynew)

        try:
            self.E = E
        except ValueError:
            Ex = self.A(X, ynew)
            logger('te: ', te)
            logger('se: ', se)
            logger('tau_min:', tau_min)
            logger('sigma_min:', sigma_min)
            if Etrial < self.E:
                E = Etrial
                logger('APPROX FAILED, go to trial point')
                U, W = stiefel_transport_operators(Y, X, tau=te)
                # transport pw along geodesic
                Xnew = U @ X
                Yt = W @ Y
                # parallel transport gradient along geodesic
                # new occupation numbers
                ynew = f + se * y
                self.E = self.A(Xnew, ynew)
                assert (np.isclose(self.E, Etrial))
            else:
                try:
                    logger('attempt an optimization step in x only')
                    # attempt to optimize only X
                    self.A(X, f)
                    coeffs, Etrial = quadratic_approximation(
                        self.A, dAdC, dAdf, Y, y, X=X, fn=f, te=te, se=0,
                        comm=self.comm, kweights=self.kweights, mag=self.mag)
                    assert coeffs[0] > 0 and coeffs[2] < 0
                    tau_min = -coeffs[2] / (2*coeffs[0])
                    U, W = stiefel_transport_operators(Y, X, tau=tau_min)
                    # transport pw along geodesic
                    Xnew = U @ X
                    Yt = W @ Y
                    ynew = f
                    E = self.A(Xnew, ynew)
                    # attempt to update
                    self.E = E
                except:
                    save_state(self.A.energy.kpointset, X, Y, f, y, tau_min, sigma_min)
                    raise Exception('giving up')

        F = stiefel_project_tangent(dAdC, X)
        WF = W @ F
        # trim..
        # Xout, fnout, sel = trim(Xnew, ynew)
        # assert np.isclose(E, self.A(Xout, fnout))
        # compute new gradients
        # self.A(Xnew, ynew)
        dAdC, dAdf = self.A.grad(Xnew, ynew)
        # project to Stiefel manifold
        Fnext = stiefel_project_tangent(dAdC, Xnew)
        # determine next direction
        gamma = self.polak_ribiere(Fnext, WF, F)

        Ynew = -Fnext + gamma * Yt

        return Xnew, Ynew, dAdC, ynew, dAdf
