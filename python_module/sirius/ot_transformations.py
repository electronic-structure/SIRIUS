import numpy as np
from numpy.linalg import eigh, solve
from .coefficient_array import PwCoeffs


def matview(x):
    return np.matrix(x, copy=False)


def constrain(p, c0):
    """
    Keyword Arguments:
    p  --
    c0 --
    """
    from .coefficient_array import CoefficientArray
    if isinstance(p, CoefficientArray):
        out = type(p)(dtype=p.dtype, ctype=p.ctype)
        for key, v in p.items():
            out[key] = constrain(v, c0[key])
        return out
    else:
        c0 = matview(c0)
        p = matview(p)
        oc = solve(c0.H * c0, c0.H * p)
        correction = -1 * c0 * oc
        return p + correction


def c(x, c0):
    """
    Keyword Arguments:
    x  -- either numpy array like or PwCoeffs
    c0 -- either numpy array like or PwCoeffs
    it must hold x.T*c0 = 0
    """
    if isinstance(x, PwCoeffs):
        assert(isinstance(c0, PwCoeffs))
        cres = PwCoeffs(dtype=np.complex)
        for key, xloc in x.items():
            c0loc = c0[key]
            cres[key] = c(xloc, c0loc)
        return cres
    else:
        x = matview(x)
        assert (np.linalg.norm(np.dot(x.H, c0), 'fro') < 1e-7)
        XX = np.dot(x.H, x)
        w, R = eigh(XX)
        w = np.sqrt(w)
        R = matview(R)
        err = np.linalg.norm(R.H * R - np.eye(*R.shape), 'fro')
        assert(err < 1e-11)

        Wsin = np.diag(np.sin(w))
        sinU = np.dot(np.dot(R, Wsin), R.H)

        Wcos = np.diag(np.cos(w))
        cosU = np.dot(np.dot(R, Wcos), R.H)
        invU = np.dot(np.dot(R, np.diag(1. / w)), R.H)

        return np.dot(c0, cosU) + np.dot(np.dot(x, invU), sinU)


class ConstrainedGradient:
    def __init__(self, hamiltonian, c0):
        self.hamiltonian = hamiltonian
        self.c0 = c0

    @staticmethod
    def _prepare_fX(x, c0):
        """
        compute matrix functions of x
        x  -- OT coefficients
        c0 -- initial pw coefficients

        Returns:
        tuple -- (c, Λ, invU, sinU)
        """
        # make sure x has type np.matrix
        x = np.matrix(x, copy=False)

        # check that x fulfills constraint condition
        assert (np.linalg.norm(np.dot(x.H, c0), 'fro') < 1e-7)

        # compute eigenvectors and eigenvalues of U
        XX = np.dot(x.H, x)
        Λ, R = eigh(XX)
        w = np.sqrt(Λ)
        R = np.matrix(R)
        # note: U = V * sqrt(Λ) * V.H = sqrt(X.T X)

        # check for orthonormality
        err = np.linalg.norm(R.H * R - np.eye(*R.shape), 'fro')
        assert (err < 1e-10)

        # pre-compute matrix functions sin, cos, and inverse of U
        Wsin = np.diag(np.sin(w))
        sinU = np.dot(np.dot(R, Wsin), R.H)
        assert(isinstance(sinU, np.matrix))
        # cos
        Wcos = np.diag(np.cos(w))
        cosU = np.dot(np.dot(R, Wcos), R.H)
        assert(isinstance(cosU, np.matrix))
        # inv
        invU = np.dot(np.dot(R, np.diag(1. / w)), R.H)
        assert(isinstance(invU, np.matrix))

        # compute c(c0, x)
        c = np.dot(c0, cosU) + np.dot(np.dot(x, invU), sinU)

        return c, Λ, invU, sinU, R

    def __call__(self, x, ki=None, ispn=None):
        """
        compute ∂E/∂x
        x    -- OT coefficients
        ki   -- (default None) k-point index
        ispn -- (default None) spin-component

        Returns

        Gx -- gradient (projected)
        Hx -- gradient (not projected)
        depending on type of x: np.matrix or PwCoeffs
        """
        if isinstance(x, PwCoeffs):
            assert(isinstance(self.c0, PwCoeffs))
            Hx_out = PwCoeffs(dtype=x.dtype)
            Gx_out = PwCoeffs(dtype=x.dtype)
            for ki in x.kvalues():
                # get a view of all x-coefficients for given k-index
                xk = x.kview(ki)
                # precompute matrix functions and pw-coeffs
                fX = {}
                # plane-wave coefficients c(x, c0)
                c = PwCoeffs(dtype=x.dtype)
                for key, val in xk.items():
                    fX[key] = {k: t for k, t in zip(['c', 'Λ', 'invU', 'sinU', 'R'],
                                                    self._prepare_fX(x=val, c0=self.c0[key]))}
                    # this makes a copy ...
                    c[key] = fX[key]['c']
                # apply Hamiltonian for all spin-components in a single k-point
                Hc = self.hamiltonian(c)
                # now call _apply_single for every element
                for key, val in xk.items():
                    Hc_loc = Hc[key]
                    Gx, Hx = self._apply_single(x=xk[key],
                                                c0=self.c0[key],
                                                Λ=fX[key]['Λ'],
                                                Hc=Hc_loc,
                                                invU=fX[key]['invU'],
                                                sinU=fX[key]['sinU'],
                                                R=fX[key]['R'])
                    Gx_out[key] = Gx
                    Hx_out[key] = Hx
            # return ∂E/∂x as PwCoeffs
            return Gx_out, Hx_out
        else:
            # valid input?
            assert(ispn is not None)
            assert(ki is not None)
            # compute matrix functions
            c, Λ, invU, sinU, R = self._prepare_fX(x, self.c0)
            # compute ∂E/∂c
            Hc = self.hamiltonian(c, ki=ki, ispn=ispn)
            # return ∂E/∂x as np.matrix
            return self._apply_single(x=x,
                                      c0=np.matrix(self.c0, copy=False),
                                      Λ=Λ,
                                      Hc=Hc,
                                      invU=invU,
                                      sinU=sinU,
                                      R=R)

    @staticmethod
    def _apply_single(x, c0, Λ, Hc, invU, sinU, R):
        """
        Computes ∂E/∂x
        x    -- OT coefficients, where PW-coefficients are given by c = c(x, c0)
        c0   -- initial coefficients
        Λ    -- eigenvalues of sqrt(x.H * x)
        Hc   -- gradient
        R    -- eigenvectors of U (in columns)
        """
        # D¹: TODO repeat formula from
        #   VandeVondele, J., & Hutter, J. . An efficient orbital transformation method
        #   for electronic structure calculations. , 118(10), 4365–4369.
        #   http://dx.doi.org/10.1063/1.1543154
        assert(isinstance(R, np.matrix))
        assert(isinstance(invU, np.matrix))
        assert(isinstance(sinU, np.matrix))
        assert(isinstance(Hc, np.matrix))
        assert(isinstance(c0, np.matrix))

        v = np.sin(np.sqrt(Λ)) / np.sqrt(Λ)
        v = v[:, np.newaxis]
        # TODO: mask diagonal elements (get rid of warnings)
        diffL = Λ[:, np.newaxis] - Λ[:, np.newaxis].T
        mask = np.abs(diffL) < 1e-10
        D1 = np.ma.masked_array(v - v.T, mask=mask) / diffL
        # fill masked entries with correct formula
        irow, icol = np.where(mask)
        # D1(x1,x2) = 1/2 ( cos(sqrt(x)) / x - sin(sqrt(x)) / x**(3/2) )  if x1==x2
        D1[irow, icol] = 0.5*(np.cos(np.sqrt(Λ[irow])) / Λ[irow] - np.sin(np.sqrt(Λ[irow])) / (Λ[irow]**(1.5)))
        # D²: TODO insert formula
        v = np.cos(np.sqrt(Λ))
        v = v[:, np.newaxis]
        D2 = np.ma.masked_array(v - v.T, mask=mask) / diffL
        # D2(x1, x2) = -1/2 sin(sqrt(x)) / sqrt(x) if x1==x2
        D2[irow, icol] = -0.5*np.sin(np.sqrt(Λ[irow])) / np.sqrt(Λ[irow])

        # compute K: TODO copy/paste formula from the paper
        RtHCtxR = np.array(R.H * (Hc.H * x) * R)
        RtHCtc0R = np.array(R.H * Hc.H * c0 * R)
        K = np.matrix(RtHCtxR*np.array(D1) + RtHCtc0R*np.array(D2))
        # compute ∂E/∂x
        dEdx = Hc*invU*sinU + x*(R*(K.H + K)*R.H)

        # Lagrange multiplier
        # TODO: this can be precomputed and stored
        lM = solve(c0.H * c0, c0.H * dEdx)
        correction = -1 * c0 * lM

        return dEdx + correction, dEdx
