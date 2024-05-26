import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias
from numpy.linalg import eigh, solve
from ..coefficient_array import PwCoeffs, allthreaded

complex_array_t: TypeAlias = NDArray[np.complex128]


def matview(x):
    return np.matrix(x, copy=False)


def lagrangeMult(p, c0, P=None):
    """
    Keyword Arguments:
    p  --
    c0 --
    """
    from ..coefficient_array import CoefficientArray

    if isinstance(p, CoefficientArray):
        out = PwCoeffs()
        for key, v in p.items():
            if P is not None:
                out[key] = lagrangeMult(v, c0[key], P[key])
            else:
                out[key] = lagrangeMult(v, c0[key])
        return out
    else:
        c0 = matview(c0)
        p = matview(p)
        if P is not None:
            oc = solve(c0.H * P * c0, c0.H * P * p)
            correction = -P * c0 * oc
        else:
            oc = solve(c0.H * c0, c0.H * p)
            correction = -c0 * oc
        return correction


def _c(x: complex_array_t, c0: complex_array_t) -> complex_array_t:
    x = matview(x)
    assert np.linalg.norm(np.dot(x.H, c0), "fro") < 1e-7
    XX = np.dot(x.H, x)
    w, R = eigh(XX)
    w = np.where(np.abs(w) < 1e-14, 0, w)
    w = np.sqrt(w)
    R = matview(R)
    err = np.linalg.norm(R.H @ R - np.eye(*R.shape), "fro")
    # TODO: remove: check that we are not losing accuracy for small x
    assert np.isclose(R * np.diag(w**2) * R.H, XX).all()
    assert err < 1e-11

    Wsinc = np.diag(np.sinc(w / np.pi))
    sincU = np.dot(np.dot(R, Wsinc), R.H)

    Wcos = np.diag(np.cos(w))
    cosU = np.dot(np.dot(R, Wcos), R.H)

    return np.dot(c0, cosU) + np.dot(x, sincU)


def c(x, c0):
    """
    Keyword Arguments:
    x  -- either numpy array like or PwCoeffs
    c0 -- either numpy array like or PwCoeffs
    it must hold x.T*c0 = 0
    """
    if isinstance(x, PwCoeffs):
        return allthreaded(_c)(x, c0)
    else:
        return _c(x, c0)


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
        tuple -- (c, Λ, sinU, R)
        """
        # make sure x has type np.matrix
        x = np.matrix(x, copy=False)

        # check that x fulfills constraint condition
        assert np.linalg.norm(np.dot(x.H, c0), "fro") < 1e-7

        # compute eigenvectors and eigenvalues of U
        XX = np.dot(x.H, x)
        Λ, R = eigh(XX)
        w = np.sqrt(Λ)
        R = np.matrix(R)
        # note: U = V * sqrt(Λ) * V.H = sqrt(X.T X)

        # check for orthonormality
        err = np.linalg.norm(R.H * R - np.eye(*R.shape), "fro")
        assert err < 1e-10

        # pre-compute matrix functions sin, cos, and inverse of U
        Wsinc = np.diag(np.sinc(w / np.pi))
        sincU = np.dot(np.dot(R, Wsinc), R.H)
        # cos
        Wcos = np.diag(np.cos(w))
        cosU = np.dot(np.dot(R, Wcos), R.H)
        assert isinstance(cosU, np.matrix)

        # compute c(c0, x)
        c = np.dot(c0, cosU) + np.dot(x, sincU)

        return c, Λ, sincU, R

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
            assert isinstance(self.c0, PwCoeffs)
            Hx_out = PwCoeffs()
            Gx_out = PwCoeffs()
            for ki in x.kvalues():
                # get a view of all x-coefficients for given k-index
                xk = x.kview(ki)
                # precompute matrix functions and pw-coeffs
                fX = {}
                # plane-wave coefficients c(x, c0)
                c = PwCoeffs()
                for key, val in xk.items():
                    fX[key] = {
                        k: t
                        for k, t in zip(
                            ["c", "Λ", "sincU", "R"],
                            self._prepare_fX(x=val, c0=self.c0[key]),
                        )
                    }
                    # this makes a copy ...
                    c[key] = fX[key]["c"]
                # apply Hamiltonian for all spin-components in a single k-point
                Hc = self.hamiltonian(c)
                # now call _apply_single for every element
                for key, val in xk.items():
                    Hc_loc = Hc[key]
                    Gx, Hx = self._apply_single(
                        x=xk[key],
                        c0=self.c0[key],
                        Λ=fX[key]["Λ"],
                        Hc=Hc_loc,
                        sincU=fX[key]["sincU"],
                        R=fX[key]["R"],
                    )
                    Gx_out[key] = Gx
                    Hx_out[key] = Hx
            # return ∂E/∂x as PwCoeffs
            return Gx_out, Hx_out
        else:
            # valid input?
            assert ispn is not None
            assert ki is not None
            # compute matrix functions
            c, Λ, sincU, R = self._prepare_fX(x, self.c0)
            # compute ∂E/∂c
            Hc = self.hamiltonian(c, ki=ki, ispn=ispn)
            # return ∂E/∂x as np.matrix
            return self._apply_single(
                x=x, c0=np.matrix(self.c0, copy=False), Λ=Λ, Hc=Hc, sincU=sincU, R=R
            )

    @staticmethod
    def _apply_single(x, c0, Λ, Hc, sincU, R):
        """
        Computes ∂E/∂x
        x    -- OT coefficients, where PW-coefficients are given by c = c(x, c0)
        c0   -- initial coefficients
        Λ    -- eigenvalues of sqrt(x.H * x)
        Hc   -- gradient
        sincU -- sinc(U)
        R    -- eigenvectors of U (in columns)
        """
        # D¹: TODO repeat formula from
        #   VandeVondele, J., & Hutter, J. . An efficient orbital transformation method
        #   for electronic structure calculations. , 118(10), 4365–4369.
        #   http://dx.doi.org/10.1063/1.1543154

        R = matview(R)
        sincU = matview(sincU)
        Hc = matview(Hc)
        c0 = matview(c0)

        # assert isinstance(R, np.matrix)
        # assert isinstance(sincU, np.matrix)
        # assert isinstance(Hc, np.matrix)
        # assert isinstance(c0, np.matrix)

        v = np.sinc(np.sqrt(Λ) / np.pi)
        v = v[:, np.newaxis]
        diffL = Λ[:, np.newaxis] - Λ[:, np.newaxis].T
        mask = np.abs(diffL) < 1e-10
        D1 = np.ma.masked_array(v - v.T, mask=mask) / diffL
        # fill masked entries with correct formula
        irow, icol = np.where(mask)

        def f1(la):
            """
            D1 matrix entries, where lambda1=lambda2 or lambda1=lambda2=0
            """
            pos_index = la > 1e-10
            out = np.zeros(la.shape, dtype=la.dtype)
            out[pos_index] = 0.5 * (
                np.cos(np.sqrt(la[pos_index])) / la[pos_index]
                - np.sin(np.sqrt(la[pos_index])) / (la[pos_index] ** (1.5))
            )
            out[~pos_index] = -1 / 6
            # if (~pos_index).any():
            #     print('OUTPUT: encountered small eigenvalue!')
            if (la < -1e-10).any():
                raise Exception
            return out

        # D1(x1,x2) = 1/2 ( cos(sqrt(x)) / x - sin(sqrt(x)) / x**(3/2) )  if x1==x2
        # D1[irow, icol] = 0.5*(np.cos(np.sqrt(Λ[irow])) / Λ[irow] - np.sin(np.sqrt(Λ[irow])) / (Λ[irow]**(1.5)))
        D1[irow, icol] = f1(Λ[irow])

        # D²: TODO insert formula
        v = np.cos(np.sqrt(Λ))
        v = v[:, np.newaxis]
        D2 = np.ma.masked_array(v - v.T, mask=mask) / diffL
        # D2(x1, x2) = -1/2 sin(sqrt(x)) / sqrt(x) if x1==x2
        D2[irow, icol] = -0.5 * np.sinc(np.sqrt(Λ[irow]) / np.pi)

        # compute K: TODO copy/paste formula from the paper
        RtHCtxR = np.array(R.H * (Hc.H * x) * R)
        RtHCtc0R = np.array(R.H * Hc.H * c0 * R)
        K = np.matrix(RtHCtxR * np.array(D1) + RtHCtc0R * np.array(D2))
        # compute ∂E/∂x
        dEdx = Hc * sincU + x * (R * (K.H + K) * R.H)

        # Lagrange multiplier
        # TODO: this can be precomputed and stored
        lM = solve(c0.H @ c0, c0.H @ dEdx)
        correction = -1 * c0 @ lM

        return dEdx + correction, dEdx


def get_c0_x(kpointset, eps=0):
    """ """
    c0 = PwCoeffs(kpointset)
    x = PwCoeffs()
    for key, c0_loc in c0.items():
        x_loc = np.zeros_like(c0_loc)
        x[key] = x_loc

    return c0, x
