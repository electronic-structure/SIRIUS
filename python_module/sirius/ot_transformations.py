import numpy as np
from numpy.linalg import eigh, solve


def matview(x):
    return np.matrix(x, copy=False)


def c(x, c0):
    """
    Keyword Arguments:
    x --
    c0 --
    it must hold x.T*c0 = 0
    """
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
        self.c0 = np.matrix(c0, copy=True)

    def __call__(self, x):
        """
        Computes ∂E/∂x

        x -- OT coefficients, where PW-coefficients are given by c = c(x, c0)
        """
        # make sure x has type np.matrix
        x = np.matrix(x, copy=False)

        # check that x fulfills constraint condition
        assert (np.linalg.norm(np.dot(x.H, self.c0), 'fro') < 1e-7)

        # compute eigenvectors and eigenvalues of U
        XX = np.dot(x.H, x)
        Λ, R = eigh(XX)
        w = np.sqrt(Λ)
        R = np.matrix(R)
        # note: U = V * sqrt(Λ) * V.H = sqrt(X.T X)

        # check that we have obtained orthonormal eigenvectors
        err = np.linalg.norm(R.H * R - np.eye(*R.shape), 'fro')
        assert (err < 1e-10)

        # pre-compute matrix functions sin, cos, and inverse of U
        # sin
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
        c = np.dot(self.c0, cosU) + np.dot(np.dot(x, invU), sinU)
        # store of c(x) in wave_function object
        # compute ∂E/∂c
        Hc = self.hamiltonian(c)

        # D¹: TODO repeat formula from
        #   VandeVondele, J., & Hutter, J. . An efficient orbital transformation method
        #   for electronic structure calculations. , 118(10), 4365–4369.
        #   http://dx.doi.org/10.1063/1.1543154
        v = np.sin(np.sqrt(Λ)) / np.sqrt(Λ)
        v = v[:, np.newaxis]
        # TODO: mask diagonal elements (get rid of warnings)
        diffL = (Λ[:, np.newaxis] - Λ[:, np.newaxis].T)
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
        RtHCtc0R = np.array(R.H * Hc.H * self.c0 * R)
        K = np.matrix(RtHCtxR*np.array(D1) + RtHCtc0R*np.array(D2))
        # compute ∂E/∂x
        dEdx = Hc*invU*sinU + x*(R*(K.H + K)*R.H)

        # Lagrange multiplier
        lagrangeMult = solve(self.c0.H * self.c0, self.c0.H * dEdx)
        correction_term = -1 * self.c0 * lagrangeMult

        return dEdx + correction_term, dEdx
