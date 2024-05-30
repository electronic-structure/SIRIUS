from scipy.sparse import dia_matrix
from ..coefficient_array import CoefficientArray, PwCoeffs
import numpy as np


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
        raise ValueError

    def __mul__(self, s):
        if self._f == -1:
            return -s
        elif self._f == 1:
            return s
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
        out = type(other)()
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
        if isinstance(s, CoefficientArray):
            out = type(s)()
            for key in s.keys():
                out[key] = self.D[key] * s[key]
            return out

    __lmul__ = __mul__
    __rmul__ = __mul__

    def __neg__(self):
        """
        """
        if isinstance(self.D, CoefficientArray):
            out_data = type(self.D)()
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
        D = type(self.D)()
        for k in self.D.keys():
            shape = self.D[k].shape
            D[k] = dia_matrix((1/self.D[k].data, 0), shape=shape)
        out = type(self)(D)
        return out


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
    P = PwCoeffs()
    for k in range(nk):
        kp = kpointset[k]
        gkvec = kp.gkvec()
        # assert (gkvec.num_gvec() == gkvec.count())
        N = gkvec.count()
        d = np.array([
            1 / (np.sum(
                (np.array(gkvec.gkvec_cart(i)))**2) + eps)
            for i in range(N)
        ])
        for ispn in range(nc):
            P[k, ispn] = dia_matrix((d, 0), shape=(N, N))
    return DiagonalPreconditioner(P)


def make_kinetic_precond2(kpointset):
    """

    Payne, M. C., Teter, M. P., Allan, D. C., Arias, T. A., & Joannopoulos, J.
    D., Iterative minimization techniques for ab initio total-energy
    calculations: molecular dynamics and conjugate gradients.
    https://dx.doi.org/10.1103/RevModPhys.64.1045

    """
    nk = len(kpointset)
    nc = kpointset.ctx().num_spins()
    P = PwCoeffs()
    for k in range(nk):
        kp = kpointset[k]
        gkvec = kp.gkvec()
        # assert (gkvec.num_gvec() == gkvec.count())
        N = gkvec.count()

        def ekin(i):
            return np.sum((np.array(gkvec.gkvec_cart(i))**2))

        def Tp(T):
            """Teter preconditioner."""
            return 16*T**4 / (27 + 18*T + 12*T**2 + 8*T**3)

        d = np.array([1 / (1 + Tp(ekin(i))) for i in range(N)])
        for ispn in range(nc):
            P[k, ispn] = dia_matrix((d, 0), shape=(N, N))
    return DiagonalPreconditioner(P)
