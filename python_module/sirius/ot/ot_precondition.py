from sirius.coefficient_array.coefficient_array import CoefficientArray
from ..coefficient_array import PwCoeffs
from scipy.sparse import dia_matrix
import numpy as np


def make_kinetic_precond(kpointset, c0, eps=0.1, asPwCoeffs=True):
    """
    Preconditioner
    P = 1 / (||k|| + ε)

    Keyword Arguments:
    kpointset --
    """

    nk = len(kpointset)
    nc = kpointset.ctx().num_spins()
    if nc == 1 and nk == 1 and not asPwCoeffs:
        # return as np.matrix
        kp = kpointset[0]
        gkvec = kp.gkvec()
        # assert (gkvec.num_gvec() == gkvec.count())
        N = gkvec.count()
        d = np.array([
            1 / (np.sum((np.array(gkvec.gkvec(i)))**2) + eps)
            for i in range(N)
        ])
        return DiagonalPreconditioner(
            D=dia_matrix((d, 0), shape=(N, N)), c0=c0)
    else:
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
        return DiagonalPreconditioner(P, c0)


class Preconditioner:
    def __init__(self):
        pass


class DiagonalPreconditioner(Preconditioner):
    """
    Apply diagonal preconditioner and project resulting gradient to satisfy the constraint.
    """

    def __init__(self, D, c0):
        super().__init__()
        self.c0 = c0
        self.D = D

    def __matmul__(self, other: PwCoeffs):
        """
        """
        from .ot_transformations import lagrangeMult

        assert(isinstance(other, PwCoeffs))

        out = PwCoeffs()
        for key, Dl in self.D.items():
            out[key] = Dl * other[key]
        ll = lagrangeMult(other, self.c0, self)
        return out + ll

    def __mul__(self, s):
        """

        """
        from ..coefficient_array import CoefficientArray
        import numpy as np

        if np.isscalar(s):
            for key, Dl in self.D.items():
                self.D[key] = s*Dl
        elif isinstance(s, CoefficientArray):
            out = PwCoeffs()
            for key in s.keys():
                out[key] = self.D[key] * s[key]
            return out

    __lmul__ = __mul__
    __rmul__ = __mul__

    def __neg__(self):
        """
        """
        from ..coefficient_array import CoefficientArray
        if isinstance(self.D, CoefficientArray):
            out_data = PwCoeffs()
            out = DiagonalPreconditioner(out_data, self.c0)
            for k, v in self.D.items():
                out.D[k] = -v
            return out
        else:
            out = DiagonalPreconditioner(self.D, self.c0)
            out.D = -self.D
            return out

    def __getitem__(self, key):
        return self.D[key]


class IdentityPreconditioner(Preconditioner):

    def __init__(self, c0, _f=1):
        super().__init__()
        self.c0 = c0
        self._f = _f

    def __matmul__(self, other):
        from .ot_transformations import lagrangeMult

        ll = lagrangeMult(other, self.c0, self)
        return self._f * other + ll

    def __mul__(self, s):
        return self._f * s

    def __neg__(self):
        return IdentityPreconditioner(self.c0, _f=-self._f)

    def __getitem__(self, _):
        return self._f

    __lmul__ = __mul__
    __rmul__ = __mul__
