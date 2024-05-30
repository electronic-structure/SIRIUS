from .py_sirius import InverseS_k, S_k, Precond_us, Hamiltonian0
# from .coefficient_array import threaded
from scipy.sparse.linalg import LinearOperator
import numpy as np


class KpointOperatorBase:
    def __init__(self):
        """"""
        self.ops = {}

    def __getitem__(self, key):
        return NotImplemented


class S_operator(KpointOperatorBase):
    """
    Description: computes  S|ψ〉
    """

    def __init__(self, ctx, potential, kpointset):
        super().__init__()
        hamiltonian0 = Hamiltonian0(potential)
        for ik, kp in enumerate(kpointset):
            for ispn in range(ctx.num_spins()):
                self.ops[ik, ispn] = S_k(
                    ctx, hamiltonian0.Q(), kp.beta_projectors(), ispn
                )

    def apply(self, cn):
        """"""
        out = type(cn)()
        for key in cn.keys():
            out[key] = np.array(self.ops[key].apply(np.asfortranarray(cn[key])))
        return out

    def __getitem__(self, key):
        def _matvec(X):
            return np.array(self.ops[key].apply(np.asfortranarray(X)))
        n = self.ops[key].size
        return LinearOperator(dtype=np.complex128, shape=(n, n), matvec=_matvec, rmatvec=_matvec)

    def __matmul__(self, cn):
        return self.apply(cn)


class Sinv_operator(KpointOperatorBase):
    """Description: computes S⁻¹|ψ〉"""

    def __init__(self, ctx, potential, kpointset):
        super().__init__()
        hamiltonian0 = Hamiltonian0(potential)
        for ik, kp in enumerate(kpointset):
            for ispn in range(ctx.num_spins()):
                self.ops[ik, ispn] = InverseS_k(
                    ctx, hamiltonian0.Q(), kp.beta_projectors(), ispn
                )

    def apply(self, cn):
        out = type(cn)()
        for key in cn.keys():
            out[key] = np.array(self.ops[key].apply(np.asfortranarray(cn[key])))
        return out

    def __getitem__(self, key):
        def _matvec(X):
            return np.array(self.ops[key].apply(np.asfortranarray(X)))
        n = self.ops[key].size
        return LinearOperator(dtype=np.complex128, shape=(n, n), matvec=_matvec, rmatvec=_matvec)

    def __matmul__(self, cn):
        return self.apply(cn)


class US_Precond(KpointOperatorBase):
    """
    Description: ultrasoft preconditioner

    Hasnip, P. J., & Pickard, C. J. Electronic energy minimisation with
    ultrasoft pseudopotentials. , 174(1), 24–29.
    http://dx.doi.org/10.1016/j.cpc.2005.07.011
    """

    def __init__(self, ctx, potential, kpointset):
        super().__init__()
        hamiltonian0 = Hamiltonian0(potential)
        for ik, kp in enumerate(kpointset):
            for ispn in range(ctx.num_spins()):
                self.ops[ik, ispn] = Precond_us(
                    ctx, hamiltonian0.Q(), ispn, kp.beta_projectors(), kp.gkvec()
                )

    def apply(self, cn):
        out = type(cn)()
        for key in cn.keys():
            out[key] = np.array(self.ops[key].apply(np.asfortranarray(cn[key])))
        return out

    def __getitem__(self, key):
        def _matvec(X):
            return np.array(self.ops[key].apply(np.asfortranarray(X)))
        n = self.ops[key].size
        return LinearOperator(dtype=np.complex128, shape=(n, n), matvec=_matvec, rmatvec=_matvec)

    def __matmul__(self, cn):
        return self.apply(cn)
