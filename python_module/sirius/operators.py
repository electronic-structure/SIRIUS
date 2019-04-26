from .py_sirius import Hamiltonian, Wave_functions
from .coefficient_array import PwCoeffs
import numpy as np

class S_operator:
    """
    Description: compute S|Psi>
    """
    def __init__(self, hamiltonian, kpointset):
        """

        """
        self.hamiltonian = hamiltonian
        self.kpointset = kpointset

    def apply(self, cn):
        """
        """
        ctx = self.hamiltonian.ctx()
        num_sc = ctx.num_spins()
        self.hamiltonian._apply_ref_inner_prepare()
        assert isinstance(cn, PwCoeffs)
        out = PwCoeffs(dtype=cn.dtype)
        for k, ispn_coeffs in cn.by_k().items():
            kpoint = self.kpointset[k]
            # spins might have different number of bands ...
            num_wf = max(ispn_coeffs, key=lambda x: x[1].shape[1])[1].shape[1]
            Psi_x = Wave_functions(kpoint.gkvec_partition(), num_wf,
                                   ctx.preferred_memory_t(), num_sc)
            Psi_y = Wave_functions(kpoint.gkvec_partition(), num_wf,
                                   ctx.preferred_memory_t(), num_sc)
            for i, val in ispn_coeffs:
                Psi_x.pw_coeffs(i)[:, :val.shape[1]] = val
            self.hamiltonian._apply_overlap_inner(self.kpointset[k], Psi_y, Psi_x)
            for i, _ in ispn_coeffs:
                out[(k, i)] = np.array(
                    Psi_y.pw_coeffs(i), copy=False)[:, :num_wf]
        self.hamiltonian._apply_ref_inner_dismiss()
        return out

    def __matmul__(self, cn):
        return self.apply(cn)
