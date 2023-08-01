from ..py_sirius import apply_U_operator, band_range, Hamiltonian0, num_mag_dims, num_bands, Wave_functions, MemoryEnum
from ..coefficient_array import PwCoeffs, zeros_like
from ..constants import spin_up, spin_dn
import numpy as np


class HubbardU:
    def __init__(self, kset, potential):
        self.kset = kset
        self.potential = potential

    def __matmul__(self, X: PwCoeffs):
        """ """
        ctx = self.kset.ctx()

        H0 = Hamiltonian0(self.potential, False)

        if not ctx.hubbard_correction:
            return zeros_like(X)

        out = PwCoeffs()
        for k, ispn_coeffs in X.by_k().items():
            kpoint = self.kset[k]
            H_k = H0.Hk(kpoint)
            # spins might have different number of bands ...
            # num_wf = max(ispn_coeffs, key=lambda x: x[1].shape[1])[1].shape[1]
            md = num_mag_dims(ctx.num_mag_dims())
            Psi_x = Wave_functions(kpoint.gkvec(), md, num_bands(ctx.num_bands()), MemoryEnum.host)
            Psi_y = Wave_functions(kpoint.gkvec(), md, num_bands(ctx.num_bands()), MemoryEnum.host)
            Psi_y.zero()

            u_op = H_k.U

            spins = [spin_up, spin_dn]
            for i, val in ispn_coeffs:
                nb = X[(k, i)].shape[1]
                Psi_x.pw_coeffs(i)[:, : val.shape[1]] = val

                apply_U_operator(
                    ctx,
                    spin_range=spins[i],
                    band_range=band_range(0, nb),
                    hub_wf=kpoint.hubbard_wave_functions_S(),
                    phi=Psi_x,
                    u_op=u_op,
                    hphi=Psi_y,
                )

                # copy result to python PwCoeffs
                out[(k, i)] = np.array(Psi_y.pw_coeffs(i), copy=False)[:, :nb]

        return out
