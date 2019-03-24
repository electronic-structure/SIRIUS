from ..py_sirius import ewald_energy, energy_bxc, Wave_functions, MemoryEnum
from ..coefficient_array import PwCoeffs
import numpy as np


def matview(x):
    return np.matrix(x, copy=False)


def pp_total_energy(potential, density, k_point_set, ctx):
    """
    Keyword Arguments:
    potential   --
    density     --
    k_point_set --
    ctx         --
    """
    gvec = ctx.gvec()
    unit_cell = ctx.unit_cell()
    # TODO: Ewald energy is constant...
    return (k_point_set.valence_eval_sum() - potential.energy_vxc(density) -
            potential.PAW_one_elec_energy() - 0.5 * potential.energy_vha() -
            energy_bxc(density, potential,
                       ctx.num_mag_dims()) + potential.energy_exc(density) +
            potential.PAW_total_energy() + ewald_energy(ctx, gvec, unit_cell))


class Energy:
    def __init__(self, kpointset, potential, density, hamiltonian, ctx=None):
        """
        Keyword Arguments:
        kpointset   --
        potential   --
        density     --
        hamiltonian -- object of type ApplyHamiltonian (c++ wrapper)
        ctx         --
        """
        self.kpointset = kpointset
        self.potential = potential
        self.density = density
        self.H = hamiltonian
        if ctx is None:
            self.ctx = kpointset.ctx()
        else:
            self.ctx = ctx

    def __call__(self, cn, ek=None):
        """
        Keyword Arguments:
        cn  --
        ek  -- band energies
        """

        if isinstance(cn, PwCoeffs):
            # update coefficients for all items in PwCoeffs
            for key, val in cn.items():
                k, ispn = key
                self.kpointset[k].spinor_wave_functions().pw_coeffs(ispn)[:, :val.shape[1]] = val
            # copy to device (if needed)
            for ki in cn.kvalues():
                psi = self.kpointset[ki].spinor_wave_functions()
                if psi.preferred_memory_t() == MemoryEnum.device:
                    psi.copy_to_gpu()
            # update density, potential
            self.density.generate(self.kpointset)
            if self.ctx.use_symmetry():
                self.density.symmetrize()
                self.density.symmetrize_density_matrix()

            self.density.generate_paw_loc_density()
            self.density.fft_transform(1)
            self.potential.generate(self.density)
            if self.ctx.use_symmetry():
                self.potential.symmetrize()

            self.potential.fft_transform(1)

            # update band energies
            if ek is None:
                yn = self.H(cn, scale=False)
                for key, val in yn.items():
                    k, ispn = key
                    benergies = np.zeros(self.ctx.num_bands(), dtype=np.complex)
                    benergies[:val.shape[1]] = np.einsum('ij,ij->j',
                                                        val,
                                                        np.conj(cn[key]))

                    for j, ek in enumerate(benergies):
                        self.kpointset[k].set_band_energy(j, ispn, np.real(ek))

                self.kpointset.sync_band_energies()
            else:
                self.kpointset.e = ek

            return pp_total_energy(self.potential, self.density,
                                   self.kpointset, self.ctx)
        else:
            assert False

class ApplyHamiltonian:
    def __init__(self, hamiltonian, kpointset):
        assert (not isinstance(hamiltonian, ApplyHamiltonian))
        self.hamiltonian = hamiltonian
        self.kpointset = kpointset

    def apply(self, cn, scale=True, ki=None, ispn=None):
        """
        Keyword Arguments:
        cn -- input coefficient array
        """
        from ..coefficient_array import PwCoeffs

        num_sc = self.hamiltonian.ctx().num_spins()
        self.hamiltonian._apply_ref_inner_prepare()
        if isinstance(cn, PwCoeffs):
            assert (ki is None)
            assert (ispn is None)
            out = PwCoeffs(dtype=cn.dtype)
            for k, ispn_coeffs in cn.by_k().items():
                kpoint = self.kpointset[k]
                # spins might have different number of bands ...
                num_wf = max(ispn_coeffs, key=lambda x: x[1].shape[1])[1].shape[1]
                Psi_x = Wave_functions(kpoint.gkvec_partition(), num_wf,
                                       MemoryEnum.host,
                                       num_sc)
                Psi_y = Wave_functions(kpoint.gkvec_partition(), num_wf,
                                       MemoryEnum.host,
                                       num_sc)
                for i, val in ispn_coeffs:
                    Psi_x.pw_coeffs(i)[:, :val.shape[1]] = val
                self.hamiltonian._apply_ref_inner(self.kpointset[k], Psi_y, Psi_x)

                w = kpoint.weight()
                # copy coefficients from Psi_y
                for i, _ in ispn_coeffs:
                    num_wf = cn[(k, i)].shape[1]
                    if scale:
                        bnd_occ = np.array(kpoint.band_occupancy(i))
                        out[(k, i)] = np.array(
                            Psi_y.pw_coeffs(i), copy=False)[:, :num_wf] * bnd_occ * w
                    else:
                        out[(k, i)] = np.array(
                            Psi_y.pw_coeffs(i), copy=False)[:, :num_wf]
                # end for
            self.hamiltonian._apply_ref_inner_dismiss()
            return out
        else:
            assert (ki in self.kpointset)
            kpoint = self.kpointset[ki]
            w = kpoint.weight()
            num_wf = cn.shape[1]
            Psi_y = Wave_functions(kpoint.gkvec_partition(), num_wf, MemoryEnum.host, num_sc)
            Psi_x = Wave_functions(kpoint.gkvec_partition(), num_wf, MemoryEnum.host, num_sc)
            bnd_occ = np.array(kpoint.band_occupancy(ispn))
            Psi_x.pw_coeffs(ispn)[:] = cn
            self.hamiltonian.apply_ref(kpoint, Psi_y, Psi_x)
            self.hamiltonian._apply_ref_inner_dismiss()
            if scale:
                return np.matrix(
                    np.array(Psi_y.pw_coeffs(ispn), copy=False) * bnd_occ * w,
                    copy=True)
            else:
                return np.matrix(
                    np.array(Psi_y.pw_coeffs(ispn), copy=False),
                    copy=True)

    def __matmul__(self, cn):
        """

        """
        from ..coefficient_array import PwCoeffs

        if not isinstance(cn, PwCoeffs):
            raise TypeError(
                'argument to ApplyHamiltonian.__matmul__ must be of type PwCoeffs'
            )
        return self.apply(cn)

    def __call__(self, cn, scale=True, ki=None, ispn=None):
        return self.apply(cn=cn, scale=scale, ki=ki, ispn=ispn)
