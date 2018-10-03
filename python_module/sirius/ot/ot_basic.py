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

    def __call__(self, cn, ki=None, ispn=0):
        """
        Keyword Arguments:
        cn  --
        ki -- the index of the k-point
        """
        from ..coefficient_array import PwCoeffs

        if isinstance(cn, PwCoeffs):
            ispn = None
            assert (ki is None)
            # update coefficients for all items in PwCoeffs
            for key, val in cn.items():
                k, ispn = key
                self.kpointset[k].spinor_wave_functions().pw_coeffs(ispn)[:] = val
                assert(np.isclose(matview(val).H*val, np.eye(val.shape[1])).all())
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
            yn = self.H(cn)
            for key, val in yn.items():
                k, ispn = key
                w = self.kpointset[k].weight()
                bnd_occ = np.array(self.kpointset[k].band_occupancy(ispn))
                if np.isclose(bnd_occ, 0).all():
                    # print('WARNING: encountered unoccupied band')
                    # TODO: handle this properly
                    continue
                # elif np.isclose(bnd_occ, 0).any():
                #     raise Exception("encountered zero band occupation")
                # scale columns by 1/bnd_occ
                benergies = np.einsum('ij,ij,j->j', val, np.conj(cn[key]),
                                      1 / (bnd_occ * w))
                for j, ek in enumerate(benergies):
                    if bnd_occ[j] > 1e-10:
                        assert (np.abs(np.imag(ek)) < 1e-10)
                        self.kpointset[k].set_band_energy(j, ispn, np.real(ek))
                    else:
                        self.kpointset[k].set_band_energy(j, ispn, 0)
                    self.kpointset.sync_band_energies()
            return pp_total_energy(self.potential, self.density,
                                   self.kpointset, self.ctx)
        else:
            k = self.kpointset[ki]
            k.spinor_wave_functions().pw_coeffs(ispn)[:] = cn
            psi = self.kpointset[k].spinor_wave_functions()
            if psi.preferred_memory_t() == MemoryEnum.device:
                psi.copy_to_gpu()
            # update density and potential at point
            self.density.generate(self.kpointset)
            self.density.generate_paw_loc_density()
            self.density.fft_transform(1)
            self.potential.generate(self.density)
            self.potential.fft_transform(1)
            # after updating H to the new position, we can compute new band energies
            bnd_occ = k.band_occupancy(ispn)
            assert (np.isclose(bnd_occ, (bnd_occ + 1e-4).astype(int)).all())
            w = k.weight()
            yn = self.H(cn, ki=ki, ispn=ispn)
            # Hc is scaled by band occupancies, need to divide here to get correct band energies
            yn = np.matrix(np.array(yn) / bnd_occ / w)
            HH = yn.H * cn
            ek = np.diag(HH)
            for i, ek in enumerate(ek):
                k.set_band_energy(i, ispn, ek)
            self.kpointset.sync_band_energies()
            return pp_total_energy(self.potential, self.density,
                                   self.kpointset, self.ctx)


class ApplyHamiltonian:
    def __init__(self, hamiltonian, kpointset):
        assert (not isinstance(hamiltonian, ApplyHamiltonian))
        self.hamiltonian = hamiltonian
        self.kpointset = kpointset

    def apply(self, cn, ki=None, ispn=None):
        """
        Keyword Arguments:
        cn -- input coefficient array
        """
        from ..coefficient_array import PwCoeffs

        num_sc = self.hamiltonian.ctx().num_spins()
        if isinstance(cn, PwCoeffs):
            assert (ki is None)
            assert (ispn is None)
            out = PwCoeffs(dtype=cn.dtype)
            cn._data.keys()
            for k, ispn_coeffs in cn.by_k().items():
                num_wf = ispn_coeffs[0][1].shape[1]
                kpoint = self.kpointset[k]
                w = kpoint.weight()
                Psi_x = Wave_functions(kpoint.gkvec_partition(), num_wf,
                                       num_sc)
                Psi_y = Wave_functions(kpoint.gkvec_partition(), num_wf,
                                       num_sc)
                for i, val in ispn_coeffs:
                    Psi_x.pw_coeffs(i)[:] = val
                self.hamiltonian.apply_ref(self.kpointset[k], Psi_y, Psi_x)
                # copy coefficients from Psi_y
                for i, _ in ispn_coeffs:
                    bnd_occ = np.array(kpoint.band_occupancy(i))
                    assert (np.isclose(bnd_occ,
                                       (bnd_occ + 1e-4).astype(int)).all())
                    out[(k, i)] = np.array(
                        Psi_y.pw_coeffs(i), copy=False) * bnd_occ * w
            # end for
            return out
        else:
            assert (ki in self.kpointset)
            kpoint = self.kpointset[ki]
            w = kpoint.weight()
            num_wf = cn.shape[1]
            Psi_y = Wave_functions(kpoint.gkvec_partition(), num_wf, num_sc)
            Psi_x = Wave_functions(kpoint.gkvec_partition(), num_wf, num_sc)
            bnd_occ = np.array(kpoint.band_occupancy(ispn))
            Psi_x.pw_coeffs(ispn)[:] = cn
            self.hamiltonian.apply_ref(kpoint, Psi_y, Psi_x)
            return np.matrix(
                np.array(Psi_y.pw_coeffs(ispn), copy=False) * bnd_occ * w,
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

    def __call__(self, cn, ki=None, ispn=None):
        return self.apply(cn=cn, ki=ki, ispn=ispn)
