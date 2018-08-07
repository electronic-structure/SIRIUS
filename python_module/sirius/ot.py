from .py_sirius import ewald_energy, Wave_functions, DeviceEnum
from .ot_transformations import ConstrainedGradient, c
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
            potential.PAW_one_elec_energy() - 0.5 * potential.energy_vha() +
            potential.energy_exc(density) + potential.PAW_total_energy() +
            ewald_energy(ctx, gvec, unit_cell))


class Energy:
    def __init__(self, kpointset, potential, density, hamiltonian, ctx):
        """
        Keyword Arguments:
        kpointset --
        potential   --
        density     --
        hamiltonian -- object of type ApplyHamiltonian (c++ wrapper)
        ctx         --
        """
        self.kpointset = kpointset
        self.potential = potential
        self.density = density
        self.H = hamiltonian
        self.ctx = ctx

    def __call__(self, cn, ki, ispn=0):
        """
        Keyword Arguments:
        cn  --
        ki -- the index of the k-point
        """
        k = self.kpointset[ki]
        k.spinor_wave_functions().pw_coeffs(ispn)[:] = cn
        if self.ctx.processing_unit() == DeviceEnum.GPU:
            k.spinor_wave_functions().copy_to_gpu()

        # update density and potential at point
        self.density.generate(self.kpointset)
        self.density.generate_paw_loc_density()
        self.density.fft_transform(1)

        self.potential.generate(self.density)
        self.potential.fft_transform(1)

        # after updating H to the new position, we can compute new band energies
        bnd_occ = k.band_occupancy(ispn)
        yn = self.H * cn
        # Hc is scaled by band occupancies, need to divide here to get correct band energies
        yn = np.matrix(np.array(yn) / bnd_occ)
        ek = np.diag(yn.H * cn)
        for i, ek in enumerate(ek):
            k.set_band_energy(i, ispn, ek)

        return pp_total_energy(self.potential, self.density, self.kpointset,
                               self.ctx)


class ApplyHamiltonian:
    def __init__(self, hamiltonian, kpoint):
        self.hamiltonian = hamiltonian
        self.kpoint = kpoint
        spinors = kpoint.spinor_wave_functions()
        self.num_wf = spinors.num_wf()
        assert (spinors.num_sc() == 1)
        self.num_sc = 1

    def apply(self, cn, ispn=0):
        """
        Keyword Arguments:
        cn -- input coefficient array
        """
        assert (cn.shape[1] == self.num_wf)
        # since assert(num_sc==1)
        device_t = self.hamiltonian.ctx().processing_unit()
        Psi_y = Wave_functions(self.kpoint.gkvec_partition(), self.num_wf,
                               self.num_sc)
        # TODO: is it necessary to set Psi_y = 0?
        Psi_y.zero_pw(device_t, ispn, 0, self.num_wf)
        Psi_x = Wave_functions(self.kpoint.gkvec_partition(), self.num_wf,
                               self.num_sc)
        bnd_occ = np.array(self.kpoint.band_occupancy(ispn))
        Psi_x.pw_coeffs(ispn)[:] = cn
        # apply Hamiltonian
        # TODO: note, applying to all ispn...
        # hamiltonian.apply* copies to GPU
        self.hamiltonian.apply_ref(self.kpoint, Psi_y, Psi_x)
        return np.matrix(
            np.array(Psi_y.pw_coeffs(ispn), copy=False) * bnd_occ, copy=True)

    def __mul__(self, x):
        return self.apply(x)

    def __call__(self, x):
        return self.apply(x)
