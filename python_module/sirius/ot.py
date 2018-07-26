from .py_sirius import ewald_energy, Wave_functions
import numpy as np
from numpy.linalg import eigh, solve
from .ot_transformations import EnergyGradient, c


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

    return (k_point_set.valence_eval_sum() - potential.energy_vxc(density) -
            potential.PAW_one_elec_energy() - 0.5 * potential.energy_vha() +
            potential.energy_exc(density) + potential.PAW_total_energy() +
            ewald_energy(ctx, gvec, unit_cell))


class Energy:
    def __init__(self, kpointset, potential, density, ctx):
        """
        Keyword Arguments:
        kpointset --
        potential   --
        density     --
        hamiltonian -- object of type Hamiltonian (c++ wrapper)
        ctx         --
        """
        self.kpointset = kpointset
        self.potential = potential
        self.density = density
        self.ctx = ctx

    def __call__(self, cn, ki):
        """
        Keyword Arguments:
        cn  --
        ki -- the index of the k-point
        """
        ispn = 0
        k = self.kpointset[ki]
        k.spinor_wave_functions().pw_coeffs(ispn)[:] = cn

        # determine band energies ...
        self.density.generate(self.kpointset)
        self.density.fft_transform(1)

        self.potential.generate(self.density)
        self.potential.fft_transform(1)

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
        # input wave function
        self.Psi_x = Wave_functions(kpoint.gkvec_partition(), self.num_wf,
                                    self.num_sc)

    def apply(self, c):
        """
        Keyword Arguments:
        x -- input coefficient array
        """
        assert (c.shape[1] == self.num_wf)
        # since assert(num_sc==1)
        ispn = 0

        # self.Psi_x.pw_coeffs(0)[:] = pw_coeffs_in
        # if self.hamiltonian.on_gpu():
        #     self.Psi_x.copy_to_gpu()
        # # apply Hamiltonian
        # print('before apply Hamiltonian')
        # Psi_y = self.hamiltonian.apply(self.kpoint, ispn, self.Psi_x)

        self.Psi_x = self.kpoint.spinor_wave_functions()
        self.Psi_x.pw_coeffs(0)[:] = c
        if self.hamiltonian.on_gpu():
            self.Psi_x.copy_to_gpu()
        # apply Hamiltonian
        Psi_y = self.hamiltonian.apply(self.kpoint, ispn, self.Psi_x)

        return np.matrix(Psi_y.pw_coeffs(0))

    def __mul__(self, x):
        return self.apply(x)

    def __call__(self, x):
        return self.apply(x)
