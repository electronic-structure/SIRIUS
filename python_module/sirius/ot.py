from .py_sirius import ewald_energy, Wave_functions
import numpy as np
from numpy.linalg import eigh, solve


def matview(x):
    return np.matrix(x, copy=False)


def c(x, c0):
    """
    TODO: optimize this

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

    return (k_point_set.valence_eval_sum()
            - potential.energy_vxc(density)
            - potential.PAW_one_elec_energy()
            - 0.5 * potential.energy_vha()
            + potential.energy_exc(density)
            + potential.PAW_total_energy()
            + ewald_energy(ctx, gvec, unit_cell))


class Energy:
    def __init__(self, k_point_set, potential, density, ctx):
        """
        Keyword Arguments:
        k_point_set --
        potential   --
        density     --
        ctx         --
        """

        self.k_point_set = k_point_set
        self.potential = potential
        self.density = density
        self.ctx = ctx

    def __call__(self, x, ki, c0):
        """
        Keyword Arguments:
        x  --
        ki --
        """
        cn = c(x, c0)
        ki.spinor_wave_functions().pw_coeffs(0)[:] = cn

        self.density.generate(self.k_point_set)
        self.density.fft_transform(1)

        self.potential.generate(self.density)
        self.potential.fft_transform(1)

        return pp_total_energy(self.potential, self.density, self.k_point_set,
                               self.ctx)


class ApplyHamiltonian:
    def __init__(self, hamiltonian, kpoint):
        self.hamiltonian = hamiltonian
        self.kpoint = kpoint
        spinors = kpoint.spinor_wave_functions()
        self.num_wf = spinors.num_wf()
        assert(spinors.num_sc() == 1)
        self.num_sc = 1
        # input wave function
        self.Psi_x = Wave_functions(kpoint.gkvec_partition(),
                                    self.num_wf,
                                    self.num_sc)

    def apply(self, c):
        """
        Keyword Arguments:
        x -- input coefficient array
        """
        assert(c.shape[1] == self.num_wf)
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


class EnergyGradient:
    def __init__(self, hamiltonian, c0):
        self.hamiltonian = hamiltonian
        self.c0 = np.matrix(c0)

    def __call__(self, x):
        """
        it takes a wave-function psi, which is stored inside the K_point class
        """
        # make sure x has type np.matrix
        x = np.matrix(x)

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
        # TODO: mask diagonal elements (get rid of warning), they are computed later
        diffL = (Λ[:, np.newaxis] - Λ[:, np.newaxis].T)
        D1 = (v - v.T) / diffL
        np.fill_diagonal(D1, 0.5*(np.cos(np.sqrt(Λ)) / Λ - np.sin(np.sqrt(Λ)) / (Λ**(1.5))))
        # D²: TODO insert formula
        v = np.cos(np.sqrt(Λ))
        v = v[:, np.newaxis]
        D2 = (v - v.T) / diffL
        np.fill_diagonal(D2, -0.5*np.sin(np.sqrt(Λ)) / np.sqrt(Λ))

        # compute K: TODO copy/paste formula from the paper
        RtHCtxR = np.array(R.H * (Hc.H * x) * R)
        RtHCtc0R = np.array(R.H * Hc.H * self.c0 * R)
        K = np.matrix(RtHCtxR*np.array(D1) + RtHCtc0R*np.array(D2))
        # compute ∂E/∂x
        dEdx = Hc*invU*sinU + x*(R*(K.H + K)*R.H)

        # Lagrange multiplier
        lagrangeMult = solve(self.c0.H * self.c0, self.c0.H * dEdx)
        correction_term = -1 * self.c0 * lagrangeMult

        return dEdx + correction_term
