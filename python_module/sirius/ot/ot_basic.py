import typing
from ..coefficient_array import CoefficientArray
from ..py_sirius import (
    ewald_energy,
    energy_bxc,
    Wave_functions,
    MemoryEnum,
    CopyEnum,
    Hamiltonian0,
    total_energy
)
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
    return (
        k_point_set.valence_eval_sum()
        - potential.energy_vxc(density)
        - potential.PAW_one_elec_energy(density)
        - 0.5 * potential.energy_vha()
        - energy_bxc(density, potential)
        + potential.energy_exc(density)
        + potential.PAW_total_energy()
        + ewald_energy(ctx, gvec, unit_cell)
    )


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

    def compute(self, X: PwCoeffs, fn: typing.Optional[CoefficientArray] = None):
        """
        Keyword Arguments:
        X  -- PW coefficients

        Returns
        Etot -- total energy
        HX   -- Hamiltonian@X
        """

        if fn:
            self.kpointset.fn = fn

        for key, val in X.items():
            k, ispn = key
            psi = self.kpointset[k].spinor_wave_functions()
            psi.pw_coeffs(ispn)[:, : val.shape[1]] = val

        # update density, potential
        self.density.generate(
            self.kpointset, symmetrize=self.ctx.use_symmetry(), transform_to_rg=True
        )
        self.potential.generate(
            self.density, use_sym=self.ctx.use_symmetry(), transform_to_rg=True
        )

        yn = self.H(X, scale=False)

        for key, val in yn.items():
            k, ispn = key
            benergies = np.zeros(self.ctx.num_bands(), dtype=np.complex128)
            benergies[: val.shape[1]] = np.einsum("ij,ij->j", val, np.conj(X[key]))

            for j, ek in enumerate(benergies):
                self.kpointset[k].set_band_energy(j, ispn, np.real(ek))

        self.kpointset.sync_band_energy()

        Etot = total_energy(self.ctx, self.kpointset, self.density, self.potential,
                            ewald_energy(self.ctx, self.ctx.gvec(), self.ctx.unit_cell()))


        return Etot, yn

    def __call__(self, cn: PwCoeffs, fn: typing.Optional[CoefficientArray] = None):

        # update occupation numbers if needed
        if fn:
            self.kpointset.fn = fn

        E, _ = self.compute(cn)
        return E


class ApplyHamiltonian:
    def __init__(self, potential, kpointset):
        assert not isinstance(potential, ApplyHamiltonian)
        self.potential = potential
        self.kpointset = kpointset

    def apply(self, cn, scale=True, ki=None, ispn=None):
        """
        Keyword Arguments:
        cn -- input coefficient array
        """
        from ..coefficient_array import PwCoeffs
        from ..py_sirius import apply_hamiltonian, num_mag_dims, num_bands

        ctx = self.kpointset.ctx()
        pmem_t = ctx.processing_unit_memory_t()
        num_sc = ctx.num_spins()
        H0 = Hamiltonian0(self.potential, False)
        if isinstance(cn, PwCoeffs):
            assert ki is None
            assert ispn is None
            out = PwCoeffs(dtype=cn.dtype)
            for k, ispn_coeffs in cn.by_k().items():
                kpoint = self.kpointset[k]
                # spins might have different number of bands ...
                # num_wf = max(ispn_coeffs, key=lambda x: x[1].shape[1])[1].shape[1]
                md = num_mag_dims(ctx.num_mag_dims())
                nb = num_bands(ctx.num_bands())
                Psi_x = Wave_functions(
                    kpoint.gkvec(), md, nb, MemoryEnum.host
                )
                Psi_y = Wave_functions(
                    kpoint.gkvec(), md, nb, MemoryEnum.host
                )
                for i, val in ispn_coeffs:
                    Psi_x.pw_coeffs(i)[:, : val.shape[1]] = val

                apply_hamiltonian(H0, kpoint, Psi_y, Psi_x)

                w = kpoint.weight()
                # copy coefficients from Psi_y
                for i, _ in ispn_coeffs:
                    num_wf = cn[(k, i)].shape[1]
                    if scale:
                        bnd_occ = np.array(kpoint.band_occupancy(i))
                        out[(k, i)] = (
                            np.array(Psi_y.pw_coeffs(i), copy=False)[:, :num_wf]
                            * bnd_occ
                            * w
                        )
                    else:
                        out[(k, i)] = np.array(Psi_y.pw_coeffs(i), copy=False)[
                            :, :num_wf
                        ]
            return out

    def __matmul__(self, cn):
        """ """
        from ..coefficient_array import PwCoeffs

        if not isinstance(cn, PwCoeffs):
            raise TypeError(
                "argument to ApplyHamiltonian.__matmul__ must be of type PwCoeffs"
            )
        return self.apply(cn)

    def __call__(self, cn, scale=True, ki=None, ispn=None):
        return self.apply(cn=cn, scale=scale, ki=ki, ispn=ispn)
