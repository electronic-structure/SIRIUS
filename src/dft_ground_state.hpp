// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file dft_ground_state.hpp
 *
 *  \brief Contains definition and partial implementation of sirius::DFT_ground_state class.
 */

#ifndef __DFT_GROUND_STATE_HPP__
#define __DFT_GROUND_STATE_HPP__

#include "K_point/k_point_set.hpp"
#include "utils/json.hpp"
#include "Hubbard/hubbard.hpp"
#include "Geometry/stress.hpp"
#include "Geometry/force.hpp"
#include "Band/band.hpp"

using json = nlohmann::json;

namespace sirius {

/// The whole DFT ground state implementation.
/** The DFT cycle consists of four basic steps: solution of the Kohn-Sham equations, summation of the occupied states
 *  in order to get a system's charge density and magnetization, mixing and finally gneneration of the effective
 *  potential and effective magnetic field. \image html dft_cycle.png "DFT self-consistency cycle"
 */
class DFT_ground_state
{
  private:
    /// Context of simulation.
    Simulation_context& ctx_;

    /// Set of k-points that are used to generate density.
    K_point_set& kset_;

    /// Alias of the unit cell.
    Unit_cell& unit_cell_;

    /// Instance of the Potential class.
    Potential potential_;

    /// Instance of the Density class.
    Density density_;

    /// Kohn-Sham Hamiltoninan.
    //Hamiltonian hamiltonian_;

    /// Lattice stress.
    Stress stress_;

    /// Atomic forces.
    Force forces_;

    /// Store Ewald energy which is computed once and which doesn't change during the run.
    double ewald_energy_{0};

    /// Compute the ion-ion electrostatic energy using Ewald method.
    /** The following contribution (per unit cell) to the total energy has to be computed:
     *  \f[
     *    E^{ion-ion} = \frac{1}{N} \frac{1}{2} \sum_{i \neq j} \frac{Z_i Z_j}{|{\bf r}_i - {\bf r}_j|} =
     *      \frac{1}{2} \sideset{}{'} \sum_{\alpha \beta {\bf T}} \frac{Z_{\alpha} Z_{\beta}}{|{\bf r}_{\alpha} -
     *      {\bf r}_{\beta} + {\bf T}|}
     *  \f]
     *  where \f$ N \f$ is the number of unit cells in the crystal.
     *  Following the idea of Ewald the Coulomb interaction is split into two terms:
     *  \f[
     *     \frac{1}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|} =
     *       \frac{{\rm erf}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} -
     *       {\bf r}_{\beta} + {\bf T}|} +
     *       \frac{{\rm erfc}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} -
     *       {\bf r}_{\beta} + {\bf T}|}
     *  \f]
     *  Second term is computed directly. First term is computed in the reciprocal space. Remembering that
     *  \f[
     *    \frac{1}{\Omega} \sum_{\bf G} e^{i{\bf Gr}} = \sum_{\bf T} \delta({\bf r - T})
     *  \f]
     *  we rewrite the first term as
     *  \f[
     *    \frac{1}{2} \sideset{}{'} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta}
     *      \frac{{\rm erf}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} -
     *        {\bf r}_{\beta} + {\bf T}|} = \frac{1}{2} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta}
     *       \frac{{\rm erf}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}
     *            {|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|} -
     *      \frac{1}{2} \sum_{\alpha} Z_{\alpha}^2 2 \sqrt{\frac{\lambda}{\pi}} = \\
     *    \frac{1}{2} \sum_{\alpha \beta} Z_{\alpha} Z_{\beta} \frac{1}{\Omega} \sum_{\bf G} \int e^{i{\bf Gr}}
     *    \frac{{\rm erf}(\sqrt{\lambda} |{\bf r}_{\alpha\beta} + {\bf r}|)}{|{\bf r}_{\alpha\beta} + {\bf r}|}
     *    d{\bf r} - \sum_{\alpha} Z_{\alpha}^2  \sqrt{\frac{\lambda}{\pi}}
     *  \f]
     *  The integral is computed using the \f$ \ell=0 \f$ term of the spherical expansion of the plane-wave:
     *  \f[
     *    \int e^{i{\bf Gr}} \frac{{\rm erf}(\sqrt{\lambda} |{\bf r}_{\alpha\beta} + {\bf r}|)}{|{\bf r}_{\alpha\beta} +
     *      {\bf r}|} d{\bf r} =
     *      \int e^{-i{\bf r}_{\alpha \beta}{\bf G}} e^{i{\bf Gr}} \frac{{\rm erf}(\sqrt{\lambda} |{\bf r}|)}{|{\bf r}|}
     *     d{\bf r} = e^{-i{\bf r}_{\alpha \beta}{\bf G}} 4 \pi \int_0^{\infty} \frac{\sin({G r})}{G}
     *    {\rm erf}(\sqrt{\lambda} r ) dr
     *  \f]
     *  We will split integral in two parts:
     *  \f[
     *    \int_0^{\infty} \sin({G r}) {\rm erf}(\sqrt{\lambda} r ) dr = \int_0^{b} \sin({G r})
     *    {\rm erf}(\sqrt{\lambda} r ) dr +
     *      \int_b^{\infty} \sin({G r}) dr = \frac{1}{G} e^{-\frac{G^2}{4 \lambda}}
     *  \f]
     *  where \f$ b \f$ is sufficiently large. To reproduce in Mathrmatica:
        \verbatim
        Limit[Limit[
          Integrate[Sin[g*x]*Erf[Sqrt[nu] * x], {x, 0, b},
              Assumptions -> {nu > 0, g >= 0, b > 0}] +
                 Integrate[Sin[g*(x + I*a)], {x, b, \[Infinity]},
                     Assumptions -> {a > 0, nu > 0, g >= 0, b > 0}], a -> 0],
                      b -> \[Infinity], Assumptions -> {nu > 0, g >= 0}]
        \endverbatim
     *  The first term of the Ewald sum thus becomes:
     *  \f[
     *    \frac{2 \pi}{\Omega} \sum_{{\bf G}} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \Big| \sum_{\alpha} Z_{\alpha}
     *      e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 - \sum_{\alpha} Z_{\alpha}^2 \sqrt{\frac{\lambda}{\pi}}
     *  \f]
     *  For \f$ G=0 \f$ the following is done:
     *  \f[
     *    \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \approx \frac{1}{G^2}-\frac{1}{4 \lambda }
     *  \f]
     *  The term \f$ \frac{1}{G^2} \f$ is compensated together with the corresponding Hartree terms in electron-electron
     *  and electron-ion interactions (cell should be neutral) and we are left with the following conribution:
     *  \f[
     *    -\frac{2\pi}{\Omega}\frac{N_{el}^2}{4 \lambda}
     *  \f]
     *  Final expression for the Ewald energy:
     *  \f[
     *    E^{ion-ion} = \frac{1}{2} \sideset{}{'} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta}
     *      \frac{{\rm erfc}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} -
     *      {\bf r}_{\beta} + {\bf T}|} + \frac{2 \pi}{\Omega} \sum_{{\bf G}\neq 0}
     *      \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \Big| \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}}
     *      \Big|^2 - \sum_{\alpha} Z_{\alpha}^2 \sqrt{\frac{\lambda}{\pi}} - \frac{2\pi}{\Omega}
     *      \frac{N_{el}^2}{4 \lambda}
     *  \f]
     */
    double ewald_energy() const;

  public:
    /// Constructor.
    DFT_ground_state(K_point_set& kset__)
        : ctx_(kset__.ctx())
        , kset_(kset__)
        , unit_cell_(ctx_.unit_cell())
        , potential_(ctx_)
        , density_(ctx_)
        , stress_(ctx_, density_, potential_, kset__)
        , forces_(ctx_, density_, potential_, kset__)

    {
        if (!ctx_.full_potential()) {
            ewald_energy_ = ewald_energy();
        }
    }

    /// Return reference to a simulation context.
    inline Simulation_context const& ctx() const
    {
        return ctx_;
    }

    inline Density& density()
    {
        return density_;
    }

    inline Potential& potential()
    {
        return potential_;
    }

    inline K_point_set& k_point_set()
    {
        return kset_;
    }

    inline Force& forces()
    {
        return forces_;
    }

    inline Stress& stress()
    {
        return stress_;
    }

    /// Generate initial densty, potential and a subspace of wave-functions.
    void initial_state();

    /// Update the parameters after the change of lattice vectors or atomic positions.
    void update();

    /// Run the SCF ground state calculation and find a total energy minimum.
    json find(double potential_tol, double energy_tol, double initial_tolerance, int num_dft_iter, bool write_state);

    /// Print the basic information (total energy, charges, moments, etc.).
    void print_info();

    /// Print an estimation of magnetic moments in case of pseudopotential.
    //void print_magnetic_moment() const;

    /// Return nucleus energy in the electrostatic field.
    /** Compute energy of nucleus in the electrostatic potential generated by the total (electrons + nuclei)
     *  charge density. Diverging self-interaction term z*z/|r=0| is excluded. */
    double energy_enuc() const;

    /// Return eigen-value sum of core states.
    double core_eval_sum() const;

    double energy_vha() const
    {
        return potential_.energy_vha();
    }

    double energy_vxc() const
    {
        return potential_.energy_vxc(density_);
    }

    double energy_exc() const
    {
        return potential_.energy_exc(density_);
    }

    double energy_bxc() const
    {
        double ebxc{0};
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            ebxc += sirius::inner(density_.magnetization(j), potential_.effective_magnetic_field(j));
        }
        return ebxc;
    }

    double energy_veff() const
    {
        return sirius::inner(density_.rho(), potential_.effective_potential());
    }

    double energy_vloc() const
    {
        return sirius::inner(potential_.local_potential(), density_.rho());
    }

    /// Full eigen-value sum (core + valence)
    double eval_sum() const
    {
        return (core_eval_sum() + kset_.valence_eval_sum());
    }

    /// Kinetic energy
    /** more doc here
     */
    double energy_kin() const
    {
        return (eval_sum() - energy_veff() - energy_bxc());
    }

    double energy_kin_sum_pw() const;

    inline double energy_ewald() const
    {
        return ewald_energy_;
    }

    /// Total energy of the electronic subsystem.
    /** <b> Full potential total energy </b>
     *
     *  From the definition of the density functional we have:
     *  \f[
     *      E[\rho] = T[\rho] + E^{H}[\rho] + E^{XC}[\rho] + E^{ext}[\rho]
     *  \f]
     *  where \f$ T[\rho] \f$ is the kinetic energy, \f$ E^{H}[\rho] \f$ - electrostatic energy of
     *  electron-electron density interaction, \f$ E^{XC}[\rho] \f$ - exchange-correlation energy
     *  and \f$ E^{ext}[\rho] \f$ - energy in the external field of nuclei.
     *
     *  Electrostatic and external field energies are grouped in the following way:
     *  \f[
     *      \frac{1}{2} \int \int \frac{\rho({\bf r})\rho({\bf r'}) d{\bf r} d{\bf r'}}{|{\bf r} - {\bf r'}|} +
     *          \int \rho({\bf r}) V^{nuc}({\bf r}) d{\bf r} = \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} +
     *          \frac{1}{2} \int \rho({\bf r}) V^{nuc}({\bf r}) d{\bf r}
     *  \f]
     *  Here \f$ V^{H}({\bf r}) \f$ is the total (electron + nuclei) electrostatic potential returned by the
     *  poisson solver. Next we transform the remaining term:
     *  \f[
     *      \frac{1}{2} \int \rho({\bf r}) V^{nuc}({\bf r}) d{\bf r} =
     *      \frac{1}{2} \int \int \frac{\rho({\bf r})\rho^{nuc}({\bf r'}) d{\bf r} d{\bf r'}}{|{\bf r} - {\bf r'}|} =
     *      \frac{1}{2} \int V^{H,el}({\bf r}) \rho^{nuc}({\bf r}) d{\bf r}
     *  \f]
     *
     *  <b> Pseudopotential total energy </b>
     *
     *  Total energy in PW-PP method has the following expression:
     *  \f[
     *    E_{tot} = \sum_{i} f_i \sum_{\sigma \sigma'} \langle \psi_i^{\sigma'} | \Big( \hat T + \sum_{\xi \xi'}
     *    |\beta_{\xi} \rangle D_{\xi \xi'}^{ion} \delta_{\sigma \sigma'} \langle \beta_{\xi'} |\Big) | \psi_i^{\sigma}
     * \rangle + \int V^{ion}({\bf r})\rho({\bf r})d{\bf r} + \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} +
     *      E^{XC}[\rho + \rho_{core}, |{\bf m}|]
     *  \f]
     *  Ionic contribution to the non-local part of pseudopotential is diagonal in spin. The following rearrangement
     *  is performed next:
     *  \f[
     *     \int \rho({\bf r}) \Big( V^{ion}({\bf r}) + \frac{1}{2} V^{H}({\bf r}) \Big) d{\bf r} = \\
     *     \int \rho({\bf r}) \Big( V^{ion}({\bf r}) + V^{H}({\bf r}) + V^{XC}({\bf r}) \Big) d{\bf r} +
     *     \int {\bf m}({\bf r}) {\bf B}^{XC}({\bf r}) d{\bf r} -
     *     \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} - \int V^{XC}({\bf r})\rho({\bf r})d{\bf r} -
     *     \int {\bf m}({\bf r}) {\bf B}^{XC}({\bf r}) d{\bf r}  = \\
     *     \sum_{\sigma \sigma'}\int \rho_{\sigma \sigma'}({\bf r}) V_{\sigma' \sigma}^{eff}({\bf r}) d{\bf r} -
     *     \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} - \int V^{XC}({\bf r})\rho({\bf r})d{\bf r} -
     *     \int {\bf m}({\bf r}) {\bf B}^{XC}({\bf r}) d{\bf r}
     *  \f]
     *  Where
     *  \f[
     *    \rho_{\sigma \sigma'}({\bf r}) = \sum_{i}^{occ} f_{i} \psi_{i}^{\sigma' *}({\bf r})\psi_{i}^{\sigma}({\bf r})
     *  \f]
     *  is a \f$ 2 \times 2 \f$ density matrix and
     *  \f[
     *    V_{\sigma\sigma'}^{eff}({\bf r})=\Big({\bf I}V^{eff}({\bf r})+{\boldsymbol \sigma}{\bf B}^{XC}({\bf r}) \Big)
     * =
     *      \left( \begin{array}{cc} V^{eff}({\bf r})+B_z^{XC}({\bf r}) & B_x^{XC}({\bf r})-iB_y^{XC}({\bf r}) \\
     *          B_x^{XC}({\bf r})+iB_y^{XC}({\bf r})  & V^{eff}({\bf r})-B_z^{XC}({\bf r}) \end{array} \right)
     *  \f]
     *  is a \f$ 2 \times 2 \f$ matrix potential (see \ref dft for the full derivation).
     *
     *  We are interested in this term:
     *  \f[
     *   \sum_{\sigma \sigma'}\int \rho_{\sigma \sigma'}({\bf r}) V_{\sigma' \sigma}^{eff}({\bf r}) d{\bf r} =
     *    \int V^{eff}({\bf r})\rho({\bf r})d{\bf r} + \int {\bf m}({\bf r}) {\bf B}^{XC}({\bf r}) d{\bf r}
     *  \f]
     *
     * We are going to split density into two contributions (sum of occupied bands \f$ \rho^{ps} \f$ and augmented
     * charge \f$ \rho^{aug} \f$) and use the definition of \f$ \rho^{aug} \f$: \f[ \sum_{\sigma \sigma'}\int
     * \rho_{\sigma \sigma'}^{aug}({\bf r}) V_{\sigma' \sigma}^{eff}({\bf r}) d{\bf r} = \sum_{\sigma \sigma'}\int
     * \sum_{i} \sum_{\xi \xi'} f_i \langle \psi_i^{\sigma'} | \beta_{\xi} \rangle Q_{\xi \xi'}({\bf r}) \langle
     * \beta_{\xi'} | \psi_i^{\sigma} \rangle V_{\sigma' \sigma}^{eff}({\bf r}) d{\bf r} = \sum_{\sigma \sigma'}
     * \sum_{i}\sum_{\xi \xi'} f_i \langle \psi_i^{\sigma'} | \beta_{\xi} \rangle D_{\xi \xi', \sigma' \sigma}^{aug}
     * \langle \beta_{\xi'} | \psi_i^{\sigma} \rangle \f] Now we can rewrite the total energy expression: \f[ E_{tot} =
     * \sum_{i} f_i \sum_{\sigma \sigma'} \langle \psi_i^{\sigma'} | \Big( \hat T + \sum_{\xi \xi'} |\beta_{\xi} \rangle
     * D_{\xi \xi'}^{ion} \delta_{\sigma \sigma'} + D_{\xi \xi', \sigma' \sigma}^{aug} \langle \beta_{\xi'} |\Big) |
     *    \psi_i^{\sigma} \rangle + \sum_{\sigma \sigma}
     *     \int V^{eff}_{\sigma' \sigma}({\bf r})\rho^{ps}_{\sigma \sigma'}({\bf r})d{\bf r} -
     *     \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} - \int V^{XC}({\bf r})\rho({\bf r}) d{\bf r} -
     *     \int {\bf m}({\bf r}) {\bf B}^{XC}({\bf r}) d{\bf r} + E^{XC}[\rho + \rho_{core}, |{\bf m}|]
     *  \f]
     *  From the Kohn-Sham equations
     *  \f[
     *    \hat T |\psi_i^{\sigma} \rangle + \sum_{\sigma'} \sum_{\xi \xi'} \Big( |\beta_{\xi}
     *    \rangle D_{\xi \xi', \sigma' \sigma} \langle \beta_{\xi'}| + \hat V^{eff}_{\sigma' \sigma} \Big)
     *    | \psi_i^{\sigma'} \rangle = \varepsilon_i \Big( 1+\hat S \Big) |\psi_i^{\sigma} \rangle
     *  \f]
     *  we immediately obtain that
     *  \f[
     *    \sum_{i} f_i \varepsilon_i = \sum_{i} f_i \sum_{\sigma \sigma'} \langle \psi_i^{\sigma'} |
     *    \Big( \hat T + \sum_{\xi \xi'} |\beta_{\xi}
     *    \rangle D_{\xi \xi', \sigma' \sigma} \langle \beta_{\xi'} |\Big) |
     *    \psi_i^{\sigma} \rangle + \sum_{\sigma \sigma}
     *     \int V^{eff}_{\sigma' \sigma}({\bf r})\rho^{ps}_{\sigma \sigma'}({\bf r})d{\bf r}
     *  \f]
     *  and the total energy expression simplifies to:
     *  \f[
     *    E_{tot} = \sum_{i} f_i \varepsilon_i - \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} -
     *    \int V^{XC}({\bf r})\rho({\bf r}) d{\bf r} - \int {\bf m}({\bf r}) {\bf B}^{XC}({\bf r}) d{\bf r} +
     *     E^{XC}[\rho + \rho_{core}, |{\bf m}|]
     *  \f]
     */
    double total_energy() const;

    json serialize();

    /// A quick check of self-constent density in case of pseudopotential.
    json check_scf_density();
};

} // namespace sirius

#endif // __DFT_GROUND_STATE_HPP__

/** \page dft Spin-polarized DFT
 *  \section s1dft Preliminary notes
 *
 *  \note Here and below sybol \f$ {\boldsymbol \sigma} \f$ is reserved for the vector of Pauli matrices. Spin
 *        components are labeled with \f$ \alpha \f$ or \f$ \beta \f$.
 *
 *  Wave-function of spin-1/2 particle is a two-component spinor:
 *  \f[
 *    {\bf \Psi}({\bf r}) = \left( \begin{array}{c} \psi^{\uparrow}({\bf r}) \\
 *                                                  \psi^{\downarrow}({\bf r}) 
 *                                  \end{array}
 *                          \right)
 *  \f] 
 *  Operator of spin: 
 *  \f[
 *    {\bf \hat S}=\frac{\hbar}{2}{\boldsymbol \sigma},
 *  \f]
 *  Pauli matrices:
 *  \f[
 *      \sigma_x=\left( \begin{array}{cc}
 *         0 & 1 \\
 *         1 & 0 \\ \end{array} \right) \,
 *       \sigma_y=\left( \begin{array}{cc}
 *         0 & -i \\
 *         i & 0 \\ \end{array} \right) \,
 *       \sigma_z=\left( \begin{array}{cc}
 *         1 & 0 \\
 *         0 & -1 \\ \end{array} \right)
 *  \f]
 *
 *  Spin moment of an electron in quantum state \f$ | \Psi \rangle \f$:
 *  \f[
 *     {\bf S}=\langle \Psi | {\bf \hat S} | \Psi \rangle  = \frac{\hbar}{2} \langle \Psi | {\boldsymbol \sigma} | \Psi
 *     \rangle 
 *  \f]
 *
 *  Spin magnetic moment of electron:
 *  \f[
 *    {\bf \mu}_e=\gamma_e {\bf S},
 *  \f]
 *  where \f$ \gamma_e \f$ is the gyromagnetic ratio for the electron.
 *  \f[
 *   \gamma_e=-\frac{g_e \mu_B}{\hbar} \;\;\; \mu_B=\frac{e\hbar}{2m_ec}
 *  \f]
 *  Here \f$ g_e \f$ is a g-factor for electron which is ~2, and \f$ \mu_B \f$ - Bohr magneton (defined as positive
 * constant). Finally, magnetic moment of electron: \f[
 *    {\bf \mu}_e=-{\bf \mu}_B \langle \Psi | {\boldsymbol \sigma} | \Psi \rangle
 *  \f]
 *  Potential energy of magnetic dipole in magnetic field:
 *  \f[
 *    U=-{\bf B}{\bf \mu}={\bf \mu}_B {\bf B} \langle \Psi | {\boldsymbol \sigma} | \Psi \rangle
 *  \f]
 *
 *  \section s2dft Density and magnetization
 *  In magnetic calculations we have charge density \f$ \rho({\bf r}) \f$ (scalar function) and magnetization density
 *  \f$ {\bf m}({\bf r}) \f$ (vector function). Density is defined as:
 *  \f[
 *      \rho({\bf r}) = \sum_{j}^{occ} \Psi_{j}^{*}({\bf r}){\bf I} \Psi_{j}({\bf r}) =
 *         \sum_{j}^{occ} \psi_{j}^{\uparrow *}({\bf r}) \psi_{j}^{\uparrow}({\bf r}) +
 *         \psi_{j}^{\downarrow *}({\bf r}) \psi_{j}^{\downarrow}({\bf r})
 *  \f]
 *  Magnetization is defined as:
 *  \f[
 *      {\bf m}({\bf r}) = \sum_{j}^{occ} \Psi_{j}^{*}({\bf r}) {\boldsymbol \sigma} \Psi_{j}({\bf r})
 *  \f]
 *  \f[
 *      m_x({\bf r}) = \sum_{j}^{occ} \psi_{j}^{\uparrow *}({\bf r}) \psi_{j}^{\downarrow}({\bf r}) +
 *        \psi_{j}^{\downarrow *}({\bf r}) \psi_{j}^{\uparrow}({\bf r})
 *  \f]
 *  \f[
 *      m_y({\bf r}) = \sum_{j}^{occ} -i \psi_{j}^{\uparrow *}({\bf r}) \psi_{j}^{\downarrow}({\bf r}) +
 *        i \psi_{j}^{\downarrow *}({\bf r}) \psi_{j}^{\uparrow}({\bf r})
 *  \f]
 *  \f[
 *      m_z({\bf r}) = \sum_{j}^{occ} \psi_{j}^{\uparrow *}({\bf r}) \psi_{j}^{\uparrow}({\bf r}) -
 *        \psi_{j}^{\downarrow *}({\bf r}) \psi_{j}^{\downarrow}({\bf r})
 *  \f]
 *  Density and magnetization can be grouped into a \f$ 2 \times 2 \f$ density matrix, which is defined as:
 *  \f[
 *      {\boldsymbol \rho}({\bf r}) = \frac{1}{2} \Big( {\bf I}\rho({\bf r}) + {\boldsymbol \sigma} {\bf m}({\bf
 * r})\Big) =
 *        \frac{1}{2} \left( \begin{array}{cc} \rho({\bf r}) + m_z({\bf r}) & m_x({\bf r}) - i m_y({\bf r}) \\
 *                                              m_x({\bf r}) + i m_y({\bf r}) & \rho({\bf r}) - m_z({\bf r}) \end{array}
 * \right) = \sum_{j}^{occ} \left( \begin{array}{cc} \psi_{j}^{\uparrow *}({\bf r}) \psi_{j}^{\uparrow}({\bf r}) &
 *                                                \psi_{j}^{\downarrow *}({\bf r}) \psi_{j}^{\uparrow}({\bf r}) \\
 *                                                \psi_{j}^{\uparrow *}({\bf r}) \psi_{j}^{\downarrow}({\bf r}) &
 *                                                \psi_{j}^{\downarrow *}({\bf r}) \psi_{j}^{\downarrow}({\bf r})
 * \end{array} \right) \f] or simply \f[ \rho_{\alpha \beta}({\bf r}) = \sum_{j}^{occ} \psi_{j}^{\beta *}({\bf
 * r})\psi_{j}^{\alpha}({\bf r}) \f] Pay attention to the order of spin indices in the \f$ 2 \times 2 \f$ density
 * matrix. External potential \f$ v^{ext}({\bf r}) \f$ and external magnetic field \f$ {\bf B}^{ext}({\bf r}) \f$ can
 *  also be grouped into a \f$ 2 \times 2 \f$ matrix:
 *  \f[
 *    V_{\alpha\beta}^{ext}({\bf r})=\Big({\bf I}v^{ext}({\bf r})+\mu_{B}{\boldsymbol \sigma}{\bf B}^{ext}({\bf r})
 * \Big) =
 *      \left( \begin{array}{cc} v^{ext}({\bf r})+\mu_{B}B_z^{ext}({\bf r}) & \mu_{B} \Big( B_x^{ext}({\bf
 * r})-iB_y^{exp}({\bf r}) \Big) \\ \mu_{B} \Big( B_x^{ext}({\bf r})+iB_y^{ext}({\bf r}) \Big) & v^{ext}({\bf
 * r})-\mu_{B}B_z^{ext}({\bf r}) \end{array} \right) \f] Let's check that potential energy in external fields can be
 * written in the following way: \f[ E^{ext}=\int Tr \Big( {\boldsymbol \rho}({\bf r}) {\bf V}^{ext}({\bf r}) \Big)
 * d{\bf r} = \sum_{\alpha\beta} \int \rho_{\alpha\beta}({\bf r})V_{\beta\alpha}^{ext}({\bf r}) d{\bf r} \f] (below \f$
 * {\bf r} \f$, \f$ ext \f$ and \f$ \int d{\bf r} \f$ are dropped for simplicity) \f[ \begin{eqnarray}
 *    \rho_{11}V_{11} &= \frac{1}{2}(\rho+m_z)(v+\mu_{B}B_z) = \frac{1}{2}(\rho v +\mu_{B} \rho B_z + m_z v +
 * \mu_{B}m_zB_z) \\
 *    \rho_{22}V_{22} &= \frac{1}{2}(\rho-m_z)(v-\mu_{B}B_z) = \frac{1}{2}(\rho v -\mu_{B} \rho B_z - m_z v +
 * \mu_{B}m_zB_z) \\
 *    \rho_{12}V_{21} &= \frac{1}{2}(m_x-im_y)\Big( \mu_{B}( B_x+iB_y) \Big) =
 * \frac{\mu_B}{2}(m_xB_x+im_xB_y-im_yB_x+m_yB_y) \\ \rho_{21}V_{12} &= \frac{1}{2}(m_x+im_y)\Big( \mu_{B}( B_x-iB_y)
 * \Big) = \frac{\mu_B}{2}(m_xB_x-im_xB_y+im_yB_x+m_yB_y) \end{eqnarray} \f] The sum of this four terms will give
 * exactly \f$ \int \rho({\bf r}) v^{ext}({\bf r}) + \mu_{B}{\bf m}({\bf r})
 *   {\bf B}^{ext}({\bf r}) d{\bf r}\f$
 *
 *  \section s3dft Total energy variation
 *
 *  To derive Kohn-Sham equations we need to write total energy functional of density matrix \f$ \rho_{\alpha\beta}({\bf
 * r}) \f$: \f[ E^{tot}[\rho_{\alpha\beta}] = E^{kin}[\rho_{\alpha\beta}] + E^{H}[\rho_{\alpha\beta}] +
 * E^{ext}[\rho_{\alpha\beta}] + E^{XC}[\rho_{\alpha\beta}] \f] Kinetic energy of non-interacting electrons is written
 * in the following way: \f[ E^{kin}[\rho_{\alpha\beta}] \equiv E^{kin}[\Psi[\rho_{\alpha\beta}]] =
 *    -\frac{1}{2} \sum_{i}^{occ}\sum_{\alpha} \int \psi_{i}^{\alpha*}({\bf r}) \nabla^{2} \psi_{i}^{\alpha}({\bf
 * r})d^3{\bf r} \f] Hartree energy: \f[ E^{H}[\rho_{\alpha\beta}]= \frac{1}{2} \iint \frac{\rho({\bf r})\rho({\bf
 * r'})}{|{\bf r}-{\bf r'}|} d{\bf r} d{\bf r'} = \frac{1}{2} \iint \sum_{\alpha\beta}\delta_{\alpha\beta}
 * \frac{\rho_{\alpha\beta}({\bf r}) \rho({\bf r'})}{|{\bf r}-{\bf r'}|} d{\bf r} d{\bf r'} \f] where \f$ \rho({\bf r})
 * = Tr \rho_{\alpha\beta}({\bf r}) \f$.
 *
 *  Exchange-correlation energy:
 *  \f[
 *    E^{XC}[\rho_{\alpha\beta}({\bf r})] \equiv E^{XC}[\rho({\bf r}),|{\bf m}({\bf r})|] =
 *     \int \rho({\bf r}) \eta_{XC}(\rho({\bf r}), m({\bf r})) d{\bf r} =
 *     \int \rho({\bf r}) \eta_{XC}(\rho^{\uparrow}({\bf r}), \rho_{\downarrow}({\bf r})) d{\bf r}
 *  \f]
 *  Now we can write the total energy variation over auxiliary orbitals with constrain of orbital normalization:
 *  \f[
 *    \frac{\delta \Big( E^{tot}+\varepsilon_i \big( 1-\sum_{\alpha} \int \psi^{\alpha*}_{i}({\bf r})
 *       \psi^{\alpha}_{i}({\bf r})d{\bf r} \big) \Big)} {\delta \psi_{i}^{\gamma*}({\bf r})} = 0
 *  \f]
 *  We will use the following chain rule:
 *  \f[
 *    \frac{\delta F[\rho_{\alpha\beta}]}{\delta \psi_{i}^{\gamma *}({\bf r})} =
 *      \sum_{\alpha' \beta'} \frac{\delta F[\rho_{\alpha\beta}]}{\delta \rho_{\alpha'\beta'}({\bf r})}
 *      \frac{\delta \rho_{\alpha'\beta'}({\bf r})}{\delta \psi_{i}^{\gamma *}({\bf r})} =
 *      \sum_{\alpha'\beta'}\frac{\delta F[\rho_{\alpha\beta}]}{\delta \rho_{\alpha'\beta'}({\bf r})}
 *        \psi_{i}^{\alpha'}({\bf r}) \delta_{\beta'\gamma} =
 *        \sum_{\alpha'}\frac{\delta F[\rho_{\alpha\beta}]}{\delta \rho_{\alpha'\gamma}({\bf r})}\psi_{i}^{\alpha'}({\bf
 * r}) \f] Variation of the normalization integral: \f[ \frac{\delta \sum_{\alpha} \int \psi_{i}^{\alpha*}({\bf r})
 * \psi_{i}^{\alpha}({\bf r}) d {\bf r} }{\delta \psi_{i}^{\gamma *}({\bf r})} =  \psi_{i}^{\gamma}({\bf r}) \f]
 *  Variation of the kinetic energy functional:
 *  \f[
 *    \frac{\delta E^{kin}}{\delta \psi_{i}^{\gamma*}({\bf r})}  = -\frac{1}{2} \sum_{\alpha} \nabla^{2}
 * \psi_{i}^{\alpha}({\bf r}) \delta_{\alpha\gamma} = -\frac{1}{2}\nabla^{2}\psi_{i}^{\gamma}({\bf r}) \f] Variation of
 * the Hartree energy functional: \f[ \frac{\delta E^{H}[\rho_{\alpha\beta}]}{\delta \psi_{i}^{\gamma *}({\bf r})} =
 *    \sum_{\alpha'} \sum_{\alpha\beta} \delta_{\alpha\beta} \frac{1}{2} \int \frac{ \rho({\bf r'})}{|{\bf r}-{\bf r'}|}
 * d{\bf r'} \delta_{\alpha\alpha'}\delta_{\beta\gamma} \psi_{i}^{\alpha'}({\bf r}) = v^{H}({\bf r})
 * \psi_{i}^{\gamma}({\bf r}) \f] Variation of the external energy functional: \f[ \frac{\delta
 * E^{ext}[\rho_{\alpha\beta}]}{\delta \psi_{i}^{\gamma*}({\bf r}) } = \sum_{\alpha'} \sum_{\alpha\beta}
 * V_{\beta\alpha}^{ext}({\bf r}) \delta_{\alpha\alpha'} \delta_{\beta\gamma} \psi_{i}^{\alpha'}({\bf r})= \sum_{\alpha}
 * V_{\gamma\alpha}^{ext}({\bf r}) \psi_{i}^{\alpha}({\bf r}) \f]
 *
 *  Variation of the exchange-correlation functional:
 *  \f[
 *    \frac{\delta E^{XC}[\rho_{\alpha\beta}]}{ \delta \psi_{i}^{\gamma*}({\bf r}) } =
 *    \frac{\delta E^{XC}[\rho_{\alpha\beta}]}{ \delta \rho({\bf r})} \frac{\delta \rho({\bf r})}{\delta
 *    \psi_{i}^{\gamma*}({\bf r})} + \frac{\delta E^{XC}[\rho_{\alpha\beta}]}{ \delta m({\bf r})} \sum_{p=x,y,z}
 *    \frac{\delta m({\bf r})}{ \delta m_p({\bf r})}
 *    \frac{\delta m_p({\bf r})}{\delta \psi_{i}^{\gamma*}({\bf r})}
 *  \f]
 *  where \f$ m({\bf r}) = |{\bf  m}({\bf r})|\f$ is the length of magnetization vector.
 *
 *  First term:
 *  \f[
 *    \frac{\delta E^{XC}[\rho_{\alpha\beta}]}{ \delta \rho({\bf r})} \frac{\delta \rho({\bf r})}{\delta
 *     \psi_{i}^{\gamma*}({\bf r})} = v^{XC}({\bf r}) \psi_{i}^{\gamma}({\bf r})
 *  \f]
 *  Second term:
 *  \f[
 *    \frac{\delta E^{XC}[\rho_{\alpha\beta}]}{ \delta m({\bf r})} \sum_{p=x,y,z} \frac{\delta m({\bf r})}{ \delta
 * m_p({\bf r})} \frac{\delta m_p({\bf r})}{\delta \psi_{i}^{\gamma*}({\bf r})} = B^{XC}({\bf r}) \hat {\bf m}
 * \sum_{\beta} {\boldsymbol \sigma}_{\gamma \beta} \psi_{i}^{\beta}({\bf r}) \f] where \f$ B^{XC}({\bf r}) =
 * \frac{\delta E^{XC}[\rho_{\alpha\beta}]}{ \delta m({\bf r})} \f$, \f$ \hat {\bf m}({\bf r}) = \frac{\delta m({\bf
 * r})}{ \delta {\bf m}({\bf r})} \f$ is the unit vector, parallel to \f$ {\bf m}({\bf r}) \f$ and \f$ {\bf m}({\bf r})
 * = \sum_{i} \sum_{\alpha \beta} \psi_{i}^{\alpha*}({\bf r}) {\boldsymbol \sigma}_{\alpha \beta} \psi_i^{\beta}({\bf
 * r}) \f$
 *
 *  Similarly to external potential, exchange-correlation potential can be grouped into \f$ 2 \times 2 \f$ matrix::
 *  \f[
 *    \frac{\delta E^{XC}[\rho_{\alpha\beta}]}{\delta \rho_{\alpha'\beta'}({\bf r})} \equiv V^{XC}_{\beta'\alpha'}({\bf
 * r})  = \Big( {\bf I}v^{XC}({\bf r}) + {\bf B}^{XC}({\bf r}) {\boldsymbol \sigma} \Big)_{\beta'\alpha'} \f] where
 * \f${\bf B}^{XC}({\bf r}) = \hat {\bf m}({\bf r})B^{XC}({\bf r}) \f$ -- exchange-correlation magnetic field, parallel
 * to \f$ {\bf m}({\bf r}) \f$ at each point in space. We can now collect \f$ v^{H}({\bf r}) \f$, \f$
 * V_{\alpha\beta}^{ext}({\bf r}) \f$ and \f$V_{\alpha\beta}^{XC}({\bf r}) \f$ to one effective potential: \f[
 *    V^{eff}_{\alpha\beta}({\bf r}) = v^{H}({\bf r})\delta_{\alpha\beta} + V_{\alpha\beta}^{ext}({\bf r}) +
 *         V_{\alpha\beta}^{XC}({\bf r}) =
 *     \Big({\bf I}\big(v^{H}({\bf r})+v^{ext}({\bf r})+v^{XC}({\bf r})\big) +
 *     {\boldsymbol \sigma}\big( \mu_{B}{\bf B}^{ext}({\bf r}) + {\bf B}^{XC}({\bf r})\big)\Big)_{\alpha\beta}
 *  \f]
 *  and finally, we arrive to the following Kohn-Sham equation for each component \f$ \gamma \f$ of spinor
 * wave-functions: \f[
 *   -\frac{1}{2}\nabla^{2}\psi_{i}^{\gamma}({\bf r}) + \sum_{\alpha} V_{\gamma\alpha}^{eff}({\bf r})
 * \psi_{i}^{\alpha}({\bf r}) = \varepsilon_i \psi_{i}^{\gamma}({\bf r}) \f] or in matrix form \f[
 *  \left( \begin{array}{cc} -\frac{1}{2}\nabla^2+V^{eff}_{\uparrow \uparrow} & V^{eff}_{\uparrow \downarrow} \\
 *    V^{eff}_{\downarrow \uparrow} & -\frac{1}{2}\nabla^2+V^{eff}_{\downarrow \downarrow} \end{array}\right)
 *    \left(\begin{array}{c} \psi_{i}^{\uparrow}({\bf r}) \\ \psi_{i}^{\downarrow} ({\bf r}) \end{array} \right) =
 * \varepsilon_i \left(\begin{array}{c} \psi_{i}^{\uparrow}({\bf r}) \\ \psi_{i}^{\downarrow}({\bf r}) \end{array}
 * \right) \f]
 *
 *  \section s4dft Second-variational approach
 *
 *  Suppose that we know first \f$ N_{fv} \f$ solutions of the following equation (so-called first variational
 * equation): \f[ \Big(-\frac{1}{2}\nabla^2+v^{H}({\bf r})+v^{ext}({\bf r})+v^{XC}({\bf r}) \Big)\phi_{i}({\bf r}) =
 * \epsilon_i \phi_{i}({\bf r}) \f] We can write expansion of the components of spinor wave-functions \f$
 * \psi^{\alpha}_j({\bf r}) \f$ in terms of first-variational states \f$ \phi_i({\bf r}) \f$: \f[ \psi_{j}^{\alpha}({\bf
 * r}) = \sum_{i}^{N_{fv}}C_{ij}^{\alpha}\phi_{i}({\bf r}) \f] Next, we switch to matrix equation: \f[
 *    \langle \Psi_{j'}| \hat H | \Psi_{j} \rangle = \varepsilon_j \delta_{j'j} \\
 *    \sum_{\alpha \alpha'} \sum_{ii'} C_{i'j'}^{\alpha'*} C_{ij}^{\alpha} \langle \phi_{i'} | \hat H_{\alpha' \alpha} |
 * \phi_{i} \rangle = \sum_{\alpha \alpha'} \sum_{ii'} C_{i'j'}^{\alpha'*} C_{ij}^{\alpha} H_{\alpha'i', \alpha i} =
 * \varepsilon_j \delta_{j'j} \f]
 *
 *  We can combine indices \f$ \{i,\alpha\} \f$ into one 'flat' index \f$ \nu \f$. If we also assume that the number of
 *  spinor wave-functions is equal to \f$ 2 N_{fv} \f$ then we arrive to the well-known eigen decomposition:
 *  \f[
 *    \sum_{\nu'\nu} C_{\nu' j'}^{*} H_{\nu'\nu} C_{\nu j} = \varepsilon_j \delta_{j'j}
 *  \f]
 *
 * The expression for second-variational Hamiltonian is simple:
 * \f[
 *   \langle \phi_{i'}|\hat H_{\alpha'\alpha} |\phi_{i} \rangle =
 *    \langle \phi_{i'} | \Big(-\frac{1}{2}\nabla^2 + v^{H}({\bf r}) + v^{ext}({\bf r}) + v^{XC}({\bf r}) \Big)
 *     \delta_{\alpha\alpha'}|\phi_{i}\rangle +
 *    \langle \phi_{i'} | {\boldsymbol \sigma}_{\alpha\alpha'} \Big( \mu_{B}{\bf B}^{ext}({\bf r})+
 *    {\bf B}^{XC}({\bf r})\Big) | \phi_{i}\rangle =\\
 *    \epsilon_{i}\delta_{i'i}\delta_{\alpha\alpha'} + {\boldsymbol \sigma}_{\alpha\alpha'} \langle \phi_{i'} |
 *    \Big( \mu_{B}{\bf B}^{ext}({\bf r}) +
 *    {\bf B}^{XC}({\bf r})\Big) | \phi_{i}\rangle
 *  \f]
 */
