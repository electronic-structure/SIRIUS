#ifndef __ENERGY_HPP__
#define __ENERGY_HPP__

#include "density/density.hpp"
#include "potential/potential.hpp"
#include "simulation_context.hpp"

namespace sirius {

/* forward declaration */
class DFT_ground_state;

/// Compute various contributions to total energy.
namespace energy {

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
double ewald(const Simulation_context& ctx);

/// Returns integral of exchange-correlation potential with density.
/** The following quantity is computed
 *  \f[
 *    \int_{\Omega} \rho({\bf r}) V^{XC}({\bf r}) d{\bf r}
 *  \f]
 *  where \f$ \Omega \f$ is the volume of the unit cell and \f$ V^{XC}({\bf r}) \f$ is the exchange-correlation
 *  potential. */
double vxc(Density const& density, Potential const& potential);

/// Returns exchange-correlation energy.
/** The following quantity is computed
 *  \f[
 *    E^{XC}[\rho] = \int_{\Omega} \rho({\bf r}) \varepsilon_{xc}\big(\rho({\bf r})\big) d{\bf r}
 *  \f]
 *  where \f$ \Omega \f$ is the volume of the unit cell and \f$ \varepsilon_{xc}(x) \f$ is the
 *  exchange-correlation energy density. See sirius::Potential::xc for details. */
double exc(Density const& density, Potential const& potential);

/// Returns Hatree potential integral with density.
/** The following quantity is computed:
 *  \f[
 *    \int_{\Omega} \rho({\bf r}) V^{H}({\bf r}) d{\bf r}
 *  \f]
 *  where \f$ V^{H}({\bf r}) \f$ is the Hartree potential. Note that the Hartree energy is equal to half of this
 *  value, i.e.
 *  \f[
 *    E^H[\rho] = \frac{1}{2} \int_{\Omega} \rho({\bf r}) V^{H}({\bf r}) d{\bf r}
 *  \f]
 *  */
double vha(Potential const& potential);

/// Return integral of the exchange-correlation magnetic field with the magnetisation.
/** The following quantity is computed:
 *  \f[
 *    \int_{\Omega} {\bf m}({\bf r}) {\bf B}^{XC}({\bf r}) d{\bf r}
 *  \f]
 * where under the integral scalar product of \f$ {\bf B}^{XC}({\bf r}) \f$ with \f$ {\bf m}({\bf r}) \f$
 * is computed. */
double bxc(const Density& density, const Potential& potential);

/// Return nucleus energy in the electrostatic field.
/** Compute energy of nucleus in the electrostatic potential generated by the total (electrons + nuclei)
 *  charge density. Diverging self-interaction term \f$ \frac{z^2}{|r=0|} \f$ is excluded. */
double nuc(Simulation_context const& ctx, Potential const& potential);

/// Return integral of the local part of pseudopotential with the charge density.
/** The following quantity is computed:
 *  \f[
 *    \int_{\Omega} \rho({\bf r}) V^{loc}({\bf r}) d{\bf r}
 *  \f]
 */
double vloc(Density const& density, Potential const& potential);

/// Return eigen-value sum of the core states.
double ecore_sum(Unit_cell const& unit_cell);

/// Returns integral of effective potential with density.
/** The following quantity is computed
 *  \f[
 *    \int_{\Omega} \rho({\bf r}) V^{eff}({\bf r}) d{\bf r}
 *  \f]
 *  where \f$ \Omega \f$ is the volume of the unit cell and \f$ V^{eff}({\bf r}) \f$ is the effective
 *  potential. */
double veff(Density const& density, Potential const& potential);

/// Return kinetic energy.
double kin(DFT_ground_state const &dft);

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
 *     = \left( \begin{array}{cc} V^{eff}({\bf r})+B_z^{XC}({\bf r}) & B_x^{XC}({\bf r})-iB_y^{XC}({\bf r}) \\
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
double total(DFT_ground_state const& dft);

double one_electron(Density const& density, Potential const& potential);

/// Projected-augmented wave contribution to total energy.
double paw(Potential const& potential__);

} // namespace energy

} // namespace sirius

#endif /* __ENERGY_HPP__ */
