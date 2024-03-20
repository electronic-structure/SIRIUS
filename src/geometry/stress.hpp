/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file stress.hpp
 *
 *  \brief Contains definition of sirius::Stress tensor class.
 */

#ifndef __STRESS_HPP__
#define __STRESS_HPP__

#include "potential/potential.hpp"

namespace sirius {

/// Stress tensor.
/** The following referenceces were particularly useful in the derivation of the stress tensor components:
 *    - Hutter, D. M. A. J. (2012). Ab Initio Molecular Dynamics (pp. 1–580).
 *    - Marx, D., & Hutter, J. (2000). Ab initio molecular dynamics: Theory and implementation.
 *      Modern Methods and Algorithms of Quantum Chemistry.
 *    - Knuth, F., Carbogno, C., Atalla, V., & Blum, V. (2015). All-electron formalism for total energy strain
 *      derivatives and stress tensor components for numeric atom-centered orbitals. Computer Physics Communications.
 *    - Willand, A., Kvashnin, Y. O., Genovese, L., Vázquez-Mayagoitia, Á., Deb, A. K., Sadeghi, A., et al. (2013).
 *      Norm-conserving pseudopotentials with chemical accuracy compared to all-electron calculations.
 *      The Journal of Chemical Physics, 138(10), 104109. http://doi.org/10.1103/PhysRevB.50.4327
 *    - Corso, A. D., & Resta, R. (1994). Density-functional theory of macroscopic stress: Gradient-corrected
 *      calculations for crystalline Se. Physical Review B.
 *
 *  Stress tensor describes a reaction of crystall to a strain:
 *  \f[
 *  \sigma_{\mu \nu} = \frac{1}{\Omega} \frac{\partial  E}{\partial \varepsilon_{\mu \nu}}
 *  \f]
 *  where \f$ \varepsilon_{\mu \nu} \f$ is a symmetric strain tensor which describes the infinitesimal
 *  deformation of a crystal:
 *  \f[
 *  r_{\mu} \rightarrow \sum_{\nu} (\delta_{\mu \nu} + \varepsilon_{\mu \nu}) r_{\nu}
 *  \f]
 *  The following expressions are helpful:
 *    - strain derivative of a general position vector component:
 *  \f[
 *  \frac{\partial r_{\tau}}{\partial \varepsilon_{\mu \nu}} = \delta_{\tau \mu} r_{\nu}
 *  \f]
 *    - strain derivative of the lengths of a general position vector:
 *  \f[
 *  \frac{\partial |{\bf r}|}{\partial \varepsilon_{\mu \nu}} = \sum_{\tau} \frac{\partial |{\bf r}|}{\partial
 *  r_{\tau}} \frac{\partial r_{\tau}}{\partial \varepsilon_{\mu \nu}} = \sum_{\tau} \frac{ r_{\tau}}{|{\bf r}|}
 *  \delta_{\tau \mu} r_{\nu} = \frac{r_{\mu} r_{\nu}}{|{\bf r}|}
 *  \f]
 *    - strain derivative of the unit cell volume:
 *  \f[
 *  \frac{\partial \Omega}{\partial \varepsilon_{\mu \nu}} = \delta_{\mu \nu} \Omega
 *  \f]
 *    - strain derivative of the inverse square root of the unit cell volume:
 *  \f[
 *  \frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{1}{\sqrt{\Omega}} = -\frac{1}{2}\frac{1}{\sqrt{\Omega}}
 *  \delta_{\mu \nu}
 *  \f]
 *    - strain derivative of a reciprocal vector:
 *  \f[
 *  \frac{\partial G_{\tau}}{\partial \varepsilon_{\mu \nu}} = -\delta_{\tau \nu} G_{\mu}
 *  \f]
 *    - strain derivative of the length of a reciprocal vector:
 *  \f[
 *  \frac{\partial |{\bf G}|}{\partial \varepsilon_{\mu \nu}} = \sum_{\tau} \frac{\partial |{\bf G}|}{\partial
 *  G_{\tau}} \frac{\partial G_{\tau}}{\partial \varepsilon_{\mu \nu}} = -\frac{1}{|{\bf G}|}G_{\nu}G_{\mu}
 *  \f]
 *  In the derivation of the stress tensor contributions it is important to know which variables are
 *  invariant under lattice distortion. This are:
 *    - scalar product of the real-space (in the Bravais lattice frame) and reciprocal
 *      (in the reciprocal lattice frame) vectors
 *    - normalized plane-wave coefficients of the wave-functions
 *  \f[
 *  \psi({\bf G}) = \frac{1}{\sqrt{\Omega}} \int e^{-i {\bf G}{\bf r}} \psi({\bf r}) d{\bf r}
 *  \f]
 *    - unnormalized plane-wave coefficients of the charge density
 *  \f[
 *  \tilde \rho({\bf G}) = \int e^{-i {\bf G}{\bf r}} \sum_{j} |\psi_j({\bf r})|^{2} d{\bf r}
 *  \f]
 */
class Stress
{
  private:
    Simulation_context& ctx_;

    Density const& density_;

    Potential& potential_;

    K_point_set& kset_;

    r3::matrix<double> stress_kin_;

    r3::matrix<double> stress_har_;

    r3::matrix<double> stress_ewald_;

    r3::matrix<double> stress_vloc_;

    r3::matrix<double> stress_nonloc_;

    r3::matrix<double> stress_us_;

    r3::matrix<double> stress_xc_;

    r3::matrix<double> stress_core_;

    r3::matrix<double> stress_hubbard_;

    r3::matrix<double> stress_total_;

    /// Non-local contribution to stress.
    /** Energy contribution from the non-local part of pseudopotential:
     *  \f[
     *  E^{nl} = \sum_{i{\bf k}} \sum_{\alpha}\sum_{\xi \xi'}P_{\xi}^{\alpha,i{\bf k}}
     *   D_{\xi\xi'}^{\alpha}P_{\xi'}^{\alpha,i{\bf k}*}
     *  \f]
     *  where
     *  \f[
     *  P_{\xi}^{\alpha,i{\bf k}} = \langle \psi_{i{\bf k}} | \beta_{\xi}^{\alpha} \rangle =
     *   \frac{1}{\Omega} \sum_{\bf G} \langle \psi_{i{\bf k}} | {\bf G+k} \rangle
     *   \langle {\bf G+k} | \beta_{\xi}^{\alpha} \rangle =
     *   \sum_{\bf G} \psi_i^{*}({\bf G+k}) \beta_{\xi}^{\alpha}({\bf G+k})
     *  \f]
     *  where
     *  \f[
     *  \beta_{\xi}^{\alpha}({\bf G+k}) = \frac{1}{\sqrt{\Omega}} \int e^{-i({\bf G+k}){\bf r}}
     *   \beta_{\xi}^{\alpha}({\bf r}) d{\bf r} = \frac{4\pi}{\sqrt{\Omega}} e^{-i{\bf (G+k) r_{\alpha}}}(-i)^{\ell}
     *   R_{\ell m}(\theta_{G+k}, \phi_{G+k}) \int \beta_{\ell}(r) j_{\ell}(|{\bf G+k}|r) r^2 dr
     *  \f]
     *  Contribution to stress tensor:
     *  \f[
     *  \sigma_{\mu \nu}^{nl} = \frac{1}{\Omega} \frac{\partial E^{nl}}{\partial \varepsilon_{\mu \nu}} =
     *   \frac{1}{\Omega}  \sum_{i{\bf k}} \sum_{\xi \xi'}
     *   \Bigg( \frac{\partial P_{\xi}^{\alpha,i{\bf k}}}{\partial \varepsilon_{\mu \nu}}
     *   D_{\xi\xi'}^{\alpha}P_{\xi'}^{\alpha,i{\bf k}*} + P_{\xi}^{\alpha,i{\bf k}}
     *   D_{\xi\xi'}^{\alpha} \frac{\partial P_{\xi'}^{\alpha,i{\bf k}*}}{\partial \varepsilon_{\mu \nu}} \Bigg)
     *  \f]
     *  We need to compute strain derivatives of \f$ P_{\xi}^{\alpha,i{\bf k}} \f$:
     *  \f[
     *   \frac{\partial}{\partial
     *  \varepsilon_{\mu \nu}} P_{\xi}^{\alpha,i{\bf k}} = \sum_{\bf G} \psi_i^{*}({\bf G+k}) \frac{\partial}{\partial
     *  \varepsilon_{\mu \nu}} \beta_{\xi}^{\alpha}({\bf G+k})
     *  \f]
     *  First way to take strain derivative of beta-projectors (here and below \f$ {\bf G+k} = {\bf q} \f$):
     *  \f[
     *  \frac{\partial}{\partial \varepsilon_{\mu \nu}}
     *  \beta_{\xi}^{\alpha}({\bf q}) =
     *    -\frac{4\pi}{2\sqrt{\Omega}} \delta_{\mu \nu} e^{-i{\bf q r_{\alpha}}}(-i)^{\ell}
     *    R_{\ell m}(\theta_q, \phi_q) \int \beta_{\ell}(r) j_{\ell}(q r) r^2 dr + \\
     *    \frac{4\pi}{\sqrt{\Omega}} e^{-i{\bf q r_{\alpha}}}(-i)^{\ell}
     *    \frac{\partial R_{\ell m}(\theta_q, \phi_q)}{\partial \varepsilon_{\mu \nu}}
     *      \int \beta_{\ell}(r) j_{\ell}(q r) r^2 dr + \\
     *    \frac{4\pi}{\sqrt{\Omega}} e^{-i{\bf q r_{\alpha}}}(-i)^{\ell} R_{\ell m}(\theta_q, \phi_q)
     *       \int \beta_{\ell}(r) \frac{\partial j_{\ell}(q r)}{\partial \varepsilon_{\mu \nu}} r^2 dr = \\
     *    \frac{4\pi}{\sqrt{\Omega}} e^{-i{\bf q r_{\alpha}}}(-i)^{\ell}
     *    \Bigg[ \int \beta_{\ell}(r) j_{\ell}(q r) r^2 dr
     *    \Big(\frac{\partial R_{\ell m}(\theta_q, \phi_q)}{\partial \varepsilon_{\mu \nu}} -
     *    \frac{1}{2} R_{\ell m}(\theta_q, \phi_q) \delta_{\mu \nu}\Big) + R_{\ell m}(\theta_q, \phi_q)
     *    \int \beta_{\ell}(r) \frac{\partial j_{\ell}(q r)}{\partial \varepsilon_{\mu \nu}} r^2 dr \Bigg]
     *  \f]
     *
     *  Strain derivative of the real spherical harmonics:
     *  \f[
     *    \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \varepsilon_{\mu \nu}} =
     *      \sum_{\tau} \frac{\partial R_{\ell m}(\theta, \phi)}{\partial q_{\tau}} \frac{\partial q_{\tau}}{\partial
     *   \varepsilon_{\mu \nu}} = -q_{\mu} \frac{\partial R_{\ell m}(\theta, \phi)}{\partial q_{\nu}}
     *  \f]
     *  For the derivatives of spherical harmonics over Cartesian components of vector please refer to
     *  the sht::dRlm_dr function.
     *
     *  Strain derivative of spherical Bessel function integral:
     *  \f[
     *    \int \beta_{\ell}(r) \frac{\partial j_{\ell}(qr) }{\partial \varepsilon_{\mu \nu}}  r^2 dr =
     *     \int \beta_{\ell}(r) \frac{\partial j_{\ell}(qr) }{\partial q} \frac{-q_{\mu} q_{\nu}}{q} r^2 dr
     *  \f]
     *  The second way to compute strain derivative of beta-projectors is trough Gaunt coefficients:
     * \f[
     *  \frac{\partial}{\partial \varepsilon_{\mu \nu}} \beta_{\xi}^{\alpha}({\bf q}) =
     *  -\frac{1}{2\sqrt{\Omega}} \delta_{\mu \nu} \int e^{-i{\bf q r}} \beta_{\xi}^{\alpha}({\bf r}) d{\bf r} +
     *  \frac{1}{\sqrt{\Omega}} \int i r_{\nu} q_{\mu} e^{-i{\bf q r}} \beta_{\xi}^{\alpha}({\bf r}) d{\bf r}
     * \f]
     * (second term comes from the strain derivative of \f$ e^{-i{\bf q r}} \f$). Remembering that \f$ {\bf r} \f$ is
     * proportional to p-like real spherical harmonics, we can rewrite the second part of beta-projector derivative as:
     * \f[
     *  \frac{1}{\sqrt{\Omega}} \int i r_{\nu} q_{\mu} e^{-i{\bf q r}} \beta_{\xi}^{\alpha}({\bf r}) d{\bf r} =
     *   \frac{1}{\sqrt{\Omega}} i q_{\mu} \int r \bar R_{1 \nu}(\theta, \phi) 4\pi \sum_{\ell_3 m_3} (-i)^{\ell_3}
     *  R_{\ell_3 m_3}(\theta_q, \phi_q) R_{\ell_3 m_3}(\theta, \phi) j_{\ell_3}(q r) \beta_{\ell_2}^{\alpha}(r)
     *  R_{\ell_2 m_2}(\theta, \phi) d{\bf r} = \\
     *  \frac{4 \pi}{\sqrt{\Omega}} i q_{\mu} \sum_{\ell_3 m_3} (-i)^{\ell_3} R_{\ell_3 m_3}(\theta_q, \phi_q)
     *     \langle \bar R_{1\nu} | R_{\ell_3 m_3} | R_{\ell_2 m_2} \rangle
     *     \int j_{\ell_3}(q r) \beta_{\ell_2}^{\alpha}(r) r^3 dr
     * \f]
     * where
     * \f[
     *   \bar R_{1 x}(\theta, \phi) = -2 \sqrt{\frac{\pi}{3}} R_{11}(\theta, \phi) = \sin(\theta) \cos(\phi) \\
     *   \bar R_{1 y}(\theta, \phi) = -2 \sqrt{\frac{\pi}{3}} R_{1-1}(\theta,\phi) = \sin(\theta) \sin(\phi) \\
     *   \bar R_{1 z}(\theta, \phi) = 2 \sqrt{\frac{\pi}{3}} R_{10}(\theta, \phi) = \cos(\theta)
     * \f]
     *
     * \tparam T  One of float, double, complex<float> or complex<double> types for generic or Gamma point case.
     */
    template <typename T, typename F>
    void
    calc_stress_nonloc_aux();

  public:
    Stress(Simulation_context& ctx__, Density& density__, Potential& potential__, K_point_set& kset__)
        : ctx_(ctx__)
        , density_(density__)
        , potential_(potential__)
        , kset_(kset__)
    {
    }

    /// Local potential contribution to stress.
    /** Energy contribution from the local part of pseudopotential:
     *  \f[
     *  E^{loc} = \int \rho({\bf r}) V^{loc}({\bf r}) d{\bf r} =
     *  \frac{1}{\Omega} \sum_{\bf G} \langle \rho | {\bf G} \rangle \langle {\bf G}| V^{loc} \rangle =
     *  \frac{1}{\Omega} \sum_{\bf G} \tilde \rho^{*}({\bf G}) \tilde
     *  V^{loc}({\bf G})
     *  \f]
     *  where
     *  \f[
     *  \tilde \rho({\bf G}) = \langle {\bf G} | \rho \rangle = \int e^{-i{\bf Gr}}\rho({\bf r}) d {\bf r}
     *  \f]
     *  and
     *  \f[
     *   \tilde V^{loc}({\bf G}) = \langle {\bf G} | V^{loc} \rangle =
     *   \int e^{-i{\bf Gr}}V^{loc}({\bf r}) d {\bf r}
     *  \f]
     *  Using the expression for \f$ \tilde V^{loc}({\bf G}) \f$, the local contribution to the total energy
     *  is rewritten as
     *  \f[
     *  E^{loc} = \frac{1}{\Omega} \sum_{\bf G} \tilde \rho^{*}({\bf G})
     *   \sum_{\alpha} e^{-{\bf G\tau}_{\alpha}} 4 \pi \int V_{\alpha}^{loc}(r)\frac{\sin(Gr)}{Gr} r^2 dr =
     *   \frac{4\pi}{\Omega}\sum_{\bf G}\tilde \rho^{*}({\bf G})\sum_{\alpha} e^{-{\bf G\tau}_{\alpha}}
     *    \Bigg( \int \Big(V_{\alpha}(r) r + Z_{\alpha}^p {\rm erf}(r) \Big)
     *   \frac{\sin(Gr)}{G} dr -  Z_{\alpha}^p \frac{e^{-\frac{G^2}{4}}}{G^2} \Bigg)
     *  \f]
     *  (see \link sirius::Potential::generate_local_potential \endlink for details).
     *
     *  Contribution to stress tensor:
     *  \f[
     *   \sigma_{\mu \nu}^{loc} = \frac{1}{\Omega} \frac{\partial  E^{loc}}{\partial \varepsilon_{\mu \nu}} =
     *   \frac{1}{\Omega} \frac{-1}{\Omega} \delta_{\mu \nu} \sum_{\bf G}\tilde \rho^{*}({\bf G}) \tilde
     *   V^{loc}({\bf G}) + \frac{4\pi}{\Omega^2} \sum_{\bf G}\tilde \rho^{*}({\bf G}) \sum_{\alpha} e^{-{\bf
     *   G\tau}_{\alpha}} \Bigg( \int \Big(V_{\alpha}(r) r + Z_{\alpha}^p {\rm erf}(r) \Big)
     *   \Big( \frac{r \cos (G r)}{G}-\frac{\sin (G r)}{G^2} \Big) \Big( -\frac{G_{\mu}G_{\nu}}{G} \Big) dr -
     *   Z_{\alpha}^p \Big(-\frac{e^{-\frac{G^2}{4}}}{2 G}-\frac{2 e^{-\frac{G^2}{4}}}{G^3} \Big)
     *   \Big( -\frac{G_{\mu}G_{\nu}}{G} \Big)  \Bigg) = \\
     *    -\delta_{\mu \nu} \sum_{\bf G}\rho^{*}({\bf G}) V^{loc}({\bf G}) + \sum_{\bf G} \rho^{*}({\bf G}) \Delta
     *   V^{loc}({\bf G}) G_{\mu}G_{\nu}
     *  \f]
     *  where \f$ \Delta V^{loc}({\bf G}) \f$ is built from the following radial integrals:
     *  \f[
     *   \int \Big(V_{\alpha}(r) r + Z_{\alpha}^p {\rm erf}(r) \Big)
     *   \Big( \frac{\sin (G r)}{G^3} - \frac{r\cos (G r)}{G^2}\Big) dr -
     *   Z_{\alpha}^p \Big( \frac{e^{-\frac{G^2}{4}}}{2 G^2} + \frac{2 e^{-\frac{G^2}{4}}}{G^4}\Big)
     *  \f]
     */
    r3::matrix<double>
    calc_stress_vloc();

    inline r3::matrix<double>
    stress_vloc() const
    {
        return stress_vloc_;
    }

    /// Hartree energy contribution to stress.
    /** Hartree energy:
     *  \f[
     *    E^{H} = \frac{1}{2} \int_{\Omega} \rho({\bf r}) V^{H}({\bf r}) d{\bf r} =
     *      \frac{1}{2} \frac{1}{\Omega} \sum_{\bf G} \langle \rho | {\bf G} \rangle \langle {\bf G}| V^{H} \rangle =
     *      \frac{2 \pi}{\Omega} \sum_{\bf G} \frac{|\tilde \rho({\bf G})|^2}{G^2}
     *  \f]
     *  where
     *  \f[
     *    \langle {\bf G} | \rho \rangle = \int e^{-i{\bf Gr}}\rho({\bf r}) d {\bf r} = \tilde \rho({\bf G})
     *  \f]
     *  and
     *  \f[
     *    \langle {\bf G} | V^{H} \rangle = \int e^{-i{\bf Gr}}V^{H}({\bf r}) d {\bf r} = \frac{4\pi}{G^2} \tilde
     * \rho({\bf G}) \f]
     *
     *  Hartree energy contribution to stress tensor:
     *  \f[
     *  \sigma_{\mu \nu}^{H} = \frac{1}{\Omega} \frac{\partial  E^{H}}{\partial \varepsilon_{\mu \nu}} =
     *   \frac{1}{\Omega} 2\pi \Big( \big( \frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{1}{\Omega} \big)
     *   \sum_{{\bf G}} \frac{|\tilde \rho({\bf G})|^2}{G^2} +
     *   \frac{1}{\Omega} \sum_{{\bf G}} |\tilde \rho({\bf G})|^2 \frac{\partial}{\partial \varepsilon_{\mu \nu}}
     *  \frac{1}{G^2} \Big) = \\ \frac{1}{\Omega} 2\pi \Big( -\frac{1}{\Omega} \delta_{\mu \nu} \sum_{{\bf G}}
     *  \frac{|\tilde \rho({\bf G})|^2}{G^2} +
     *  \frac{1}{\Omega} \sum_{\bf G} |\tilde \rho({\bf G})|^2 \sum_{\tau} \frac{-2 G_{\tau}}{G^4} \frac{\partial
     *  G_{\tau}}{\partial \varepsilon_{\mu \nu}} \Big) = \\ 2\pi \sum_{\bf G} \frac{|\rho({\bf G})|^2}{G^2} \Big(
     *  -\delta_{\mu \nu} + \frac{2}{G^2} G_{\nu} G_{\mu} \Big)
     * \f]
     */
    r3::matrix<double>
    calc_stress_har();

    inline r3::matrix<double>
    stress_har() const
    {
        return stress_har_;
    }

    /// Ewald energy contribution to stress.
    /** Ewald energy:
     *  \f[
     *  E^{ion-ion} = \frac{1}{2} \sideset{}{'} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta}
     *    \frac{{\rm erfc}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} - {\bf
     *   r}_{\beta} + {\bf T}|} + \frac{2 \pi}{\Omega} \sum_{{\bf G}} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \Big|
     *  \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 - \sum_{\alpha} Z_{\alpha}^2
     *  \sqrt{\frac{\lambda}{\pi}} - \frac{2\pi}{\Omega}\frac{N_{el}^2}{4 \lambda}
     *  \f]
     * (check \link sirius::DFT_ground_state::ewald_energy \endlink for details).\n
     * Contribution to stress tensor:
     * \f[ \sigma_{\mu \nu}^{ion-ion} = \frac{1}{\Omega} \frac{\partial
     *   E^{ion-ion}}{\partial \varepsilon_{\mu \nu}}
     * \f]
     * Derivative of the first part:
     * \f[
     *  \frac{1}{\Omega}\frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{1}{2} \sideset{}{'}
     * \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta} \frac{{\rm erfc}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf
     * r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|}  = \frac{1}{2\Omega} \sideset{}{'}
     * \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta} \Big( -2e^{-\lambda |{\bf r'}|^2} \sqrt{\frac{\lambda}{\pi}}
     * \frac{1}{|{\bf r'}|^2} - {\rm erfc}(\sqrt{\lambda} |{\bf r'}|) \frac{1}{|{\bf r'}|^3} \Big) r'_{\mu} r'_{\nu}
     * \f]
     *  where \f$ {\bf r'} = {\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T} \f$.
     *
     *  Derivative of the second part:
     *  \f[
     *   \frac{1}{\Omega}\frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{2\pi}{\Omega} \sum_{{\bf G}}
     *  \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \Big| \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 =
     *  -\frac{2\pi}{\Omega^2} \delta_{\mu \nu} \sum_{{\bf G}} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \Big|
     *  \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 +
     *  \frac{2\pi}{\Omega^2} \sum_{\bf G} G_{\mu} G_{\nu} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} 2
     * \frac{\frac{G^2}{4\lambda} + 1}{G^2} \Big| \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2
     * \f]
     *
     *  Derivative of the fourth part:
     *  \f[
     *   -\frac{1}{\Omega}\frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{2\pi}{\Omega}\frac{N_{el}^2}{4 \lambda}
     *   = \frac{2\pi}{\Omega^2}\frac{N_{el}^2}{4 \lambda} \delta_{\mu \nu}
     *  \f]
     */
    r3::matrix<double>
    calc_stress_ewald();

    inline r3::matrix<double>
    stress_ewald() const
    {
        return stress_ewald_;
    }

    /// Kinetic energy contribution to stress.
    /** Kinetic energy:
     *  \f[
     *    E^{kin} = \sum_{{\bf k}} w_{\bf k} \sum_j f_j \frac{1}{2} |{\bf G+k}|^2 |\psi_j({\bf G + k})|^2
     *  \f]
     *  Contribution to the stress tensor
     *  \f[
     *  \sigma_{\mu \nu}^{kin} = \frac{1}{\Omega} \frac{\partial E^{kin}}{\partial \varepsilon_{\mu \nu}} =
     *  \frac{1}{\Omega} \sum_{{\bf k}} w_{\bf k} \sum_j f_j \frac{1}{2} 2 |{\bf G+k}| \Big( -\frac{1}{|{\bf G+k}|}
     * (G+k)_{\mu} (G+k)_{\nu} \Big)  |\psi_j({\bf G + k})|^2 =\\
     *  -\frac{1}{\Omega} \sum_{{\bf k}} w_{\bf k} (G+k)_{\mu} (G+k)_{\nu} \sum_j f_j  |\psi_j({\bf G + k})|^2
     *  \f]
     */
    template <typename T>
    void
    calc_stress_kin_aux();

    r3::matrix<double>
    calc_stress_kin();

    inline r3::matrix<double>
    stress_kin() const
    {
        return stress_kin_;
    }

    r3::matrix<double>
    calc_stress_nonloc();

    inline r3::matrix<double>
    stress_nonloc() const
    {
        return stress_nonloc_;
    }

    /// Contribution to the stress tensor from the augmentation operator.
    /** Total energy in ultrasoft pseudopotential contains this term:
     *  \f[
     *  \int V^{eff}({\bf r})\rho^{aug}({\bf r})d{\bf r} =
     *   \sum_{\alpha} \sum_{\xi \xi'} n_{\xi \xi'}^{\alpha} \int V^{eff}({\bf r}) Q_{\xi \xi'}^{\alpha}({\bf r})
     *   d{\bf r}
     * \f]
     * The derivatives of beta-projectors (hidden in the desnity matrix expression) are taken into account
     * in \link sirius::Stress::calc_stress_nonloc \endlink. Here we need to compute the remaining contribution
     * from the
     * \f$ Q_{\xi \xi'}({\bf r}) \f$ itself. We are interested in the integral:
     * \f[
     * \int V^{eff}({\bf r}) Q_{\xi \xi'}^{\alpha}({\bf r}) d{\bf r} =
     *  \sum_{\bf G} V^{eff}({\bf G}) \tilde Q_{\xi \xi'}^{\alpha}({\bf G})
     *  \f]
     * where
     *  \f[
     *     \tilde Q_{\xi \xi'}^{\alpha}({\bf G}) = \int e^{-i{\bf Gr}} Q_{\xi \xi'}^{\alpha}({\bf r}) d{\bf r} =
     *      4\pi \sum_{\ell m} (-i)^{\ell} R_{\ell m}(\hat{\bf G}) \langle R_{\ell_{\xi} m_{\xi}} | R_{\ell m} |
     *  R_{\ell_{\xi'} m_{\xi'}} \rangle \int Q_{\ell_{\xi} \ell_{\xi'}}^{\ell}(r) j_{\ell}(Gr) r^2 dr
     * \f]
     * Strain derivative of \f$ \tilde Q_{\xi \xi'}^{\alpha}({\bf G}) \f$ is:
     * \f[
     *  \frac{\partial}{\partial \varepsilon_{\mu \nu}} \tilde Q_{\xi \xi'}^{\alpha}({\bf G}) =
     *   4\pi \sum_{\ell m} (-i)^{\ell} \langle R_{\ell_{\xi} m_{\xi}} |
     *   R_{\ell m} | R_{\ell_{\xi'} m_{\xi'}} \rangle
     *  \Big( \frac{\partial R_{\ell m}(\hat{\bf G})}{\partial \varepsilon_{\mu \nu}}
     *  \int Q_{\ell_{\xi} \ell_{\xi'}}^{\ell}(r) j_{\ell}(Gr) r^2 dr + R_{\ell m}(\hat{\bf G})
     *  \int Q_{\ell_{\xi} \ell_{\xi'}}^{\ell}(r)
     *  \frac{\partial j_{\ell}(Gr)}{\partial \varepsilon_{\mu \nu}} r^2 dr \Big)
     * \f]
     * For strain derivatives of spherical harmonics and Bessel functions see
     * \link sirius::Stress::calc_stress_nonloc \endlink. We can pull the common multiplier \f$ -G_{\mu} / G \f$
     * from both terms and arrive to the following expression:
     * \f[
     *  \frac{\partial}{\partial \varepsilon_{\mu \nu}} \tilde Q_{\xi \xi'}^{\alpha}({\bf G}) =
     *      -\frac{G_{\mu}}{G}  4\pi \sum_{\ell m} (-i)^{\ell} \langle R_{\ell_{\xi} m_{\xi}} | R_{\ell m} |
     *   R_{\ell_{\xi'} m_{\xi'}} \rangle \Big( \big(\nabla_{G} R_{\ell m}(\hat{\bf G})\big)_{\nu} \int Q_{\ell_{\xi}
     *  \ell_{\xi'}}^{\ell}(r) j_{\ell}(Gr) r^2 dr + R_{\ell m}(\hat{\bf G}) \int Q_{\ell_{\xi} \ell_{\xi'}}^{\ell}(r)
     *  \frac{\partial j_{\ell}(Gr)}{\partial G} G_{\nu} r^2 dr \Big)
     * \f]
     */
    r3::matrix<double>
    calc_stress_us();

    inline auto
    stress_us() const
    {
        return stress_us_;
    }

    inline auto
    stress_us_nl() const
    {
        return stress_nonloc_ + stress_us_;
    }

    /// XC contribution to stress.
    /** XC contribution has the following expression:
     *  \f[
     *  \frac{\partial E_{xc}}{\partial \varepsilon_{\mu \nu}} = \delta_{\mu \nu} \int \Big( \epsilon_{xc}({\bf r}) -
     *  v_{xc}({\bf r}) \Big) \rho({\bf r})d{\bf r} - \int \frac{\partial \epsilon_{xc} \big( \rho({\bf r}), \nabla
     *  \rho({\bf r})\big) }{\nabla_{\mu} \rho({\bf r})} \nabla_{\nu}\rho({\bf r}) d{\bf r}
     *  \f]
     */
    r3::matrix<double>
    calc_stress_xc();

    inline auto
    stress_xc() const
    {
        return stress_xc_;
    }

    /// Non-linear core correction to stress tensor.
    r3::matrix<double>
    calc_stress_core();

    inline auto
    stress_core() const
    {
        return stress_core_;
    }

    r3::matrix<double>
    calc_stress_hubbard();

    inline auto
    stress_hubbard() const
    {
        return stress_hubbard_;
    }

    r3::matrix<double>
    calc_stress_total();

    inline auto
    stress_total() const
    {
        return stress_total_;
    }

    void
    print_info(std::ostream& out__, int verbosity__) const;
};

} // namespace sirius

#endif
