// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file stress.hpp
 *
 *  \brief Stress tensor calculation.
 */

#ifndef __STRESS_HPP__
#define __STRESS_HPP__

#include "../Beta_projectors/beta_projectors_strain_deriv.h"
#include "non_local_functor.hpp"

namespace sirius {

/// Stress tensor.
/** The following referenceces were particularly useful in the derivation of the stress tensor components:
 *    - Hutter, D. M. A. J. (2012). Ab Initio Molecular Dynamics (pp. 1–580).
 *    - Marx, D., & Hutter, J. (2000). Ab initio molecular dynamics: Theory and implementation.
 *      Modern Methods and Algorithms of Quantum Chemistry.
 *    - Knuth, F., Carbogno, C., Atalla, V., & Blum, V. (2015). All-electron formalism for total energy strain derivatives 
 *      and stress tensor components for numeric atom-centered orbitals. Computer Physics Communications.
 *    - Willand, A., Kvashnin, Y. O., Genovese, L., Vázquez-Mayagoitia, Á., Deb, A. K., Sadeghi, A., et al. (2013).
 *      Norm-conserving pseudopotentials with chemical accuracy compared to all-electron calculations.
 *      The Journal of Chemical Physics, 138(10), 104109. http://doi.org/10.1103/PhysRevB.50.4327
 *    - Corso, A. D., & Resta, R. (1994). Density-functional theory of macroscopic stress: Gradient-corrected calculations 
 *      for crystalline Se. Physical Review B.
 *
 * Stress tensor describes a reaction of crystall to a strain:
 *  \f[
 *    \sigma_{\mu \nu} = \frac{1}{\Omega} \frac{\partial  E}{\partial \varepsilon_{\mu \nu}}
 *  \f]
 *  where \f$ \varepsilon_{\mu \nu} \f$ is a symmetric strain tensor which describes the infinitesimal deformation of a crystal:
 *  \f[
 *    r_{\mu} \rightarrow \sum_{\nu} (\delta_{\mu \nu} + \varepsilon_{\mu \nu}) r_{\nu}
 *  \f]
 *  The following expressions are helpful:
 *    - strain derivative of a general position vector component:
 *  \f[
 *    \frac{\partial r_{\tau}}{\partial \varepsilon_{\mu \nu}} = \delta_{\tau \mu} r_{\nu}
 *  \f]
 *    - strain derivative of the lengths of a general position vector:
 *  \f[
 *    \frac{\partial |{\bf r}|}{\partial \varepsilon_{\mu \nu}} = \sum_{\tau} \frac{\partial |{\bf r}|}{\partial r_{\tau}} 
 *      \frac{\partial r_{\tau}}{\partial \varepsilon_{\mu \nu}} = \sum_{\tau} \frac{ r_{\tau}}{|{\bf r}|}
 *      \delta_{\tau \mu} r_{\nu} = \frac{r_{\mu} r_{\nu}}{|{\bf r}|} 
 *  \f]
 *    - strain derivative of the unit cell volume:
 *  \f[
 *    \frac{\partial \Omega}{\partial \varepsilon_{\mu \nu}} = \delta_{\mu \nu} \Omega
 *  \f]
 *    - strain derivative of the inverse square root of the unit cell volume:
 *  \f[
 *    \frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{1}{\sqrt{\Omega}} = -\frac{1}{2}\frac{1}{\sqrt{\Omega}} \delta_{\mu \nu}
 *  \f]
 *    - strain derivative of a reciprocal vector:
 *  \f[
 *    \frac{\partial G_{\tau}}{\partial \varepsilon_{\mu \nu}} = -\delta_{\tau \nu} G_{\mu}
 *  \f]
 *    - strain derivative of the length of a reciprocal vector:
 *  \f[
 *    \frac{\partial |{\bf G}|}{\partial \varepsilon_{\mu \nu}} = \sum_{\tau} \frac{\partial |{\bf G}|}{\partial G_{\tau}}
 *      \frac{\partial G_{\tau}}{\partial \varepsilon_{\mu \nu}} = -\frac{1}{|{\bf G}|}G_{\nu}G_{\mu}
 *  \f]
 *  In the derivation of the stress tensor contributions it is important to know which variables are invariant under lattice
 *  distortion. This are:
 *    - scalar product of the real-space (in the Bravais lattice frame) and reciprocal (in the reciprocal lattice frame) vectors
 *    - normalized plane-wave coefficients of the wave-functions
 *  \f[
 *    \psi({\bf G}) = \frac{1}{\sqrt{\Omega}} \int e^{-i {\bf G}{\bf r}} \psi({\bf r}) d{\bf r}
 *  \f]
 *    - unnormalized plane-wave coefficients of the charge density
 *  \f[
 *    \tilde \rho({\bf G}) = \int e^{-i {\bf G}{\bf r}} \sum_{j} |\psi_j({\bf r})|^{2} d{\bf r}
 *  \f]
 */
class Stress {
  private:
    Simulation_context& ctx_;
    
    K_point_set& kset_;

    Density& density_;

    Potential& potential_;

    matrix3d<double> stress_kin_;

    matrix3d<double> stress_har_;

    matrix3d<double> stress_ewald_;

    matrix3d<double> stress_vloc_;

    matrix3d<double> stress_nonloc_;

    matrix3d<double> stress_us_;

    matrix3d<double> stress_xc_;

    matrix3d<double> stress_core_;
    
    /// Non-local contribution to stress.
    /** Energy contribution from the non-local part of pseudopotential:
     *  \f[
     *    E^{nl} = \sum_{i{\bf k}} \sum_{\alpha}\sum_{\xi \xi'}P_{\xi}^{\alpha,i{\bf k}} D_{\xi\xi'}^{\alpha}P_{\xi'}^{\alpha,i{\bf k}*}
     *  \f]
     *  where
     *  \f[
     *    P_{\xi}^{\alpha,i{\bf k}} = \langle \psi_{i{\bf k}} | \beta_{\xi}^{\alpha} \rangle =
     *      \frac{1}{\Omega} \sum_{\bf G} \langle \psi_{i{\bf k}} | {\bf G+k} \rangle \langle {\bf G+k} | \beta_{\xi}^{\alpha} \rangle  = 
     *      \sum_{\bf G} \psi_i^{*}({\bf G+k}) \beta_{\xi}^{\alpha}({\bf G+k}) 
     *  \f]
     *  where
     *  \f[
     *     \beta_{\xi}^{\alpha}({\bf G+k}) = \frac{1}{\sqrt{\Omega}} \int e^{-i({\bf G+k}){\bf r}} \beta_{\xi}^{\alpha}({\bf r}) d{\bf r} = 
     *      \frac{4\pi}{\sqrt{\Omega}} e^{-i{\bf (G+k) r_{\alpha}}}(-i)^{\ell} R_{\ell m}(\theta_{G+k}, \phi_{G+k})
     *      \int \beta_{\ell}(r) j_{\ell}(|{\bf G+k}|r) r^2 dr
     *  \f]
     *  Contribution to stress tensor:
     *  \f[
     *     \sigma_{\mu \nu}^{nl} = \frac{1}{\Omega} \frac{\partial  E^{nl}}{\partial \varepsilon_{\mu \nu}} =
     *       \frac{1}{\Omega}  \sum_{i{\bf k}} \sum_{\xi \xi'} \Bigg( 
     *       \frac{\partial P_{\xi}^{\alpha,i{\bf k}}}{\partial \varepsilon_{\mu \nu}} D_{\xi\xi'}^{\alpha}P_{\xi'}^{\alpha,i{\bf k}*} + 
     *        P_{\xi}^{\alpha,i{\bf k}} D_{\xi\xi'}^{\alpha} \frac{\partial P_{\xi'}^{\alpha,i{\bf k}*}}{\partial \varepsilon_{\mu \nu}} \Bigg)
     *  \f]
     *  We need to compute strain derivatives of \f$ P_{\xi}^{\alpha,i{\bf k}} \f$:
     *  \f[
     *    \frac{\partial}{\partial \varepsilon_{\mu \nu}} P_{\xi}^{\alpha,i{\bf k}} = 
     *      \sum_{\bf G} \psi_i^{*}({\bf G+k}) \frac{\partial}{\partial \varepsilon_{\mu \nu}} \beta_{\xi}^{\alpha}({\bf G+k})  
     *  \f]
     *  First way to take strain derivative of beta-projectors (here and below \f$ {\bf G+k} = {\bf q} \f$):
     *  \f[
     *    \frac{\partial}{\partial \varepsilon_{\mu \nu}} \beta_{\xi}^{\alpha}({\bf q}) = 
     *    -\frac{4\pi}{2\sqrt{\Omega}} \delta_{\mu \nu} e^{-i{\bf q r_{\alpha}}}(-i)^{\ell} R_{\ell m}(\theta_q, \phi_q)
     *      \int \beta_{\ell}(r) j_{\ell}(q r) r^2 dr + \\
     *    \frac{4\pi}{\sqrt{\Omega}} e^{-i{\bf q r_{\alpha}}}(-i)^{\ell} \frac{\partial R_{\ell m}(\theta_q, \phi_q)}{\partial \varepsilon_{\mu \nu}}
     *      \int \beta_{\ell}(r) j_{\ell}(q r) r^2 dr + \\
     *    \frac{4\pi}{\sqrt{\Omega}} e^{-i{\bf q r_{\alpha}}}(-i)^{\ell} R_{\ell m}(\theta_q, \phi_q)
     *       \int \beta_{\ell}(r) \frac{\partial j_{\ell}(q r)}{\partial \varepsilon_{\mu \nu}} r^2 dr = \\
     *    \frac{4\pi}{\sqrt{\Omega}} e^{-i{\bf q r_{\alpha}}}(-i)^{\ell} \Bigg[ \int \beta_{\ell}(r) j_{\ell}(q r) r^2 dr
     *       \Big(\frac{\partial R_{\ell m}(\theta_q, \phi_q)}{\partial \varepsilon_{\mu \nu}} - \frac{1}{2} R_{\ell m}(\theta_q, \phi_q) \delta_{\mu \nu}\Big) +
     *        R_{\ell m}(\theta_q, \phi_q)  \int \beta_{\ell}(r) \frac{\partial j_{\ell}(q r)}{\partial \varepsilon_{\mu \nu}} r^2 dr 
     *     \Bigg]
     *  \f]
     *  
     *  Strain derivative of the real spherical harmonics:
     *  \f[
     *    \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \varepsilon_{\mu \nu}} = 
     *      \sum_{\tau} \frac{\partial R_{\ell m}(\theta, \phi)}{\partial q_{\tau}} \frac{\partial q_{\tau}}{\partial \varepsilon_{\mu \nu}} =
     *      -q_{\mu} \frac{\partial R_{\ell m}(\theta, \phi)}{\partial q_{\nu}} 
     *  \f]
     *  For the derivatives of spherical harmonics over Cartesian components of vector please refer to the SHT::dRlm_dr function.
     *  
     *  Strain derivative of spherical Bessel function integral:
     *  \f[
     *    \int \beta_{\ell}(r) \frac{\partial j_{\ell}(qr) }{\partial \varepsilon_{\mu \nu}}  r^2 dr = 
     *     \int \beta_{\ell}(r) \frac{\partial j_{\ell}(qr) }{\partial q} \frac{-q_{\mu} q_{\nu}}{q} r^2 dr  
     *  \f]
     *  The second way to compute strain derivative of beta-projectors is trough Gaunt coefficients:
     * \f[
     *    \frac{\partial}{\partial \varepsilon_{\mu \nu}} \beta_{\xi}^{\alpha}({\bf q}) = 
     *    -\frac{1}{2\sqrt{\Omega}} \delta_{\mu \nu} \int e^{-i{\bf q r}} \beta_{\xi}^{\alpha}({\bf r}) d{\bf r} + 
     *    \frac{1}{\sqrt{\Omega}} \int i r_{\nu} q_{\mu} e^{-i{\bf q r}} \beta_{\xi}^{\alpha}({\bf r}) d{\bf r}
     * \f]
     * (second term comes from the strain derivative of \f$ e^{-i{\bf q r}} \f$). Remembering that \f$ {\bf r} \f$ is
     * proportional to p-like real spherical harmonics, we can rewrite the second part of beta-projector derivative as:
     * \f[
     *   \frac{1}{\sqrt{\Omega}} \int i r_{\nu} q_{\mu} e^{-i{\bf q r}} \beta_{\xi}^{\alpha}({\bf r}) d{\bf r} = 
     *    \frac{1}{\sqrt{\Omega}} i q_{\mu} \int r \bar R_{1 \nu}(\theta, \phi) 4\pi \sum_{\ell_3 m_3} (-i)^{\ell_3} R_{\ell_3 m_3}(\theta_q, \phi_q)
     *    R_{\ell_3 m_3}(\theta, \phi) j_{\ell_3}(q r) \beta_{\ell_2}^{\alpha}(r) R_{\ell_2 m_2}(\theta, \phi) d{\bf r} = \\
     *     \frac{4 \pi}{\sqrt{\Omega}} i q_{\mu} \sum_{\ell_3 m_3} (-i)^{\ell_3} R_{\ell_3 m_3}(\theta_q, \phi_q) 
     *     \langle \bar R_{1\nu} | R_{\ell_3 m_3} | R_{\ell_2 m_2} \rangle
     *     \int j_{\ell_3}(q r) \beta_{\ell_2}^{\alpha}(r) r^3 dr 
     * \f]
     * where
     * \f[
     *   \bar R_{1 x}(\theta, \phi) = -2 \sqrt{\frac{\pi}{3}} R_{11}(\theta, \phi) = \sin(\theta) \cos(\phi) \\
     *   \bar R_{1 y}(\theta, \phi) = -2 \sqrt{\frac{\pi}{3}} R_{1-1}(\theta,\phi) = \sin(\theta) \sin(\phi) \\
     *   \bar R_{1 z}(\theta, \phi) = 2 \sqrt{\frac{\pi}{3}} R_{10}(\theta, \phi) = \cos(\theta)
     * \f]
     */
    template <typename T>
    inline void calc_stress_nonloc_aux()
    {
        PROFILE("sirius::Stress|nonloc");

        mdarray<double, 2> collect_result(9, ctx_.unit_cell().num_atoms());
        collect_result.zero();

        stress_nonloc_.zero();

        for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
            int ik = kset_.spl_num_kpoints(ikloc);
            auto kp = kset_[ik];
            if (kp->gkvec().reduced()) {
                TERMINATE("reduced G+k vectors are not implemented for non-local stress: fix this");
            }

#ifdef __GPU
            if (ctx_.processing_unit() == GPU && !keep_wf_on_gpu) {
                int nbnd = ctx_.num_bands();
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    /* allocate GPU memory */
                    kp->spinor_wave_functions().pw_coeffs(ispn).allocate_on_device();
                    kp->spinor_wave_functions().pw_coeffs(ispn).copy_to_device(0, nbnd);
                }
            }
#endif

            Beta_projectors_strain_deriv bp_strain_deriv(ctx_, kp->gkvec(), kp->igk_loc());

            Non_local_functor<T, 9> nlf(ctx_, bp_strain_deriv);

            nlf.add_k_point_contribution(*kp, collect_result);

#ifdef __GPU
            if (ctx_.processing_unit() == GPU && !keep_wf_on_gpu) {
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    /* deallocate GPU memory */
                    kp->spinor_wave_functions().pw_coeffs(ispn).deallocate_on_device();
                }
            }
#endif
        }

        #pragma omp parallel
        {
            matrix3d<double> tmp_stress;

            #pragma omp for
            for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        tmp_stress(i, j) -= collect_result(j * 3 + i, ia);
                    }
                }
            }

            #pragma omp critical
            stress_nonloc_ += tmp_stress;
        }

        ctx_.comm().allreduce(&stress_nonloc_(0, 0), 9);

        stress_nonloc_ *= (1.0 / ctx_.unit_cell().omega());

        symmetrize(stress_nonloc_);
    }

    inline void symmetrize(matrix3d<double>& mtrx__) const
    {
        if (!ctx_.use_symmetry()) {
            return;
        }

        matrix3d<double> result;

        for (int i = 0; i < ctx_.unit_cell().symmetry().num_mag_sym(); i++) {
            auto R = ctx_.unit_cell().symmetry().magnetic_group_symmetry(i).spg_op.rotation;
            result = result + transpose(R) * mtrx__ * R;
        }

        mtrx__ = result * (1.0 / ctx_.unit_cell().symmetry().num_mag_sym());

        std::vector<std::array<int, 2>> idx = {{0, 1}, {0, 2}, {1, 2}};
        for (auto e: idx) {
            mtrx__(e[0], e[1]) = mtrx__(e[1], e[0]) = 0.5 * (mtrx__(e[0], e[1]) + mtrx__(e[1], e[0]));
        }
    }

  public:
    Stress(Simulation_context& ctx__,
           K_point_set& kset__,
           Density& density__,
           Potential& potential__)
        : ctx_(ctx__)
        , kset_(kset__)
        , density_(density__)
        , potential_(potential__)
    {
    }

    /// Local potential contribution to stress.
    /** Energy contribution from the local part of pseudopotential:
     *  \f[
     *    E^{loc} = \int \rho({\bf r}) V^{loc}({\bf r}) d{\bf r} = \frac{1}{\Omega} \sum_{\bf G} \langle \rho | {\bf G} \rangle \langle {\bf G}| V^{loc} \rangle = 
     *      \frac{1}{\Omega} \sum_{\bf G} \tilde \rho^{*}({\bf G}) \tilde V^{loc}({\bf G})
     *  \f]
     *  where
     *  \f[
     *    \tilde \rho({\bf G}) = \langle {\bf G} | \rho \rangle = \int e^{-i{\bf Gr}}\rho({\bf r}) d {\bf r}
     *  \f]
     *  and
     *  \f[
     *    \tilde V^{loc}({\bf G}) = \langle {\bf G} | V^{loc} \rangle = \int e^{-i{\bf Gr}}V^{loc}({\bf r}) d {\bf r} 
     *  \f]
     *  Using the expression for \f$ \tilde V^{loc}({\bf G}) \f$, the local contribution to the total energy is rewritten as
     *  \f[
     *    E^{loc} = \frac{1}{\Omega} \sum_{\bf G} \tilde \rho^{*}({\bf G}) \sum_{\alpha} e^{-{\bf G\tau}_{\alpha}} 4 \pi 
     *      \int V_{\alpha}^{loc}(r)\frac{\sin(Gr)}{Gr} r^2 dr = 
     *      \frac{4\pi}{\Omega}\sum_{\bf G}\tilde \rho^{*}({\bf G})\sum_{\alpha} e^{-{\bf G\tau}_{\alpha}} 
     *      \Bigg( \int \Big(V_{\alpha}(r) r + Z_{\alpha}^p {\rm erf}(r) \Big) \frac{\sin(Gr)}{G} dr -  Z_{\alpha}^p \frac{e^{-\frac{G^2}{4}}}{G^2} \Bigg)
     *  \f]
     *  (see \link sirius::Potential::generate_local_potential \endlink for details).
     *  
     *  Contribution to stress tensor:
     *  \f[
     *     \sigma_{\mu \nu}^{loc} = \frac{1}{\Omega} \frac{\partial  E^{loc}}{\partial \varepsilon_{\mu \nu}} =
     *       \frac{1}{\Omega} \frac{-1}{\Omega} \delta_{\mu \nu} \sum_{\bf G}\tilde \rho^{*}({\bf G}) \tilde V^{loc}({\bf G}) + 
     *       \frac{4\pi}{\Omega^2} \sum_{\bf G}\tilde \rho^{*}({\bf G}) \sum_{\alpha} e^{-{\bf G\tau}_{\alpha}}
     *       \Bigg( \int \Big(V_{\alpha}(r) r + Z_{\alpha}^p {\rm erf}(r) \Big) \Big( \frac{r \cos (G r)}{G}-\frac{\sin (G r)}{G^2} \Big)
     *       \Big( -\frac{G_{\mu}G_{\nu}}{G} \Big) dr -  Z_{\alpha}^p \Big( -\frac{e^{-\frac{G^2}{4}}}{2 G}-\frac{2 e^{-\frac{G^2}{4}}}{G^3} \Big) 
     *       \Big( -\frac{G_{\mu}G_{\nu}}{G} \Big)  \Bigg) = \\
     *       -\delta_{\mu \nu} \sum_{\bf G}\rho^{*}({\bf G}) V^{loc}({\bf G}) + \sum_{\bf G} \rho^{*}({\bf G}) \Delta V^{loc}({\bf G}) G_{\mu}G_{\nu}
     *  \f]
     *  where \f$ \Delta V^{loc}({\bf G}) \f$ is built from the following radial integrals:
     *  \f[
     *    \int \Big(V_{\alpha}(r) r + Z_{\alpha}^p {\rm erf}(r) \Big) \Big( \frac{\sin (G r)}{G^3} - \frac{r \cos (G r)}{G^2}\Big) dr - 
     *      Z_{\alpha}^p \Big( \frac{e^{-\frac{G^2}{4}}}{2 G^2} + \frac{2 e^{-\frac{G^2}{4}}}{G^4} \Big)  
     *  \f]
     */
    inline void calc_stress_vloc()
    {
        PROFILE("sirius::Stress|vloc");

        stress_vloc_.zero();

        Radial_integrals_vloc<false> ri_vloc(ctx_.unit_cell(), ctx_.pw_cutoff(), ctx_.settings().nprii_vloc_);
        Radial_integrals_vloc<true> ri_vloc_dg(ctx_.unit_cell(), ctx_.pw_cutoff(), ctx_.settings().nprii_vloc_);

        auto v = ctx_.make_periodic_function<index_domain_t::local>([&ri_vloc](int iat, double g)
                                                                    {
                                                                        return ri_vloc.value(iat, g);
                                                                    });

        auto dv = ctx_.make_periodic_function<index_domain_t::local>([&ri_vloc_dg](int iat, double g)
                                                                     {
                                                                         return ri_vloc_dg.value(iat, g);
                                                                     });
        
        double sdiag{0};

        int ig0 = (ctx_.comm().rank() == 0) ? 1 : 0;
        for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {

            int ig = ctx_.gvec().offset() + igloc;
            
            auto G = ctx_.gvec().gvec_cart(ig);

            for (int mu: {0, 1, 2}) {
                for (int nu: {0, 1, 2}) {
                    stress_vloc_(mu, nu) += std::real(std::conj(density_.rho().f_pw_local(igloc)) * dv[igloc]) * G[mu] * G[nu];
                }
            }

            sdiag += std::real(std::conj(density_.rho().f_pw_local(igloc)) * v[igloc]);
        }
        
        if (ctx_.gvec().reduced()) {
            stress_vloc_ *= 2;
            sdiag *= 2;
        }
        if (ctx_.comm().rank() == 0) {
            sdiag += std::real(std::conj(density_.rho().f_pw_local(0)) * v[0]);
        }

        for (int mu: {0, 1, 2}) {
            stress_vloc_(mu, mu) -= sdiag;
        }

        ctx_.comm().allreduce(&stress_vloc_(0, 0), 9);

        symmetrize(stress_vloc_);
    }

    inline matrix3d<double> stress_vloc() const
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
     *    \langle {\bf G} | V^{H} \rangle = \int e^{-i{\bf Gr}}V^{H}({\bf r}) d {\bf r} = \frac{4\pi}{G^2} \tilde \rho({\bf G})
     *  \f]
     *  
     *  Hartree energy contribution to stress tensor:
     *  \f[
     *     \sigma_{\mu \nu}^{H} = \frac{1}{\Omega} \frac{\partial  E^{H}}{\partial \varepsilon_{\mu \nu}} = 
     *        \frac{1}{\Omega} 2\pi \Big( \big( \frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{1}{\Omega} \big) 
     *          \sum_{{\bf G}} \frac{|\tilde \rho({\bf G})|^2}{G^2} + 
     *          \frac{1}{\Omega} \sum_{{\bf G}} |\tilde \rho({\bf G})|^2 \frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{1}{G^2} \Big) = \\
     *     \frac{1}{\Omega} 2\pi \Big( -\frac{1}{\Omega} \delta_{\mu \nu} \sum_{{\bf G}} \frac{|\tilde \rho({\bf G})|^2}{G^2} + 
     *       \frac{1}{\Omega} \sum_{\bf G} |\tilde \rho({\bf G})|^2 \sum_{\tau} \frac{-2 G_{\tau}}{G^4} \frac{\partial G_{\tau}}{\partial \varepsilon_{\mu \nu}} \Big) = \\
     *     2\pi \sum_{\bf G} \frac{|\rho({\bf G})|^2}{G^2} \Big( -\delta_{\mu \nu} + \frac{2}{G^2} G_{\nu} G_{\mu} \Big)
     *  \f]
     */
    inline void calc_stress_har()
    {
        PROFILE("sirius::Stress|har");

        stress_har_.zero();
        
        int ig0 = (ctx_.comm().rank() == 0) ? 1 : 0;
        for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;

            auto G = ctx_.gvec().gvec_cart(ig);
            double g2 = std::pow(G.length(), 2);
            auto z = density_.rho().f_pw_local(igloc);
            double d = twopi * (std::pow(z.real(), 2) + std::pow(z.imag(), 2)) / g2;

            for (int mu: {0, 1, 2}) {
                for (int nu: {0, 1, 2}) {
                    stress_har_(mu, nu) += d * 2 * G[mu] * G[nu] / g2;
                }
            }
            for (int mu: {0, 1, 2}) {
                stress_har_(mu, mu) -= d;
            }
        }

        if (ctx_.gvec().reduced()) {
            stress_har_ *= 2;
        } 

        ctx_.comm().allreduce(&stress_har_(0, 0), 9);

        symmetrize(stress_har_);
    }


    inline matrix3d<double> stress_har() const
    {
        return stress_har_;
    }

    /// Ewald energy contribution to stress.
    /** Ewald energy:
     *  \f[
     *    E^{ion-ion} = \frac{1}{2} \sideset{}{'} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta} 
     *      \frac{{\rm erfc}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|} +
     *      \frac{2 \pi}{\Omega} \sum_{{\bf G}} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \Big| \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 -
     *      \sum_{\alpha} Z_{\alpha}^2 \sqrt{\frac{\lambda}{\pi}} - \frac{2\pi}{\Omega}\frac{N_{el}^2}{4 \lambda}
     *  \f]
     * (check sirius::DFT_ground_state::ewald_energy for details).\n
     *  Contribution to stress tensor:
     *  \f[
     *    \sigma_{\mu \nu}^{ion-ion} = \frac{1}{\Omega} \frac{\partial  E^{ion-ion}}{\partial \varepsilon_{\mu \nu}} 
     *  \f]
     *  Derivative of the first part:
     *  \f[
     *    \frac{1}{\Omega}\frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{1}{2} \sideset{}{'} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta} 
     *      \frac{{\rm erfc}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|}  = 
     *      \frac{1}{2\Omega} \sideset{}{'} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta} \Big( -2e^{-\lambda |{\bf r'}|^2} 
     *      \sqrt{\frac{\lambda}{\pi}} \frac{1}{|{\bf r'}|^2} - {\rm erfc}(\sqrt{\lambda} |{\bf r'}|) \frac{1}{|{\bf r'}|^3} \Big) r'_{\mu} r'_{\nu}
     *  \f]
     *  where \f$ {\bf r'} = {\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T} \f$.
     *  
     *  Derivative of the second part:
     *  \f[
     *    \frac{1}{\Omega}\frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{2\pi}{\Omega} \sum_{{\bf G}} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} 
     *    \Big| \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 = 
     *    -\frac{2\pi}{\Omega^2} \delta_{\mu \nu} \sum_{{\bf G}} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \Big| 
     *    \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 + 
     *    \frac{2\pi}{\Omega^2} \sum_{\bf G} G_{\mu} G_{\nu} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} 2 \frac{\frac{G^2}{4\lambda} + 1}{G^2}  
     *    \Big| \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 
     *  \f]
     *  
     *  Derivative of the fourth part:
     *  \f[
     *    -\frac{1}{\Omega}\frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{2\pi}{\Omega}\frac{N_{el}^2}{4 \lambda} = 
     *       \frac{2\pi}{\Omega^2}\frac{N_{el}^2}{4 \lambda} \delta_{\mu \nu}
     *  \f]
     */
    inline void calc_stress_ewald()
    {
        PROFILE("sirius::Stress|ewald");

        stress_ewald_.zero();

        double lambda = ctx_.ewald_lambda();

        auto& uc = ctx_.unit_cell();

        int ig0 = (ctx_.comm().rank() == 0) ? 1 : 0;
        for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;

            auto G = ctx_.gvec().gvec_cart(ig);
            double g2 = std::pow(G.length(), 2);
            double g2lambda = g2 / 4.0 / lambda;

            double_complex rho(0, 0);

            for (int ia = 0; ia < uc.num_atoms(); ia++) {
                rho += ctx_.gvec_phase_factor(ig, ia) * static_cast<double>(uc.atom(ia).zn());
            }

            double a1 = twopi * std::pow(std::abs(rho) / uc.omega(), 2) * std::exp(-g2lambda) / g2;
            
            for (int mu: {0, 1, 2}) {
                for (int nu: {0, 1, 2}) {
                    stress_ewald_(mu, nu) += a1 * G[mu] * G[nu] * 2 * (g2lambda + 1) / g2;
                }
            }

            for (int mu: {0, 1, 2}) {
                stress_ewald_(mu, mu) -= a1;
            }
        }

        if (ctx_.gvec().reduced()) {
            stress_ewald_ *= 2;
        } 

        ctx_.comm().allreduce(&stress_ewald_(0, 0), 9);
        
        for (int mu: {0, 1, 2}) {
            stress_ewald_(mu, mu) += twopi * std::pow(uc.num_electrons() / uc.omega(), 2) / 4 / lambda;
        }

        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            for (int i = 1; i < uc.num_nearest_neighbours(ia); i++) {
                int ja = uc.nearest_neighbour(i, ia).atom_id;
                double d = uc.nearest_neighbour(i, ia).distance;

                vector3d<double> v1 = uc.atom(ja).position() - uc.atom(ia).position() +
                                      vector3d<int>(uc.nearest_neighbour(i, ia).translation);
                auto r1 = uc.get_cartesian_coordinates(v1);
                double len = r1.length();

                if (std::abs(d - len) > 1e-12) {
                    STOP(); 
                }

                double a1 = (0.5 * uc.atom(ia).zn() * uc.atom(ja).zn() / uc.omega() / std::pow(len, 3)) *
                            (-2 * std::exp(-lambda * std::pow(len, 2)) * std::sqrt(lambda / pi) * len - gsl_sf_erfc(std::sqrt(lambda) * len));

                for (int mu: {0, 1, 2}) {
                    for (int nu: {0, 1, 2}) {
                        stress_ewald_(mu, nu) += a1 * r1[mu] * r1[nu];
                    }
                }
            }
        }

        symmetrize(stress_ewald_);
    }

    inline matrix3d<double> stress_ewald() const
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
     *    \sigma_{\mu \nu}^{kin} = \frac{1}{\Omega} \frac{\partial E^{kin}}{\partial \varepsilon_{\mu \nu}} = 
     *     \frac{1}{\Omega} \sum_{{\bf k}} w_{\bf k} \sum_j f_j \frac{1}{2} 2 |{\bf G+k}| \Big( -\frac{1}{|{\bf G+k}|}  (G+k)_{\mu} (G+k)_{\nu} \Big)  |\psi_j({\bf G + k})|^2 =\\
     *     -\frac{1}{\Omega} \sum_{{\bf k}} w_{\bf k} (G+k)_{\mu} (G+k)_{\nu} \sum_j f_j  |\psi_j({\bf G + k})|^2 
     *  \f]
     */
    inline void calc_stress_kin()
    {
        PROFILE("sirius::Stress|kin");

        stress_kin_.zero();

        for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
            int ik = kset_.spl_num_kpoints(ikloc);
            auto kp = kset_[ik];
            if (kp->gkvec().reduced()) {
                TERMINATE("fix this");
            }

            for (int igloc = 0; igloc < kp->num_gkvec_loc(); igloc++) {
                int ig = kp->idxgk(igloc);
                auto Gk = kp->gkvec().gkvec_cart(ig);

                double d{0};
                for (int ispin = 0; ispin < ctx_.num_spins(); ispin++ ) {
                    for (int i = 0; i < kp->num_occupied_bands(ispin); i++) {
                        double f = kp->band_occupancy(i, ispin);
                        auto z = kp->spinor_wave_functions().pw_coeffs(ispin).prime(igloc, i);
                        d += f * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
                    }
                }
                d *= kp->weight();
                for (int mu: {0, 1, 2}) {
                    for (int nu: {0, 1, 2}) {
                        stress_kin_(mu, nu) += Gk[mu] * Gk[nu] * d;
                    }
                }
            } // igloc
        } // ikloc

        ctx_.comm().allreduce(&stress_kin_(0, 0), 9);

        stress_kin_ *= (-1.0 / ctx_.unit_cell().omega());

        symmetrize(stress_kin_);
    }

    inline matrix3d<double> stress_kin() const
    {
        return stress_kin_;
    }

    inline void calc_stress_nonloc()
    {
        if (ctx_.gamma_point()) {
            calc_stress_nonloc_aux<double>();
        } else {
            calc_stress_nonloc_aux<double_complex>();
        }
    }

    inline matrix3d<double> stress_nonloc() const
    {
        return stress_nonloc_;
    }

    /// Contribution to the stress tensor from the augmentation operator.
    /** Total energy in ultrasoft pseudopotential contains this term:
     *  \f[
     *    \int V^{eff}({\bf r})\rho^{aug}({\bf r})d{\bf r} =
     *      \sum_{\alpha} \sum_{\xi \xi'} n_{\xi \xi'}^{\alpha} \int V^{eff}({\bf r}) Q_{\xi \xi'}^{\alpha}({\bf r}) d{\bf r}  
     *  \f]
     *  The derivatives of beta-projectors (hidden in the desnity matrix expression) are taken into account in sirius::Stress::calc_stress_nonloc.
     *  Here we need to compute the remaining contribution from the \f$ Q_{\xi \xi'}({\bf r}) \f$ itself. We are interested in the integral:
     *  \f[
     *     \int V^{eff}({\bf r}) Q_{\xi \xi'}^{\alpha}({\bf r}) d{\bf r} = \sum_{\bf G} V^{eff}({\bf G}) \tilde Q_{\xi \xi'}^{\alpha}({\bf G})
     *  \f]
     *  where
     *  \f[
     *     \tilde Q_{\xi \xi'}^{\alpha}({\bf G}) = \int e^{-i{\bf Gr}} Q_{\xi \xi'}^{\alpha}({\bf r}) d{\bf r} = 
     *      4\pi \sum_{\ell m} (-i)^{\ell} R_{\ell m}(\hat{\bf G}) \langle R_{\ell_{\xi} m_{\xi}} | R_{\ell m} | R_{\ell_{\xi'} m_{\xi'}} \rangle \int Q_{\ell_{\xi} \ell_{\xi'}}^{\ell}(r) j_{\ell}(Gr) r^2 dr
     *  \f]
     *  Strain derivative of \f$ \tilde Q_{\xi \xi'}^{\alpha}({\bf G}) \f$ is:
     *  \f[
     *    \frac{\partial}{\partial \varepsilon_{\mu \nu}}  \tilde Q_{\xi \xi'}^{\alpha}({\bf G}) = 
     *      4\pi \sum_{\ell m} (-i)^{\ell} \langle R_{\ell_{\xi} m_{\xi}} | R_{\ell m} | R_{\ell_{\xi'} m_{\xi'}} \rangle 
     *      \Big( \frac{\partial R_{\ell m}(\hat{\bf G})}{\partial  \varepsilon_{\mu \nu}} \int Q_{\ell_{\xi} \ell_{\xi'}}^{\ell}(r) j_{\ell}(Gr) r^2 dr +
     *       R_{\ell m}(\hat{\bf G}) \int Q_{\ell_{\xi} \ell_{\xi'}}^{\ell}(r) \frac{\partial j_{\ell}(Gr)}{\partial \varepsilon_{\mu \nu}} r^2 dr \Big)
     *  \f]
     *  For strain derivatives of spherical harmonics and Bessel functions see sirius::Stress::calc_stress_nonloc. We can pull the
     *  common multiplier \f$ -G_{\mu} / G \f$ from both terms and arrive to the following expression:
     *  \f[
     *    \frac{\partial}{\partial \varepsilon_{\mu \nu}} \tilde Q_{\xi \xi'}^{\alpha}({\bf G}) = 
     *      -\frac{G_{\mu}}{G}  4\pi \sum_{\ell m} (-i)^{\ell} \langle R_{\ell_{\xi} m_{\xi}} | R_{\ell m} | R_{\ell_{\xi'} m_{\xi'}} \rangle 
     *      \Big( \big(\nabla_{G} R_{\ell m}(\hat{\bf G})\big)_{\nu} \int Q_{\ell_{\xi} \ell_{\xi'}}^{\ell}(r) j_{\ell}(Gr) r^2 dr +
     *       R_{\ell m}(\hat{\bf G}) \int Q_{\ell_{\xi} \ell_{\xi'}}^{\ell}(r) \frac{\partial j_{\ell}(Gr)}{\partial G} G_{\nu} r^2 dr \Big)
     *  \f]
     */
    inline void calc_stress_us()
    {
        PROFILE("sirius::Stress|us");

        stress_us_.zero();

        potential_.effective_potential()->fft_transform(-1);

        Radial_integrals_aug<false> const& ri = ctx_.aug_ri();
        Radial_integrals_aug<true> ri_dq(ctx_.unit_cell(), ctx_.pw_cutoff(), ctx_.settings().nprii_aug_);

        /* pack v effective in one array of pointers*/
        Periodic_function<double>* vfield_eff[4];
        vfield_eff[0] = potential_.effective_potential();
        vfield_eff[0]->fft_transform(-1);
        for (int imag = 0; imag < ctx_.num_mag_dims(); imag++){
            vfield_eff[imag + 1] = potential_.effective_magnetic_field(imag);
            vfield_eff[imag + 1]->fft_transform(-1);
        }

        Augmentation_operator_gvec_deriv q_deriv(ctx_);

        for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
            auto& atom_type = ctx_.unit_cell().atom_type(iat);
            if (!atom_type.augment()) {
                continue;
            }

            int nbf = atom_type.mt_basis_size();

            /* get auxiliary density matrix */
            auto dm = density_.density_matrix_aux(iat);

            mdarray<double_complex, 2> phase_factors(atom_type.num_atoms(), ctx_.gvec().count());

            sddk::timer t0("sirius::Stress|us|phase_fac");
            #pragma omp parallel for schedule(static)
            for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
                int ig = ctx_.gvec().offset() + igloc;
                for (int i = 0; i < atom_type.num_atoms(); i++) {
                    int ia = atom_type.atom_id(i);
                    phase_factors(i, igloc) = ctx_.gvec_phase_factor(ig, ia);
                }
            }
            t0.stop();
            mdarray<double, 2> v_tmp(atom_type.num_atoms(), ctx_.gvec().count() * 2);
            mdarray<double, 2> tmp(nbf * (nbf + 1) / 2, atom_type.num_atoms());
            /* over spin components, can be from 1 to 4*/
            for (int ispin = 0; ispin < ctx_.num_mag_dims() + 1; ispin++ ){
                for (int nu = 0; nu < 3; nu++) {
                    q_deriv.generate_pw_coeffs(iat, ri, ri_dq, nu);

                    for (int mu = 0; mu < 3; mu++) {
                        sddk::timer t2("sirius::Stress|us|prepare");
                        int igloc0{0};
                        if (ctx_.comm().rank() == 0) {
                            for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                                v_tmp(ia, 0) = v_tmp(ia, 1) = 0;
                            }
                            igloc0 = 1;
                        }
                        #pragma omp parallel for schedule(static)
                        for (int igloc = igloc0; igloc < ctx_.gvec().count(); igloc++) {
                            int ig = ctx_.gvec().offset() + igloc;
                            auto gvc = ctx_.gvec().gvec_cart(ig);
                            double g = gvc.length();

                            for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                                //auto z = phase_factors(ia, igloc) * std::conj(potential_.effective_potential()->f_pw_local(igloc));
                                auto z = phase_factors(ia, igloc) * vfield_eff[ispin]->f_pw_local(igloc) * (-gvc[mu] / g);
                                v_tmp(ia, 2 * igloc)     = z.real();
                                v_tmp(ia, 2 * igloc + 1) = z.imag();
                            }
                        }
                        t2.stop();

                        sddk::timer t1("sirius::Stress|us|gemm");
                        linalg<CPU>::gemm(0, 1, nbf * (nbf + 1) / 2, atom_type.num_atoms(), 2 * ctx_.gvec().count(),
                                          q_deriv.q_pw(), v_tmp, tmp);
                        t1.stop();
                        for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                            for (int i = 0; i < nbf * (nbf + 1) / 2; i++) {
                                stress_us_(mu, nu) += tmp(i, ia) * dm(i, ia, ispin) * q_deriv.sym_weight(i);
                            }

                        }
                    }
                }
            }
        }

        ctx_.comm().allreduce(&stress_us_(0, 0), 9);
        if (ctx_.gvec().reduced()) {
            stress_us_ *= 2;
        }

        stress_us_ *= (1.0 / ctx_.unit_cell().omega());

        symmetrize(stress_us_);
    }

    inline matrix3d<double> stress_us() const
    {
        return stress_us_;
    }

    inline matrix3d<double> stress_us_nl() const
    {
        return stress_nonloc_ + stress_us_;
    }

    /// XC contribution to stress.
    /** XC contribution has the following expression:
     *  \f[
     *    \frac{\partial E_{xc}}{\partial \varepsilon_{\mu \nu}} = \delta_{\mu \nu} \int \Big( \epsilon_{xc}({\bf r}) - v_{xc}({\bf r}) \Big) \rho({\bf r})d{\bf r} - 
     *      \int \frac{\partial \epsilon_{xc} \big( \rho({\bf r}), \nabla \rho({\bf r})\big) }{\nabla_{\beta} \rho({\bf r})} \nabla_{\alpha}\rho({\bf r}) d{\bf r}
     *  \f]
     */
    void calc_stress_xc()
    {
        stress_xc_.zero();

        double e = potential_.energy_exc(density_) - potential_.energy_vxc(density_);

        for (int l = 0; l < 3; l++) {
            stress_xc_(l, l) = e / ctx_.unit_cell().omega();
        }

        if (potential_.is_gradient_correction()) {

            Smooth_periodic_function<double> rhovc(ctx_.fft(), ctx_.gvec_partition());
            rhovc.zero();
            rhovc.add(density_.rho());
            rhovc.add(density_.rho_pseudo_core());
            
            /* transform to PW domain */
            rhovc.fft_transform(-1);

            /* generate pw coeffs of the gradient */
            auto grad_rho = gradient(rhovc);

            /* gradient in real space */
            for (int x: {0, 1, 2}) {
                grad_rho[x].fft_transform(1);
            }

            matrix3d<double> t;
            for (int irloc = 0; irloc < ctx_.fft().local_size(); irloc++) {
                for (int mu = 0; mu < 3; mu++) {
                    for (int nu = 0; nu < 3; nu++) {
                        t(mu, nu) += grad_rho[mu].f_rg(irloc) * grad_rho[nu].f_rg(irloc) * potential_.vsigma(0).f_rg(irloc);
                    }
                }
            }
            ctx_.fft().comm().allreduce(&t(0, 0), 9);
            t *= (-2.0 / ctx_.fft().size()); // factor 2 comes from the derivative of sigma (which is grad(rho) * grad(rho)) 
                                             // with respect to grad(rho) components
            stress_xc_ += t;
        }

        symmetrize(stress_xc_);
    }

    inline matrix3d<double> stress_xc() const
    {
        return stress_xc_;
    }
    
    /// Non-linear core correction to stress tensor.
    void calc_stress_core()
    {
        stress_core_.zero();

        potential_.xc_potential()->fft_transform(-1);

        Radial_integrals_rho_core_pseudo<true> ri_dg(ctx_.unit_cell(), ctx_.pw_cutoff(), ctx_.settings().nprii_rho_core_);

        auto drhoc = ctx_.make_periodic_function<index_domain_t::local>([&ri_dg](int iat, double g)
                                                                        {
                                                                            return ri_dg.value<int>(iat, g);
                                                                        });
        double sdiag{0};
        int ig0 = (ctx_.comm().rank() == 0) ? 1 : 0;

        for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;
            
            auto G = ctx_.gvec().gvec_cart(ig);
            auto g = G.length();

            for (int mu: {0, 1, 2}) {
                for (int nu: {0, 1, 2}) {
                    stress_core_(mu, nu) -= std::real(std::conj(potential_.xc_potential()->f_pw_local(igloc)) * drhoc[igloc]) * G[mu] * G[nu] / g;
                }
            }

            sdiag += std::real(std::conj(potential_.xc_potential()->f_pw_local(igloc)) * density_.rho_pseudo_core().f_pw_local(igloc));
        }
        
        if (ctx_.gvec().reduced()) {
            stress_core_ *= 2;
            sdiag *= 2;
        }
        if (ctx_.comm().rank() == 0) {
            sdiag += std::real(std::conj(potential_.xc_potential()->f_pw_local(0)) * density_.rho_pseudo_core().f_pw_local(0));
        }

        for (int mu: {0, 1, 2}) {
            stress_core_(mu, mu) -= sdiag;
        }

        ctx_.comm().allreduce(&stress_core_(0, 0), 9);

        symmetrize(stress_core_);
    }

    inline matrix3d<double> stress_core() const
    {
        return stress_core_;
    }

    inline void calc_stress_total()
    {
        calc_stress_kin();
        calc_stress_har();
        calc_stress_ewald();
        calc_stress_vloc();
        calc_stress_core();
        calc_stress_xc();
        calc_stress_us();
        calc_stress_nonloc();
    }

    inline void print_info() const
    {
        if (ctx_.comm().rank() == 0) {
            
            auto print_stress = [&](matrix3d<double> const& s)
            {
                for (int mu: {0, 1, 2}) {
                    printf("%12.6f %12.6f %12.6f\n", s(mu, 0), s(mu, 1), s(mu, 2));
                }
            };

            const double au2kbar = 2.94210119E5;
            auto stress_kin    = stress_kin_    * au2kbar;
            auto stress_har    = stress_har_    * au2kbar;
            auto stress_ewald  = stress_ewald_  * au2kbar;
            auto stress_vloc   = stress_vloc_   * au2kbar;
            auto stress_nonloc = stress_nonloc_ * au2kbar;
            auto stress_us     = stress_us_     * au2kbar;

            printf("== stress_kin ==\n");
            print_stress(stress_kin);

            printf("== stress_har ==\n");
            print_stress(stress_har);

            printf("== stress_ewald ==\n");
            print_stress(stress_ewald);

            printf("== stress_vloc ==\n");
            print_stress(stress_vloc);

            printf("== stress_nonloc ==\n");
            print_stress(stress_nonloc);

            printf("== stress_us ==\n");
            print_stress(stress_us);

            stress_us = stress_us + stress_nonloc;
            printf("== stress_us_nl ==\n");
            print_stress(stress_us);
        }
    }
};

}

#endif
