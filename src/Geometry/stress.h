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

/** \file stress.h
 *
 *  \brief Stress tensor calculation.
 */

#ifndef __STRESS_H__
#define __STRESS_H__

#include "../Beta_projectors/beta_projectors_strain_deriv.h"

namespace sirius {

/// Stress tensor.
/** Stress tensor describes a reaction of crystall to a strain:
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
    inline matrix3d<double> calc_stress_kin() const
    {
        PROFILE("sirius::Stress|kin");

        matrix3d<double> s;

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
                for (int i = 0; i < ctx_.num_bands(); i++) {
                    double f = kp->band_occupancy(i);
                    if (f > 1e-12) {
                        auto z = kp->spinor_wave_functions(0).pw_coeffs().prime(igloc, i);
                        d += f * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
                    }
                }
                d *= kp->weight();
                for (int mu: {0, 1, 2}) {
                    for (int nu: {0, 1, 2}) {
                        s(mu, nu) += Gk[mu] * Gk[nu] * d;
                    }
                }
            } // igloc
        } // ikloc

        ctx_.comm().allreduce(&s(0, 0), 9);

        s *= (-1.0 / ctx_.unit_cell().omega());

        symmetrize(s);

        return s;
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

        for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;
            if (!ig) {
                continue;
            }

            auto G = ctx_.gvec().gvec_cart(ig);
            double g2 = std::pow(G.length(), 2);
            auto z = density_.rho()->f_pw_local(igloc);
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

        double lambda = 2.5;

        auto& uc = ctx_.unit_cell();

        for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;
            if (!ig) {
                continue;
            }

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

                vector3d<double> v1 = uc.atom(ja).position() - uc.atom(ia).position() + uc.nearest_neighbour(i, ia).translation;
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

        Radial_integrals_vloc ri_vloc(ctx_.unit_cell(), ctx_.pw_cutoff(), 100);
        Radial_integrals_vloc_dg ri_vloc_dg(ctx_.unit_cell(), ctx_.pw_cutoff(), 100);

        auto v = ctx_.make_periodic_function<index_domain_t::local>([&ri_vloc](int iat, double g)
                                                                    {
                                                                        return ri_vloc.value(iat, g);
                                                                    });

        auto dv = ctx_.make_periodic_function<index_domain_t::local>([&ri_vloc_dg](int iat, double g)
                                                                     {
                                                                         return ri_vloc_dg.value(iat, g);
                                                                     });
        
        double sdiag{0};

        for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;
            
            if (!ig) {
                continue;
            }

            auto G = ctx_.gvec().gvec_cart(ig);

            for (int mu: {0, 1, 2}) {
                for (int nu: {0, 1, 2}) {
                    stress_vloc_(mu, nu) += std::real(std::conj(density_.rho()->f_pw_local(igloc)) * dv[igloc]) * G[mu] * G[nu];
                }
            }

            sdiag += std::real(std::conj(density_.rho()->f_pw_local(igloc)) * v[igloc]);
        }
        
        if (ctx_.gvec().reduced()) {
            stress_vloc_ *= 2;
            sdiag *= 2;
        }
        if (ctx_.comm().rank() == 0) {
            sdiag += std::real(std::conj(density_.rho()->f_pw_local(0)) * v[0]);
        }

        for (int mu: {0, 1, 2}) {
            stress_vloc_(mu, mu) -= sdiag;
        }

        ctx_.comm().allreduce(&stress_vloc_(0, 0), 9);

        symmetrize(stress_vloc_);
    }

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
     *      \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \theta} \sum_{\tau} \frac{\partial \theta}{\partial q_{\tau}} \frac{\partial{q_{\tau}}}{\partial \varepsilon_{\mu \nu}} + 
     *      \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \phi} \sum_{\tau} \frac{\partial \phi}{\partial q_{\tau}}\frac{\partial q_{\tau}}{\partial \varepsilon_{\mu \nu}} = 
     *      -q_{\mu} \Big(  \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \theta} \frac{\partial \theta}{\partial q_{\nu}} + 
     *              \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \phi} \frac{\partial \phi}{\partial q_{\nu}}\Big)
     *  \f]
     *  
     *  The derivatives of angles are:
     *  \f[
     *     \frac{\partial \theta}{\partial q_{x}} = \frac{\cos(\phi) \cos(\theta)}{q} \\
     *     \frac{\partial \theta}{\partial q_{y}} = \frac{\cos(\theta) \sin(\phi)}{q} \\
     *     \frac{\partial \theta}{\partial q_{z}} = -\frac{\sin(\theta)}{q}
     *  \f]
     *  and
     *  \f[
     *     \frac{\partial \phi}{\partial q_{x}} = -\frac{\sin(\phi)}{\sin(\theta) q} \\
     *     \frac{\partial \phi}{\partial q_{y}} = \frac{\cos(\phi)}{\sin(\theta) q} \\
     *     \frac{\partial \phi}{\partial q_{z}} = 0
     *  \f]
     *  The derivative of \f$ \phi \f$ has discontinuities at \f$ \theta = 0, \theta=\pi \f$. This, however, is not a problem, because
     *  multiplication by the the derivative of \f$ R_{\ell m} \f$ removes it. The following functions have to be hardcoded:
     *  \f[
     *    \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \theta} \\
     *    \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \phi} \frac{1}{\sin(\theta)} 
     *  \f]
     *  
     *  Mathematica script for spherical harmonic derivatives:
        \verbatim
        Rlm[l_, m_, th_, ph_] := 
         If[m > 0, Sqrt[2]*ComplexExpand[Re[SphericalHarmonicY[l, m, th, ph]]],
           If[m < 0, Sqrt[2]*ComplexExpand[Im[SphericalHarmonicY[l, m, th, ph]]], 
             If[m == 0, ComplexExpand[Re[SphericalHarmonicY[l, 0, th, ph]]]]
           ]
         ]
        Do[Print[FullSimplify[D[Rlm[l, m, theta, phi], theta]]], {l, 0, 4}, {m, -l, l}]
        Do[Print[FullSimplify[TrigExpand[D[Rlm[l, m, theta, phi], phi]/Sin[theta]]]], {l, 0, 4}, {m, -l, l}]
        \endverbatim
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
     * (second term comes from the strain derivative of \f$ e^{-i{\bf q r}} \f$).
     */
    template <typename T>
    inline void calc_stress_nonloc()
    {
        PROFILE("sirius::Stress|nonloc");

        auto& bchunk = ctx_.beta_projector_chunks();

        for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
            int ik = kset_.spl_num_kpoints(ikloc);
            auto kp = kset_[ik];
            if (kp->gkvec().reduced()) {
                TERMINATE("fix this");
            }

            auto d_mtrx = [&](Atom& atom, int ibnd, int i, int j)
            {
                if (atom.type().pp_desc().augment) {
                    return atom.d_mtrx(i, j, 0) - kp->band_energy(ibnd) * ctx_.augmentation_op(atom.type_id()).q_mtrx(i, j);
                } else {
                    return atom.d_mtrx(i, j, 0);
                }
            };

            Beta_projectors_strain_deriv bp_strain_deriv(ctx_, kp->gkvec());

            kp->beta_projectors().prepare();
            bp_strain_deriv.prepare();

            for (int ichunk = 0; ichunk < bchunk.num_chunks(); ichunk++) {
                kp->beta_projectors().generate(ichunk);

                int nbnd = kp->num_occupied_bands(0);
                splindex<block> spl_nbnd(nbnd, kp->comm().size(), kp->comm().rank());

                /* compute <beta|psi> */
                auto beta_psi = kp->beta_projectors().inner<T>(ichunk, kp->spinor_wave_functions(0), 0, nbnd);

                for (int mu = 0; mu < 3; mu++) {
                    for (int nu = 0; nu < 3; nu++) {
                        bp_strain_deriv.generate(ichunk, mu + nu * 3);
                        auto dbeta_psi = bp_strain_deriv.inner<T>(ichunk, kp->spinor_wave_functions(0), 0, nbnd);

                        for (int i = 0; i < bchunk(ichunk).num_atoms_; i++) {
                            int ia   = bchunk(ichunk).desc_(beta_desc_idx::ia, i);
                            int offs = bchunk(ichunk).desc_(beta_desc_idx::offset, i);
                            int nbf  = bchunk(ichunk).desc_(beta_desc_idx::nbf, i);

                            auto& atom = ctx_.unit_cell().atom(ia);

                            for (int ibnd_loc = 0; ibnd_loc < spl_nbnd.local_size(); ibnd_loc++) {
                                int ibnd = spl_nbnd[ibnd_loc];
                                double w = kp->weight() * kp->band_occupancy(ibnd);

                                for (int xi1 = 0; xi1 < nbf; xi1++) {
                                    for (int xi2 = 0; xi2 < nbf; xi2++) {
                                        stress_nonloc_(mu, nu) += w * std::real(
                                            std::conj(dbeta_psi(offs + xi1, ibnd)) * d_mtrx(atom, ibnd, xi1, xi2) * beta_psi(offs + xi2, ibnd) +
                                            std::conj(beta_psi(offs + xi1, ibnd)) * d_mtrx(atom, ibnd, xi1, xi2) * dbeta_psi(offs + xi2, ibnd)
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
            kp->beta_projectors().dismiss();
        }
        ctx_.comm().allreduce(&stress_nonloc_(0, 0), 9);

        stress_nonloc_ *= (1.0 / ctx_.unit_cell().omega());

        symmetrize(stress_nonloc_);

        std::vector<std::array<int, 2>> idx = {{0, 1}, {0, 2}, {1, 2}};
        for (auto e: idx) {
            stress_nonloc_(e[0], e[1]) = stress_nonloc_(e[1], e[0]) = 0.5 * (stress_nonloc_(e[0], e[1]) + stress_nonloc_(e[1], e[0]));
        }
    }

    inline void calc_stress_us()
    {
        PROFILE("sirius::Stress|us");

        potential_.effective_potential()->fft_transform(-1);

        Radial_integrals_aug<false> ri(ctx_.unit_cell(), ctx_.pw_cutoff(), 20);
        Radial_integrals_aug<true> ri_dq(ctx_.unit_cell(), ctx_.pw_cutoff(), 20);
        
        Augmentation_operator_gvec_deriv q_deriv(ctx_);

        for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
            auto& atom_type = ctx_.unit_cell().atom_type(iat);
            if (!atom_type.pp_desc().augment) {
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
                    phase_factors(i, igloc) = std::conj(ctx_.gvec_phase_factor(ig, ia));
                }
            }
            t0.stop();
            mdarray<double, 2> q_tmp(nbf * (nbf + 1) / 2, ctx_.gvec().count() * 2);
            mdarray<double, 2> v_tmp(atom_type.num_atoms(), ctx_.gvec().count() * 2);
            mdarray<double, 2> tmp(nbf * (nbf + 1) / 2, atom_type.num_atoms());
            for (int nu = 0; nu < 3; nu++) {
                q_deriv.generate_pw_coeffs(iat, ri, ri_dq, nu);

                for (int mu = 0; mu < 3; mu++) {
                    sddk::timer t2("sirius::Stress|us|prepare");
                    int igloc0{0};
                    if (ctx_.comm().rank() == 0) {
                        for (int i = 0; i < nbf * (nbf + 1) / 2; i++) {
                            q_tmp(i, 0) = q_tmp(i, 1) = 0;
                        }
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
                        for (int i = 0; i < nbf * (nbf + 1) / 2; i++) {
                            auto z = double_complex(q_deriv.q_pw(i, 2 * igloc), q_deriv.q_pw(i, 2 * igloc + 1)) * (-gvc[mu] / g);
                            q_tmp(i, 2 * igloc)     = z.real();
                            q_tmp(i, 2 * igloc + 1) = z.imag();
                        }
                        for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                            auto z = phase_factors(ia, igloc) * std::conj(potential_.effective_potential()->f_pw_local(igloc));
                            v_tmp(ia, 2 * igloc)     = z.real();
                            v_tmp(ia, 2 * igloc + 1) = z.imag();
                        }
                    }
                    t2.stop();
                    
                    //tmp.zero();
                    //for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                    //    int iaglob = atom_type.atom_id(ia);
                    //    for (int i = 0; i < nbf * (nbf + 1) / 2; i++) {
                    //        for (int igloc = igloc0; igloc < ctx_.gvec().count(); igloc++) {
                    //            int ig = ctx_.gvec().offset() + igloc;
                    //            auto gvc = ctx_.gvec().gvec_cart(ig);
                    //            double g = gvc.length();
                    //            tmp(i, ia) += std::real(
                    //                    double_complex(q_deriv.q_pw(i, 2 * igloc), q_deriv.q_pw(i, 2 * igloc + 1)) *
                    //                    std::conj(potential_.effective_potential()->f_pw_local(igloc)) * (-gvc[mu] / g) *
                    //                    double_complex(phase_factors(ia, 2 * igloc), phase_factors(ia, 2 * igloc + 1))
                    //                    );
                    //        }
                    //    }
                    //}
                    
                    /* canonical code */
                    //for (int igloc = igloc0; igloc < ctx_.gvec().count(); igloc++) {
                    //    int ig = ctx_.gvec().offset() + igloc;
                    //    auto gvc = ctx_.gvec().gvec_cart(ig);
                    //    double g = gvc.length();
                    //    for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                    //        int iaglob = atom_type.atom_id(ia);
                    //        for (int i = 0; i < nbf * (nbf + 1) / 2; i++) {
                    //            stress_us_(mu, nu) += std::real(
                    //                double_complex(q_deriv.q_pw(i, 2 * igloc), q_deriv.q_pw(i, 2 * igloc + 1)) *
                    //                std::conj(potential_.effective_potential()->f_pw_local(igloc)) * (-gvc[mu] / g) *
                    //                double_complex(phase_factors(ia, 2 * igloc), phase_factors(ia, 2 * igloc + 1)) *
                    //                dm(i, ia, 0) * q_deriv.sym_weight(i)
                    //            );

                    //        }
                    //    }
                    //}

                    sddk::timer t1("sirius::Stress|us|gemm");
                    linalg<CPU>::gemm(0, 1, nbf * (nbf + 1) / 2, atom_type.num_atoms(), 2 * ctx_.gvec().count(),
                                      q_tmp, v_tmp, tmp);
                    t1.stop();
                    for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                        for (int i = 0; i < nbf * (nbf + 1) / 2; i++) {
                            stress_us_(mu, nu) += tmp(i, ia) * dm(i, ia, 0) * q_deriv.sym_weight(i);
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

        std::vector<std::array<int, 2>> idx = {{0, 1}, {0, 2}, {1, 2}};
        for (auto e: idx) {
            stress_us_(e[0], e[1]) = stress_us_(e[1], e[0]) = 0.5 * (stress_us_(e[0], e[1]) + stress_us_(e[1], e[0]));
        }
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
        stress_kin_ = calc_stress_kin();
        calc_stress_har();
        calc_stress_ewald();
        calc_stress_vloc();
        if (ctx_.gamma_point()) {
            calc_stress_nonloc<double>();
        } else {
            calc_stress_nonloc<double_complex>();
        }
        calc_stress_us();
    }

    inline matrix3d<double> stress_vloc() const
    {
        return stress_vloc_;
    }

    inline matrix3d<double> stress_har() const
    {
        return stress_har_;
    }

    inline matrix3d<double> stress_ewald() const
    {
        return stress_ewald_;
    }

    inline matrix3d<double> stress_kin() const
    {
        return stress_kin_;
    }

    inline matrix3d<double> stress_nl() const
    {
        return stress_nonloc_ + stress_us_;
    }

    inline void print_info() const
    {
        if (ctx_.comm().rank() == 0) {
            const double au2kbar = 2.94210119E5;
            auto stress_kin    = stress_kin_ * au2kbar;
            auto stress_har    = stress_har_ * au2kbar;
            auto stress_ewald  = stress_ewald_ * au2kbar;
            auto stress_vloc   = stress_vloc_ * au2kbar;
            auto stress_nonloc = stress_nonloc_ * au2kbar;
            auto stress_us     = stress_us_ * au2kbar;

            printf("== stress_kin ==\n");
            for (int mu: {0, 1, 2}) {
                printf("%12.6f %12.6f %12.6f\n", stress_kin(mu, 0), stress_kin(mu, 1), stress_kin(mu, 2));
            }
            printf("== stress_har ==\n");
            for (int mu: {0, 1, 2}) {
                printf("%12.6f %12.6f %12.6f\n", stress_har(mu, 0), stress_har(mu, 1), stress_har(mu, 2));
            }
            printf("== stress_ewald ==\n");
            for (int mu: {0, 1, 2}) {
                printf("%12.6f %12.6f %12.6f\n", stress_ewald(mu, 0), stress_ewald(mu, 1), stress_ewald(mu, 2));
            }
            printf("== stress_vloc ==\n");
            for (int mu: {0, 1, 2}) {
                printf("%12.6f %12.6f %12.6f\n", stress_vloc(mu, 0), stress_vloc(mu, 1), stress_vloc(mu, 2));
            }
            printf("== stress_nonloc ==\n");
            for (int mu: {0, 1, 2}) {
                printf("%12.6f %12.6f %12.6f\n", stress_nonloc(mu, 0), stress_nonloc(mu, 1), stress_nonloc(mu, 2));
            }
            printf("== stress_us ==\n");
            for (int mu: {0, 1, 2}) {
                printf("%12.6f %12.6f %12.6f\n", stress_us(mu, 0), stress_us(mu, 1), stress_us(mu, 2));
            }
            stress_us = stress_us + stress_nonloc;
            printf("== stress_us_tot ==\n");
            for (int mu: {0, 1, 2}) {
                printf("%12.6f %12.6f %12.6f\n", stress_us(mu, 0), stress_us(mu, 1), stress_us(mu, 2));
            }
        }
    }
};

}

#endif

/** 
\page stress Stress tensor
\section stress_section1 Preliminary notes
Derivative of the G-vector in Cartesian coordinates over the lattice vector components:
\f[
  \frac{\partial G_{\tau}}{\partial a_{\mu\nu}} + ({\bf a}^{-1})_{\nu \tau} G_{\mu} = 0
\f]
Mathematica proof script:
\verbatim
A = Table[Subscript[a, i, j], {i, 1, 3}, {j, 1, 3}];
invA = Inverse[A];
B = 2*Pi*Transpose[Inverse[A]];
G = Table[Subscript[g, i], {i, 1, 3}];
gvec = B.G;
Do[
  Print[FullSimplify[
   D[gvec[[tau]], Subscript[a, mu, nu]] + invA[[nu]][[tau]]*gvec[[mu]]]],
{beta, 1, 3}, {mu, 1, 3}, {nu, 1,3}]
\endverbatim
Another relation:
\f[
  \frac{\partial}{\partial a_{\mu \nu}} \frac{1}{\sqrt{\Omega}} + \frac{1}{2} \frac{1}{\sqrt{\Omega}} ({\bf a}^{-1})_{\nu \mu} = 0
\f]
Mathematica proof script:
\verbatim
A = Table[Subscript[a, i, j], {i, 1, 3}, {j, 1, 3}];
invA = Inverse[A];
Do[
 Print[FullSimplify[
   D[1/Sqrt[Det[A]], Subscript[a, mu, nu]] + (1/2)*(1/Sqrt[Det[A]]) * invA[[nu]][[mu]]
   ]
  ],
{mu, 1, 3}, {nu, 1, 3}]
\endverbatim

Strain tensor describes the deformation of a crystal:
\f[
  r_{\mu} \rightarrow \sum_{\nu} (\delta_{\mu \nu} + \varepsilon_{\mu \nu}) r_{\nu}
\f]
Strain derivative of general position vector component:
\f[
  \frac{\partial r_{\tau}}{\partial \varepsilon_{\mu \nu}} = \delta_{\tau \mu} r_{\nu}
\f]
Strain derivative of the lengths of general position vector:
\f[
  \frac{\partial |{\bf r}|}{\partial \varepsilon_{\mu \nu}} = \sum_{\tau} \frac{\partial |{\bf r}|}{\partial r_{\tau}} 
    \frac{\partial r_{\tau}}{\partial \varepsilon_{\mu \nu}} = \sum_{\tau} \frac{ r_{\tau}}{|{\bf r}|}
    \delta_{\tau \mu} r_{\nu} = \frac{r_{\mu} r_{\nu}}{|{\bf r}|} 
\f]
Strain derivative of unit cell volume:
\f[
  \frac{\partial \Omega}{\partial \varepsilon_{\mu \nu}} = \delta_{\mu \nu} \Omega
\f]
Strain derivative of reciprocal vector:
\f[
  \frac{\partial G_{\tau}}{\partial \varepsilon_{\mu \nu}} = -\delta_{\tau \nu} G_{\mu}
\f]
Strain derivative of the length of reciprocal vector:
\f[
  \frac{\partial |{\bf G}|}{\partial \varepsilon_{\mu \nu}} = \sum_{\tau} \frac{\partial |{\bf G}|}{\partial G_{\tau}}
    \frac{\partial G_{\tau}}{\partial \varepsilon_{\mu \nu}} = -\frac{1}{|{\bf G}|}G_{\nu}G_{\mu}
\f]
Stress tensor is a reaction to strain:
\f[
  \sigma_{\mu \nu} = \frac{1}{\Omega} \frac{\partial E_{tot}}{\partial \varepsilon_{\mu \nu}} =
    \frac{1}{\Omega} \sum_{\mu' \nu'} \frac{\partial E_{tot}}{\partial a_{\mu' \nu'}} \frac{\partial a_{\mu' \nu'}}{\partial  \varepsilon_{\mu \nu}} = 
    \frac{1}{\Omega} \sum_{\nu'} \frac{\partial E_{tot}}{\partial a_{\mu \nu'}} a_{\nu \nu'} 
\f]

\section stress_section2 Derivative of beta-projectors.
We need to compute derivative of beta-projectors
\f[
  \frac{\partial}{\partial a_{\mu \nu}} \langle {\bf G+k} | \beta_{\ell m} \rangle
\f]

Derivative of the G-vector real spherical harmonics over the lattice vector components:
\f[
  \frac{\partial R_{\ell m}(\theta, \phi)}{\partial a_{\mu \nu}} = 
    \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \theta} \frac{\partial \theta} {\partial a_{\mu \nu}} + 
    \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \phi} \frac{\partial \phi} {\partial a_{\mu \nu}}
\f]
Derivatives of the \f$ R_{\ell m} \f$ with respect to the \f$ \theta,\, \phi\f$ angles can be tabulated up to a given \f$ \ell_{max} \f$.
The derivatives of angles are computed as following:
\f[
 \frac{\partial \theta} {\partial a_{\mu \nu}} = \sum_{\beta=1}^{3} \frac{\partial \theta}{\partial G_{\beta}} \frac{\partial G_{\beta}} {\partial a_{\mu \nu}}
\f]
\f[
 \frac{\partial \phi} {\partial a_{\mu \nu}} = \sum_{\beta=1}^{3} \frac{\partial \phi}{\partial G_{\beta}} \frac{\partial G_{\beta}} {\partial a_{\mu \nu}}
\f]
where
\f[
   \frac{\partial \theta}{\partial G_{x}} = \frac{\cos(\phi) \cos(\theta)}{G} \\
   \frac{\partial \theta}{\partial G_{y}} = \frac{\cos(\theta) \sin(\phi)}{G} \\
   \frac{\partial \theta}{\partial G_{z}} = -\frac{\sin(\theta)}{G}
\f]
and
\f[
   \frac{\partial \phi}{\partial G_{x}} = -\frac{\sin(\phi)}{\sin(\theta) G} \\
   \frac{\partial \phi}{\partial G_{y}} = \frac{\cos(\phi)}{\sin(\theta) G} \\
   \frac{\partial \phi}{\partial G_{z}} = 0
\f]
The derivative of \f$ phi \f$ has discontinuities at \f$ \theta = 0, \theta=\pi \f$. This, however, is not a problem, because
multiplication by the the derivative of \f$ R_{\ell m} \f$ removes it. The following functions have to be hardcoded:
\f[
  \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \theta} \\
  \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \phi} \frac{1}{\sin(\theta)} 
\f]

Derivatives of the spherical Bessel functions are computed in the same fashion:
\f[
  \frac{\partial j_{\ell}(Gx)}{\partial a_{\mu \nu}} =
  \frac{\partial j_{\ell}(Gx)}{\partial G} \frac{\partial G} {\partial a_{\mu \nu}} = 
  \frac{\partial j_{\ell}(Gx)}{\partial G} \sum_{\beta=1}^{3}\frac{\partial G}{\partial G_{\beta}} \frac{\partial G_{\beta}} {\partial a_{\mu \nu}} 
\f]
The derivatives of \f$ G \f$ are:
\f[
  \frac{\partial G}{\partial G_{x}} = \sin(\theta)\cos(\phi) \\
  \frac{\partial G}{\partial G_{y}} = \sin(\theta)\sin(\phi) \\
  \frac{\partial G}{\partial G_{z}} = \cos(\theta)
\f]

Let's write the full expression for the derivative of beta-projector matrix elements with respect to lattice vector 
components:
\f[
  \frac{\partial \langle {\bf G+k}|\beta_{\ell m} \rangle} {\partial a_{\mu \nu}} = 
    \frac{\partial} {\partial a_{\mu \nu}} \frac{4\pi}{\sqrt{\Omega}}(-i)^{\ell} R_{\ell m}(\theta_{G+k}, \phi_{G+k}) \int \beta_{\ell}(r) j_{\ell}(Gr) r^2 dr =\\
    \frac{4\pi}{\sqrt{\Omega}} (-i)^{\ell} \Bigg[ \int \beta_{\ell}(r) j_{\ell}(Gr) r^2 dr 
      \Big( \frac{\partial R_{\ell m}(\theta, \phi)}{\partial a_{\mu \nu}} - \frac{1}{2} R_{\ell m}(\theta, \phi) ({\bf a}^{-1})_{\nu \mu}  \Big) + 
       R_{\ell m}(\theta, \phi) \int \beta_{\ell}(r) \frac{\partial j_{\ell}(Gr)}{\partial a_{\mu \nu}} r^2 dr  \Bigg]
\f]

\section stress_section3 Kinetic energy contribution to stress.
Kinetic contribution to stress tensor:
\f[
  E^{kin} = \sum_{{\bf k}} w_{\bf k} \sum_i f_i \frac{1}{2} |{\bf G+k}|^2 |\psi_i({\bf G + k})|^2
\f]
Derivative over lattice vectors
\f[
  \frac{\partial  E^{kin}}{\partial a_{\mu \nu}} =
    \sum_{{\bf k}} w_{\bf k} \sum_i f_i \sum_{\tau} (G+k)_{\tau} (-{\bf a}^{-1})_{\nu \tau} (G+k)_{\mu}  |\psi_i({\bf G + k})|^2
\f]
Contribution to the stress tensor
\f[
  \sigma_{\mu \nu}^{kin} = \frac{1}{\Omega} \sum_{\nu'} \frac{\partial E^{kin}}{\partial a_{\mu \nu'}} a_{\nu \nu'} = 
    \frac{1}{\Omega} \sum_{\nu'}  \sum_{{\bf k}} w_{\bf k} \sum_i f_i \sum_{\tau} (G+k)_{\tau} (-{\bf a}^{-1})_{\nu' \tau} (G+k)_{\mu}  |\psi_i({\bf G + k})|^2 a_{\nu \nu'}  = 
    -\frac{1}{\Omega} \sum_{{\bf k}} w_{\bf k} \sum_i (G+k)_{\nu} (G+k)_{\mu} f_i |\psi_i({\bf G + k})|^2 
\f]

\section stress_section4 Hartree energy contribution to stress.
Hartree energy:
\f[
  E^{H} = \frac{1}{2} \int_{\Omega} \rho({\bf r}) V^{H}({\bf r}) d{\bf r} = 
    \frac{1}{2} \frac{1}{\Omega} \sum_{\bf G} \langle \rho | {\bf G} \rangle \langle {\bf G}| V^{H} \rangle = 
    \frac{2 \pi}{\Omega} \sum_{\bf G} \frac{|\tilde \rho({\bf G})|^2}{G^2}
\f]
where 
\f[
  \langle {\bf G} | \rho \rangle = \int e^{-i{\bf Gr}}\rho({\bf r}) d {\bf r} = \tilde \rho({\bf G})
\f]
and
\f[
  \langle {\bf G} | V^{H} \rangle = \int e^{-i{\bf Gr}}V^{H}({\bf r}) d {\bf r} = \frac{4\pi}{G^2} \tilde \rho({\bf G})
\f]

Hartree energy contribution to stress tensor:
\f[
   \sigma_{\mu \nu}^{H} = \frac{1}{\Omega} \frac{\partial  E^{H}}{\partial \varepsilon_{\mu \nu}} = 
      \frac{1}{\Omega} 2\pi \Big( \big( \frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{1}{\Omega} \big) 
        \sum_{{\bf G}} \frac{|\tilde \rho({\bf G})|^2}{G^2} + 
        \frac{1}{\Omega} \sum_{{\bf G}} |\tilde \rho({\bf G})|^2 \frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{1}{G^2} \Big) = \\
   \frac{1}{\Omega} 2\pi \Big( -\frac{1}{\Omega} \delta_{\mu \nu} \sum_{{\bf G}} \frac{|\tilde \rho({\bf G})|^2}{G^2} + 
     \frac{1}{\Omega} \sum_{\bf G} |\tilde \rho({\bf G})|^2 \sum_{\tau} \frac{-2 G_{\tau}}{G^4} \frac{\partial G_{\tau}}{\partial \varepsilon_{\mu \nu}} \Big) = \\
   2\pi \sum_{\bf G} \frac{|\rho({\bf G})|^2}{G^2} \Big( -\delta_{\mu \nu} + \frac{2}{G^2} G_{\nu} G_{\mu} \Big)
\f]

\section stress_section5 Ewald energy contribution to stress.
Ewald energy:
\f[
  E^{ion-ion} = \frac{1}{2} \sideset{}{'} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta} 
    \frac{{\rm erfc}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|} +
    \frac{2 \pi}{\Omega} \sum_{{\bf G}} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \Big| \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 -
    \sum_{\alpha} Z_{\alpha}^2 \sqrt{\frac{\lambda}{\pi}} - \frac{2\pi}{\Omega}\frac{N_{el}^2}{4 \lambda}
\f]
Contribution to stress tensor:
\f[
  \sigma_{\mu \nu}^{ion-ion} = \frac{1}{\Omega} \frac{\partial  E^{ion-ion}}{\partial \varepsilon_{\mu \nu}} 
\f]
Derivative of the first part:
\f[
  \frac{1}{\Omega}\frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{1}{2} \sideset{}{'} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta} 
    \frac{{\rm erfc}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|}  = 
    \frac{1}{2\Omega} \sideset{}{'} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta} \Big( -2e^{-\lambda |{\bf r'}|^2} 
    \sqrt{\frac{\lambda}{\pi}} \frac{1}{|{\bf r'}|^2} - {\rm erfc}(\sqrt{\lambda} |{\bf r'}|) \frac{1}{|{\bf r'}|^3} \Big) r'_{\mu} r'_{\nu}
\f]
where \f$ {\bf r'} = {\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T} \f$.

Derivative of the second part:
\f[
  \frac{1}{\Omega}\frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{2\pi}{\Omega} \sum_{{\bf G}} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} 
  \Big| \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 = 
  -\frac{2\pi}{\Omega^2} \delta_{\mu \nu} \sum_{{\bf G}} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \Big| 
  \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 + 
  \frac{2\pi}{\Omega^2} \sum_{\bf G} G_{\mu} G_{\nu} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} 2 \frac{\frac{G^2}{4\lambda} + 1}{G^2}  
  \Big| \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 
\f]

Derivative of the fourth part:
\f[
  -\frac{1}{\Omega}\frac{\partial}{\partial \varepsilon_{\mu \nu}} \frac{2\pi}{\Omega}\frac{N_{el}^2}{4 \lambda} = 
     \frac{2\pi}{\Omega^2}\frac{N_{el}^2}{4 \lambda} \delta_{\mu \nu}
\f]

\section stress_section6 Local potential contribution to stress.
Energy contribution from the local part of pseudopotential:
\f[
  E^{loc} = \int \rho({\bf r}) V^{loc}({\bf r}) d{\bf r} = \frac{1}{\Omega} \sum_{\bf G} \langle \rho | {\bf G} \rangle \langle {\bf G}| V^{loc} \rangle = 
    \frac{1}{\Omega} \sum_{\bf G} \tilde \rho^{*}({\bf G}) \tilde V^{loc}({\bf G})
\f]
where
\f[
  \tilde \rho({\bf G}) = \langle {\bf G} | \rho \rangle = \int e^{-i{\bf Gr}}\rho({\bf r}) d {\bf r}
\f]
and
\f[
  \tilde V^{loc}({\bf G}) = \langle {\bf G} | V^{loc} \rangle = \int e^{-i{\bf Gr}}V^{loc}({\bf r}) d {\bf r} 
\f]
Using the expression for \f$ \tilde V^{loc}({\bf G}) \f$, the local contribution to the total energy is rewritten as
\f[
  E^{loc} = \frac{1}{\Omega} \sum_{\bf G} \tilde \rho^{*}({\bf G}) \sum_{\alpha} e^{-{\bf G\tau}_{\alpha}} 4 \pi 
    \int V_{\alpha}^{loc}(r)\frac{\sin(Gr)}{Gr} r^2 dr = 
    \frac{4\pi}{\Omega}\sum_{\bf G}\tilde \rho^{*}({\bf G})\sum_{\alpha} e^{-{\bf G\tau}_{\alpha}} 
    \Bigg( \int \Big(V_{\alpha}(r) r + Z_{\alpha}^p {\rm erf}(r) \Big) \frac{\sin(Gr)}{G} dr -  Z_{\alpha}^p \frac{e^{-\frac{G^2}{4}}}{G^2} \Bigg)
\f]
(see \link sirius::Potential::generate_local_potential \endlink for details).

Contribution to stress tensor:
\f[
   \sigma_{\mu \nu}^{loc} = \frac{1}{\Omega} \frac{\partial  E^{loc}}{\partial \varepsilon_{\mu \nu}} =
     \frac{1}{\Omega} \frac{-1}{\Omega} \delta_{\mu \nu} \sum_{\bf G}\tilde \rho^{*}({\bf G}) \tilde V^{loc}({\bf G}) + 
     \frac{4\pi}{\Omega^2} \sum_{\bf G}\tilde \rho^{*}({\bf G}) \sum_{\alpha} e^{-{\bf G\tau}_{\alpha}}
     \Bigg( \int \Big(V_{\alpha}(r) r + Z_{\alpha}^p {\rm erf}(r) \Big) \Big( \frac{r \cos (G r)}{G}-\frac{\sin (G r)}{G^2} \Big)
     \Big( -\frac{G_{\mu}G_{\nu}}{G} \Big) dr -  Z_{\alpha}^p \Big( -\frac{e^{-\frac{G^2}{4}}}{2 G}-\frac{2 e^{-\frac{G^2}{4}}}{G^3} \Big) 
     \Big( -\frac{G_{\mu}G_{\nu}}{G} \Big)  \Bigg) = \\
     -\delta_{\mu \nu} \sum_{\bf G}\rho^{*}({\bf G}) V^{loc}({\bf G}) + \sum_{\bf G} \rho^{*}({\bf G}) \Delta V^{loc}({\bf G}) G_{\mu}G_{\nu}
\f]
where \f$ \Delta V^{loc}({\bf G}) \f$ is built from the following radial integrals:
\f[
  \int \Big(V_{\alpha}(r) r + Z_{\alpha}^p {\rm erf}(r) \Big) \Big( \frac{\sin (G r)}{G^3} - \frac{r \cos (G r)}{G^2}\Big) dr - 
    Z_{\alpha}^p \Big( \frac{e^{-\frac{G^2}{4}}}{2 G^2} + \frac{2 e^{-\frac{G^2}{4}}}{G^4} \Big)  
\f]

\section stress_section7 Non-local contribution to stress.
Energy contribution from the non-local part of pseudopotential:
\f[
  E^{nl} = \sum_{i{\bf k}} \sum_{\alpha}\sum_{\xi \xi'}P_{\xi}^{\alpha,i{\bf k}} D_{\xi\xi'}^{\alpha}P_{\xi'}^{\alpha,i{\bf k}*}
\f]
where
\f[
  P_{\xi}^{\alpha,i{\bf k}} = \langle \psi_{i{\bf k}} | \beta_{\xi}^{\alpha} \rangle =
    \frac{1}{\Omega} \sum_{\bf G} \langle \psi_{i{\bf k}} | {\bf G+k} \rangle \langle {\bf G+k} | \beta_{\xi}^{\alpha} \rangle  = 
    \frac{1}{\Omega} \sum_{\bf G} \tilde \psi_i^{*}({\bf G+k}) \tilde \beta_{\xi}^{\alpha}({\bf G+k}) 
\f]
We need to compute strain derivatives of \f$ P_{\xi}^{\alpha,i{\bf k}} \f$:
\f[
  \frac{\partial}{\partial \varepsilon_{\mu \nu}} P_{\xi}^{\alpha,i{\bf k}} = -\frac{1}{\Omega}\delta_{\mu \nu}
    \sum_{\bf G} \tilde \psi_i^{*}({\bf G+k}) \tilde \beta_{\xi}^{\alpha}({\bf G+k})  + \frac{1}{\Omega}
    \sum_{\bf G} \tilde \psi_i^{*}({\bf G+k}) \frac{\partial}{\partial \varepsilon_{\mu \nu}} \tilde \beta_{\xi}^{\alpha}({\bf G+k})  
\f]
\f[
  \tilde \beta_{\xi}^{\alpha}({\bf G+k}) = 4\pi e^{-i{\bf G r_{\alpha}}}(-i)^{\ell} R_{\ell m}(\theta_{G+k}, \phi_{G+k})
    \int \beta_{\ell}(r) j_{\ell}(|{\bf G+k}|r) r^2 dr
\f]
First way to take strain derivative of beta-projectors:
\f[
  \frac{\partial}{\partial \varepsilon_{\mu \nu}} \tilde \beta_{\xi}^{\alpha}({\bf G+k}) = 4\pi e^{-i{\bf G r_{\alpha}}}(-i)^{\ell} 
  \Big( \frac{\partial R_{\ell m}(\theta_{G+k}, \phi_{G+k})}{\partial \varepsilon_{\mu \nu}} 
    \int \beta_{\ell}(r) j_{\ell}(|{\bf G+k}|r) r^2 dr + R_{\ell m}(\theta_{G+k}, \phi_{G+k})
    \int \beta_{\ell}(r)    \frac{\partial j_{\ell}(|{\bf G+k}|r) }{\partial \varepsilon_{\mu \nu}}  r^2 dr 
  \Big)
\f]

Strain derivative of the real spherical harmonics (here and below \f$ {\bf G+k} = {\bf q} \f$):
\f[
  \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \varepsilon_{\mu \nu}} = 
    \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \theta} \sum_{\tau} \frac{\partial \theta}{\partial q_{\tau}} \frac{\partial{q_{\tau}}}{\partial \varepsilon_{\mu \nu}} + 
    \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \phi} \sum_{\tau} \frac{\partial \phi}{\partial q_{\tau}}\frac{\partial q_{\tau}}{\partial \varepsilon_{\mu \nu}} = 
    -q_{\mu} \Big(  \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \theta} \frac{\partial \theta}{\partial q_{\nu}} + 
            \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \phi} \frac{\partial \phi}{\partial q_{\nu}}\Big)
\f]

The derivatives of angles are:
\f[
   \frac{\partial \theta}{\partial q_{x}} = \frac{\cos(\phi) \cos(\theta)}{q} \\
   \frac{\partial \theta}{\partial q_{y}} = \frac{\cos(\theta) \sin(\phi)}{q} \\
   \frac{\partial \theta}{\partial q_{z}} = -\frac{\sin(\theta)}{q}
\f]
and
\f[
   \frac{\partial \phi}{\partial q_{x}} = -\frac{\sin(\phi)}{\sin(\theta) q} \\
   \frac{\partial \phi}{\partial q_{y}} = \frac{\cos(\phi)}{\sin(\theta) q} \\
   \frac{\partial \phi}{\partial q_{z}} = 0
\f]
The derivative of \f$ \phi \f$ has discontinuities at \f$ \theta = 0, \theta=\pi \f$. This, however, is not a problem, because
multiplication by the the derivative of \f$ R_{\ell m} \f$ removes it. The following functions have to be hardcoded:
\f[
  \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \theta} \\
  \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \phi} \frac{1}{\sin(\theta)} 
\f]

Mathematica script for spherical harmonic derivatives:
\verbatim
Rlm[l_, m_, th_, ph_] := 
 If[m > 0, Sqrt[2]*ComplexExpand[Re[SphericalHarmonicY[l, m, th, ph]]],
   If[m < 0, Sqrt[2]*ComplexExpand[Im[SphericalHarmonicY[l, m, th, ph]]], 
     If[m == 0, ComplexExpand[Re[SphericalHarmonicY[l, 0, th, ph]]]]
   ]
 ]
Do[Print[FullSimplify[D[Rlm[l, m, theta, phi], theta]]], {l, 0, 4}, {m, -l, l}]
Do[Print[FullSimplify[TrigExpand[D[Rlm[l, m, theta, phi], phi]/Sin[theta]]]], {l, 0, 4}, {m, -l, l}]
\endverbatim

Strain derivative of spherical Bessel function integral:
\f[
  \int \beta_{\ell}(r) \frac{\partial j_{\ell}(qr) }{\partial \varepsilon_{\mu \nu}}  r^2 dr = 
   \int \beta_{\ell}(r) \frac{\partial j_{\ell}(qr) }{\partial q} \frac{-q_{\mu} q_{\nu}}{q} r^2 dr  
\f]

 */
