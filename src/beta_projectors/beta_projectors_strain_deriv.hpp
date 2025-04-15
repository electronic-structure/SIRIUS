/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file beta_projectors_strain_deriv.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Beta_projectors_strain_deriv class.
 */

#ifndef __BETA_PROJECTORS_STRAIN_DERIV_HPP__
#define __BETA_PROJECTORS_STRAIN_DERIV_HPP__

#include "beta_projectors_base.hpp"

namespace sirius {

/// Compute derivative of beta-projectors with respect to strain tensor.
/**
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
*  Second way to compute strain derivative of beta-projectors is trough Gaunt coefficients:
*  \f[
*   \frac{\partial}{\partial \varepsilon_{\mu \nu}} \beta_{\xi}^{\alpha}({\bf q}) =
*   -\frac{1}{2\sqrt{\Omega}} \delta_{\mu \nu} \int e^{-i{\bf q r}} \beta_{\xi}^{\alpha}({\bf r}) d{\bf r} +
*   \frac{1}{\sqrt{\Omega}} \int i r_{\nu} q_{\mu} e^{-i{\bf q r}} \beta_{\xi}^{\alpha}({\bf r}) d{\bf r}
*  \f]
*  (second term comes from the strain derivative of \f$ e^{-i{\bf q r}} \f$). Remembering that \f$ {\bf r} \f$ is
*  proportional to p-like real spherical harmonics, we can rewrite the second part of beta-projector derivative as:
*  \f[
*   \frac{1}{\sqrt{\Omega}} \int i r_{\nu} q_{\mu} e^{-i{\bf q r}} \beta_{\xi}^{\alpha}({\bf r}) d{\bf r} =
*    \frac{1}{\sqrt{\Omega}} i q_{\mu} \int r \bar R_{1 \nu}(\theta, \phi) 4\pi \sum_{\ell_3 m_3} (-i)^{\ell_3}
*   R_{\ell_3 m_3}(\theta_q, \phi_q) R_{\ell_3 m_3}(\theta, \phi) j_{\ell_3}(q r) \beta_{\ell_2}^{\alpha}(r)
*   R_{\ell_2 m_2}(\theta, \phi) d{\bf r} = \\
*   \frac{4 \pi}{\sqrt{\Omega}} i q_{\mu} \sum_{\ell_3 m_3} (-i)^{\ell_3} R_{\ell_3 m_3}(\theta_q, \phi_q)
*      \langle \bar R_{1\nu} | R_{\ell_3 m_3} | R_{\ell_2 m_2} \rangle
*      \int j_{\ell_3}(q r) \beta_{\ell_2}^{\alpha}(r) r^3 dr
*  \f]
*  where
*  \f[
*    \bar R_{1 x}(\theta, \phi) = -2 \sqrt{\frac{\pi}{3}} R_{11}(\theta, \phi) = \sin(\theta) \cos(\phi) \\
*    \bar R_{1 y}(\theta, \phi) = -2 \sqrt{\frac{\pi}{3}} R_{1-1}(\theta,\phi) = \sin(\theta) \sin(\phi) \\
*    \bar R_{1 z}(\theta, \phi) = 2 \sqrt{\frac{\pi}{3}} R_{10}(\theta, \phi) = \cos(\theta)
*  \f]
*/
template <typename T>
class Beta_projectors_strain_deriv : public Beta_projectors_base<T>
{
  private:
    void
    generate_pw_coefs_t()
    {
        PROFILE("sirius::Beta_projectors_strain_deriv::generate_pw_coefs_t");

        if (!this->num_beta_t()) {
            return;
        }

        auto& uc = this->ctx_.unit_cell();

        std::vector<int> offset_t(uc.num_atom_types());
        std::generate(offset_t.begin(), offset_t.end(), [n = 0, iat = 0, &uc]() mutable {
            int offs = n;
            n += uc.atom_type(iat++).mt_basis_size();
            return offs;
        });

        auto& beta_ri0 = *this->ctx_.ri().beta_;
        auto& beta_ri1 = *this->ctx_.ri().beta_djl_;

        int lmax  = uc.lmax();
        int lmmax = sf::lmmax(lmax);

        mdarray<double, 2> rlm_g({lmmax, this->num_gkvec_loc()});
        mdarray<double, 3> rlm_dg({lmmax, 3, this->num_gkvec_loc()});

        /* array of real spherical harmonics and derivatives for each G-vector */
        #pragma omp parallel for schedule(static)
        for (int igkloc = 0; igkloc < this->num_gkvec_loc(); igkloc++) {
            auto gvc = this->gkvec_.gkvec_cart(gvec_index_t::local(igkloc));
            auto rtp = r3::spherical_coordinates(gvc);

            double theta = rtp[1];
            double phi   = rtp[2];

            sf::spherical_harmonics(lmax, theta, phi, &rlm_g(0, igkloc));
            mdarray<double, 2> rlm_dg_tmp({lmmax, 3}, &rlm_dg(0, 0, igkloc));
            sf::dRlm_dr(lmax, gvc, rlm_dg_tmp);
        }

        this->pw_coeffs_t_.zero(memory_t::host);

        /* compute d <G+k|beta> / d epsilon_{mu, nu} */
        #pragma omp parallel for schedule(static)
        for (int igkloc = 0; igkloc < this->num_gkvec_loc(); igkloc++) {
            auto gvc = this->gkvec_.gkvec_cart(gvec_index_t::local(igkloc));
            /* vs = {r, theta, phi} */
            auto gvs = r3::spherical_coordinates(gvc);

            auto inv_len = (gvs[0] < 1e-10) ? 0 : 1.0 / gvs[0];

            for (int iat = 0; iat < uc.num_atom_types(); iat++) {
                auto& atom_type = uc.atom_type(iat);

                auto ri0 = beta_ri0.values(iat, gvs[0]);
                auto ri1 = beta_ri1.values(iat, gvs[0]);

                for (int nu = 0; nu < 3; nu++) {
                    for (int mu = 0; mu < 3; mu++) {
                        double p = (mu == nu) ? 0.5 : 0;

                        for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                            int l     = atom_type.indexb(xi).am.l();
                            int lm    = atom_type.indexb(xi).lm;
                            int idxrf = atom_type.indexb(xi).idxrf;

                            auto z = std::pow(std::complex<double>(0, -1), l) * fourpi / std::sqrt(uc.omega());

                            auto d1 = ri0(idxrf) * (-gvc[mu] * rlm_dg(lm, nu, igkloc) - p * rlm_g(lm, igkloc));

                            auto d2 = ri1(idxrf) * rlm_g(lm, igkloc) * (-gvc[mu] * gvc[nu] * inv_len);

                            this->pw_coeffs_t_(igkloc, offset_t[atom_type.id()] + xi, mu + nu * 3) =
                                    static_cast<std::complex<T>>(z * (d1 + d2));
                        }
                    }
                }
            }
        }
    }

  public:
    Beta_projectors_strain_deriv(Simulation_context& ctx__, fft::Gvec const& gkvec__)
        : Beta_projectors_base<T>(ctx__, gkvec__, 9)
    {
        generate_pw_coefs_t();
    }
};

} // namespace sirius

#endif
