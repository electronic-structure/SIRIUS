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

/** \file beta_projectors_strain_deriv.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Beta_projectors_strain_deriv class.
 */

#ifndef __BETA_PROJECTORS_STRAIN_DERIV_HPP__
#define __BETA_PROJECTORS_STRAIN_DERIV_HPP__

#include "beta_projectors_base.hpp"

namespace sirius {

template <typename T>
class Beta_projectors_strain_deriv : public Beta_projectors_base<T>
{
  private:

    void generate_pw_coefs_t()
    {
        PROFILE("sirius::Beta_projectors_strain_deriv::generate_pw_coefs_t");

        if (!this->num_beta_t()) {
            return;
        }

        auto& beta_ri0 = this->ctx_.beta_ri();
        auto& beta_ri1 = this->ctx_.beta_ri_djl();

        int lmax = this->ctx_.unit_cell().lmax();
        int lmmax = utils::lmmax(lmax);

        sddk::mdarray<double, 2> rlm_g(lmmax, this->num_gkvec_loc());
        sddk::mdarray<double, 3> rlm_dg(lmmax, 3, this->num_gkvec_loc());

        /* array of real spherical harmonics and derivatives for each G-vector */
        #pragma omp parallel for schedule(static)
        for (int igkloc = 0; igkloc < this->num_gkvec_loc(); igkloc++) {
            auto gvc = this->gkvec_.template gkvec_cart<sddk::index_domain_t::local>(igkloc);
            auto rtp = SHT::spherical_coordinates(gvc);

            double theta = rtp[1];
            double phi   = rtp[2];

            sf::spherical_harmonics(lmax, theta, phi, &rlm_g(0, igkloc));
            sddk::mdarray<double, 2> rlm_dg_tmp(&rlm_dg(0, 0, igkloc), lmmax, 3);
            sf::dRlm_dr(lmax, gvc, rlm_dg_tmp);
        }

        /* compute d <G+k|beta> / d epsilon_{mu, nu} */
        #pragma omp parallel for schedule(static)
        for (int igkloc = 0; igkloc < this->num_gkvec_loc(); igkloc++) {
            auto gvc = this->gkvec_.template gkvec_cart<sddk::index_domain_t::local>(igkloc);
            /* vs = {r, theta, phi} */
            auto gvs = SHT::spherical_coordinates(gvc);

            /* |G+k|=0 case */
            if (gvs[0] < 1e-10) {
                for (int nu = 0; nu < 3; nu++) {
                    for (int mu = 0; mu < 3; mu++) {
                        double p = (mu == nu) ? 0.5 : 0;

                        for (int iat = 0; iat < this->ctx_.unit_cell().num_atom_types(); iat++) {
                            auto& atom_type = this->ctx_.unit_cell().atom_type(iat);
                            auto ri0 = beta_ri0.values(iat, gvs[0]);

                            for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                                int l     = atom_type.indexb(xi).l;
                                int idxrf = atom_type.indexb(xi).idxrf;

                                if (l == 0) {
                                    auto z = fourpi / std::sqrt(this->ctx_.unit_cell().omega());

                                    auto d1 = ri0(idxrf) * (-p * y00);

                                    this->pw_coeffs_t_(igkloc, atom_type.offset_lo() + xi, mu + nu * 3) = z * d1;
                                } else {
                                    this->pw_coeffs_t_(igkloc, atom_type.offset_lo() + xi, mu + nu * 3) = 0;
                                }
                            }
                        }
                    }
                }
                continue;
            }

            for (int iat = 0; iat < this->ctx_.unit_cell().num_atom_types(); iat++) {
                auto& atom_type = this->ctx_.unit_cell().atom_type(iat);

                auto ri0 = beta_ri0.values(iat, gvs[0]);
                auto ri1 = beta_ri1.values(iat, gvs[0]);

                for (int nu = 0; nu < 3; nu++) {
                    for (int mu = 0; mu < 3; mu++) {
                        double p = (mu == nu) ? 0.5 : 0;

                        for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                            int l     = atom_type.indexb(xi).l;
                            int lm    = atom_type.indexb(xi).lm;
                            int idxrf = atom_type.indexb(xi).idxrf;

                            auto z = std::pow(std::complex<double>(0, -1), l) * fourpi / std::sqrt(this->ctx_.unit_cell().omega());

                            auto d1 = ri0(idxrf) * (-gvc[mu] * rlm_dg(lm, nu, igkloc) - p * rlm_g(lm, igkloc));

                            auto d2 = ri1(idxrf) * rlm_g(lm, igkloc) * (-gvc[mu] * gvc[nu] / gvs[0]);

                            this->pw_coeffs_t_(igkloc, atom_type.offset_lo() + xi, mu + nu * 3) = static_cast<std::complex<T>>(z * (d1 + d2));
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

}

#endif

