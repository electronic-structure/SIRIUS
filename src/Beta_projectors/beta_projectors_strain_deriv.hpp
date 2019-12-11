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

class Beta_projectors_strain_deriv : public Beta_projectors_base
{
  private:

    void generate_pw_coefs_t(std::vector<int> const& igk__)
    {
        PROFILE("sirius::Beta_projectors_strain_deriv::generate_pw_coefs_t");

        if (!num_beta_t()) {
            return;
        }

        auto& beta_ri0 = ctx_.beta_ri();
        auto& beta_ri1 = ctx_.beta_ri_djl();

        int lmax = ctx_.unit_cell().lmax();
        int lmmax = utils::lmmax(lmax);

        mdarray<double, 2> rlm_g(lmmax, num_gkvec_loc());
        mdarray<double, 3> rlm_dg(lmmax, 3, num_gkvec_loc());

        /* array of real spherical harmonics and derivatives for each G-vector */
        #pragma omp parallel for schedule(static)
        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) {
            auto gvc = gkvec_.gkvec_cart<index_domain_t::global>(igk__[igkloc]);
            auto rtp = SHT::spherical_coordinates(gvc);

            double theta = rtp[1];
            double phi   = rtp[2];

            sf::spherical_harmonics(lmax, theta, phi, &rlm_g(0, igkloc));
            mdarray<double, 2> rlm_dg_tmp(&rlm_dg(0, 0, igkloc), lmmax, 3);
            sf::dRlm_dr(lmax, gvc, rlm_dg_tmp);
        }

        /* compute d <G+k|beta> / d epsilon_{mu, nu} */
        #pragma omp parallel for schedule(static)
        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) {
            auto gvc = gkvec_.gkvec_cart<index_domain_t::global>(igk__[igkloc]);
            /* vs = {r, theta, phi} */
            auto gvs = SHT::spherical_coordinates(gvc);

            /* |G+k|=0 case */
            if (gvs[0] < 1e-10) {
                for (int nu = 0; nu < 3; nu++) {
                    for (int mu = 0; mu < 3; mu++) {
                        double p = (mu == nu) ? 0.5 : 0;

                        for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
                            auto& atom_type = ctx_.unit_cell().atom_type(iat);
                            for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                                int l     = atom_type.indexb(xi).l;
                                int idxrf = atom_type.indexb(xi).idxrf;

                                if (l == 0) {
                                    auto z = fourpi / std::sqrt(ctx_.unit_cell().omega());

                                    auto d1 = beta_ri0.value<int, int>(idxrf, iat, gvs[0]) * (-p * y00);

                                    pw_coeffs_t_(igkloc, atom_type.offset_lo() + xi, mu + nu * 3) = z * d1;
                                } else {
                                    pw_coeffs_t_(igkloc, atom_type.offset_lo() + xi, mu + nu * 3) = 0;
                                }
                            }
                        }
                    }
                }
                continue;
            }

            for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
                auto& atom_type = ctx_.unit_cell().atom_type(iat);

                auto ri0 = beta_ri0.values(iat, gvs[0]);
                auto ri1 = beta_ri1.values(iat, gvs[0]);

                for (int nu = 0; nu < 3; nu++) {
                    for (int mu = 0; mu < 3; mu++) {
                        double p = (mu == nu) ? 0.5 : 0;

                        for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                            int l     = atom_type.indexb(xi).l;
                            int lm    = atom_type.indexb(xi).lm;
                            int idxrf = atom_type.indexb(xi).idxrf;

                            auto z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(ctx_.unit_cell().omega());

                            auto d1 = ri0(idxrf) * (-gvc[mu] * rlm_dg(lm, nu, igkloc) - p * rlm_g(lm, igkloc));

                            auto d2 = ri1(idxrf) * rlm_g(lm, igkloc) * (-gvc[mu] * gvc[nu] / gvs[0]);

                            pw_coeffs_t_(igkloc, atom_type.offset_lo() + xi, mu + nu * 3) = z * (d1 + d2);
                        }
                    }
                }
            }
        }
    }

    //void generate_pw_coefs_t_v2()
    //{
    //    PROFILE("sirius::Beta_projectors_strain_deriv::generate_pw_coefs_t_v2");

    //    auto& bchunk = ctx_.beta_projector_chunks();
    //    if (!bchunk.num_beta_t()) {
    //        return;
    //    }

    //    Radial_integrals_beta<false> beta_ri0(ctx_.unit_cell(), ctx_.gk_cutoff(), ctx_.settings().nprii_beta_);
    //    Radial_integrals_beta_jl beta_ri1(ctx_.unit_cell(), ctx_.gk_cutoff(), ctx_.settings().nprii_beta_);

    //    vector3d<int> r_m({1, -1, 0});
    //    vector3d<double> r_f({-2 * std::sqrt(pi / 3), -2 * std::sqrt(pi / 3), 2 * std::sqrt(pi / 3)});

    //    auto& comm = gkvec_.comm();

    //    /* zero array */
    //    for (int i = 0; i < 9; i++) {
    //        pw_coeffs_t_[i].zero();
    //    }

    //    std::vector<double_complex> zil(lmax_beta_ + 2);
    //    for (int l = 0; l < lmax_beta_ + 2; l++) {
    //        zil[l] = std::pow(double_complex(0, -1), l);
    //    }

    //    Gaunt_coefficients<double> gc(1, lmax_beta_ + 2, lmax_beta_, SHT::gaunt_rlm);

    //    /* compute d <G+k|beta> / d epsilon_{mu, nu} */
    //    #pragma omp parallel for schedule(static)
    //    for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) {
    //        int igk  = gkvec_.gvec_offset(comm.rank()) + igkloc;
    //        auto gvc = gkvec_.gkvec_cart(igk);
    //        /* vs = {r, theta, phi} */
    //        auto gvs = SHT::spherical_coordinates(gvc);

    //        /* compute real spherical harmonics for G+k vector */
    //        std::vector<double> gkvec_rlm(utils::lmmax(lmax_beta_ + 2));
    //        SHT::spherical_harmonics(lmax_beta_ + 2, gvs[1], gvs[2], &gkvec_rlm[0]);

    //        mdarray<double, 3> tmp(ctx_.unit_cell().max_mt_radial_basis_size(), lmax_beta_ + 3, ctx_.unit_cell().num_atom_types());
    //        for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
    //            for (int l = 0; l <= lmax_beta_ + 2; l++) {
    //                for (int j = 0; j < ctx_.unit_cell().atom_type(iat).mt_radial_basis_size(); j++) {
    //                    tmp(j, l, iat) = beta_ri1.value<int, int, int>(j, l, iat, gvs[0]);
    //                }
    //            }
    //        }

    //        for (int nu = 0; nu < 3; nu++) {
    //            for (int mu = 0; mu < 3; mu++) {
    //                double p = (mu == nu) ? 0.5 : 0;

    //                auto z1 = fourpi / std::sqrt(ctx_.unit_cell().omega());
    //                auto z2 = z1 * gvc[mu] * double_complex(0, 1) * r_f[nu];

    //                for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
    //                    auto& atom_type = ctx_.unit_cell().atom_type(iat);
    //                    for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
    //                        int l     = atom_type.indexb(xi).l;
    //                        int lm    = atom_type.indexb(xi).lm;
    //                        int idxrf = atom_type.indexb(xi).idxrf;
    //                        
    //                        double_complex z3(0, 0);
    //                        for (int k = 0; k < gc.num_gaunt(Utils::lm_by_l_m(1, r_m[nu]), lm); k++) {
    //                            auto& c = gc.gaunt(Utils::lm_by_l_m(1, r_m[nu]), lm, k);
    //                            int l3 = c.l3;
    //                            z3 += zil[l3] * gkvec_rlm[c.lm3] * c.coef * tmp(idxrf, l3, iat);
    //                        }

    //                            //pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) += 
    //                            //    gvc[mu] * std::pow(double_complex(0, -1), l3) * z * double_complex(0, 1) * 
    //                            //    gkvec_rlm[c.lm3] * c.coef * tmp(idxrf, l3, iat) * r_f[nu];
    //                        pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) += z3 * z2;

    //                        auto d2 = beta_ri0.value<int, int>(idxrf, iat, gvs[0]) * (-p * gkvec_rlm[lm]);

    //                        pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) += z1 * d2 * zil[l];
    //                    }
    //                }
    //            } // mu
    //            //for (int mu = 0; mu <= nu; mu++) {
    //            //    for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
    //            //        pw_coeffs_t_[nu + mu * 3](igkloc, atom_type.offset_lo() + xi) = pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi);
    //            //    }
    //            //}
    //        } // nu
    //    }
    //}

  public:
    Beta_projectors_strain_deriv(Simulation_context&     ctx__,
                                 Gvec const&             gkvec__,
                                 std::vector<int> const& igk__)
        : Beta_projectors_base(ctx__, gkvec__, igk__, 9)
    {
        generate_pw_coefs_t(igk__);
        //generate_pw_coefs_t_v2();

        //if (ctx__.processing_unit() == GPU) {
        //    for (int j = 0; j < 9; j++) {
        //        pw_coeffs_t_[j].copy<memory_t::host, memory_t::device>();
        //    }
        //}
    }
};

}

#endif

