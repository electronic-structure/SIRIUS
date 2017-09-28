#ifndef __BETA_PROJECTORS_STRAIN_DERIV_H__
#define __BETA_PROJECTORS_STRAIN_DERIV_H__

#include "beta_projectors_base.h"

namespace sirius {

class Beta_projectors_strain_deriv : public Beta_projectors_base<9>
{
  private:

    void generate_pw_coefs_t()
    {
        PROFILE("sirius::Beta_projectors_strain_deriv::generate_pw_coefs_t");

        auto& bchunk = ctx_.beta_projector_chunks();
        if (!bchunk.num_beta_t()) {
            return;
        }

        auto& beta_ri0 = ctx_.beta_ri();
        auto& beta_ri1 = ctx_.beta_ri_djl();

        auto& comm = gkvec_.comm();

        int lmax = ctx_.unit_cell().lmax();
        int lmmax = Utils::lmmax(lmax);

        mdarray<double, 2> rlm_g(lmmax, num_gkvec_loc());
        mdarray<double, 3> rlm_dg(lmmax, 3, num_gkvec_loc());

        /* array of real spherical harmonics and derivatives for each G-vector */
        #pragma omp parallel for schedule(static)
        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) {
            int igk  = gkvec_.gvec_offset(comm.rank()) + igkloc;
            auto gvc = gkvec_.gkvec_cart(igk);
            auto rtp = SHT::spherical_coordinates(gvc);

            double theta = rtp[1];
            double phi   = rtp[2];
            //vector3d<double> dtheta_dq({std::cos(phi) * std::cos(theta), std::cos(theta) * std::sin(phi), -std::sin(theta)});
            //vector3d<double> dphi_dq({-std::sin(phi), std::cos(phi), 0.0});

            SHT::spherical_harmonics(lmax, theta, phi, &rlm_g(0, igkloc));
            mdarray<double, 2> rlm_dg_tmp(&rlm_dg(0, 0, igkloc), lmmax, 3);
            if (rtp[0] > 1e-12) {
                SHT::dRlm_dr(lmax, gvc, rlm_dg_tmp);
            } else {
                rlm_dg_tmp.zero();
            }
            
            //mdarray<double, 1> dRlm_dtheta(lmmax);
            //mdarray<double, 1> dRlm_dphi_sin_theta(lmmax);

            //SHT::dRlm_dtheta(lmax, theta, phi, dRlm_dtheta);
            //SHT::dRlm_dphi_sin_theta(lmax, theta, phi, dRlm_dphi_sin_theta);
            //for (int nu = 0; nu < 3; nu++) {
            //    for (int lm = 0; lm < lmmax; lm++) {
            //        rlm_dg(lm, nu, igkloc) = dRlm_dtheta[lm] * dtheta_dq[nu] + dRlm_dphi_sin_theta[lm] * dphi_dq[nu];
            //    }
            //}
        }

        /* compute d <G+k|beta> / d epsilon_{mu, nu} */
        #pragma omp parallel for schedule(static)
        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) {
            int igk  = gkvec_.gvec_offset(comm.rank()) + igkloc;
            auto gvc = gkvec_.gkvec_cart(igk);
            /* vs = {r, theta, phi} */
            auto gvs = SHT::spherical_coordinates(gvc);

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

                                    pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) = z * d1;
                                } else {
                                    pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) = 0;
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

                            pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) = z * (d1 + d2);
                        }
                    }
                }
            }
        }
    }

    void generate_pw_coefs_t_v2()
    {
        PROFILE("sirius::Beta_projectors_strain_deriv::generate_pw_coefs_t_v2");

        auto& bchunk = ctx_.beta_projector_chunks();
        if (!bchunk.num_beta_t()) {
            return;
        }

        Radial_integrals_beta<false> beta_ri0(ctx_.unit_cell(), ctx_.gk_cutoff(), ctx_.settings().nprii_beta_);
        Radial_integrals_beta_jl beta_ri1(ctx_.unit_cell(), ctx_.gk_cutoff(), ctx_.settings().nprii_beta_);

        vector3d<int> r_m({1, -1, 0});
        vector3d<double> r_f({-2 * std::sqrt(pi / 3), -2 * std::sqrt(pi / 3), 2 * std::sqrt(pi / 3)});

        auto& comm = gkvec_.comm();

        /* zero array */
        for (int i = 0; i < 9; i++) {
            pw_coeffs_t_[i].zero();
        }

        std::vector<double_complex> zil(lmax_beta_ + 2);
        for (int l = 0; l < lmax_beta_ + 2; l++) {
            zil[l] = std::pow(double_complex(0, -1), l);
        }

        Gaunt_coefficients<double> gc(1, lmax_beta_ + 2, lmax_beta_, SHT::gaunt_rlm);

        /* compute d <G+k|beta> / d epsilon_{mu, nu} */
        #pragma omp parallel for schedule(static)
        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) {
            int igk  = gkvec_.gvec_offset(comm.rank()) + igkloc;
            auto gvc = gkvec_.gkvec_cart(igk);
            /* vs = {r, theta, phi} */
            auto gvs = SHT::spherical_coordinates(gvc);

            /* compute real spherical harmonics for G+k vector */
            std::vector<double> gkvec_rlm(Utils::lmmax(lmax_beta_ + 2));
            SHT::spherical_harmonics(lmax_beta_ + 2, gvs[1], gvs[2], &gkvec_rlm[0]);

            mdarray<double, 3> tmp(ctx_.unit_cell().max_mt_radial_basis_size(), lmax_beta_ + 3, ctx_.unit_cell().num_atom_types());
            for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
                for (int l = 0; l <= lmax_beta_ + 2; l++) {
                    for (int j = 0; j < ctx_.unit_cell().atom_type(iat).mt_radial_basis_size(); j++) {
                        tmp(j, l, iat) = beta_ri1.value<int, int, int>(j, l, iat, gvs[0]);
                    }
                }
            }

            for (int nu = 0; nu < 3; nu++) {
                for (int mu = 0; mu < 3; mu++) {
                    double p = (mu == nu) ? 0.5 : 0;

                    auto z1 = fourpi / std::sqrt(ctx_.unit_cell().omega());
                    auto z2 = z1 * gvc[mu] * double_complex(0, 1) * r_f[nu];

                    for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
                        auto& atom_type = ctx_.unit_cell().atom_type(iat);
                        for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                            int l     = atom_type.indexb(xi).l;
                            int lm    = atom_type.indexb(xi).lm;
                            int idxrf = atom_type.indexb(xi).idxrf;
                            
                            double_complex z3(0, 0);
                            for (int k = 0; k < gc.num_gaunt(Utils::lm_by_l_m(1, r_m[nu]), lm); k++) {
                                auto& c = gc.gaunt(Utils::lm_by_l_m(1, r_m[nu]), lm, k);
                                int l3 = c.l3;
                                z3 += zil[l3] * gkvec_rlm[c.lm3] * c.coef * tmp(idxrf, l3, iat);
                            }

                                //pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) += 
                                //    gvc[mu] * std::pow(double_complex(0, -1), l3) * z * double_complex(0, 1) * 
                                //    gkvec_rlm[c.lm3] * c.coef * tmp(idxrf, l3, iat) * r_f[nu];
                            pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) += z3 * z2;

                            auto d2 = beta_ri0.value<int, int>(idxrf, iat, gvs[0]) * (-p * gkvec_rlm[lm]);

                            pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) += z1 * d2 * zil[l];
                        }
                    }
                } // mu
                //for (int mu = 0; mu <= nu; mu++) {
                //    for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                //        pw_coeffs_t_[nu + mu * 3](igkloc, atom_type.offset_lo() + xi) = pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi);
                //    }
                //}
            } // nu
        }
    }

  public:
    Beta_projectors_strain_deriv(Simulation_context& ctx__,
                                 Gvec const&         gkvec__)
        : Beta_projectors_base<9>(ctx__, gkvec__)
    {
        generate_pw_coefs_t();
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

