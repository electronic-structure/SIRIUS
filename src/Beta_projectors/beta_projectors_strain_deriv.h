#ifndef __BETA_PROJECTORS_STRAIN_DERIV_H__
#define __BETA_PROJECTORS_STRAIN_DERIV_H__

#include "beta_projectors_base.h"

namespace sirius {

class Beta_projectors_strain_deriv : public Beta_projectors_base<9>
{
  private:

    void generate_pw_coefs_t()
    {
        auto& bchunk = ctx_.beta_projector_chunks();
        if (!bchunk.num_beta_t()) {
            return;
        }

        Radial_integrals_beta beta_ri0(ctx_.unit_cell(), ctx_.gk_cutoff(), 20);
        Radial_integrals_beta_dg beta_ri1(ctx_.unit_cell(), ctx_.gk_cutoff(), 20);

        auto& comm = gkvec_.comm();

        /* allocate array */
        for (int i = 0; i < 9; i++) {
            pw_coeffs_t_[i] = matrix<double_complex>(num_gkvec_loc(), bchunk.num_beta_t());
        }

        auto dRlm_deps = [this](int lm, vector3d<double>& gvs, int mu, int nu)
        {
            double theta = gvs[1];
            double phi   = gvs[2];
            
            if (lm == 0) {
                return 0.0;
            }

            vector3d<double> q({std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)});
            vector3d<double> dtheta_dq({std::cos(phi) * std::cos(theta), std::cos(theta) * std::sin(phi), -std::sin(theta)});
            vector3d<double> dphi_dq({-std::sin(phi), std::cos(phi), 0.0});

            return -q[mu] * (SHT::dRlm_dtheta(lm, theta, phi) * dtheta_dq[nu] +
                             SHT::dRlm_dphi_sin_theta(lm, theta, phi) * dphi_dq[nu]);
        };

        //auto dRlm_deps_v2 = [this](int lm, vector3d<double>& gvc, vector3d<double>& gvs, int mu, int nu)
        //{
        //    int lmax = 4;
        //    int lmmax = Utils::lmmax(lmax);

        //    double dg = 1e-6 * gvs[0];

        //    mdarray<double, 2>drlm(lmmax, 3);

        //    for (int x = 0; x < 3; x++) {
        //        vector3d<double> g1 = gvc;
        //        g1[x] += dg;
        //        vector3d<double> g2 = gvc;
        //        g2[x] -= dg;
        //        
        //        auto gs1 = SHT::spherical_coordinates(g1);
        //        auto gs2 = SHT::spherical_coordinates(g2);
        //        std::vector<double> rlm1(lmmax);
        //        std::vector<double> rlm2(lmmax);
        //        
        //        SHT::spherical_harmonics(lmax, gs1[1], gs1[2], &rlm1[0]);
        //        SHT::spherical_harmonics(lmax, gs2[1], gs2[2], &rlm2[0]);
        //        
        //        for (int lm = 0; lm < lmmax; lm++) {
        //            drlm(lm, x) = (rlm1[lm] - rlm2[lm]) / 2 / dg;
        //        }
        //    }

        //    return -gvc[mu] * drlm(lm, nu);
        //};

        /* compute d <G+k|beta> / d epsilon_{mu, nu} */
        #pragma omp parallel for
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

                                    auto d1 = beta_ri0.value(idxrf, iat, gvs[0]) * (-p * y00);

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

            for (int nu = 0; nu < 3; nu++) {
                for (int mu = 0; mu < 3; mu++) {
                    double p = (mu == nu) ? 0.5 : 0;
                    /* compute real spherical harmonics for G+k vector */
                    std::vector<double> gkvec_rlm(Utils::lmmax(lmax_beta_));
                    std::vector<double> gkvec_drlm(Utils::lmmax(lmax_beta_));

                    SHT::spherical_harmonics(lmax_beta_, gvs[1], gvs[2], &gkvec_rlm[0]);

                    for (int lm = 0; lm < Utils::lmmax(lmax_beta_); lm++) {
                        gkvec_drlm[lm] = dRlm_deps(lm, gvs, mu, nu);
                        //gkvec_drlm[lm] = dRlm_deps_v2(lm, gvc, gvs, mu, nu);
                    }

                    for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
                        auto& atom_type = ctx_.unit_cell().atom_type(iat);
                        for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                            int l     = atom_type.indexb(xi).l;
                            int lm    = atom_type.indexb(xi).lm;
                            int idxrf = atom_type.indexb(xi).idxrf;

                            auto z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(ctx_.unit_cell().omega());

                            auto d1 = beta_ri0.value(idxrf, iat, gvs[0]) * (gkvec_drlm[lm] - p * gkvec_rlm[lm]);

                            auto d2 = beta_ri1.value(idxrf, iat, gvs[0]) * gkvec_rlm[lm];

                            pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) = z * (d1 - d2 * gvc[mu] * gvc[nu] / gvs[0]);
                        }
                    }
                }
            }
        }
    }

    void generate_pw_coefs_t_v2()
    {
        auto& bchunk = ctx_.beta_projector_chunks();
        if (!bchunk.num_beta_t()) {
            return;
        }

        Radial_integrals_beta beta_ri0(ctx_.unit_cell(), ctx_.gk_cutoff(), 20);
        Radial_integrals_beta_jl beta_ri1(ctx_.unit_cell(), ctx_.gk_cutoff(), 20);

        vector3d<int> r_m({1, -1, 0});
        vector3d<double> r_f({-2 * std::sqrt(pi / 3), -2 * std::sqrt(pi / 3), 2 * std::sqrt(pi / 3)});

        auto& comm = gkvec_.comm();

        /* allocate array */
        for (int i = 0; i < 9; i++) {
            pw_coeffs_t_[i] = matrix<double_complex>(num_gkvec_loc(), bchunk.num_beta_t());
            pw_coeffs_t_[i].zero();
        }

        /* compute d <G+k|beta> / d epsilon_{mu, nu} */
        #pragma omp parallel for
        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) {
            int igk  = gkvec_.gvec_offset(comm.rank()) + igkloc;
            auto gvc = gkvec_.gkvec_cart(igk);
            /* vs = {r, theta, phi} */
            auto gvs = SHT::spherical_coordinates(gvc);

            for (int nu = 0; nu < 3; nu++) {
                for (int mu = 0; mu < 3; mu++) {
                    double p = (mu == nu) ? 0.5 : 0;

                    /* compute real spherical harmonics for G+k vector */
                    std::vector<double> gkvec_rlm(Utils::lmmax(lmax_beta_ + 2));

                    SHT::spherical_harmonics(lmax_beta_ + 2, gvs[1], gvs[2], &gkvec_rlm[0]);
                    
                    auto z = fourpi / std::sqrt(ctx_.unit_cell().omega());

                    for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
                        auto& atom_type = ctx_.unit_cell().atom_type(iat);
                        for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                            int l     = atom_type.indexb(xi).l;
                            int m     = atom_type.indexb(xi).m;
                            int lm    = atom_type.indexb(xi).lm;
                            int idxrf = atom_type.indexb(xi).idxrf;

                            for (int l1 = 0; l1 <= l + 1; l1++) {    
                                auto d1 = beta_ri1.value(idxrf, l1, iat, gvs[0]);
                                for (int m1 = -l1; m1 <= l1; m1++) {
                                    auto c = SHT::gaunt_rlm(l1, 1, l, m1, r_m[nu], m);
                                    pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) += 
                                        gvc[mu] * std::pow(double_complex(0, -1), l1) * z * double_complex(0, 1) * 
                                        gkvec_rlm[Utils::lm_by_l_m(l1, m1)] * c * d1 * r_f[nu];
                                }
                            }
                            
                            auto d2 = beta_ri0.value(idxrf, iat, gvs[0]) * (-p * gkvec_rlm[lm]);

                            pw_coeffs_t_[mu + nu * 3](igkloc, atom_type.offset_lo() + xi) += z * d2 * std::pow(double_complex(0, -1), l);
                        }
                    }
                }
            }
        }

    }


  public:
    Beta_projectors_strain_deriv(Simulation_context& ctx__,
                                 Gvec         const& gkvec__)
        : Beta_projectors_base<9>(ctx__, gkvec__)
    {
        //generate_pw_coefs_t();
        generate_pw_coefs_t_v2();
    }
    
    /// Generate strain derivatives of beta-projectors for a chunk of atoms.
    void generate(int ichunk__)
    {
        auto& bchunk = ctx_.beta_projector_chunks();

        int num_beta = bchunk(ichunk__).num_beta_;

        auto& comm = gkvec_.comm();

        /* allocate array */
        for (int i = 0; i < 9; i++) {
            pw_coeffs_a_[i] = matrix<double_complex>(num_gkvec_loc(), num_beta);
        }

        #pragma omp for
        for (int i = 0; i < bchunk(ichunk__).num_atoms_; i++) {
            int ia = bchunk(ichunk__).desc_(beta_desc_idx::ia, i);

            double phase = twopi * (gkvec_.vk() * ctx_.unit_cell().atom(ia).position());
            double_complex phase_k = std::exp(double_complex(0.0, phase));

            std::vector<double_complex> phase_gk(num_gkvec_loc());
            for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++) {
                int igk = gkvec_.gvec_offset(comm.rank()) + igk_loc;
                auto G = gkvec_.gvec(igk);
                phase_gk[igk_loc] = std::conj(ctx_.gvec_phase_factor(G, ia) * phase_k);
            }

            for (int nu = 0; nu < 3; nu++) {
                for (int mu = 0; mu < 3; mu++) {
                    for (int xi = 0; xi < bchunk(ichunk__).desc_(beta_desc_idx::nbf, i); xi++) {
                        for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
                            pw_coeffs_a_[mu + nu * 3](igk_loc, bchunk(ichunk__).desc_(beta_desc_idx::offset, i) + xi) = 
                                pw_coeffs_t_[mu + nu * 3](igk_loc, bchunk(ichunk__).desc_(beta_desc_idx::offset_t, i) + xi) * phase_gk[igk_loc];
                        }
                    }
                }
            }
        }
    }

    inline int num_gkvec_loc() const
    {
        return num_gkvec_loc_;
    }

};

}

#endif

