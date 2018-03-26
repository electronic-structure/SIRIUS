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

/** \file augmentation_operator.h
 *
 *  \brief Contains implementation of sirius::Augmentation_operator class.
 */

#ifndef __AUGMENTATION_OPERATOR_H__
#define __AUGMENTATION_OPERATOR_H__

#include "radial_integrals.h"

namespace sirius {

class Augmentation_operator
{
    private:
        
        Simulation_context_base const& ctx_;

        Communicator const& comm_;

        Atom_type const& atom_type_;

        mdarray<double, 2> q_mtrx_;

        mdarray<double, 2> q_pw_;

        mdarray<double, 1> sym_weight_;

        void generate_pw_coeffs(double omega__, Gvec const& gvec__, Radial_integrals_aug<false> const& radial_integrals__)
        {
            PROFILE("sirius::Augmentation_operator::generate_pw_coeffs");
        
            double fourpi_omega = fourpi / omega__;

            /* maximum l of beta-projectors */
            int lmax_beta = atom_type_.indexr().lmax();
            int lmmax = Utils::lmmax(2 * lmax_beta);

            auto l_by_lm = Utils::l_by_lm(2 * lmax_beta);
        
            std::vector<double_complex> zilm(lmmax);
            for (int l = 0, lm = 0; l <= 2 * lmax_beta; l++) {
                for (int m = -l; m <= l; m++, lm++) {
                    zilm[lm] = std::pow(double_complex(0, 1), l);
                }
            }

            /* Gaunt coefficients of three real spherical harmonics */
            Gaunt_coefficients<double> gaunt_coefs(lmax_beta, 2 * lmax_beta, lmax_beta, SHT::gaunt_rlm);
            
            /* split G-vectors between ranks */
            int gvec_count = gvec__.count();
            int gvec_offset = gvec__.offset();
            
            /* array of real spherical harmonics for each G-vector */
            mdarray<double, 2> gvec_rlm(Utils::lmmax(2 * lmax_beta), gvec_count);
            #pragma omp parallel for schedule(static)
            for (int igloc = 0; igloc < gvec_count; igloc++) {
                int ig = gvec_offset + igloc;
                auto rtp = SHT::spherical_coordinates(gvec__.gvec_cart(ig));
                SHT::spherical_harmonics(2 * lmax_beta, rtp[1], rtp[2], &gvec_rlm(0, igloc));
            }
        
            /* number of beta-projectors */
            int nbf = atom_type_.mt_basis_size();
            
            /* array of plane-wave coefficients */
            q_pw_ = mdarray<double, 2>(nbf * (nbf + 1) / 2, 2 * gvec_count, memory_t::host_pinned, "q_pw_");
            #pragma omp parallel for schedule(static)
            for (int igloc = 0; igloc < gvec_count; igloc++) {
                int ig = gvec_offset + igloc;
                double g = gvec__.gvec_len(ig);
                
                std::vector<double_complex> v(lmmax);
                
                auto ri = radial_integrals__.values(atom_type_.id(), g);

                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    int lm2 = atom_type_.indexb(xi2).lm;
                    int idxrf2 = atom_type_.indexb(xi2).idxrf;
        
                    for (int xi1 = 0; xi1 <= xi2; xi1++) {
                        int lm1 = atom_type_.indexb(xi1).lm;
                        int idxrf1 = atom_type_.indexb(xi1).idxrf;
                        
                        /* packed orbital index */
                        int idx12 = Utils::packed_index(xi1, xi2);
                        /* packed radial-function index */
                        int idxrf12 = Utils::packed_index(idxrf1, idxrf2);
                        
                        for (int lm3 = 0; lm3 < lmmax; lm3++) {
                            v[lm3] = std::conj(zilm[lm3]) * gvec_rlm(lm3, igloc) * ri(idxrf12, l_by_lm[lm3]);
                        }

                        double_complex z = fourpi_omega * gaunt_coefs.sum_L3_gaunt(lm2, lm1, &v[0]);
                        q_pw_(idx12, 2 * igloc)     = z.real();
                        q_pw_(idx12, 2 * igloc + 1) = z.imag();
                    }
                }
            }

            sym_weight_ = mdarray<double, 1>(nbf * (nbf + 1) / 2, memory_t::host_pinned, "sym_weight_");
            for (int xi2 = 0; xi2 < nbf; xi2++) {
                for (int xi1 = 0; xi1 <= xi2; xi1++) {
                    /* packed orbital index */
                    int idx12 = Utils::packed_index(xi1, xi2);
                    sym_weight_(idx12) = (xi1 == xi2) ? 1 : 2;
                }
            }
    
            q_mtrx_ = mdarray<double, 2>(nbf, nbf);
            q_mtrx_.zero();

            if (comm_.rank() == 0) {
                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    for (int xi1 = 0; xi1 <= xi2; xi1++) {
                        /* packed orbital index */
                        int idx12 = Utils::packed_index(xi1, xi2);
                        q_mtrx_(xi1, xi2) = q_mtrx_(xi2, xi1) = omega__ * q_pw_(idx12, 0);
                    }
                }
            }
            /* broadcast from rank#0 */
            comm_.bcast(&q_mtrx_(0, 0), nbf * nbf , 0);

            if (ctx_.control().print_checksum_) {
                double cs = q_pw_.checksum();
                comm_.allreduce(&cs, 1);
                if (comm_.rank() == 0) {
                    print_checksum("q_pw", cs);
                }
            }
        }

    public:
       
        Augmentation_operator(Simulation_context_base const& ctx__,
                              int iat__)
            : ctx_(ctx__)
            , comm_(ctx__.comm())
            , atom_type_(ctx__.unit_cell().atom_type(iat__))
        {
            if (atom_type_.augment()) {
                generate_pw_coeffs(ctx__.unit_cell().omega(), ctx__.gvec(), ctx__.aug_ri());
            }
        }

        void prepare(int stream_id__)
        {
            if (atom_type_.parameters().processing_unit() == GPU && atom_type_.augment()) {
                sym_weight_.allocate(memory_t::device);
                sym_weight_.async_copy<memory_t::host, memory_t::device>(stream_id__);

                q_pw_.allocate(memory_t::device);
                q_pw_.async_copy<memory_t::host, memory_t::device>(stream_id__);
            }
        }

        void dismiss()
        {
            if (atom_type_.parameters().processing_unit() == GPU && atom_type_.augment()) {
                q_pw_.deallocate(memory_t::device);
                sym_weight_.deallocate(memory_t::device);
            }
        }

        mdarray<double, 2> const& q_pw() const
        {
            return q_pw_;
        }

        double q_pw(int i__, int ig__) const
        {
            return q_pw_(i__, ig__);
        }

        double const& q_mtrx(int xi1__, int xi2__) const
        {
            return q_mtrx_(xi1__, xi2__);
        }

        double& q_mtrx(int xi1__, int xi2__)
        {
            return q_mtrx_(xi1__, xi2__);
        }

        inline mdarray<double, 1> const& sym_weight() const
        {
            return sym_weight_;
        }
        
        /// Weight of Q_{\xi,\xi'}. 
        /** 2 if off-diagonal (xi != xi'), 1 if diagonal (xi=xi') */
        inline double sym_weight(int idx__) const
        {
            return sym_weight_(idx__);
        }
};

class Augmentation_operator_gvec_deriv
{
    private:

        Simulation_context_base const& ctx_;

        Communicator const& comm_;

        mdarray<double, 2> q_pw_;

        mdarray<double, 1> sym_weight_;

        mdarray<double, 2> rlm_g_;
        mdarray<double, 3> rlm_dg_;
        std::unique_ptr<Gaunt_coefficients<double>> gaunt_coefs_;

    public:
       
        Augmentation_operator_gvec_deriv(Simulation_context_base const& ctx__)
            : ctx_(ctx__)
            , comm_(ctx__.comm())
        {
            PROFILE("sirius::Augmentation_operator_gvec_deriv|constructor");

            int lmax = ctx__.unit_cell().lmax();
            int lmmax = Utils::lmmax(2 * lmax);

            /* Gaunt coefficients of three real spherical harmonics */
            gaunt_coefs_ = std::unique_ptr<Gaunt_coefficients<double>>(new Gaunt_coefficients<double>(lmax, 2 * lmax, lmax, SHT::gaunt_rlm));
            
            /* split G-vectors between ranks */
            int gvec_count = ctx_.gvec().count();
            int gvec_offset = ctx_.gvec().offset();

            rlm_g_ = mdarray<double, 2>(lmmax, gvec_count);
            rlm_dg_ = mdarray<double, 3>(lmmax, 3, gvec_count);

            /* array of real spherical harmonics and derivatives for each G-vector */
            #pragma omp parallel for schedule(static)
            for (int igloc = 0; igloc < gvec_count; igloc++) {
                int ig = gvec_offset + igloc;
                auto rtp = SHT::spherical_coordinates(ctx_.gvec().gvec_cart(ig));

                double theta = rtp[1];
                double phi   = rtp[2];
                vector3d<double> dtheta_dq({std::cos(phi) * std::cos(theta), std::cos(theta) * std::sin(phi), -std::sin(theta)});
                vector3d<double> dphi_dq({-std::sin(phi), std::cos(phi), 0.0});

                SHT::spherical_harmonics(2 * lmax, theta, phi, &rlm_g_(0, igloc));
                
                mdarray<double, 1> dRlm_dtheta(lmmax);
                mdarray<double, 1> dRlm_dphi_sin_theta(lmmax);

                SHT::dRlm_dtheta(2 * lmax, theta, phi, dRlm_dtheta);
                SHT::dRlm_dphi_sin_theta(2 * lmax, theta, phi, dRlm_dphi_sin_theta);
                for (int nu = 0; nu < 3; nu++) {
                    for (int lm = 0; lm < lmmax; lm++) {
                        rlm_dg_(lm, nu, igloc) = dRlm_dtheta[lm] * dtheta_dq[nu] + dRlm_dphi_sin_theta[lm] * dphi_dq[nu];
                    }
                }
            }
        }

        void generate_pw_coeffs(int iat__,
                                Radial_integrals_aug<false> const& ri__,
                                Radial_integrals_aug<true>  const& ri_dq__,
                                int                                nu__)
        {
            PROFILE("sirius::Augmentation_operator_gvec_deriv::generate_pw_coeffs");

            auto& atom_type = ctx_.unit_cell().atom_type(iat__);
        
            /* maximum l of beta-projectors */
            int lmax_beta = atom_type.indexr().lmax();
            int lmmax = Utils::lmmax(2 * lmax_beta);

            auto l_by_lm = Utils::l_by_lm(2 * lmax_beta);
        
            std::vector<double_complex> zilm(lmmax);
            for (int l = 0, lm = 0; l <= 2 * lmax_beta; l++) {
                for (int m = -l; m <= l; m++, lm++) {
                    zilm[lm] = std::pow(double_complex(0, 1), l);
                }
            }

            /* split G-vectors between ranks */
            int gvec_count = ctx_.gvec().count();
            int gvec_offset = ctx_.gvec().offset();

            /* number of beta-projectors */
            int nbf = atom_type.mt_basis_size();
            
            /* array of plane-wave coefficients */
            q_pw_ = mdarray<double, 2>(nbf * (nbf + 1) / 2, 2 * gvec_count, memory_t::host_pinned, "q_pw_dg_");
            sddk::timer t2("sirius::Augmentation_operator_gvec_deriv::generate_pw_coeffs|qpw");
            #pragma omp parallel for schedule(static)
            for (int igloc = 0; igloc < gvec_count; igloc++) {
                int ig = gvec_offset + igloc;
                double g = ctx_.gvec().gvec_len(ig);
                auto gvc = ctx_.gvec().gvec_cart(ig);

                std::vector<double_complex> v(lmmax);
                auto ri = ri__.values(atom_type.id(), g);
                auto ri_dg = ri_dq__.values(atom_type.id(), g);

                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    int lm2 = atom_type.indexb(xi2).lm;
                    int idxrf2 = atom_type.indexb(xi2).idxrf;
        
                    for (int xi1 = 0; xi1 <= xi2; xi1++) {
                        int lm1 = atom_type.indexb(xi1).lm;
                        int idxrf1 = atom_type.indexb(xi1).idxrf;
                        
                        /* packed orbital index */
                        int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                        /* packed radial-function index */
                        int idxrf12 = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
                        
                        for (int lm3 = 0; lm3 < lmmax; lm3++) {
                            v[lm3] = std::conj(zilm[lm3]) * (rlm_dg_(lm3, nu__, igloc) * ri(idxrf12, l_by_lm[lm3]) +
                                                             rlm_g_(lm3, igloc) * ri_dg(idxrf12, l_by_lm[lm3]) * gvc[nu__]);
                        }

                        double_complex z = fourpi * gaunt_coefs_->sum_L3_gaunt(lm2, lm1, &v[0]);
                        q_pw_(idx12, 2 * igloc)     = z.real();
                        q_pw_(idx12, 2 * igloc + 1) = z.imag();
                    }
                }
            }
            t2.stop();

            sym_weight_ = mdarray<double, 1>(nbf * (nbf + 1) / 2, memory_t::host_pinned, "sym_weight_");
            for (int xi2 = 0; xi2 < nbf; xi2++) {
                for (int xi1 = 0; xi1 <= xi2; xi1++) {
                    /* packed orbital index */
                    int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                    sym_weight_(idx12) = (xi1 == xi2) ? 1 : 2;
                }
            }
        }

        //void prepare(int stream_id__) const
        //{
        //    #ifdef __GPU
        //    if (atom_type_.parameters().processing_unit() == GPU && atom_type_.pp_desc().augment) {
        //        sym_weight_.allocate(memory_t::device);
        //        sym_weight_.async_copy_to_device(stream_id__);

        //        q_pw_.allocate(memory_t::device);
        //        q_pw_.async_copy_to_device(stream_id__);
        //    }
        //    #endif
        //}

        //void dismiss() const
        //{
        //    #ifdef __GPU
        //    if (atom_type_.parameters().processing_unit() == GPU && atom_type_.pp_desc().augment) {
        //        q_pw_.deallocate_on_device();
        //        sym_weight_.deallocate_on_device();
        //    }
        //    #endif
        //}

        //mdarray<double, 2> const& q_pw() const
        //{
        //    return q_pw_;
        //}

        mdarray<double, 2> const& q_pw() const
        {
            return q_pw_;
        }

        double q_pw(int i__, int ig__) const
        {
            return q_pw_(i__, ig__);
        }

        //inline mdarray<double, 1> const& sym_weight() const
        //{
        //    return sym_weight_;
        //}
        //

        /// Weight of Q_{\xi,\xi'}. 
        /** 2 if off-diagonal (xi != xi'), 1 if diagonal (xi=xi') */
        inline double sym_weight(int idx__) const
        {
            return sym_weight_(idx__);
        }
};

}

#endif // __AUGMENTATION_OPERATOR_H__
