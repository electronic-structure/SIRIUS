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

        Communicator const& comm_;

        Atom_type const& atom_type_;

        mdarray<double, 2> q_mtrx_;

        mdarray<double, 2> q_pw_;

        mdarray<double, 1> sym_weight_;

        void generate_pw_coeffs(double omega__, Gvec const& gvec__, Radial_integrals_aug const& radial_integrals__)
        {
            PROFILE("sirius::Augmentation_operator::generate_pw_coeffs");
        
            double fourpi_omega = fourpi / omega__;

            /* maximum l of beta-projectors */
            int lmax_beta = atom_type_.indexr().lmax();
            int lmmax = Utils::lmmax(2 * lmax_beta);

            std::vector<int> l_by_lm = Utils::l_by_lm(2 * lmax_beta);
        
            std::vector<double_complex> zilm(lmmax);
            for (int l = 0, lm = 0; l <= 2 * lmax_beta; l++) {
                for (int m = -l; m <= l; m++, lm++) {
                    zilm[lm] = std::pow(double_complex(0, 1), l);
                }
            }

            /* Gaunt coefficients of three real spherical harmonics */
            Gaunt_coefficients<double> gaunt_coefs(lmax_beta, 2 * lmax_beta, lmax_beta, SHT::gaunt_rlm);
            
            /* split G-vectors between ranks */
            int gvec_count = gvec__.gvec_count(comm_.rank());
            int gvec_offset = gvec__.gvec_offset(comm_.rank());
            
            /* array of real spherical harmonics for each G-vector */
            mdarray<double, 2> gvec_rlm(Utils::lmmax(2 * lmax_beta), gvec_count);
            for (int igloc = 0; igloc < gvec_count; igloc++) {
                int ig = gvec_offset + igloc;
                auto rtp = SHT::spherical_coordinates(gvec__.gvec_cart(ig));
                SHT::spherical_harmonics(2 * lmax_beta, rtp[1], rtp[2], &gvec_rlm(0, igloc));
            }
        
            /* number of beta-projectors */
            int nbf = atom_type_.mt_basis_size();
            
            q_mtrx_ = mdarray<double, 2>(nbf, nbf);
            q_mtrx_.zero();

            /* array of plane-wave coefficients */
            q_pw_ = mdarray<double, 2>(nbf * (nbf + 1) / 2, 2 * gvec_count, memory_t::host_pinned, "q_pw_");
            #pragma omp parallel for
            for (int igloc = 0; igloc < gvec_count; igloc++) {
                int ig = gvec_offset + igloc;
                double g = gvec__.gvec_len(ig);
                
                std::vector<double_complex> v(lmmax);

                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    int lm2 = atom_type_.indexb(xi2).lm;
                    int idxrf2 = atom_type_.indexb(xi2).idxrf;
        
                    for (int xi1 = 0; xi1 <= xi2; xi1++) {
                        int lm1 = atom_type_.indexb(xi1).lm;
                        int idxrf1 = atom_type_.indexb(xi1).idxrf;
                        
                        /* packed orbital index */
                        int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                        /* packed radial-function index */
                        int idxrf12 = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
                        
                        for (int lm3 = 0; lm3 < lmmax; lm3++) {
                            v[lm3] = std::conj(zilm[lm3]) * gvec_rlm(lm3, igloc) * radial_integrals__.value(idxrf12, l_by_lm[lm3], atom_type_.id(), g);
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
                    int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                    sym_weight_(idx12) = (xi1 == xi2) ? 1 : 2;
                }
            }
    
            if (comm_.rank() == 0) {
                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    for (int xi1 = 0; xi1 <= xi2; xi1++) {
                        /* packed orbital index */
                        int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                        q_mtrx_(xi1, xi2) = q_mtrx_(xi2, xi1) = omega__ * q_pw_(idx12, 0);
                    }
                }
            }
            /* broadcast from rank#0 */
            comm_.bcast(&q_mtrx_(0, 0), nbf * nbf , 0);

            #ifdef __PRINT_OBJECT_CHECKSUM
            double cs = q_pw_.checksum();
            comm_.allreduce(&cs, 1);
            DUMP("checksum(q_pw) : %18.10f", cs);
            #endif
        }

    public:
       
        Augmentation_operator(Simulation_context_base& ctx__,
                              int iat__,
                              Radial_integrals_aug const& ri__)
            : comm_(ctx__.comm())
            , atom_type_(ctx__.unit_cell().atom_type(iat__))
        {
            if (atom_type_.pp_desc().augment) {
                generate_pw_coeffs(ctx__.unit_cell().omega(), ctx__.gvec(), ri__);
            }
        }

        void prepare(int stream_id__) const
        {
            #ifdef __GPU
            if (atom_type_.parameters().processing_unit() == GPU && atom_type_.pp_desc().augment) {
                sym_weight_.allocate(memory_t::device);
                sym_weight_.async_copy_to_device(stream_id__);

                q_pw_.allocate(memory_t::device);
                q_pw_.async_copy_to_device(stream_id__);
            }
            #endif
        }

        void dismiss() const
        {
            #ifdef __GPU
            if (atom_type_.parameters().processing_unit() == GPU && atom_type_.pp_desc().augment) {
                q_pw_.deallocate_on_device();
                sym_weight_.deallocate_on_device();
            }
            #endif
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

};

#endif // __AUGMENTATION_OPERATOR_H__
