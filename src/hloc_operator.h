// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file hloc_operator.h
 *   
 *  \brief Contains declaration and implementation of sirius::Hloc_operator class.
 */

#ifndef __HLOC_OPERATOR_H__
#define __HLOC_OPERATOR_H__

#include "wave_functions.h"
#include "periodic_function.h"

namespace sirius {

class Hloc_operator
{
    private:

        FFT3D& fft_;

        Gvec_partition const& gkvec_;

        Communicator const& comm_col_;

        std::vector<double> pw_ekin_;

        mdarray<double, 2> veff_vec_;

        mdarray<double_complex, 1> vphi1_;

        mdarray<double_complex, 1> vphi2_;
        
        /// V(G=0) matrix elements.
        double v0_[2];

    public:
        
        /** This is used internally to benchmark and profile the Hloc kernel */
        Hloc_operator(FFT3D& fft__,
                      Gvec_partition const& gkvec__,
                      Communicator const& comm_col__,
                      std::vector<double> veff__)
            : fft_(fft__),
              gkvec_(gkvec__),
              comm_col_(comm_col__)
        {
            pw_ekin_ = std::vector<double>(gkvec_.gvec_count_fft(), 0);
            
            veff_vec_ = mdarray<double, 2>(fft_.local_size(), 1);
            std::memcpy(&veff_vec_[0], &veff__[0], fft_.local_size() * sizeof(double));
            vphi1_ = mdarray<double_complex, 1>(gkvec_.gvec_count_fft());
            if (gkvec_.reduced()) {
                vphi2_ = mdarray<double_complex, 1>(gkvec_.gvec_count_fft());
            }
            #ifdef __GPU
            if (fft_.hybrid()) {
                veff_vec_.allocate_on_device();
                veff_vec_.copy_to_device();
            }
            #endif
        }

        /** \param [in] fft FFT driver for the coarse grid used to apply effective field.
         *  \param [in] gkvec Partitioning of G-vectors for the FFT.
         *  \param [in] comm_col Column communicator to swap wave-functions.
         *  \param [in] num_mag_dims Number of magnetic dimensions.
         *  \param [in] gvec_coarse G-vectors of the coarse FFT grid.
         *  \param [in] effective_potential \f$ V_{eff}({\bf r}) \f$ on the fine real-space mesh.
         *  \param [in] effective_magnetic_field \f$ {\bf B}_{eff}({\bf r}) \f$ on the fine real-space mesh.
         */
        Hloc_operator(FFT3D& fft__,
                      Gvec_partition const& gkvec__,
                      Communicator const& comm_col__,
                      int num_mag_dims__,
                      Gvec const& gvec_coarse__,
                      Periodic_function<double>* effective_potential__,
                      Periodic_function<double>* effective_magnetic_field__[3]) 
            : fft_(fft__),
              gkvec_(gkvec__),
              comm_col_(comm_col__)
        {
            PROFILE();

            /* cache kinteic energy of plane-waves */
            pw_ekin_ = std::vector<double>(gkvec_.gvec_count_fft());
            for (int ig_loc = 0; ig_loc < gkvec_.gvec_count_fft(); ig_loc++) {
                /* global index of G-vector */
                int ig = gkvec_.gvec_offset_fft() + ig_loc;
                /* get G+k in Cartesian coordinates */
                auto gv = gkvec_.gvec().gkvec_cart(ig);
                pw_ekin_[ig_loc] = 0.5 * (gv * gv);
            }
            #ifdef __PRINT_OBJECT_CHECKSUM
            {
                auto cs = std::accumulate(pw_ekin_.begin(), pw_ekin_.end(), 0.0);
                fft_.comm().allreduce(&cs, 1);
                DUMP("checksum(pw_ekin): %18.10f", cs);
            }
            #endif
            
            /* group effective fields into single vector */
            std::vector<Periodic_function<double>*> veff_vec(num_mag_dims__ + 1);
            veff_vec[0] = effective_potential__;
            for (int j = 0; j < num_mag_dims__; j++) {
                veff_vec[1 + j] = effective_magnetic_field__[j];
            }

            veff_vec_ = mdarray<double, 2>(fft_.local_size(), num_mag_dims__ + 1);
        
            fft_.prepare(gvec_coarse__.partition());
            /* map components of effective potential to a corase grid */
            for (int j = 0; j < num_mag_dims__ + 1; j++) {
                /* low-frequency part of PW coefficients */
                std::vector<double_complex> v_pw_coarse(gvec_coarse__.partition().gvec_count_fft());
                /* loop over low-frequency G-vectors */
                for (int ig = 0; ig < gvec_coarse__.partition().gvec_count_fft(); ig++) {
                    /* G-vector in fractional coordinates */
                    auto G = gvec_coarse__.gvec(ig + gvec_coarse__.partition().gvec_offset_fft());
                    v_pw_coarse[ig] = veff_vec[j]->f_pw(G);
                }
                fft_.transform<1>(gvec_coarse__.partition(), &v_pw_coarse[0]);
                fft_.output(&veff_vec_(0, j));
                #ifdef __PRINT_OBJECT_CHECKSUM
                {
                    auto cs2 = mdarray<double, 1>(&veff_vec_(0, j), fft_.local_size()).checksum();
                    fft_.comm().allreduce(&cs2, 1);
                    DUMP("checksum(v_rg_coarse): %18.10f", cs2);
                }
                #endif
            }
            fft_.dismiss();

            if (num_mag_dims__) {
                for (int ir = 0; ir < fft_.local_size(); ir++) {
                    double v0 = veff_vec_(ir, 0);
                    double v1 = veff_vec_(ir, 1);
                    veff_vec_(ir, 0) = v0 + v1; // v + Bz
                    veff_vec_(ir, 1) = v0 - v1; // v - Bz
                }
            }

            if (num_mag_dims__ == 0) {
                v0_[0] = veff_vec[0]->f_pw(0).real();
            } else {
                v0_[0] = veff_vec[0]->f_pw(0).real() + veff_vec[1]->f_pw(0).real();
                v0_[1] = veff_vec[0]->f_pw(0).real() - veff_vec[1]->f_pw(0).real();
            }

            vphi1_ = mdarray<double_complex, 1>(gkvec_.gvec_count_fft());
            if (gkvec_.reduced()) {
                vphi2_ = mdarray<double_complex, 1>(gkvec_.gvec_count_fft());
            }

            #ifdef __GPU
            if (fft_.hybrid()) {
                veff_vec_.allocate_on_device();
                veff_vec_.copy_to_device();
            }
            #endif
        }
        
        void apply(int ispn__, Wave_functions<false>& hphi__, int idx0__, int n__)
        {
            PROFILE_WITH_TIMER("sirius::Hloc_operator::apply");

            hphi__.swap_forward(idx0__, n__, gkvec_, comm_col_);

            int first{0};
            /* if G-vectors are reduced, wave-functions are real and 
             * we can transform two of them at once */
            if (gkvec_.reduced()) {
                int npairs = hphi__.spl_num_swapped().local_size() / 2;

                for (int i = 0; i < npairs; i++) {
                    /* phi(G) -> phi(r) */
                    fft_.transform<1>(gkvec_, hphi__[2 * i], hphi__[2 * i + 1]);
                    /* multiply by effective potential */
                    if (fft_.hybrid()) {
                        #ifdef __GPU
                        scale_matrix_rows_gpu(fft_.local_size(), 1, fft_.buffer<GPU>(), veff_vec_.at<GPU>(0, ispn__));
                        #else
                        TERMINATE_NO_GPU
                        #endif
                    } else {
                        #pragma omp parallel for
                        for (int ir = 0; ir < fft_.local_size(); ir++) {
                            fft_.buffer(ir) *= veff_vec_(ir, ispn__);
                        }
                    }
                    /* V(r)phi(r) -> [V*phi](G) */
                    fft_.transform<-1>(gkvec_, &vphi1_[0], &vphi2_[0]);

                    /* add kinetic energy */
                    #pragma omp parallel for
                    for (int ig = 0; ig < gkvec_.gvec_count_fft(); ig++) {
                        hphi__[2 * i    ][ig] = hphi__[2 * i    ][ig] * pw_ekin_[ig] + vphi1_[ig];
                        hphi__[2 * i + 1][ig] = hphi__[2 * i + 1][ig] * pw_ekin_[ig] + vphi2_[ig];
                    }
                }
                
                /* check if we have to do last wave-function which had no pair */
                first = (hphi__.spl_num_swapped().local_size() % 2) ? hphi__.spl_num_swapped().local_size() - 1
                                                                    : hphi__.spl_num_swapped().local_size();
            }
            
            /* if we don't have G-vector reductions, first = 0 and we start a normal loop */
            for (int i = first; i < hphi__.spl_num_swapped().local_size(); i++) {
                /* phi(G) -> phi(r) */
                fft_.transform<1>(gkvec_, hphi__[i]);
                /* multiply by effective potential */
                if (fft_.hybrid()) {
                    #ifdef __GPU
                    scale_matrix_rows_gpu(fft_.local_size(), 1, fft_.buffer<GPU>(), veff_vec_.at<GPU>(0, ispn__));
                    #else
                    TERMINATE_NO_GPU
                    #endif
                } else {
                    #pragma omp parallel for
                    for (int ir = 0; ir < fft_.local_size(); ir++) {
                        fft_.buffer(ir) *= veff_vec_(ir, ispn__);
                    }
                }
                /* V(r)phi(r) -> [V*phi](G) */
                fft_.transform<-1>(gkvec_, &vphi1_[0]);
                /* add kinetic energy */
                #pragma omp parallel for
                for (int ig = 0; ig < gkvec_.gvec_count_fft(); ig++) {
                    hphi__[i][ig] = hphi__[i][ig] * pw_ekin_[ig] + vphi1_[ig];
                }
            }

            hphi__.swap_backward(idx0__, n__, gkvec_, comm_col_);
        }

        inline double v0(int ispn__)
        {
            return v0_[ispn__];
        }
};

};

#endif
