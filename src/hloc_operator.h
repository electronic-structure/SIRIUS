// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

        Gvec const& gkvec_;

        std::vector<double> pw_ekin_;

        mdarray<double, 2> veff_vec_;

        mdarray<double_complex, 1> vphi_;
        
        /// V(G=0) matrix elements.
        double v0_[2];

    public:
        
        /** This is used internally to benchmark and profile the Hloc kernel */
        Hloc_operator(FFT3D& fft__,
                      Gvec const& gkvec__,
                      std::vector<double> veff__)
            : fft_(fft__),
              gkvec_(gkvec__)
        {
            pw_ekin_ = std::vector<double>(gkvec_.num_gvec_fft(), 0);
            
            veff_vec_ = mdarray<double, 2>(fft_.local_size(), 1);
            std::memcpy(&veff_vec_[0], &veff__[0], fft_.local_size() * sizeof(double));
            vphi_ = mdarray<double_complex, 1>(gkvec__.num_gvec_fft());
            #ifdef __GPU
            if (fft_ctx_.pu() == GPU)
            {
                veff_vec_.allocate_on_device();
                veff_vec_.copy_to_device();
            }
            #endif
        }

        /** \param [in] fft_ctx FFT context of the coarse grid used to apply effective field.
         *  \param [in] gvec G-vectors of the coarse FFT grid.
         *  \param [in] gkvec G-vectors of the wave-functions.
         */
        Hloc_operator(FFT3D& fft__,
                      Gvec const& gvec__,
                      Gvec const& gkvec__,
                      int num_mag_dims__,
                      Periodic_function<double>* effective_potential__,
                      Periodic_function<double>* effective_magnetic_field__[3]) 
            : fft_(fft__),
              gkvec_(gkvec__)
        {
            /* cache kinteic energy of plane-waves */
            pw_ekin_ = std::vector<double>(gkvec_.num_gvec_fft());
            for (int ig_loc = 0; ig_loc < gkvec_.num_gvec_fft(); ig_loc++)
            {
                /* global index of G-vector */
                int ig = gkvec_.offset_gvec_fft() + ig_loc;
                /* get G+k in Cartesian coordinates */
                auto gv = gkvec_.cart_shifted(ig);
                pw_ekin_[ig_loc] = 0.5 * (gv * gv);
            }
            
            /* group effective fields into single vector */
            std::vector<Periodic_function<double>*> veff_vec(num_mag_dims__ + 1);
            veff_vec[0] = effective_potential__;
            for (int j = 0; j < num_mag_dims__; j++) veff_vec[1 + j] = effective_magnetic_field__[j];

            veff_vec_ = mdarray<double, 2>(fft_.local_size(), num_mag_dims__ + 1);

            /* map components of effective potential to a corase grid */
            for (int j = 0; j < num_mag_dims__ + 1; j++)
            {
                std::vector<double_complex> v_pw_coarse(gvec__.num_gvec_fft());

                for (int ig = 0; ig < gvec__.num_gvec_fft(); ig++)
                {
                    auto G = gvec__[ig + gvec__.offset_gvec_fft()];
                    v_pw_coarse[ig] = veff_vec[j]->f_pw(G);
                }
                fft_.transform<1>(gvec__, &v_pw_coarse[0]);
                fft_.output(&veff_vec_(0, j));
            }

            if (num_mag_dims__)
            {
                for (int ir = 0; ir < fft_.local_size(); ir++)
                {
                    double v0 = veff_vec_(ir, 0);
                    double v1 = veff_vec_(ir, 1);
                    veff_vec_(ir, 0) = v0 + v1; // v + Bz
                    veff_vec_(ir, 1) = v0 - v1; // v - Bz
                }
            }

            if (num_mag_dims__ == 0)
            {
                v0_[0] = veff_vec[0]->f_pw(0).real();
            }
            else
            {
                v0_[0] = veff_vec[0]->f_pw(0).real() + veff_vec[1]->f_pw(0).real();
                v0_[1] = veff_vec[0]->f_pw(0).real() - veff_vec[1]->f_pw(0).real();
            }

            vphi_ = mdarray<double_complex, 1>(gkvec__.num_gvec_fft());

            #ifdef __GPU
            if (fft_ctx_.pu() == GPU)
            {
                veff_vec_.allocate_on_device();
                veff_vec_.copy_to_device();
            }
            #endif
        }
        
        void apply(int ispn__, Wave_functions<false>& hphi__, int idx0__, int n__)
        {
            PROFILE_WITH_TIMER("sirius::Hloc_operator::apply");

            hphi__.swap_forward(idx0__, n__);

            for (int i = 0; i < hphi__.spl_num_swapped().local_size(); i++)
            {
                /* phi(G) -> phi(r) */
                fft_.transform<1>(gkvec_, hphi__[i]);
                /* multiply by effective potential */
                if (fft_.hybrid())
                {
                    #ifdef __GPU
                    scale_matrix_rows_gpu(fft_.local_size(), 1, fft_.buffer<GPU>(), veff_vec_.at<GPU>(0, ispn__));
                    #else
                    TERMINATE_NO_GPU
                    #endif
                }
                else
                {
                    #pragma omp parallel for
                    for (int ir = 0; ir < fft_.local_size(); ir++) fft_.buffer(ir) *= veff_vec_(ir, ispn__);

                }
                /* V(r)phi(r) -> [V*phi](G) */
                fft_.transform<-1>(gkvec_, &vphi_[0]);

                /* add kinetic energy */
                #pragma omp parallel for
                for (int ig = 0; ig < gkvec_.num_gvec_fft(); ig++)
                    hphi__[i][ig] = hphi__[i][ig] * pw_ekin_[ig] + vphi_[ig];
            }

            hphi__.swap_backward(idx0__, n__);
        }

        inline double v0(int ispn__)
        {
            return v0_[ispn__];
        }
};

};

#endif
