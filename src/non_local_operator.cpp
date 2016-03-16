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

/** \file non_local_operator.cpp
 *   
 *  \brief Contains implementation of sirius::Non_local_operator::apply() method.
 */

#include "non_local_operator.h"

namespace sirius {

template<>
void Non_local_operator<double_complex>::apply(int chunk__, int ispn__, Wave_functions<false>& op_phi__, int idx0__, int n__)
{
    PROFILE_WITH_TIMER("sirius::Non_local_operator::apply");

    assert(op_phi__.num_gvec_loc() == beta_.num_gkvec_loc());

    auto beta_phi = beta_.beta_phi<double_complex>(chunk__, n__);
    auto& beta_gk = beta_.beta_gk();
    int num_gkvec_loc = beta_.num_gkvec_loc();
    int nbeta = beta_.beta_chunk(chunk__).num_beta_;

    if (static_cast<size_t>(nbeta * n__) > work_.size())
    {
        work_ = mdarray<double_complex, 1>(nbeta * n__);
        #ifdef __GPU
        if (pu_ == GPU) work_.allocate_on_device();
        #endif
    }

    if (pu_ == CPU)
    {
        #pragma omp parallel for
        for (int i = 0; i < beta_.beta_chunk(chunk__).num_atoms_; i++)
        {
            /* number of beta functions for a given atom */
            int nbf = beta_.beta_chunk(chunk__).desc_(0, i);
            int offs = beta_.beta_chunk(chunk__).desc_(1, i);
            int ia = beta_.beta_chunk(chunk__).desc_(3, i);

            /* compute O * <beta|phi> */
            linalg<CPU>::gemm(0, 0, nbf, n__, nbf,
                              op_.at<CPU>(packed_mtrx_offset_(ia), ispn__), nbf,
                              beta_phi.at<CPU>(offs, 0), nbeta,
                              work_.at<CPU>(offs), nbeta);
        }
        
        /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
        linalg<CPU>::gemm(0, 0, num_gkvec_loc, n__, nbeta, double_complex(1, 0),
                          beta_gk.at<CPU>(), num_gkvec_loc, work_.at<CPU>(), nbeta, double_complex(1, 0),
                          &op_phi__(0, idx0__), num_gkvec_loc);
    }
    #ifdef __GPU
    if (pu_ == GPU)
    {
        #pragma omp parallel for
        for (int i = 0; i < beta_.beta_chunk(chunk__).num_atoms_; i++)
        {
            /* number of beta functions for a given atom */
            int nbf = beta_.beta_chunk(chunk__).desc_(0, i);
            int offs = beta_.beta_chunk(chunk__).desc_(1, i);
            int ia = beta_.beta_chunk(chunk__).desc_(3, i);

            /* compute O * <beta|phi> */
            linalg<GPU>::gemm(0, 0, nbf, n__, nbf,
                              op_.at<GPU>(packed_mtrx_offset_(ia), ispn__), nbf, 
                              beta_phi.at<GPU>(offs, 0), nbeta,
                              work_.at<GPU>(offs), nbeta,
                              omp_get_thread_num());

        }
        cuda_device_synchronize();
        double_complex alpha(1, 0);
        
        /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
        linalg<GPU>::gemm(0, 0, num_gkvec_loc, n__, nbeta, &alpha,
                          beta_gk.at<GPU>(), beta_gk.ld(), work_.at<GPU>(), nbeta, &alpha, 
                          op_phi__.coeffs().at<GPU>(0, idx0__), op_phi__.coeffs().ld());
        
        cuda_device_synchronize();
    }
    #endif
}

template<>
void Non_local_operator<double>::apply(int chunk__, int ispn__, Wave_functions<false>& op_phi__, int idx0__, int n__)
{
    PROFILE_WITH_TIMER("sirius::Non_local_operator::apply");

    assert(op_phi__.num_gvec_loc() == beta_.num_gkvec_loc());

    auto beta_phi = beta_.beta_phi<double>(chunk__, n__);
    auto& beta_gk = beta_.beta_gk();
    int num_gkvec_loc = beta_.num_gkvec_loc();
    int nbeta = beta_.beta_chunk(chunk__).num_beta_;

    if (static_cast<size_t>(nbeta * n__) > work_.size())
    {
        work_ = mdarray<double, 1>(nbeta * n__);
        #ifdef __GPU
        if (pu_ == GPU) work_.allocate_on_device();
        #endif
    }

    if (pu_ == CPU)
    {
        #pragma omp parallel for
        for (int i = 0; i < beta_.beta_chunk(chunk__).num_atoms_; i++)
        {
            /* number of beta functions for a given atom */
            int nbf = beta_.beta_chunk(chunk__).desc_(0, i);
            int offs = beta_.beta_chunk(chunk__).desc_(1, i);
            int ia = beta_.beta_chunk(chunk__).desc_(3, i);

            /* compute O * <beta|phi> */
            linalg<CPU>::gemm(0, 0, nbf, n__, nbf,
                              op_.at<CPU>(packed_mtrx_offset_(ia), ispn__), nbf,
                              beta_phi.at<CPU>(offs, 0), nbeta,
                              work_.at<CPU>(offs), nbeta);
        }
        
        /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
        linalg<CPU>::gemm(0, 0, 2 * num_gkvec_loc, n__, nbeta, 1.0,
                          (double*)beta_gk.at<CPU>(), 2 * num_gkvec_loc, work_.at<CPU>(), nbeta, 1.0,
                          (double*)&op_phi__(0, idx0__), 2 * num_gkvec_loc);
    }
    #ifdef __GPU
    if (pu_ == GPU)
    {
        STOP();
        //#pragma omp parallel for
        //for (int i = 0; i < beta_.beta_chunk(chunk__).num_atoms_; i++)
        //{
        //    /* number of beta functions for a given atom */
        //    int nbf = beta_.beta_chunk(chunk__).desc_(0, i);
        //    int offs = beta_.beta_chunk(chunk__).desc_(1, i);
        //    int ia = beta_.beta_chunk(chunk__).desc_(3, i);

        //    /* compute O * <beta|phi> */
        //    linalg<GPU>::gemm(0, 0, nbf, n__, nbf,
        //                      (double_complex*)op_.at<GPU>(2 * packed_mtrx_offset_(ia), ispn__), nbf, 
        //                      beta_phi.at<GPU>(offs, 0), nbeta,
        //                      (double_complex*)work_.at<GPU>(2 * offs), nbeta,
        //                      omp_get_thread_num());

        //}
        //cuda_device_synchronize();
        //double_complex alpha(1, 0);
        //
        ///* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
        //linalg<GPU>::gemm(0, 0, num_gkvec_loc, n__, nbeta, &alpha,
        //                  beta_gk.at<GPU>(), beta_gk.ld(), (double_complex*)work_.at<GPU>(), nbeta, &alpha, 
        //                  op_phi__.coeffs().at<GPU>(0, idx0__), op_phi__.coeffs().ld());
        //
        //cuda_device_synchronize();
    }
    #endif
}

};
