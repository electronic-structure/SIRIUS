// Copyright (c) 2013-2015 Anton Kozhevnikov, Thomas Schulthess
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

/** \file non_local_operator.h
 *   
 *  \brief Contains declaration and implementation of sirius::Non_local_operator class.
 */

#ifndef __NON_LOCAL_OPERATOR_H__
#define __NON_LOCAL_OPERATOR_H__

namespace sirius {

class Non_local_operator
{
    protected:

        Beta_projectors& beta_;

        processing_unit_t pu_;
        
        int packed_mtrx_size_;

        mdarray<int, 1> packed_mtrx_offset_;
        
        mdarray<double_complex, 2> op_;
        mdarray<double_complex, 1> work_;

        Non_local_operator& operator=(Non_local_operator const& src) = delete;
        Non_local_operator(Non_local_operator const& src) = delete;

    public:

        Non_local_operator(Beta_projectors& beta__, processing_unit_t pu__) : beta_(beta__), pu_(pu__)
        {
            PROFILE();

            auto& uc = beta_.unit_cell();
            packed_mtrx_offset_ = mdarray<int, 1>(uc.num_atoms());
            packed_mtrx_size_ = 0;
            for (int ia = 0; ia < uc.num_atoms(); ia++)
            {   
                int nbf = uc.atom(ia).mt_basis_size();
                packed_mtrx_offset_(ia) = packed_mtrx_size_;
                packed_mtrx_size_ += nbf * nbf;
            }

            #ifdef __GPU
            if (pu_ == GPU)
            {
                packed_mtrx_offset_.allocate_on_device();
                packed_mtrx_offset_.copy_to_device();
            }
            #endif
        }

        ~Non_local_operator()
        {
        }

        void apply(int chunk__, int ispn__, Wave_functions<false>& op_phi__, int idx0__, int n__)
        {
            PROFILE_WITH_TIMER("sirius::Non_local_operator::apply");

            assert(op_phi__.num_gvec_loc() == beta_.num_gkvec_loc());

            auto& beta_phi = beta_.beta_phi();
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
                                      beta_phi.at<CPU>(offs), nbeta,
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
                                      beta_phi.at<GPU>(offs), nbeta,
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

        inline double_complex operator()(int xi1__, int xi2__, int ia__)
        {
            int nbf = beta_.unit_cell().atom(ia__).mt_basis_size();
            return op_(packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, 0);
        }

        inline double_complex operator()(int xi1__, int xi2__, int ispn__, int ia__)
        {
            int nbf = beta_.unit_cell().atom(ia__).mt_basis_size();
            return op_(packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, ispn__);
        }
};

class D_operator: public Non_local_operator
{
    public:

        D_operator(Beta_projectors& beta__, int num_mag_dims__, processing_unit_t pu__) : Non_local_operator(beta__, pu__)
        {
            op_ = mdarray<double_complex, 2>(packed_mtrx_size_, num_mag_dims__ + 1);
            auto& uc = beta_.unit_cell();

            for (int j = 0; j < num_mag_dims__ + 1; j++)
            {
                for (int ia = 0; ia < uc.num_atoms(); ia++)
                {
                    int nbf = uc.atom(ia).mt_basis_size();
                    for (int xi2 = 0; xi2 < nbf; xi2++)
                    {
                        for (int xi1 = 0; xi1 < nbf; xi1++)
                        {
                            op_(packed_mtrx_offset_(ia) + xi2 * nbf + xi1, j) = uc.atom(ia).d_mtrx(xi1, xi2, j);
                        }
                    }
                }
            }
            if (num_mag_dims__)
            {
                for (int ia = 0; ia < uc.num_atoms(); ia++)
                {
                    int nbf = uc.atom(ia).mt_basis_size();
                    for (int xi2 = 0; xi2 < nbf; xi2++)
                    {
                        for (int xi1 = 0; xi1 < nbf; xi1++)
                        {
                            auto v0 = op_(packed_mtrx_offset_(ia) + xi2 * nbf + xi1, 0); 
                            auto v1 = op_(packed_mtrx_offset_(ia) + xi2 * nbf + xi1, 1); 
                            op_(packed_mtrx_offset_(ia) + xi2 * nbf + xi1, 0) = v0 + v1;
                            op_(packed_mtrx_offset_(ia) + xi2 * nbf + xi1, 1) = v0 - v1;
                        }
                    }
                }
            }
            #ifdef __GPU
            if (pu_ == GPU)
            {
                op_.allocate_on_device();
                op_.copy_to_device();
            }
            #endif
        }
};

class Q_operator: public Non_local_operator
{
    public:
        
        Q_operator(Beta_projectors& beta__, processing_unit_t pu__) : Non_local_operator(beta__, pu__)
        {
            /* Q-operator is independent of spin */
            op_ = mdarray<double_complex, 2>(packed_mtrx_size_, 1);
            auto& uc = beta_.unit_cell();
            for (int ia = 0; ia < uc.num_atoms(); ia++)
            {
                int nbf = uc.atom(ia).mt_basis_size();
                for (int xi2 = 0; xi2 < nbf; xi2++)
                {
                    for (int xi1 = 0; xi1 < nbf; xi1++)
                    {
                        op_(packed_mtrx_offset_(ia) + xi2 * nbf + xi1, 0) = uc.atom(ia).type().uspp().q_mtrx(xi1, xi2);
                    }
                }
            }
            #ifdef __GPU
            if (pu_ == GPU)
            {
                op_.allocate_on_device();
                op_.copy_to_device();
            }
            #endif
        }
};

}

#endif
