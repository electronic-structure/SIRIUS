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
        
        mdarray<double_complex, 1> op_;

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
                int nbf = uc.atom(ia)->mt_basis_size();
                packed_mtrx_offset_(ia) = packed_mtrx_size_;
                packed_mtrx_size_ += nbf * nbf;
            }
            op_ = mdarray<double_complex, 1>(packed_mtrx_size_);

            #ifdef __GPU
            if (pu_ == GPU)
            {
                packed_mtrx_offset_.allocate_on_device();
                packed_mtrx_offset_.copy_to_device();
            }
            #endif
        }

        void apply(int chunk__, Wave_functions& op_phi__, int idx0__, int n__)
        {
            PROFILE();

            assert(op_phi__.num_gvec_loc() == beta_.num_gkvec_loc());

            auto& beta_phi = beta_.beta_phi();
            auto& beta_gk = beta_.beta_gk();
            int num_gkvec_loc = beta_.num_gkvec_loc();
            int nbeta = beta_.beta_chunk(chunk__).num_beta_;
            matrix<double_complex> work(nbeta, n__);

            if (pu_ == CPU)
            {
                #pragma omp parallel for
                for (int i = 0; i < beta_.beta_chunk(chunk__).num_atoms_; i++)
                {
                    /* number of beta functions for a given atom */
                    int nbf = beta_.beta_chunk(chunk__).desc_(0, i);
                    int ofs = beta_.beta_chunk(chunk__).desc_(1, i);
                    int ia  = beta_.beta_chunk(chunk__).desc_(3, i);

                    /* compute O * <beta|phi> */
                    linalg<CPU>::gemm(0, 0, nbf, n__, nbf,
                                      op_.at<CPU>(packed_mtrx_offset_(ia)), nbf,
                                      beta_phi.at<CPU>(ofs), nbeta,
                                      work.at<CPU>(ofs, 0), work.ld());
                }
                
                /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
                linalg<CPU>::gemm(0, 0, num_gkvec_loc, n__, nbeta, double_complex(1, 0),
                                  beta_gk.at<CPU>(), num_gkvec_loc, work.at<CPU>(), work.ld(), double_complex(1, 0),
                                  &op_phi__(0, idx0__), num_gkvec_loc);
            }
            #ifdef __GPU
            if (pu_ == GPU)
            {
                work.allocate_on_device();
                #pragma omp parallel for
                for (int i = 0; i < beta_.beta_chunk(chunk__).num_atoms_; i++)
                {
                    /* number of beta functions for a given atom */
                    int nbf = beta_.beta_chunk(chunk__).desc_(0, i);
                    int ofs = beta_.beta_chunk(chunk__).desc_(1, i);
                    int ia  = beta_.beta_chunk(chunk__).desc_(3, i);

                    /* compute O * <beta|phi> */
                    linalg<GPU>::gemm(0, 0, nbf, n__, nbf,
                                      op_.at<GPU>(packed_mtrx_offset_(ia)), nbf, 
                                      beta_phi.at<GPU>(ofs), nbeta,
                                      work.at<GPU>(ofs, 0), work.ld(),
                                      omp_get_thread_num());

                }
                cuda_device_synchronize();
                double_complex alpha(1, 0);
                
                /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
                linalg<GPU>::gemm(0, 0, num_gkvec_loc, n__, nbeta, &alpha,
                                  beta_gk.at<GPU>(), beta_gk.ld(), work.at<GPU>(), work.ld(), &alpha, 
                                  op_phi__.coeffs().at<GPU>(0, idx0__), op_phi__.coeffs().ld());
                
                cuda_device_synchronize();
            }
            #endif
        }
};

class D_operator: public Non_local_operator
{
    public:

        D_operator(Beta_projectors& beta__, processing_unit_t pu__) : Non_local_operator(beta__, pu__)
        {
            auto& uc = beta_.unit_cell();
            for (int ia = 0; ia < uc.num_atoms(); ia++)
            {
                int nbf = uc.atom(ia)->mt_basis_size();
                for (int xi2 = 0; xi2 < nbf; xi2++)
                {
                    for (int xi1 = 0; xi1 < nbf; xi1++)
                    {
                        op_(packed_mtrx_offset_(ia) + xi2 * nbf + xi1) = uc.atom(ia)->d_mtrx(xi1, xi2);
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
            auto& uc = beta_.unit_cell();
            for (int ia = 0; ia < uc.num_atoms(); ia++)
            {
                int nbf = uc.atom(ia)->mt_basis_size();
                for (int xi2 = 0; xi2 < nbf; xi2++)
                {
                    for (int xi1 = 0; xi1 < nbf; xi1++)
                    {
                        op_(packed_mtrx_offset_(ia) + xi2 * nbf + xi1) = uc.atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
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
