#ifndef __NON_LOCAL_OPERATOR_H__
#define __NON_LOCAL_OPERATOR_H__

namespace sirius {

class Non_local_operator
{
    protected:

        Beta_projectors& beta_;
        
        int packed_mtrx_size_;

        mdarray<int, 1> packed_mtrx_offset_;
        
        mdarray<double_complex, 1> op_;

        Non_local_operator& operator=(Non_local_operator const& src) = delete;
        Non_local_operator(Non_local_operator const& src) = delete;

    public:

        Non_local_operator(Beta_projectors& beta__) : beta_(beta__)
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
        }

        void apply(Wave_functions& op_phi__, int idx0__, int n__)
        {
            PROFILE();

            int ib = 0;

            auto& beta_phi = beta_.beta_phi();
            auto& uc = beta_.unit_cell();
            int num_gkvec_loc = beta_.num_gkvec_loc();
            
            matrix<double_complex> work(uc.mt_basis_size(), n__);

            #pragma omp parallel for
            for (int i = 0; i < beta_.beta_chunk(ib).num_atoms_; i++)
            {
                /* number of beta functions for a given atom */
                int nbf = beta_.beta_chunk(ib).desc_(0, i);
                int ofs = beta_.beta_chunk(ib).desc_(1, i);
                int ia  = beta_.beta_chunk(ib).desc_(3, i);

                /* compute O * <beta|phi> */
                linalg<CPU>::gemm(0, 0, nbf, n__, nbf,
                                  op_.at<CPU>(packed_mtrx_offset_(ia)), nbf,
                                  beta_phi.at<CPU>(ofs, 0), beta_phi.ld(),
                                  work.at<CPU>(ofs, 0), work.ld());
            }

            int ng = beta_.num_gkvec_loc();
            auto& beta_gk = beta_.beta_gk();
            
            /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
            linalg<CPU>::gemm(0, 0, ng, n__, beta_.beta_chunk(ib).num_beta_, double_complex(1, 0),
                              beta_gk.at<CPU>(), num_gkvec_loc, work.at<CPU>(), work.ld(), double_complex(1, 0),
                              &op_phi__(0, idx0__), num_gkvec_loc);
        }
};

class D_operator: public Non_local_operator
{
    public:

        D_operator(Beta_projectors& beta__) : Non_local_operator(beta__)
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
        }
};

class Q_operator: public Non_local_operator
{
    public:
        
        Q_operator(Beta_projectors& beta__) : Non_local_operator(beta__)
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
        }
};

}

#endif
