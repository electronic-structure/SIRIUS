#include "non_local_operator.h"

namespace sirius {

template<>
void Non_local_operator::apply<double_complex>(int chunk__, int ispn__, Wave_functions<false>& op_phi__, int idx0__, int n__)
{
    PROFILE_WITH_TIMER("sirius::Non_local_operator::apply");

    assert(op_phi__.num_gvec_loc() == beta_.num_gkvec_loc());

    auto beta_phi = beta_.beta_phi<double_complex>(chunk__, n__);
    auto& beta_gk = beta_.beta_gk();
    int num_gkvec_loc = beta_.num_gkvec_loc();
    int nbeta = beta_.beta_chunk(chunk__).num_beta_;

    if (static_cast<size_t>(2 * nbeta * n__) > work_.size())
    {
        work_ = mdarray<double, 1>(2 * nbeta * n__);
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
                              (double_complex*)op_.at<CPU>(2 * packed_mtrx_offset_(ia), ispn__), nbf,
                              beta_phi.at<CPU>(offs, 0), nbeta,
                              (double_complex*)work_.at<CPU>(2 * offs), nbeta);
        }
        
        /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
        linalg<CPU>::gemm(0, 0, num_gkvec_loc, n__, nbeta, double_complex(1, 0),
                          beta_gk.at<CPU>(), num_gkvec_loc, (double_complex*)work_.at<CPU>(), nbeta, double_complex(1, 0),
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
                              (double_complex*)op_.at<GPU>(2 * packed_mtrx_offset_(ia), ispn__), nbf, 
                              beta_phi.at<GPU>(offs, 0), nbeta,
                              (double_complex*)work_.at<GPU>(2 * offs), nbeta,
                              omp_get_thread_num());

        }
        cuda_device_synchronize();
        double_complex alpha(1, 0);
        
        /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
        linalg<GPU>::gemm(0, 0, num_gkvec_loc, n__, nbeta, &alpha,
                          beta_gk.at<GPU>(), beta_gk.ld(), (double_complex*)work_.at<GPU>(), nbeta, &alpha, 
                          op_phi__.coeffs().at<GPU>(0, idx0__), op_phi__.coeffs().ld());
        
        cuda_device_synchronize();
    }
    #endif
}

template<>
void Non_local_operator::apply<double>(int chunk__, int ispn__, Wave_functions<false>& op_phi__, int idx0__, int n__)
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
