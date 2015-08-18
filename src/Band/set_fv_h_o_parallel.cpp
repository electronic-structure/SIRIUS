#include "band.h"

namespace sirius {

void Band::set_fv_h_o_parallel(int N__,
                               int n__,
                               K_point* kp__,
                               dmatrix<double_complex>& phi_slab__,
                               dmatrix<double_complex>& hphi_slab__,
                               dmatrix<double_complex>& ophi_slab__,
                               dmatrix<double_complex>& h__,
                               dmatrix<double_complex>& o__,
                               dmatrix<double_complex>& h_old__,
                               dmatrix<double_complex>& o_old__,
                               mdarray<double_complex, 1>& kappa__)
{
    PROFILE();

    Timer t("sirius::Band::set_fv_h_o_parallel", kp__->comm());

    splindex<block_cyclic> s0_col(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s1_col(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s0_row(N__,       kp__->num_ranks_row(), kp__->rank_row(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s1_row(N__ + n__, kp__->num_ranks_row(), kp__->rank_row(), parameters_.cyclic_block_size());

    /* copy old Hamiltonian and overlap */
    for (int i = 0; i < (int)s0_col.local_size(); i++)
    {
        memcpy(&h__(0, i), &h_old__(0, i), s0_row.local_size() * sizeof(double_complex));
        memcpy(&o__(0, i), &o_old__(0, i), s0_row.local_size() * sizeof(double_complex));
    }

    matrix<double_complex> tmp;
    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            tmp = matrix<double_complex>(kappa__.at<CPU>(), N__ + n__, n__);
            break;
        }
        case GPU:
        {
            #ifdef __GPU
            tmp = matrix<double_complex>(kappa__.at<CPU>(), kappa__.at<GPU>(), N__ + n__, n__);
            #endif
            break;
        }
    }
    
    int col_offs = (int)s0_col.local_size();
    Timer t2("sirius::Band::set_fv_h_o_parallel|zgemm_eff", kp__->comm());
    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            linalg<CPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec_loc(), phi_slab__.at<CPU>(), phi_slab__.ld(), 
                              hphi_slab__.at<CPU>(0, N__), hphi_slab__.ld(), tmp.at<CPU>(), tmp.ld());
            break;
        }
        case GPU:
        {
            #ifdef __GPU
            linalg<GPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec_loc(), phi_slab__.at<GPU>(), phi_slab__.ld(),
                              hphi_slab__.at<GPU>(0, N__), hphi_slab__.ld(), tmp.at<GPU>(), tmp.ld());
            tmp.copy_to_host();
            #endif
            break;
        }
    }
    kp__->comm().allreduce(tmp.at<CPU>(), (int)tmp.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)(s1_col.local_size() - col_offs); i++)
    {
        for (int j = 0; j < (int)s1_row.local_size(); j++)
        {
            h__(j, col_offs + i) = tmp(s1_row[j], s1_col[col_offs + i] - N__);
        }
    }

    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            linalg<CPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec_loc(), phi_slab__.at<CPU>(), phi_slab__.ld(), 
                              ophi_slab__.at<CPU>(0, N__), ophi_slab__.ld(), tmp.at<CPU>(), tmp.ld());
            break;
        }
        case GPU:
        {
            #ifdef __GPU
            linalg<GPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec_loc(), phi_slab__.at<GPU>(), phi_slab__.ld(),
                              ophi_slab__.at<GPU>(0, N__), ophi_slab__.ld(), tmp.at<GPU>(), tmp.ld());
            tmp.copy_to_host();
            #endif
            break;
        }
    }
    kp__->comm().allreduce(tmp.at<CPU>(), (int)tmp.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)(s1_col.local_size() - col_offs); i++)
    {
        for (int j = 0; j < (int)s1_row.local_size(); j++)
        {
            o__(j, col_offs + i) = tmp(s1_row[j], s1_col[col_offs + i] - N__);
        }
    }
    double tval = t2.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        DUMP("effective zgemm with M, N, K: %6i %6i %6i for H and O: %12.4f sec, %12.4f GFlops/rank",
             N__ + n__, n__, kp__->num_gkvec(), tval,
             2 * 8e-9 * (N__ + n__) * n__ * kp__->num_gkvec() / tval / kp__->num_ranks());
    }
    
    /* restore the bottom block of the matrix */
    if (N__ != 0)
    {
        linalg<CPU>::tranc(n__, N__, h__, 0, N__, h__, N__, 0);
        linalg<CPU>::tranc(n__, N__, o__, 0, N__, o__, N__, 0);
    }

    /* save Hamiltonian and overlap */
    for (int i = 0; i < (int)s1_col.local_size(); i++)
    {
        memcpy(&h_old__(0, i), &h__(0, i), s1_row.local_size() * sizeof(double_complex));
        memcpy(&o_old__(0, i), &o__(0, i), s1_row.local_size() * sizeof(double_complex));
    }
}

};
