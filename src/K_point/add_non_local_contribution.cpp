#include "k_point.h"

namespace sirius {

void K_point::add_non_local_contribution(int num_atoms__,
                                         int num_beta__,
                                         mdarray<int, 2> const& beta_desc__,
                                         matrix<double_complex>& beta_gk__,
                                         mdarray<double_complex, 1>& op_mtrx_packed__,
                                         mdarray<int, 1> const& packed_mtrx_offset__,
                                         matrix<double_complex>& beta_phi__,
                                         matrix<double_complex>& op_phi__,
                                         int nphi__,
                                         int offs__,
                                         double_complex alpha,
                                         matrix<double_complex>& work__)
{
    Timer t("sirius::K_point::add_non_local_contribution");

    double_complex beta = complex_one;

    if (parameters_.processing_unit() == CPU)
    {
        #pragma omp parallel for
        for (int i = 0; i < num_atoms__; i++)
        {
            /* number of beta functions for a given atom */
            int nbf = beta_desc__(0, i);
            int ofs = beta_desc__(1, i);
            int ia  = beta_desc__(3, i);

            /* compute O * <beta|phi> */
            linalg<CPU>::gemm(0, 0, nbf, nphi__, nbf,
                              op_mtrx_packed__.at<CPU>(packed_mtrx_offset__(ia)), nbf,
                              beta_phi__.at<CPU>(ofs, 0), beta_phi__.ld(),
                              work__.at<CPU>(ofs, 0), work__.ld());
        }
        
        /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
        linalg<CPU>::gemm(0, 0, num_gkvec_loc(), nphi__, num_beta__, alpha,
                          beta_gk__.at<CPU>(), beta_gk__.ld(), work__.at<CPU>(), work__.ld(), beta,
                          op_phi__.at<CPU>(0, offs__), op_phi__.ld());
    }

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef __GPU
        #pragma omp parallel for
        for (int i = 0; i < num_atoms__; i++)
        {
            /* number of beta functions for a given atom */
            int nbf = beta_desc__(0, i);
            int ofs = beta_desc__(1, i);
            int ia  = beta_desc__(3, i);

            /* compute O * <beta|phi> */
            linalg<GPU>::gemm(0, 0, nbf, nphi__, nbf,
                              op_mtrx_packed__.at<GPU>(packed_mtrx_offset__(ia)), nbf, 
                              beta_phi__.at<GPU>(ofs, 0), beta_phi__.ld(),
                              work__.at<GPU>(ofs, 0), work__.ld(), 
                              Platform::thread_id());

        }
        cuda_device_synchronize();
        
        /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
        linalg<GPU>::gemm(0, 0, num_gkvec_loc(), nphi__, num_beta__, &alpha,
                          beta_gk__.at<GPU>(), beta_gk__.ld(), work__.at<GPU>(), work__.ld(), &beta, 
                          op_phi__.at<GPU>(0, offs__), op_phi__.ld());
        
        cuda_device_synchronize();
        #else
        TERMINATE_NO_GPU
        #endif
    }
}

};
