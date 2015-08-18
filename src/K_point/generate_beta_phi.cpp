#include "k_point.h"

namespace sirius {

void K_point::generate_beta_phi(int nbeta__,
                                matrix<double_complex>& phi__,
                                int nphi__,
                                int offs__,
                                matrix<double_complex>& beta_gk__,
                                matrix<double_complex>& beta_phi__) // TODO: pass num_gkvec_loc or num_gkvec_row
{
    Timer t("sirius::K_point::generate_beta_phi");
    #ifdef __GPU
    #ifdef __GPU_DIRECT
    // allrecue with gpu-direct is broken at the moment
    bool gpu_direct = false;
    #else
    bool gpu_direct = false;
    #endif
    #endif

    if (parameters_.processing_unit() == CPU)
    {
        double wt = -MPI_Wtime();
        /* compute <beta|phi> */
        linalg<CPU>::gemm(2, 0, nbeta__, nphi__, num_gkvec_loc(), 
                          beta_gk__.at<CPU>(), beta_gk__.ld(), 
                          phi__.at<CPU>(0, offs__), phi__.ld(), 
                          beta_phi__.at<CPU>(), beta_phi__.ld());

        comm().allreduce(beta_phi__.at<CPU>(), (int)beta_phi__.size());
        wt += MPI_Wtime();
        PRINT("effective zgemm performance for <beta|phi> %f GFlops", 8e-9 * nbeta__ * nphi__ * num_gkvec() / wt);
    }

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef __GPU
        /* compute <beta|phi> */
        linalg<GPU>::gemm(2, 0, nbeta__, nphi__, num_gkvec_loc(), 
                          beta_gk__.at<GPU>(), beta_gk__.ld(), 
                          phi__.at<GPU>(0, offs__), phi__.ld(), 
                          beta_phi__.at<GPU>(), beta_phi__.ld());
        
        if (comm().size() > 1)
        {
            if (gpu_direct)
            {
                comm().allreduce(beta_phi__.at<GPU>(), (int)beta_phi__.size());
            }
            else
            {
                beta_phi__.copy_to_host();
                comm().allreduce(beta_phi__.at<CPU>(), (int)beta_phi__.size());
                beta_phi__.copy_to_device();
            }
        }

        cuda_device_synchronize();
        #else
        TERMINATE_NO_GPU
        #endif
    }
}

};
