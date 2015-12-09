#include "band.h"

namespace sirius {

void Band::diag_h_o(K_point* kp__,
                    int N__,
                    int num_bands__,
                    matrix<double_complex>& hmlt__,
                    matrix<double_complex>& ovlp__,
                    matrix<double_complex>& evec__,
                    dmatrix<double_complex>& hmlt_dist__,
                    dmatrix<double_complex>& ovlp_dist__,
                    dmatrix<double_complex>& evec_dist__,
                    std::vector<double>& eval__)
{
    Timer t("sirius::Band::diag_h_o");

    if (kp__->comm().size() > 1 && gen_evp_solver()->parallel())
    {
        for (int jloc = 0; jloc < hmlt_dist__.num_cols_local(); jloc++)
        {
            int j = hmlt_dist__.icol(jloc);
            for (int iloc = 0; iloc < hmlt_dist__.num_rows_local(); iloc++)
            {
                int i = hmlt_dist__.irow(iloc);
                hmlt_dist__(iloc, jloc) = (i > j) ? std::conj(hmlt__(j, i)) : hmlt__(i, j);
                ovlp_dist__(iloc, jloc) = (i > j) ? std::conj(ovlp__(j, i)) : ovlp__(i, j);
            }
        }
    }

    if (gen_evp_solver()->parallel())
    {
        gen_evp_solver()->solve(N__, hmlt_dist__.num_rows_local(), hmlt_dist__.num_cols_local(), num_bands__, 
                                hmlt_dist__.at<CPU>(), hmlt_dist__.ld(), ovlp_dist__.at<CPU>(), ovlp_dist__.ld(), 
                                &eval__[0], evec_dist__.at<CPU>(), evec_dist__.ld());
    }
    else
    {
        gen_evp_solver()->solve(N__, num_bands__, num_bands__, num_bands__, hmlt__.at<CPU>(), hmlt__.ld(),
                                ovlp__.at<CPU>(), ovlp__.ld(), &eval__[0], evec__.at<CPU>(), evec__.ld());
    }

    if (kp__->comm().size() > 1 && gen_evp_solver()->parallel())
    {
        evec__.zero();
        for (int i = 0; i < evec_dist__.num_cols_local(); i++)
        {
            for (int j = 0; j < evec_dist__.num_rows_local(); j++)
            {
                evec__(evec_dist__.irow(j), evec_dist__.icol(i)) = evec_dist__(j, i);
            }
        }
        kp__->comm().allreduce(evec__.at<CPU>(), evec__.ld() * num_bands__);
    }
    /* copy eigen-vectors to GPU */
    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
        acc::copyin(evec__.at<GPU>(), evec__.ld(), evec__.at<CPU>(), evec__.ld(), N__, num_bands__);
    #endif
}

};
