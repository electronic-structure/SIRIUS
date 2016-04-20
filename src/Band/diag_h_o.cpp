#include "band.h"

namespace sirius {

template <>
void Band::diag_h_o<double_complex>(K_point* kp__,
                                    int N__,
                                    int num_bands__,
                                    matrix<double_complex>& hmlt__,
                                    matrix<double_complex>& ovlp__,
                                    matrix<double_complex>& evec__,
                                    dmatrix<double_complex>& hmlt_dist__,
                                    dmatrix<double_complex>& ovlp_dist__,
                                    dmatrix<double_complex>& evec_dist__,
                                    std::vector<double>& eval__) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_h_o");

    runtime::Timer t1("sirius::Band::diag_h_o|load");
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
    t1.stop();

    runtime::Timer t2("sirius::Band::diag_h_o|diag");
    int result;
    if (gen_evp_solver()->parallel())
    {
        result = gen_evp_solver()->solve(N__,  num_bands__, hmlt_dist__.at<CPU>(), hmlt_dist__.ld(),
                                         ovlp_dist__.at<CPU>(), ovlp_dist__.ld(), &eval__[0], evec_dist__.at<CPU>(),
                                         evec_dist__.ld(), hmlt_dist__.num_rows_local(), hmlt_dist__.num_cols_local());
    }
    else
    {
        result = gen_evp_solver()->solve(N__, num_bands__, hmlt__.at<CPU>(), hmlt__.ld(), ovlp__.at<CPU>(), ovlp__.ld(),
                                         &eval__[0], evec__.at<CPU>(), evec__.ld());
    }
    if (result)
    {
        std::stringstream s;
        s << "error in diagonalziation for k-point (" << kp__->vk()[0] << " " << kp__->vk()[1] << " " << kp__->vk()[2] << ")";
        TERMINATE(s);
    }
    t2.stop();

    runtime::Timer t3("sirius::Band::diag_h_o|gather");
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
    t3.stop();
    
    //// --== DEBUG ==--
    //printf("checking evec\n");
    //for (int i = 0; i < num_bands__; i++)
    //{
    //    for (int j = 0; j < N__; j++)
    //    {
    //        if (std::abs(evec__(j, i).imag()) > 1e-12)
    //        {
    //            printf("evec(%i, %i) = %20.16f %20.16f\n", i, j, evec__(j, i).real(), evec__(j, i).real());
    //        }
    //    }
    //}
    //printf("done.\n");

    /* copy eigen-vectors to GPU */
    #ifdef __GPU
    if (ctx_.processing_unit() == GPU)
        acc::copyin(evec__.at<GPU>(), evec__.ld(), evec__.at<CPU>(), evec__.ld(), N__, num_bands__);
    #endif
}

template <>
void Band::diag_h_o<double>(K_point* kp__,
                            int N__,
                            int num_bands__,
                            matrix<double>& hmlt__,
                            matrix<double>& ovlp__,
                            matrix<double>& evec__,
                            dmatrix<double>& hmlt_dist__,
                            dmatrix<double>& ovlp_dist__,
                            dmatrix<double>& evec_dist__,
                            std::vector<double>& eval__) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_h_o");

    runtime::Timer t1("sirius::Band::diag_h_o|load");
    if (kp__->comm().size() > 1 && std_evp_solver()->parallel())
    {
        for (int jloc = 0; jloc < hmlt_dist__.num_cols_local(); jloc++)
        {
            int j = hmlt_dist__.icol(jloc);
            for (int iloc = 0; iloc < hmlt_dist__.num_rows_local(); iloc++)
            {
                int i = hmlt_dist__.irow(iloc);
                hmlt_dist__(iloc, jloc) = (i > j) ? hmlt__(j, i) : hmlt__(i, j);
            }
        }
    }
    t1.stop();

    runtime::Timer t2("sirius::Band::diag_h_o|diag");
    int result;
    if (std_evp_solver()->parallel())
    {
        result = std_evp_solver()->solve(N__,  num_bands__, hmlt_dist__.at<CPU>(), hmlt_dist__.ld(),
                                         &eval__[0], evec_dist__.at<CPU>(), evec_dist__.ld(),
                                         hmlt_dist__.num_rows_local(), hmlt_dist__.num_cols_local());
    }
    else
    {
        result = std_evp_solver()->solve(N__, num_bands__, hmlt__.at<CPU>(), hmlt__.ld(), &eval__[0], evec__.at<CPU>(), evec__.ld());
    }
    if (result) TERMINATE("error in diagonalziation");
    t2.stop();

    runtime::Timer t3("sirius::Band::diag_h_o|gather");
    if (kp__->comm().size() > 1 && std_evp_solver()->parallel())
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
    t3.stop();

    /* copy eigen-vectors to GPU */
    #ifdef __GPU
    if (ctx_.processing_unit() == GPU)
        acc::copyin(evec__.at<GPU>(), evec__.ld(), evec__.at<CPU>(), evec__.ld(), N__, num_bands__);
    #endif
}

};
