#include "band.h"

namespace sirius {

#ifdef __SCALAPACK
void Band::residuals_parallel(int N__,
                              int num_bands__,
                              K_point* kp__,
                              std::vector<double>& eval__,
                              matrix<double_complex>& evec__,
                              dmatrix<double_complex>& hphi__,
                              dmatrix<double_complex>& ophi__,
                              dmatrix<double_complex>& hpsi__,
                              dmatrix<double_complex>& opsi__,
                              dmatrix<double_complex>& res__,
                              std::vector<double>& h_diag__,
                              std::vector<double>& o_diag__,
                              std::vector<double>& res_norm__,
                              mdarray<double_complex, 1>& kappa__)
{
    PROFILE();

    Timer t("sirius::Band::residuals_parallel", kp__->comm());

    int num_gkvec_loc = kp__->num_gkvec_loc();
    
    Timer t2("sirius::Band::residuals_parallel|zgemm");
    if (parameters_.processing_unit() == CPU)
    {
        /* compute H\Psi_{i} = H\phi_{mu} * Z_{mu, i} */
        linalg<CPU>::gemm(0, 0, num_gkvec_loc, num_bands__, N__, hphi__.panel(), evec__, hpsi__.panel());
        /* compute O\Psi_{i} = O\phi_{mu} * Z_{mu, i} */
        linalg<CPU>::gemm(0, 0, num_gkvec_loc, num_bands__, N__, ophi__.panel(), evec__, opsi__.panel());
    }

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef __GPU
        linalg<GPU>::gemm(0, 0, num_gkvec_loc, num_bands__, N__, hphi__.at<GPU>(), hphi__.ld(),
                          evec__.at<GPU>(), evec__.ld(), hpsi__.at<GPU>(), hpsi__.ld());

        cublas_get_matrix(num_gkvec_loc, num_bands__, sizeof(double_complex), hpsi__.at<GPU>(), hpsi__.ld(),
                          hpsi__.at<CPU>(), hpsi__.ld());

        linalg<GPU>::gemm(0, 0, num_gkvec_loc, num_bands__, N__, ophi__.at<GPU>(), ophi__.ld(),
                          evec__.at<GPU>(), evec__.ld(), opsi__.at<GPU>(), opsi__.ld());

        cublas_get_matrix(num_gkvec_loc, num_bands__, sizeof(double_complex), opsi__.at<GPU>(), opsi__.ld(),
                          opsi__.at<CPU>(), opsi__.ld());
        #endif
    }
    double tval = t2.stop();


    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        DUMP("effective zgemm with M, N, K: %6i %6i %6i for hpsi and opsi: %12.4f sec, %12.4f GFlops/rank\n",
             kp__->num_gkvec(), num_bands__, N__, tval,
             2 * 8e-9 * kp__->num_gkvec() * num_bands__ * N__ / tval / kp__->num_ranks());
    }
    
    memset(&res_norm__[0], 0, num_bands__ * sizeof(double));
    /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm squared */
    #pragma omp parallel for
    for (int i = 0; i < num_bands__; i++)
    {
        double norm2 = 0;
        for (int igk = 0; igk < num_gkvec_loc; igk++) 
        {
            res__(igk, i) = hpsi__(igk, i) - eval__[i] * opsi__(igk, i);
            norm2 += real(conj(res__(igk, i)) * res__(igk, i));
        }
        res_norm__[i] = norm2;
    }
    kp__->comm().allreduce(res_norm__);
    
    /* compute norm */
    for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);

    /* apply preconditioner */
    #pragma omp parallel for
    for (int i = 0; i < num_bands__; i++)
    {
        for (int igk = 0; igk < num_gkvec_loc; igk++)
        {
            double p = h_diag__[igk] - eval__[i] * o_diag__[igk];

            p *= 2; // QE formula is in Ry; here we convert to Ha
            p = 0.25 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
            res__(igk, i) /= p;
        }
    }
    
    std::vector<double> norm2(num_bands__, 0);
    /* Normalize new basis functions */
    #pragma omp parallel for
    for (int i = 0; i < num_bands__; i++)
    {
        double d = 0;
        for (int igk = 0; igk < num_gkvec_loc; igk++) 
            d += real(conj(res__(igk, i)) * res__(igk, i));
        norm2[i] = d;
    }
    kp__->comm().allreduce(norm2);
    #pragma omp parallel for
    for (int i = 0; i < num_bands__; i++)
    {
        double d = 1.0 / std::sqrt(norm2[i]);
        for (int igk = 0; igk < num_gkvec_loc; igk++) res__(igk, i) *= d;
    }
}
#endif // __SCALAPACK

};
