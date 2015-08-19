#include "band.h"

namespace sirius {

#ifdef __GPU
extern "C" void compute_residuals_gpu(int num_gkvec_row,
                                      int num_res_local,
                                      int const* res_idx,
                                      double const* eval,
                                      cuDoubleComplex const* hpsi,
                                      cuDoubleComplex const* opsi,
                                      cuDoubleComplex* res,
                                      double* res_norm);

extern "C" void apply_preconditioner_gpu(int num_gkvec_row,
                                         int num_res_local,
                                         int const* res_idx,
                                         double const* eval,
                                         double const* h_diag,
                                         double const* o_diag,
                                         cuDoubleComplex* res,
                                         double* res_norm);

extern "C" void normalize_residuals_gpu(int num_gkvec_row,
                                        int num_res_local,
                                        int const* res_idx,
                                        double const* norm2,
                                        cuDoubleComplex* res);
#endif

void Band::residuals_serial(K_point* kp__,
                            int N__,
                            int num_bands__,
                            std::vector<double>& eval__,
                            matrix<double_complex>& evec__,
                            matrix<double_complex>& hphi__,
                            matrix<double_complex>& ophi__,
                            matrix<double_complex>& hpsi__,
                            matrix<double_complex>& opsi__,
                            matrix<double_complex>& res__,
                            std::vector<double>& h_diag__,
                            std::vector<double>& o_diag__,
                            std::vector<double>& res_norm__,
                            mdarray<double_complex, 1>& kappa__)
{
    Timer t("sirius::Band::residuals_serial");

    auto pu = parameters_.processing_unit();
    #ifdef __GPU
    bool economize_gpu_memory = (kappa__.size() != 0);
    #endif

    if (pu == CPU)
    {
        /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} */
        linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, &hphi__(0, 0), hphi__.ld(), &evec__(0, 0), evec__.ld(), 
                          &hpsi__(0, 0), hpsi__.ld());

        /* compute O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
        linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, &ophi__(0, 0), ophi__.ld(), &evec__(0, 0), evec__.ld(), 
                          &opsi__(0, 0), opsi__.ld());
    }

    if (pu == GPU)
    {
        #ifdef __GPU
        if (!economize_gpu_memory)
        {
            /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} */
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, hphi__.at<GPU>(), hphi__.ld(),
                              evec__.at<GPU>(), evec__.ld(), hpsi__.at<GPU>(), hpsi__.ld());

            /* compute O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, ophi__.at<GPU>(), ophi__.ld(),
                              evec__.at<GPU>(), evec__.ld(), opsi__.at<GPU>(), opsi__.ld());
        }
        else
        {
            /* copy hphi to device */
            matrix<double_complex> hphi(hphi__.at<CPU>(), kappa__.at<GPU>(), kp__->num_gkvec(), N__);
            hphi.copy_to_device();

            matrix<double_complex> hpsi(hpsi__.at<CPU>(), kappa__.at<GPU>(hphi.size()), kp__->num_gkvec(), num_bands__);

            /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} */
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, hphi.at<GPU>(), hphi.ld(),
                              evec__.at<GPU>(), evec__.ld(), hpsi.at<GPU>(), hpsi.ld());

            hpsi.copy_to_host();

            /* copy ophi to device */
            matrix<double_complex> ophi(ophi__.at<CPU>(), kappa__.at<GPU>(kp__->num_gkvec() * num_bands__), kp__->num_gkvec(), N__);
            ophi.copy_to_device();

            matrix<double_complex> opsi(opsi__.at<CPU>(), kappa__.at<GPU>(), kp__->num_gkvec(), num_bands__);
            /* compute O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, ophi.at<GPU>(), ophi.ld(),
                              evec__.at<GPU>(), evec__.ld(), opsi.at<GPU>(), opsi.ld());
            
            /* kappa(0, 0) contains opsi */
            opsi.copy_to_host();

            /* kappa(0, num_bands) contains hpsi */
            hpsi = matrix<double_complex>(hpsi__.at<CPU>(), kappa__.at<GPU>(kp__->num_gkvec() * num_bands__), kp__->num_gkvec(), num_bands__);
            hpsi.copy_to_device();
        }
        #else
        TERMINATE_NO_GPU
        #endif
    }

    if (pu == CPU)
    {
        /* compute residuals norm and apply preconditioner */
        #pragma omp parallel for
        for (int i = 0; i < num_bands__; i++)
        {
            double r = 0;
            for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            {
                /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
                res__(igk, i) = hpsi__(igk, i) - eval__[i] * opsi__(igk, i);
                r += std::real(std::conj(res__(igk, i)) * res__(igk, i));
            }
            res_norm__[i] = std::sqrt(r);
            
            /* apply preconditioner */
            for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            {
                double p = h_diag__[igk] - eval__[i] * o_diag__[igk];

                //if (std::abs(p) < 0.5e-5) p = copysign(0.5e-5, p);

                p *= 2; // QE formula is in Ry; here we convert to Ha
                p = 0.25 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                res__(igk, i) /= p;
            }
        }

        /* Normalize new basis functions */
        #pragma omp parallel for
        for (int i = 0; i < num_bands__; i++)
        {
            double d = 0;
            for (int igk = 0; igk < kp__->num_gkvec(); igk++) d += real(conj(res__(igk, i)) * res__(igk, i));
            //printf("res: %4i, norm: %18.12f\n", i, std::sqrt(d));
            d = 1.0 / std::sqrt(d);
            for (int igk = 0; igk < kp__->num_gkvec(); igk++) res__(igk, i) *= d;
        }
    }

    if (pu == GPU)
    {
        #ifdef __GPU
        double_complex* hpsi_ptr;
        double_complex* opsi_ptr;
        double_complex* res_ptr;

        if (economize_gpu_memory)
        {
            hpsi_ptr = kappa__.at<GPU>(kp__->num_gkvec() * num_bands__);
            opsi_ptr = kappa__.at<GPU>();
            res_ptr = kappa__.at<GPU>(kp__->num_gkvec() * 2 * num_bands__);
        }
        else
        {
            hpsi_ptr = hpsi__.at<GPU>();
            opsi_ptr = opsi__.at<GPU>();
            res_ptr = res__.at<GPU>();
        }

        mdarray<double, 1> res_norm_gpu(&res_norm__[0], num_bands__);
        res_norm_gpu.allocate_on_device();
        res_norm_gpu.zero_on_device();

        mdarray<double, 1> eval_gpu(&eval__[0], num_bands__);
        eval_gpu.allocate_on_device();
        eval_gpu.copy_to_device();

        /* global index of residual */
        mdarray<int, 1> res_idx_gpu(num_bands__);
        for (int i = 0; i < num_bands__; i++) res_idx_gpu(i) = i;
        res_idx_gpu.allocate_on_device();
        res_idx_gpu.copy_to_device();

        compute_residuals_gpu(kp__->num_gkvec(), num_bands__, res_idx_gpu.at<GPU>(), eval_gpu.at<GPU>(),
                              hpsi_ptr, opsi_ptr, res_ptr, res_norm_gpu.at<GPU>());
        res_norm_gpu.copy_to_host();

        /* compute norm */
        for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);

        mdarray<double, 1> hdiag_gpu(&h_diag__[0], kp__->num_gkvec_row());
        hdiag_gpu.allocate_on_device();
        hdiag_gpu.copy_to_device();

        mdarray<double, 1> odiag_gpu(&o_diag__[0], kp__->num_gkvec_row());
        odiag_gpu.allocate_on_device();
        odiag_gpu.copy_to_device();

        mdarray<double, 1> norm2(num_bands__);
        norm2.allocate_on_device();
        norm2.zero_on_device();

        apply_preconditioner_gpu(kp__->num_gkvec(), num_bands__, res_idx_gpu.at<GPU>(), eval_gpu.at<GPU>(),
                                 hdiag_gpu.at<GPU>(), odiag_gpu.at<GPU>(), res_ptr, norm2.at<GPU>());

        normalize_residuals_gpu(kp__->num_gkvec_row(), num_bands__, res_idx_gpu.at<GPU>(), norm2.at<GPU>(), res_ptr);

        //== if (economize_gpu_memory)
        //== {
        //==     cublas_get_matrix(kp__->num_gkvec(), num_bands__, sizeof(double_complex), res_ptr, kp__->num_gkvec(),
        //==                       res__.at<CPU>(), res__.ld());
        //== }
        #else
        TERMINATE_NO_GPU
        #endif
    }
}

};
