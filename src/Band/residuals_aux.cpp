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

void Band::residuals_aux(K_point* kp__,
                         int num_bands__,
                         std::vector<double>& eval__,
                         Wave_functions& hpsi__,
                         Wave_functions& opsi__,
                         Wave_functions& res__,
                         std::vector<double>& h_diag__,
                         std::vector<double>& o_diag__,
                         std::vector<double>& res_norm__,
                         mdarray<double_complex, 1>& kappa__)
{
    Timer t("sirius::Band::residuals_serial");

    auto pu = parameters_.processing_unit();

    /* compute residuals norm and apply preconditioner */
    if (pu == CPU || pu == GPU)
    {
        std::fill(res_norm__.begin(), res_norm__.end(), 0);
        /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm squared */
        #pragma omp parallel for
        for (int i = 0; i < num_bands__; i++)
        {
            double norm2 = 0;
            for (int ig = 0; ig < res__.num_gvec_loc(); ig++) 
            {
                /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
                res__(ig, i) = hpsi__(ig, i) - eval__[i] * opsi__(ig, i);
                norm2 += std::real(std::conj(res__(ig, i)) * res__(ig, i));
            }
            res_norm__[i] = norm2;
        }
        res__.comm().allreduce(res_norm__);

        /* compute norm */
        for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);

        /* apply preconditioner */
        #pragma omp parallel for
        for (int i = 0; i < num_bands__; i++)
        {
            for (int ig = 0; ig < res__.num_gvec_loc(); ig++)
            {
                double p = h_diag__[ig] - eval__[i] * o_diag__[ig];
                p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                res__(ig, i) /= p;
            }
        }

        std::vector<double> norm2(num_bands__, 0);
        /* normalize new basis functions */
        #pragma omp parallel for
        for (int i = 0; i < num_bands__; i++)
        {
            double d = 0;
            for (int ig = 0; ig < res__.num_gvec_loc(); ig++) 
                d += std::real(std::conj(res__(ig, i)) * res__(ig, i));
            norm2[i] = d;
        }
        res__.comm().allreduce(norm2);
        #pragma omp parallel for
        for (int i = 0; i < num_bands__; i++)
        {
            double d = 1.0 / std::sqrt(norm2[i]);
            for (int ig = 0; ig < res__.num_gvec_loc(); ig++) res__(ig, i) *= d;
        }
    }

    //if (pu == GPU)
    //{
    //    STOP();
    //    //#ifdef __GPU
    //    //double_complex* hpsi_ptr;
    //    //double_complex* opsi_ptr;
    //    //double_complex* res_ptr;

    //    //if (economize_gpu_memory)
    //    //{
    //    //    hpsi_ptr = kappa__.at<GPU>(kp__->num_gkvec() * num_bands__);
    //    //    opsi_ptr = kappa__.at<GPU>();
    //    //    res_ptr = kappa__.at<GPU>(kp__->num_gkvec() * 2 * num_bands__);
    //    //}
    //    //else
    //    //{
    //    //    hpsi_ptr = hpsi__.at<GPU>();
    //    //    opsi_ptr = opsi__.at<GPU>();
    //    //    res_ptr = res__.at<GPU>();
    //    //}

    //    //mdarray<double, 1> res_norm_gpu(&res_norm__[0], num_bands__);
    //    //res_norm_gpu.allocate_on_device();
    //    //res_norm_gpu.zero_on_device();

    //    //mdarray<double, 1> eval_gpu(&eval__[0], num_bands__);
    //    //eval_gpu.allocate_on_device();
    //    //eval_gpu.copy_to_device();

    //    ///* global index of residual */
    //    //mdarray<int, 1> res_idx_gpu(num_bands__);
    //    //for (int i = 0; i < num_bands__; i++) res_idx_gpu(i) = i;
    //    //res_idx_gpu.allocate_on_device();
    //    //res_idx_gpu.copy_to_device();

    //    //compute_residuals_gpu(kp__->num_gkvec(), num_bands__, res_idx_gpu.at<GPU>(), eval_gpu.at<GPU>(),
    //    //                      hpsi_ptr, opsi_ptr, res_ptr, res_norm_gpu.at<GPU>());
    //    //res_norm_gpu.copy_to_host();

    //    ///* compute norm */
    //    //for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);

    //    //mdarray<double, 1> hdiag_gpu(&h_diag__[0], kp__->num_gkvec_row());
    //    //hdiag_gpu.allocate_on_device();
    //    //hdiag_gpu.copy_to_device();

    //    //mdarray<double, 1> odiag_gpu(&o_diag__[0], kp__->num_gkvec_row());
    //    //odiag_gpu.allocate_on_device();
    //    //odiag_gpu.copy_to_device();

    //    //mdarray<double, 1> norm2(num_bands__);
    //    //norm2.allocate_on_device();
    //    //norm2.zero_on_device();

    //    //apply_preconditioner_gpu(kp__->num_gkvec(), num_bands__, res_idx_gpu.at<GPU>(), eval_gpu.at<GPU>(),
    //    //                         hdiag_gpu.at<GPU>(), odiag_gpu.at<GPU>(), res_ptr, norm2.at<GPU>());

    //    //normalize_residuals_gpu(kp__->num_gkvec_row(), num_bands__, res_idx_gpu.at<GPU>(), norm2.at<GPU>(), res_ptr);

    //    ////== if (economize_gpu_memory)
    //    ////== {
    //    ////==     cublas_get_matrix(kp__->num_gkvec(), num_bands__, sizeof(double_complex), res_ptr, kp__->num_gkvec(),
    //    ////==                       res__.at<CPU>(), res__.ld());
    //    ////== }
    //    //#else
    //    //TERMINATE_NO_GPU
    //    //#endif
    //}
}

};
