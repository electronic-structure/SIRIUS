#include "band.h"

namespace sirius {

int Band::residuals(K_point* kp__,
                    int N__,
                    int num_bands__,
                    std::vector<double>& eval__,
                    std::vector<double>& eval_old__,
                    matrix<double_complex>& evec__,
                    Wave_functions& hphi__,
                    Wave_functions& ophi__,
                    Wave_functions& hpsi__,
                    Wave_functions& opsi__,
                    Wave_functions& res__,
                    std::vector<double>& h_diag__,
                    std::vector<double>& o_diag__,
                    mdarray<double_complex, 1>& kappa__)
{
    PROFILE();

    Timer t("sirius::Band::residuals");

    auto& itso = kp__->iterative_solver_input_section_;
    bool converge_by_energy = (itso.converge_by_energy_ == 1);

    /* norm of residuals */
    std::vector<double> res_norm(num_bands__);

    int n = 0;
    if (converge_by_energy)
    {
        std::vector<double> eval_tmp(num_bands__);

        /* main trick here: first estimate energy difference, and only then compute unconverged residuals */
        double tol = ctx_.iterative_solver_tolerance();
        for (int i = 0; i < num_bands__; i++)
        {
            if (kp__->band_occupancy(i) > 1e-10 && std::abs(eval__[i] - eval_old__[i]) > tol)
            {
                std::memcpy(&evec__(0, num_bands__ + n), &evec__(0, i), N__ * sizeof(double_complex));
                eval_tmp[n++] = eval__[i];
            }
        }

        //#ifdef __GPU
        //if (parameters_.processing_unit() == GPU)
        //    cublas_set_matrix(N, n, sizeof(double_complex), evec_tmp.at<CPU>(), evec_tmp.ld(), evec_tmp.at<GPU>(), evec_tmp.ld());
        //#endif

        matrix<double_complex> evec_tmp(&evec__(0, num_bands__), evec__.ld(), n);

        residuals_serial(kp__, N__, n, eval_tmp, evec_tmp, hphi__, ophi__, hpsi__, opsi__, res__, h_diag__, o_diag__, res_norm, kappa__);

        //#ifdef __GPU
        //if (parameters_.processing_unit() == GPU && economize_gpu_memory)
        //{
        //    /* copy residuals to CPU because the content of kappa array can be destroyed */
        //    cublas_get_matrix(ngk, n, sizeof(double_complex),
        //                      kappa.at<GPU>(ngk * 2 * n), ngk,
        //                      res.at<CPU>(), res.ld());
        //}
        //#endif
    }
    else
    {
        residuals_serial(kp__, N__, num_bands__, eval__, evec__, hphi__, ophi__, hpsi__, opsi__, res__, h_diag__, o_diag__, res_norm, kappa__);

        //#ifdef __GPU
        //matrix<double_complex> res_tmp;
        //if (parameters_.processing_unit() == GPU)
        //{
        //    if (economize_gpu_memory)
        //    {
        //        res_tmp = matrix<double_complex>(nullptr, kappa.at<GPU>(ngk * 2 * num_bands), ngk, num_bands);
        //    }
        //    else
        //    {
        //        res_tmp = matrix<double_complex>(nullptr, res.at<GPU>(), ngk, num_bands);
        //    }
        //}
        //#endif
        
        Timer t1("sirius::Band::diag_fv_pseudo_potential|sort_res");

        for (int i = 0; i < num_bands__; i++)
        {
            /* take the residual if it's norm is above the threshold */
            if (res_norm[i] > ctx_.iterative_solver_tolerance() && kp__->band_occupancy(i) > 1e-10)
            {
                /* shift unconverged residuals to the beginning of array */
                if (n != i)
                {
                    switch (parameters_.processing_unit())
                    {
                        case CPU:
                        {
                            std::memcpy(&res__(0, n), &res__(0, i), res__.num_gvec_loc() * sizeof(double_complex));
                            break;
                        }
                        //case GPU:
                        //{
                        //    #ifdef __GPU
                        //    cuda_copy_device_to_device(res_tmp.at<GPU>(0, n), res_tmp.at<GPU>(0, i), ngk * sizeof(double_complex));
                        //    #else
                        //    TERMINATE_NO_GPU
                        //    #endif
                        //    break;
                        //}
                    }
                }
                n++;
            }
        }
        //#ifdef __GPU
        //if (parameters_.processing_unit() == GPU && economize_gpu_memory)
        //{
        //    /* copy residuals to CPU because the content of kappa array will be destroyed */
        //    cublas_get_matrix(ngk, n, sizeof(double_complex), res_tmp.at<GPU>(), res_tmp.ld(),
        //                      res.at<CPU>(), res.ld());
        //}
        //#endif
    }

    return n;
}

};
