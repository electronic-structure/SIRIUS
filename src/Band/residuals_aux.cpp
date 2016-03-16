#include "band.h"

namespace sirius {

#ifdef __GPU
extern "C" void residuals_aux_gpu(int num_gvec_loc__,
                                  int num_res_local__,
                                  int* res_idx__,
                                  double* eval__,
                                  cuDoubleComplex const* hpsi__,
                                  cuDoubleComplex const* opsi__,
                                  double const* h_diag__,
                                  double const* o_diag__,
                                  cuDoubleComplex* res__,
                                  double* res_norm__,
                                  double* p_norm__);
#endif

void Band::residuals_aux(K_point* kp__,
                         int num_bands__,
                         std::vector<double>& eval__,
                         Wave_functions<false>& hpsi__,
                         Wave_functions<false>& opsi__,
                         Wave_functions<false>& res__,
                         std::vector<double>& h_diag__,
                         std::vector<double>& o_diag__,
                         std::vector<double>& res_norm__) const
{
    PROFILE_WITH_TIMER("sirius::Band::residuals_aux");

    auto pu = ctx_.processing_unit();

    mdarray<double, 1> res_norm(&res_norm__[0], num_bands__);
    mdarray<double, 1> p_norm(num_bands__);

    #ifdef __GPU
    mdarray<int, 1> res_idx;
    mdarray<double, 1> eval;
    mdarray<double, 1> h_diag;
    mdarray<double, 1> o_diag;
    if (pu == GPU)
    {
        /* global index of residual */
        res_idx = mdarray<int, 1>(num_bands__);
        for (int i = 0; i < num_bands__; i++) res_idx[i] = i;
        res_idx.allocate_on_device();
        res_idx.copy_to_device();

        eval = mdarray<double, 1>(&eval__[0], num_bands__);
        eval.allocate_on_device();
        eval.copy_to_device();

        h_diag = mdarray<double, 1>(&h_diag__[0], kp__->num_gkvec_row());
        h_diag.allocate_on_device();
        h_diag.copy_to_device();

        o_diag = mdarray<double, 1>(&o_diag__[0], kp__->num_gkvec_row());
        o_diag.allocate_on_device();
        o_diag.copy_to_device();

        res_norm.allocate_on_device();
        p_norm.allocate_on_device();
    }
    #endif

    /* compute residuals norm and apply preconditioner */
    if (pu == CPU)
    {
        if (kp__->gkvec().reduced())
        {
            #pragma omp parallel for
            for (int i = 0; i < num_bands__; i++)
            {
                res_norm[i] = 0;
                p_norm[i] = 0;
                for (int ig = 0; ig < res__.num_gvec_loc(); ig++) 
                {
                    /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
                    res__(ig, i) = hpsi__(ig, i) - eval__[i] * opsi__(ig, i);
                    res_norm__[i] += 2 * (std::pow(res__(ig, i).real(), 2) + std::pow(res__(ig, i).imag(), 2));
                }
                if (kp__->comm().rank() == 0)
                    res_norm__[i] -= std::pow(res__(0, i).real(), 2);

                for (int ig = 0; ig < res__.num_gvec_loc(); ig++) 
                {
                    double p = h_diag__[ig] - eval__[i] * o_diag__[ig];
                    p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                    res__(ig, i) /= p;
                    /* norm of the preconditioned residual */
                    p_norm[i] += 2 * (std::pow(res__(ig, i).real(), 2) + std::pow(res__(ig, i).imag(), 2));
                }

                if (kp__->comm().rank() == 0)
                    p_norm[i] -= std::pow(res__(0, i).real(), 2);
            }
        }
        else
        {
            /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm squared */
            #pragma omp parallel for
            for (int i = 0; i < num_bands__; i++)
            {
                res_norm__[i] = 0;
                p_norm[i] = 0;
                for (int ig = 0; ig < res__.num_gvec_loc(); ig++) 
                {
                    /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
                    res__(ig, i) = hpsi__(ig, i) - eval__[i] * opsi__(ig, i);
                    /* norm of the original (not preconditioned) residual */
                    res_norm__[i] += (std::pow(res__(ig, i).real(), 2) + std::pow(res__(ig, i).imag(), 2));
                    /* apply preconditioner */
                    double p = h_diag__[ig] - eval__[i] * o_diag__[ig];
                    p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                    res__(ig, i) /= p;
                    /* norm of the preconditioned residual */
                    p_norm[i] += (std::pow(res__(ig, i).real(), 2) + std::pow(res__(ig, i).imag(), 2));
                }
            }
        }
    }
    #ifdef __GPU
    if (pu == GPU)
    {
        residuals_aux_gpu(res__.num_gvec_loc(), num_bands__, res_idx.at<GPU>(), eval.at<GPU>(),
                          hpsi__.coeffs().at<GPU>(), opsi__.coeffs().at<GPU>(),
                          h_diag.at<GPU>(), o_diag.at<GPU>(), res__.coeffs().at<GPU>(),
                          res_norm.at<GPU>(), p_norm.at<GPU>());
        res_norm.copy_to_host();
        p_norm.copy_to_host();
    }
    #endif

    kp__->comm().allreduce(res_norm__);
    kp__->comm().allreduce(&p_norm[0], num_bands__);

    for (int i = 0; i < num_bands__; i++) p_norm[i] = 1.0 / std::sqrt(p_norm[i]);

    /* normalize preconditioned residuals */
    if (pu == CPU)
    {
        #pragma omp parallel for
        for (int i = 0; i < num_bands__; i++)
        {
            for (int ig = 0; ig < res__.num_gvec_loc(); ig++) res__(ig, i) *= p_norm[i];
        }
    }
    #ifdef __GPU
    if (pu == GPU)
    {
        p_norm.copy_to_device();
        scale_matrix_columns_gpu(res__.num_gvec_loc(), num_bands__, res__.coeffs().at<GPU>(), p_norm.at<GPU>());
    }
    #endif
}

};
