#include "band.h"

namespace sirius {

template <typename T>
int Band::residuals(K_point* kp__,
                    int ispn__,
                    int N__,
                    int num_bands__,
                    std::vector<double>& eval__,
                    std::vector<double>& eval_old__,
                    matrix<T>& evec__,
                    Wave_functions<false>& hphi__,
                    Wave_functions<false>& ophi__,
                    Wave_functions<false>& hpsi__,
                    Wave_functions<false>& opsi__,
                    Wave_functions<false>& res__,
                    std::vector<double>& h_diag__,
                    std::vector<double>& o_diag__) const
{
    PROFILE_WITH_TIMER("sirius::Band::residuals");

    auto& itso = ctx_.iterative_solver_input_section();
    bool converge_by_energy = (itso.converge_by_energy_ == 1);

    /* norm of residuals */
    std::vector<double> res_norm(num_bands__);

    int n = 0;
    if (converge_by_energy)
    {
        std::vector<double> eval_tmp(num_bands__);

        /* main trick here: first estimate energy difference, and only then compute unconverged residuals */
        for (int i = 0; i < num_bands__; i++)
        {
            bool take_res = true;
            if (itso.converge_occupied_ && kp__->band_occupancy(i + ispn__ * ctx_.num_fv_states()) < 1e-10) take_res = false;

            if (take_res && std::abs(eval__[i] - eval_old__[i]) > itso.energy_tolerance_)
            {
                std::memcpy(&evec__(0, num_bands__ + n), &evec__(0, i), N__ * sizeof(T));
                eval_tmp[n++] = eval__[i];
            }
        }
        // TODO: do this on GPU

        /* create alias for eigen-vectors corresponding to unconverged residuals */
        matrix<T> evec_tmp;
        if (ctx_.processing_unit() == CPU)
        {
            evec_tmp = matrix<T>(&evec__(0, num_bands__), evec__.ld(), n);
        }
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU)
        {
            evec_tmp = matrix<T>(evec__.template at<CPU>(0, num_bands__), evec__.template at<GPU>(0, num_bands__), evec__.ld(), n);
            /* move matrix of eigen-vectors to GPU */
            acc::copyin(evec_tmp.template at<GPU>(), evec_tmp.ld(), evec_tmp.template at<CPU>(), evec_tmp.ld(), N__, n);
        }
        #endif

        /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} */
        hpsi__.transform_from<T>(hphi__, N__, evec_tmp, n);
        /* compute O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
        opsi__.transform_from<T>(ophi__, N__, evec_tmp, n);

        residuals_aux(kp__, n, eval_tmp, hpsi__, opsi__, res__, h_diag__, o_diag__, res_norm);

        int nmax = n;
        n = 0;
        for (int i = 0; i < nmax; i++)
        {
            /* take the residual if it's norm is above the threshold */
            if (res_norm[i] > itso.residual_tolerance_)
            {
                /* shift unconverged residuals to the beginning of array */
                if (n != i)
                {
                    switch (ctx_.processing_unit())
                    {
                        case CPU:
                        {
                            std::memcpy(&res__(0, n), &res__(0, i), res__.num_gvec_loc() * sizeof(double_complex));
                            break;
                        }
                        case GPU:
                        {
                            #ifdef __GPU
                            acc::copy(res__.coeffs().at<GPU>(0, n), res__.coeffs().at<GPU>(0, i), res__.num_gvec_loc());
                            #else
                            TERMINATE_NO_GPU
                            #endif
                            break;
                        }
                    }
                }
                n++;
            }
        }
        #if (__VERBOSITY > 2)
        if (kp__->comm().rank() == 0)
        {
            DUMP("initial and final number of residuals : %i %i", nmax, n);
        }
        #endif
    }
    else
    {
        /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} */
        hpsi__.transform_from<T>(hphi__, N__, evec__, num_bands__);
        /* compute O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
        opsi__.transform_from<T>(ophi__, N__, evec__, num_bands__);

        residuals_aux(kp__, num_bands__, eval__, hpsi__, opsi__, res__, h_diag__, o_diag__, res_norm);

        for (int i = 0; i < num_bands__; i++)
        {
            bool take_res = true;
            if (itso.converge_occupied_ && kp__->band_occupancy(i + ispn__ * ctx_.num_fv_states()) < 1e-10) take_res = false;

            /* take the residual if it's norm is above the threshold */
            if (take_res && res_norm[i] > itso.residual_tolerance_)
            {
                /* shift unconverged residuals to the beginning of array */
                if (n != i)
                {
                    switch (ctx_.processing_unit())
                    {
                        case CPU:
                        {
                            std::memcpy(&res__(0, n), &res__(0, i), res__.num_gvec_loc() * sizeof(double_complex));
                            break;
                        }
                        case GPU:
                        {
                            #ifdef __GPU
                            acc::copy(res__.coeffs().at<GPU>(0, n), res__.coeffs().at<GPU>(0, i), res__.num_gvec_loc());
                            #else
                            TERMINATE_NO_GPU
                            #endif
                            break;
                        }
                    }
                }
                n++;
            }
        }
    }
    
    //== if (!kp__->gkvec().reduced())
    //== {
    //==     // --== DEBUG ==--
    //==     for (int i = 0; i < n; i++)
    //==     {
    //==         for (int igk = 0; igk < kp__->num_gkvec(); igk++)
    //==         {
    //==             auto G = kp__->gkvec()[igk] * (-1);
    //==             int igk1 = kp__->gkvec().index_by_gvec(G);
    //==             auto z1 = res__(igk, i);
    //==             auto z2 = res__(igk1, i);
    //==             res__(igk, i) = 0.5 * (z1 + std::conj(z2));
    //==             res__(igk1, i) = std::conj(res__(igk, i));
    //==         }
    //==     }
    //== }

    //// --== DEBUG ==--
    //if (kp__->gkvec().reduced())
    //{
    //    matrix<double> ovlp(n, n);
    //    res__.inner<double>(0, n, res__, 0, n, ovlp, 0, 0);

    //    Utils::write_matrix("ovlp_res_real.txt", true, ovlp);
    //}
    //else
    //{
    //    matrix<double_complex> ovlp(n, n);
    //    res__.inner<double_complex>(0, n, res__, 0, n, ovlp, 0, 0);
    //    Utils::write_matrix("ovlp_res_cmplx.txt", true, ovlp);
    //}

    return n;
}

template int Band::residuals<double_complex>(K_point* kp__,
                                             int ispn__,
                                             int N__,
                                             int num_bands__,
                                             std::vector<double>& eval__,
                                             std::vector<double>& eval_old__,
                                             matrix<double_complex>& evec__,
                                             Wave_functions<false>& hphi__,
                                             Wave_functions<false>& ophi__,
                                             Wave_functions<false>& hpsi__,
                                             Wave_functions<false>& opsi__,
                                             Wave_functions<false>& res__,
                                             std::vector<double>& h_diag__,
                                             std::vector<double>& o_diag__) const;

template int Band::residuals<double>(K_point* kp__,
                                     int ispn__,
                                     int N__,
                                     int num_bands__,
                                     std::vector<double>& eval__,
                                     std::vector<double>& eval_old__,
                                     matrix<double>& evec__,
                                     Wave_functions<false>& hphi__,
                                     Wave_functions<false>& ophi__,
                                     Wave_functions<false>& hpsi__,
                                     Wave_functions<false>& opsi__,
                                     Wave_functions<false>& res__,
                                     std::vector<double>& h_diag__,
                                     std::vector<double>& o_diag__) const;

};
