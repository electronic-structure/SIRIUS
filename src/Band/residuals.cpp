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
                    std::vector<double>& o_diag__)
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

    return n;
}

template <>
int Band::orthogonalize<double_complex>(K_point* kp__,
                                        int N__,
                                        int n__,
                                        Wave_functions<false>& phi__,
                                        Wave_functions<false>& res__)
{
    return n__;
}

template <>
int Band::orthogonalize<double>(K_point* kp__,
                                int N__,
                                int n__,
                                Wave_functions<false>& phi__,
                                Wave_functions<false>& res__)
{
    //return n__;

    /* { phi_i, i in [0, N) }
       { res_i, i in [0, n) }

       take res_0, orthogonalize to all phi, check norm, accept or rejec
       take res_1, orthogonalize to all phi and res_0, check norm, accept or rejec


    */

    auto inner = [kp__](double_complex* f, double_complex* g)
    {
        double prod = 0;
        for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++)
                prod += 2.0 * (f[igk].real() * g[igk].real() + f[igk].imag() * g[igk].imag());
        prod -= f[0].real() * g[0].real();
        return prod;
    };


    int n = 0;
    /* loop over all available residuals */
    for (int i = 0; i < n__; i++)
    {
        if (i != n) std::memcpy(&res__(0, n), &res__(0, i), kp__->num_gkvec_loc() * sizeof(double_complex));

        /*loop over current basis */
        for (int j = 0; j < N__; j++)
        {
            double prod = inner(&phi__(0, j), &res__(0, n));

            for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++)
                res__(igk, n) -= prod * phi__(igk, j);
        }

        for (int j = 0; j < n; j++)
        {
            double prod = inner(&res__(0, j), &res__(0, n));

            for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++)
                res__(igk, n) -= prod * res__(igk, j);
        }
        double norm = inner(&res__(0, n), &res__(0, n));

        if (std::abs(norm) > 0.01)
        {
            norm = 1.0 / std::sqrt(norm);
            for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++)
                res__(igk, n) *= norm;
            n++;
        }
    }
    
    printf("initial and final number of residuals: %i %i\n", n__, n);

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
                                             std::vector<double>& o_diag__);

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
                                     std::vector<double>& o_diag__);

//template int Band::orthogonalize<double_complex>(K_point* kp__,
//                                                 int N__,
//                                                 int n__,
//                                                 Wave_functions<false>& phi__,
//                                                 Wave_functions<false>& res__);
//
//template int Band::orthogonalize<double>(K_point* kp__,
//                                         int N__,
//                                         int n__,
//                                         Wave_functions<false>& phi__,
//                                         Wave_functions<false>& res__);
};
