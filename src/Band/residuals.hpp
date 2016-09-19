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
                                  double* p_norm__,
                                  int gkvec_reduced__,
                                  int mpi_rank__);
#endif

inline mdarray<double, 1>
Band::residuals_aux(K_point* kp__,
                    int num_bands__,
                    std::vector<double>& eval__,
                    wave_functions& hpsi__,
                    wave_functions& opsi__,
                    wave_functions& res__,
                    std::vector<double>& h_diag__,
                    std::vector<double>& o_diag__) const
{
    PROFILE_WITH_TIMER("sirius::Band::residuals_aux");

    assert(kp__->num_gkvec_loc() == res__.pw_coeffs().num_rows_loc());
    assert(kp__->num_gkvec_loc() == hpsi__.pw_coeffs().num_rows_loc());
    assert(kp__->num_gkvec_loc() == opsi__.pw_coeffs().num_rows_loc());

    auto pu = ctx_.processing_unit();

    #ifdef __GPU
    mdarray<int, 1> res_idx;
    mdarray<double, 1> eval;
    mdarray<double, 1> h_diag;
    mdarray<double, 1> o_diag;
    if (pu == GPU) {
        STOP();
        ///* global index of residual */
        //res_idx = mdarray<int, 1>(num_bands__);
        //for (int i = 0; i < num_bands__; i++) res_idx[i] = i;
        //res_idx.allocate(memory_t::device);
        //res_idx.copy_to_device();

        //eval = mdarray<double, 1>(&eval__[0], num_bands__);
        //eval.allocate(memory_t::device);
        //eval.copy_to_device();

        //h_diag = mdarray<double, 1>(&h_diag__[0], kp__->num_gkvec_row());
        //h_diag.allocate(memory_t::device);
        //h_diag.copy_to_device();

        //o_diag = mdarray<double, 1>(&o_diag__[0], kp__->num_gkvec_row());
        //o_diag.allocate(memory_t::device);
        //o_diag.copy_to_device();

        //res_norm.allocate(memory_t::device);
        //p_norm.allocate(memory_t::device);
    }
    #endif

    /* compute residuals norm and apply preconditioner */
    if (pu == CPU) {
        /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
        #pragma omp parallel for
        for (int i = 0; i < num_bands__; i++) {
            for (int ig = 0; ig < res__.pw_coeffs().num_rows_loc(); ig++) {
                res__.pw_coeffs().prime(ig, i) = hpsi__.pw_coeffs().prime(ig, i) - eval__[i] * opsi__.pw_coeffs().prime(ig, i);
            }
            if (ctx_.full_potential()) {
                for (int j = 0; j < res__.mt_coeffs().num_rows_loc(); j++) {
                    res__.mt_coeffs().prime(j, i) = hpsi__.mt_coeffs().prime(j, i) - eval__[i] * opsi__.mt_coeffs().prime(j, i);
                }
            }
        }
    }
    /* compute norm */
    auto res_norm = res__.l2norm(num_bands__);

    if (pu == CPU) {
        /* apply preconditioner */
        #pragma omp parallel for
        for (int i = 0; i < num_bands__; i++) {
            /* apply preconditioner */
            for (int ig = 0; ig < res__.pw_coeffs().num_rows_loc(); ig++) {
                double p = h_diag__[ig] - eval__[i] * o_diag__[ig];
                p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                res__.pw_coeffs().prime(ig, i) /= p;
            }
            if (ctx_.full_potential()) {
                for (int j = 0; j < res__.mt_coeffs().num_rows_loc(); j++) {
                    double p = h_diag__[kp__->num_gkvec_loc() + j] - eval__[i] * o_diag__[kp__->num_gkvec_loc() + j];
                    p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                    res__.mt_coeffs().prime(j, i) /= p;
                }
            }
        }
        
        //== #pragma omp parallel for
        //== for (int i = 0; i < num_bands__; i++) {
        //==     res_norm[i] = 0;
        //==     p_norm[i] = 0;
        //==     for (int ig = 0; ig < res__.pw_coeffs().num_rows_loc(); ig++) {
        //==         /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
        //==         res__.pw_coeffs().prime(ig, i) = hpsi__.pw_coeffs().prime(ig, i) - eval__[i] * opsi__.pw_coeffs().prime(ig, i);
        //==         res_norm__[i] += (std::pow(res__.pw_coeffs().prime(ig, i).real(), 2) + std::pow(res__.pw_coeffs().prime(ig, i).imag(), 2));
        //==     }
        //==     if (kp__->gkvec().reduced()) {
        //==         if (kp__->comm().rank() == 0) {
        //==             res_norm__[i] = 2 * res_norm__[i] - std::pow(res__.pw_coeffs().prime(0, i).real(), 2);
        //==         } else {
        //==             res_norm__[i] *= 2;
        //==         }
        //==     }
        //==     /* apply preconditioner */
        //==     for (int ig = 0; ig < res__.pw_coeffs().num_rows_loc(); ig++) {
        //==         double p = h_diag__[ig] - eval__[i] * o_diag__[ig];
        //==         p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
        //==         res__.pw_coeffs().prime(ig, i) /= p;
        //==         /* norm of the preconditioned residual */
        //==         p_norm[i] += (std::pow(res__.pw_coeffs().prime(ig, i).real(), 2) + std::pow(res__.pw_coeffs().prime(ig, i).imag(), 2));
        //==     }
        //==     if (kp__->gkvec().reduced()) {
        //==         if (kp__->comm().rank() == 0) {
        //==             p_norm[i] = 2 * p_norm[i] - std::pow(res__.pw_coeffs().prime(0, i).real(), 2);
        //==         } else {
        //==             p_norm[i] *= 2;
        //==         }
        //==     }
        //== }
    }
    #ifdef __GPU
    if (pu == GPU)
    {
        residuals_aux_gpu(res__.num_rows_loc(), num_bands__, res_idx.at<GPU>(), eval.at<GPU>(),
                          hpsi__.coeffs().at<GPU>(), opsi__.coeffs().at<GPU>(),
                          h_diag.at<GPU>(), o_diag.at<GPU>(), res__.coeffs().at<GPU>(),
                          res_norm.at<GPU>(), p_norm.at<GPU>(), kp__->gkvec().reduced(), kp__->comm().rank());
        res_norm.copy_to_host();
        p_norm.copy_to_host();
    }
    #endif

    //kp__->comm().allreduce(res_norm__);
    //kp__->comm().allreduce(&p_norm[0], num_bands__);

    //for (int i = 0; i < num_bands__; i++) p_norm[i] = 1.0 / std::sqrt(p_norm[i]);
    
    auto p_norm = res__.l2norm(num_bands__);
    /* normalize preconditioned residuals */
    if (pu == CPU) {
        #pragma omp parallel for
        for (int i = 0; i < num_bands__; i++) {
            double a = 1.0 / p_norm[i];
            for (int ig = 0; ig < res__.pw_coeffs().num_rows_loc(); ig++) {
                res__.pw_coeffs().prime(ig, i) *= a;
            }
            if (ctx_.full_potential()) {
                for (int j = 0; j < res__.mt_coeffs().num_rows_loc(); j++) {
                    res__.mt_coeffs().prime(j, i) *= a;
                }
            }
        }
    }
    #ifdef __GPU
    if (pu == GPU) {
        p_norm.copy_to_device();
        scale_matrix_columns_gpu(res__.num_gvec_loc(), num_bands__, res__.coeffs().at<GPU>(), p_norm.at<GPU>());
    }
    #endif

    return std::move(res_norm);
}

template <typename T>
inline int Band::residuals(K_point* kp__,
                           int ispn__,
                           int N__,
                           int num_bands__,
                           std::vector<double>& eval__,
                           std::vector<double>& eval_old__,
                           matrix<T>& evec__,
                           dmatrix<T>& evec_dist__,
                           wave_functions& hphi__,
                           wave_functions& ophi__,
                           wave_functions& hpsi__,
                           wave_functions& opsi__,
                           wave_functions& res__,
                           std::vector<double>& h_diag__,
                           std::vector<double>& o_diag__) const
{
    PROFILE_WITH_TIMER("sirius::Band::residuals");

    auto& itso = ctx_.iterative_solver_input_section();
    bool converge_by_energy = (itso.converge_by_energy_ == 1);

    int n{0};
    if (converge_by_energy) {

        std::vector<int> ev_idx;
        for (int i = 0; i < num_bands__; i++) {
            bool take_res = true;
            if (itso.converge_occupied_ && kp__->band_occupancy(i + ispn__ * ctx_.num_fv_states()) < 1e-10) {
                take_res = false;
            }

            if (take_res && std::abs(eval__[i] - eval_old__[i]) > itso.energy_tolerance_) {
                ev_idx.push_back(i);
            }
        }
        n = static_cast<int>(ev_idx.size());
        

        std::vector<double> eval_tmp(n);

        ///* main trick here: first estimate energy difference, and only then compute unconverged residuals */
        //for (int j = 0; j < n; j++) {
        //    std::memcpy(&evec__(0, num_bands__ + j), &evec__(0, ev_idx[j]), N__ * sizeof(T));
        //    eval_tmp[j] = eval__[ev_idx[j]];
        //}
        //// TODO: do this on GPU

        ///* create alias for eigen-vectors corresponding to unconverged residuals */
        //matrix<T> evec_tmp;
        //if (ctx_.processing_unit() == CPU) {
        //    evec_tmp = matrix<T>(&evec__(0, num_bands__), evec__.ld(), n);
        //}
        //#ifdef __GPU
        //if (ctx_.processing_unit() == GPU) {
        //    evec_tmp = matrix<T>(evec__.template at<CPU>(0, num_bands__), evec__.template at<GPU>(0, num_bands__), evec__.ld(), n);
        //    /* copy matrix of eigen-vectors to GPU */
        //    acc::copyin(evec_tmp.template at<GPU>(), evec_tmp.ld(), evec_tmp.template at<CPU>(), evec_tmp.ld(), N__, n);
        //}
        //#endif

        int bs = ctx_.cyclic_block_size();
        dmatrix<T> evec1(N__, n, ctx_.blacs_grid(), bs, bs);
        for (int j = 0; j < n; j++) {
            eval_tmp[j] = eval__[ev_idx[j]];
            auto pos_src = evec_dist__.spl_col().location(ev_idx[j]);
            auto pos_dest = evec1.spl_col().location(j);

            if (pos_src.second == kp__->comm_col().rank()) {
                kp__->comm_col().isend(&evec_dist__(0, pos_src.first), evec1.num_rows_local(), pos_dest.second, ev_idx[j]);
            }
            if (pos_dest.second == kp__->comm_col().rank()) {
               kp__->comm_col().recv(&evec1(0, pos_dest.first), evec1.num_rows_local(), pos_src.second, ev_idx[j]); 
            }
        }

        /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} */
        //hpsi__.transform_from<T>(hphi__, N__, evec_tmp, n);
        /* compute O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
        //opsi__.transform_from<T>(ophi__, N__, evec_tmp, n);
        transform<T>({&hphi__, &ophi__}, 0, N__, evec1, 0, 0, {&hpsi__, &opsi__}, 0, n);

        auto res_norm = residuals_aux(kp__, n, eval_tmp, hpsi__, opsi__, res__, h_diag__, o_diag__);

        int nmax = n;
        n = 0;
        for (int i = 0; i < nmax; i++) {
            /* take the residual if it's norm is above the threshold */
            if (res_norm[i] > itso.residual_tolerance_) {
                /* shift unconverged residuals to the beginning of array */
                if (n != i) {
                    res__.copy_from(res__, i, 1, n);
                }
                n++;
            }
        }
        #if (__VERBOSITY > 2)
        if (kp__->comm().rank() == 0) {
            DUMP("initial and final number of residuals : %i %i", nmax, n);
        }
        #endif
    } else {
        transform<T>({&hphi__, &ophi__}, 0, N__, evec_dist__, 0, 0, {&hpsi__, &opsi__}, 0, num_bands__);
        ///* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} */
        //hpsi__.transform_from<T>(hphi__, N__, evec__, num_bands__);
        ///* compute O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
        //opsi__.transform_from<T>(ophi__, N__, evec__, num_bands__);

        auto res_norm = residuals_aux(kp__, num_bands__, eval__, hpsi__, opsi__, res__, h_diag__, o_diag__);

        for (int i = 0; i < num_bands__; i++) {
            bool take_res = true;
            if (itso.converge_occupied_ && kp__->band_occupancy(i + ispn__ * ctx_.num_fv_states()) < 1e-10) {
                take_res = false;
            }

            /* take the residual if it's norm is above the threshold */
            if (take_res && res_norm[i] > itso.residual_tolerance_) {
                /* shift unconverged residuals to the beginning of array */
                if (n != i) {
                    res__.copy_from(res__, i, 1, n);
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
