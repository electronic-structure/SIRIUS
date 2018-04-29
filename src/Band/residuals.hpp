#ifdef __GPU
extern "C" void residuals_aux_gpu(int num_gvec_loc__,
                                  int num_res_local__,
                                  int* res_idx__,
                                  double* eval__,
                                  double_complex const* hpsi__,
                                  double_complex const* opsi__,
                                  double const* h_diag__,
                                  double const* o_diag__,
                                  double_complex* res__,
                                  double* res_norm__,
                                  double* p_norm__,
                                  int gkvec_reduced__,
                                  int mpi_rank__);

extern "C" void compute_residuals_gpu(double_complex* hpsi__,
                                      double_complex* opsi__,
                                      double_complex* res__,
                                      int num_gvec_loc__,
                                      int num_bands__,
                                      double* eval__);

extern "C" void apply_preconditioner_gpu(double_complex* res__,
                                         int num_rows_loc__,
                                         int num_bands__,
                                         double* eval__,
                                         double* h_diag__,
                                         double* o_diag__);

extern "C" void make_real_g0_gpu(double_complex* res__,
                                 int ld__,
                                 int n__);
#endif

/// Compute r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
static void compute_res(device_t            pu__,
                        int                 ispn__,
                        int                 num_bands__,
                        mdarray<double, 1>& eval__,
                        Wave_functions&     hpsi__,
                        Wave_functions&     opsi__,
                        Wave_functions&     res__)
{
    int s0 = (ispn__ == 2) ? 0 : ispn__;
    int s1 = (ispn__ == 2) ? 1 : ispn__;

    for (int ispn = s0; ispn <= s1; ispn++) {
        switch (pu__) {
            case CPU: {
                /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
                #pragma omp parallel for
                for (int i = 0; i < num_bands__; i++) {
                    for (int ig = 0; ig < res__.pw_coeffs(ispn).num_rows_loc(); ig++) {
                        res__.pw_coeffs(ispn).prime(ig, i) = hpsi__.pw_coeffs(ispn).prime(ig, i) -
                            eval__[i] * opsi__.pw_coeffs(ispn).prime(ig, i);
                    }
                    if (res__.has_mt()) {
                        for (int j = 0; j < res__.mt_coeffs(ispn).num_rows_loc(); j++) {
                            res__.mt_coeffs(ispn).prime(j, i) = hpsi__.mt_coeffs(ispn).prime(j, i) -
                                eval__[i] * opsi__.mt_coeffs(ispn).prime(j, i);
                        }
                    }
                }
                break;
            }
            case GPU: {
                #ifdef __GPU
                compute_residuals_gpu(hpsi__.pw_coeffs(ispn).prime().at<GPU>(),
                                      opsi__.pw_coeffs(ispn).prime().at<GPU>(),
                                      res__.pw_coeffs(ispn).prime().at<GPU>(),
                                      res__.pw_coeffs(ispn).num_rows_loc(),
                                      num_bands__,
                                      eval__.at<GPU>());
                if (res__.has_mt()) {
                    compute_residuals_gpu(hpsi__.mt_coeffs(ispn).prime().at<GPU>(),
                                          opsi__.mt_coeffs(ispn).prime().at<GPU>(),
                                          res__.mt_coeffs(ispn).prime().at<GPU>(),
                                          res__.mt_coeffs(ispn).num_rows_loc(),
                                          num_bands__,
                                          eval__.at<GPU>());
                }
                #endif
            }
        }
    }
}

/// Apply preconditioner to the residuals.
static void apply_p(device_t            pu__,
                    int                 ispn__,
                    int                 num_bands__,
                    Wave_functions&     res__,
                    mdarray<double, 2>& h_diag__,
                    mdarray<double, 1>& o_diag__,
                    mdarray<double, 1>& eval__)
{
    int s0 = (ispn__ == 2) ? 0 : ispn__;
    int s1 = (ispn__ == 2) ? 1 : ispn__;

    for (int ispn = s0; ispn <= s1; ispn++) {
        switch (pu__) {
            case CPU: {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < num_bands__; i++) {
                    for (int ig = 0; ig < res__.pw_coeffs(ispn).num_rows_loc(); ig++) {
                        double p = h_diag__(ig, ispn) - o_diag__[ig] * eval__[i];
                        p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                        res__.pw_coeffs(ispn).prime(ig, i) /= p;
                    }
                    if (res__.has_mt()) {
                        for (int j = 0; j < res__.mt_coeffs(ispn).num_rows_loc(); j++) {
                            double p = h_diag__(res__.pw_coeffs(ispn).num_rows_loc() + j, ispn) - 
                                       o_diag__[res__.pw_coeffs(ispn).num_rows_loc() + j] * eval__[i];
                            p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                            res__.mt_coeffs(ispn).prime(j, i) /= p;
                        }
                    }
                }
                break;
            }
            case GPU: {
                #ifdef __GPU
                apply_preconditioner_gpu(res__.pw_coeffs(ispn).prime().at<GPU>(),
                                         res__.pw_coeffs(ispn).num_rows_loc(),
                                         num_bands__,
                                         eval__.at<GPU>(),
                                         h_diag__.at<GPU>(0, ispn),
                                         o_diag__.at<GPU>());
                if (res__.has_mt()) {
                    apply_preconditioner_gpu(res__.mt_coeffs(ispn).prime().at<GPU>(),
                                             res__.mt_coeffs(ispn).num_rows_loc(),
                                             num_bands__,
                                             eval__.at<GPU>(),
                                             h_diag__.at<GPU>(res__.pw_coeffs(ispn).num_rows_loc(), ispn),
                                             o_diag__.at<GPU>(res__.pw_coeffs(ispn).num_rows_loc()));
                }
                break;
                #endif
            }
        }
    }
}

/// Normalize residuals.
/** This not strictly necessary as the wave-function orthoronormalization can take care of this.
 *  However, normalization of residuals is harmless and gives a better numerical stability. */
static void normalize_res(device_t            pu__,
                          int                 ispn__,
                          int                 num_bands__,
                          Wave_functions&     res__,
                          mdarray<double, 1>& p_norm__)
{
    int s0 = (ispn__ == 2) ? 0 : ispn__;
    int s1 = (ispn__ == 2) ? 1 : ispn__;

    for (int ispn = s0; ispn <= s1; ispn++) {
        switch (pu__) {
            case CPU: {
            #pragma omp parallel for schedule(static)
                for (int i = 0; i < num_bands__; i++) {
                    for (int ig = 0; ig < res__.pw_coeffs(ispn).num_rows_loc(); ig++) {
                        res__.pw_coeffs(ispn).prime(ig, i) *= p_norm__[i];
                    }
                    if (res__.has_mt()) {
                        for (int j = 0; j < res__.mt_coeffs(ispn).num_rows_loc(); j++) {
                            res__.mt_coeffs(ispn).prime(j, i) *= p_norm__[i];
                        }
                    }
                }
                break;
            }
            case GPU: {
                #ifdef __GPU
                scale_matrix_columns_gpu(res__.pw_coeffs(ispn).num_rows_loc(),
                                         num_bands__,
                                         res__.pw_coeffs(ispn).prime().at<GPU>(),
                                         p_norm__.at<GPU>());

                if (res__.has_mt()) {
                    scale_matrix_columns_gpu(res__.mt_coeffs(ispn).num_rows_loc(),
                                             num_bands__,
                                             res__.mt_coeffs(ispn).prime().at<GPU>(),
                                             p_norm__.at<GPU>());
                }
                #endif
                break;
            }
        }
    }
}

inline mdarray<double, 1>
Band::residuals_aux(K_point*             kp__,
                    int                  ispn__,
                    int                  num_bands__,
                    std::vector<double>& eval__,
                    Wave_functions&      hpsi__,
                    Wave_functions&      opsi__,
                    Wave_functions&      res__,
                    mdarray<double, 2>&  h_diag__,
                    mdarray<double, 1>&  o_diag__) const
{
    PROFILE("sirius::Band::residuals_aux");

    assert(num_bands__ != 0);

    auto pu = ctx_.processing_unit();

    mdarray<double, 1> eval(eval__.data(), num_bands__, "residuals_aux::eval");
    if (pu == GPU) {
        eval.allocate(memory_t::device);
        eval.copy<memory_t::host, memory_t::device>();
    }

    /* compute residuals */
    compute_res(pu, ispn__, num_bands__, eval, hpsi__, opsi__, res__);

    /* compute norm */
    auto res_norm = res__.l2norm(pu, ispn__, num_bands__);

    apply_p(pu, ispn__, num_bands__, res__, h_diag__, o_diag__, eval);

    auto p_norm = res__.l2norm(pu, ispn__, num_bands__);
    for (int i = 0; i < num_bands__; i++) {
        p_norm[i] = 1.0 / p_norm[i];
    }
    if (pu == GPU) {
        p_norm.copy<memory_t::host, memory_t::device>();
    }

    /* normalize preconditioned residuals */
    normalize_res(pu, ispn__, num_bands__, res__, p_norm);
    
    if (ctx_.control().verbosity_ >= 5) {
        auto n_norm = res__.l2norm(pu, ispn__, num_bands__);
        if (kp__->comm().rank() == 0) {
            for (int i = 0; i < num_bands__; i++) {
                printf("norms of residual %3i: %18.14f %24.14f %18.14f", i, res_norm[i], p_norm[i], n_norm[i]);
                if (res_norm[i] > ctx_.iterative_solver_input().residual_tolerance_) {
                    printf(" +");
                }
                printf("\n");
            }
        }
    }

   return std::move(res_norm);
}

template <typename T>
inline int Band::residuals(K_point*             kp__,
                           int                  ispn__,
                           int                  N__,
                           int                  num_bands__,
                           std::vector<double>& eval__,
                           std::vector<double>& eval_old__,
                           dmatrix<T>&          evec__,
                           Wave_functions&      hphi__,
                           Wave_functions&      ophi__,
                           Wave_functions&      hpsi__,
                           Wave_functions&      opsi__,
                           Wave_functions&      res__,
                           mdarray<double, 2>&  h_diag__,
                           mdarray<double, 1>&  o_diag__) const
{
    PROFILE("sirius::Band::residuals");

    assert(N__ != 0);

    auto& itso = ctx_.iterative_solver_input();
    bool converge_by_energy = (itso.converge_by_energy_ == 1);

    int n{0};
    if (converge_by_energy) {

        /* main trick here: first estimate energy difference, and only then compute unconverged residuals */
        auto get_ev_idx = [&](double tol__)
        {
            std::vector<int> ev_idx;
            int s = ispn__ == 2 ? 0 : ispn__;
            for (int i = 0; i < num_bands__; i++) {
                double o1 = std::abs(kp__->band_occupancy(i, s) / ctx_.max_occupancy());
                double o2 = std::abs(1 - o1);

                double tol = o1 * tol__ + o2 * (tol__ + itso.empty_states_tolerance_);
                if (std::abs(eval__[i] - eval_old__[i]) > tol) {
                    ev_idx.push_back(i);
                }
            }
            return std::move(ev_idx);
        };

        auto ev_idx = get_ev_idx(itso.energy_tolerance_);

        n = static_cast<int>(ev_idx.size());

        if (n) {
            std::vector<double> eval_tmp(n);

            int bs = ctx_.cyclic_block_size();
            dmatrix<T> evec_tmp(N__, n, ctx_.blacs_grid(), bs, bs);
            int num_rows_local = evec_tmp.num_rows_local();
            for (int j = 0; j < n; j++) {
                eval_tmp[j] = eval__[ev_idx[j]];
                if (ctx_.blacs_grid().comm().size() == 1) {
                    /* do a local copy */
                    std::copy(&evec__(0, ev_idx[j]), &evec__(0, ev_idx[j]) + num_rows_local, &evec_tmp(0, j));
                } else {
                    auto pos_src  = evec__.spl_col().location(ev_idx[j]);
                    auto pos_dest = evec_tmp.spl_col().location(j);
                    /* do MPI send / recieve */
                    if (pos_src.rank == kp__->comm_col().rank()) {
                        kp__->comm_col().isend(&evec__(0, pos_src.local_index), num_rows_local, pos_dest.rank, ev_idx[j]);
                    }
                    if (pos_dest.rank == kp__->comm_col().rank()) {
                       kp__->comm_col().recv(&evec_tmp(0, pos_dest.local_index), num_rows_local, pos_src.rank, ev_idx[j]);
                    }
                }
            }
            if (ctx_.processing_unit() == GPU && evec__.blacs_grid().comm().size() == 1) {
                evec_tmp.allocate(memory_t::device);
            }
            /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} and O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
            transform<T>(ctx_.processing_unit(), ispn__, {&hphi__, &ophi__}, 0, N__, evec_tmp, 0, 0, {&hpsi__, &opsi__}, 0, n);

            auto res_norm = residuals_aux(kp__, ispn__, n, eval_tmp, hpsi__, opsi__, res__, h_diag__, o_diag__);

            int nmax = n;
            n = 0;
            for (int i = 0; i < nmax; i++) {
                /* take the residual if it's norm is above the threshold */
                if (res_norm[i] > itso.residual_tolerance_) {
                    /* shift unconverged residuals to the beginning of array */
                    if (n != i) {
                        int s0{0}, s1{1};
                        if (ispn__ != 2) {
                            s0 = s1 = ispn__;
                        }
                        for (int ispn = s0; ispn <= s1; ispn++) {
                            res__.copy_from(ctx_.processing_unit(), 1, res__, ispn, i, ispn, n);
                        }
                    }
                    n++;
                }
            }
            if (ctx_.control().verbosity_ >= 3 && kp__->comm().rank() == 0) {
                printf("initial and final number of residuals : %i %i\n", nmax, n);
            }
        }
    } else {
        /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} and O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
        transform<T>(ctx_.processing_unit(), ispn__, {&hphi__, &ophi__}, 0, N__, evec__, 0, 0, {&hpsi__, &opsi__}, 0, num_bands__);

        auto res_norm = residuals_aux(kp__, ispn__, num_bands__, eval__, hpsi__, opsi__, res__, h_diag__, o_diag__);

        for (int i = 0; i < num_bands__; i++) {
            //int s = ispn__ == 2 ? 0 : ispn__;
            double tol = itso.residual_tolerance_;// + 1e-3 * std::abs(kp__->band_occupancy(i + s * ctx_.num_fv_states()) / ctx_.max_occupancy() - 1);
            /* take the residual if its norm is above the threshold */
            if (res_norm[i] > tol) {
                /* shift unconverged residuals to the beginning of array */
                if (n != i) {
                    int s0{0}, s1{1};
                    if (ispn__ != 2) {
                        s0 = s1 = ispn__;
                    }
                    for (int ispn = s0; ispn <= s1; ispn++) {
                        res__.copy_from(ctx_.processing_unit(), 1, res__, ispn, i, ispn, n);
                    }
                }
                n++;
            }
        }
        if (ctx_.control().verbosity_ >= 3 && kp__->comm().rank() == 0) {
            printf("number of residuals : %i\n", n);
        }
    }
                                        
    /* prevent numerical noise */
    /* this only happens for real wave-functions (Gamma-point case), non-magnetic or collinear magnetic */
    if (std::is_same<T, double>::value && kp__->comm().rank() == 0 && n != 0) {
        assert(ispn__ == 0 || ispn__ == 1);
        switch (ctx_.processing_unit()) {
            case CPU: {
                for (int i = 0; i < n; i++) {
                    res__.pw_coeffs(ispn__).prime(0, i) = res__.pw_coeffs(ispn__).prime(0, i).real();
                }
                break;
            }
            case GPU: {
#ifdef __GPU
                make_real_g0_gpu(res__.pw_coeffs(ispn__).prime().at<GPU>(), res__.pw_coeffs(ispn__).prime().ld(), n);
#endif
                break;
            }
        }
    }
    
    /* print checksums */
    if (ctx_.control().print_checksum_ && n != 0) {
        int s0 = (ispn__ == 2) ? 0 : ispn__;
        int s1 = (ispn__ == 2) ? 1 : ispn__;
        for (int ispn = s0; ispn <= s1; ispn++) {
            auto cs = res__.checksum(ctx_.processing_unit(), ispn, 0, n);
            auto cs1 = hpsi__.checksum(ctx_.processing_unit(), ispn, 0, n);
            auto cs2 = opsi__.checksum(ctx_.processing_unit(), ispn, 0, n);
            if (kp__->comm().rank() == 0) {
                std::stringstream s;
                s << "res_" << ispn;
                print_checksum(s.str(), cs);
                s.str("");
                s << "hpsi_" << ispn;
                print_checksum(s.str(), cs1);
                s.str("");
                s << "opsi_" << ispn;
                print_checksum(s.str(), cs2);
            }
        }
    }

    return n;
}

template <typename T>
void Band::check_residuals(K_point* kp__, Hamiltonian& H__) const
{
    if (kp__->comm().rank() == 0) {
        printf("checking residuals\n");
    }

    const bool nc_mag = (ctx_.num_mag_dims() == 3);
    const int num_sc = nc_mag ? 2 : 1;

    auto& psi = kp__->spinor_wave_functions();
    Wave_functions hpsi(kp__->gkvec_partition(), ctx_.num_bands(), num_sc);
    Wave_functions spsi(kp__->gkvec_partition(), ctx_.num_bands(), num_sc);
    Wave_functions res(kp__->gkvec_partition(), ctx_.num_bands(), num_sc);

    /* compute residuals */
    for (int ispin_step = 0; ispin_step < ctx_.num_spin_dims(); ispin_step++) {
        if (nc_mag) {
            /* apply Hamiltonian and S operators to the wave-functions */
            H__.apply_h_s<T>(kp__, 2, 0, ctx_.num_bands(), psi, hpsi, spsi);
        } else {
            Wave_functions phi(&psi.pw_coeffs(ispin_step).prime(0, 0), kp__->gkvec_partition(), ctx_.num_bands(), 1);
            /* apply Hamiltonian and S operators to the wave-functions */
            H__.apply_h_s<T>(kp__, ispin_step, 0, ctx_.num_bands(), phi, hpsi, spsi);
        }
        
        for (int ispn = 0; ispn < num_sc; ispn++) {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < ctx_.num_bands(); j++) {
                for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++) {
                    res.pw_coeffs(ispn).prime(ig, j) = hpsi.pw_coeffs(ispn).prime(ig, j) -
                                                       spsi.pw_coeffs(ispn).prime(ig, j) * 
                                                       kp__->band_energy(j, ispin_step);
                }
            }
        }
        /* get the norm */
        auto l2norm = res.l2norm(ctx_.processing_unit(), nc_mag ? 2 : 0, ctx_.num_bands());

        if (kp__->comm().rank() == 0) {
            for (int j = 0; j < ctx_.num_bands(); j++) {
                printf("band: %3i, residual l2norm: %18.12f\n", j, l2norm[j]);
            }
        }
    }
}
