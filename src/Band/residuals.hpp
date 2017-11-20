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

static void compute_res(device_t            pu__,
                        int                 num_bands__,
                        mdarray<double, 1>& eval__,
                        wave_functions&     hpsi__,
                        wave_functions&     opsi__,
                        wave_functions&     res__)
{
    switch (pu__) {
        case CPU: {
            /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
            #pragma omp parallel for
            for (int i = 0; i < num_bands__; i++) {
                for (int ig = 0; ig < res__.pw_coeffs().num_rows_loc(); ig++) {
                    res__.pw_coeffs().prime(ig, i) = hpsi__.pw_coeffs().prime(ig, i) - eval__[i] * opsi__.pw_coeffs().prime(ig, i);
                }
                if (res__.has_mt() && res__.mt_coeffs().num_rows_loc()) {
                    for (int j = 0; j < res__.mt_coeffs().num_rows_loc(); j++) {
                        res__.mt_coeffs().prime(j, i) = hpsi__.mt_coeffs().prime(j, i) - eval__[i] * opsi__.mt_coeffs().prime(j, i);
                    }
                }
            }
            break;
        }
        case GPU: {
            #ifdef __GPU
            compute_residuals_gpu(hpsi__.pw_coeffs().prime().at<GPU>(),
                                  opsi__.pw_coeffs().prime().at<GPU>(),
                                  res__.pw_coeffs().prime().at<GPU>(),
                                  res__.pw_coeffs().num_rows_loc(),
                                  num_bands__,
                                  eval__.at<GPU>());
            if (res__.has_mt() && res__.mt_coeffs().num_rows_loc()) {
                compute_residuals_gpu(hpsi__.mt_coeffs().prime().at<GPU>(),
                                      opsi__.mt_coeffs().prime().at<GPU>(),
                                      res__.mt_coeffs().prime().at<GPU>(),
                                      res__.mt_coeffs().num_rows_loc(),
                                      num_bands__,
                                      eval__.at<GPU>());
            }
            #endif
        }
    }
}

/// Apply preconditioner to the residuals.
static void apply_p(device_t            pu__,
                    int                 num_bands__,
                    int                 ispn__,
                    wave_functions&     res__,
                    mdarray<double, 2>& h_diag__,
                    mdarray<double, 1>& o_diag__,
                    mdarray<double, 1>& eval__)
{
    switch (pu__) {
        case CPU: {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_bands__; i++) {
                for (int ig = 0; ig < res__.pw_coeffs().num_rows_loc(); ig++) {
                    double p = h_diag__(ig, ispn__) - o_diag__[ig] * eval__[i];
                    p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                    res__.pw_coeffs().prime(ig, i) /= p;
                }
                if (res__.has_mt()) {
                    for (int j = 0; j < res__.mt_coeffs().num_rows_loc(); j++) {
                        double p = h_diag__(res__.pw_coeffs().num_rows_loc() + j, ispn__) - 
                                   o_diag__[res__.pw_coeffs().num_rows_loc() + j] * eval__[i];
                        p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                        res__.mt_coeffs().prime(j, i) /= p;
                    }
                }
            }
            break;
        }
        case GPU: {
            #ifdef __GPU
            apply_preconditioner_gpu(res__.pw_coeffs().prime().at<GPU>(),
                                     res__.pw_coeffs().num_rows_loc(),
                                     num_bands__,
                                     eval__.at<GPU>(),
                                     h_diag__.at<GPU>(0, ispn__),
                                     o_diag__.at<GPU>());
            if (res__.has_mt() && res__.mt_coeffs().num_rows_loc()) {
                apply_preconditioner_gpu(res__.mt_coeffs().prime().at<GPU>(),
                                         res__.mt_coeffs().num_rows_loc(),
                                         num_bands__,
                                         eval__.at<GPU>(),
                                         h_diag__.at<GPU>(res__.pw_coeffs().num_rows_loc(), ispn__),
                                         o_diag__.at<GPU>(res__.pw_coeffs().num_rows_loc()));
            }
            break;
            #endif
        }
    }

}

/// Normalize residuals.
/** This not strictly necessary as the wave-function orthoronormalization can take care of this.
 *  However, normalization of residuals is harmless and gives a better numerical stability. */
static void normalize_res(device_t            pu__,
                          int                 num_bands__,
                          wave_functions&     res__,
                          mdarray<double, 1>& p_norm__)
{
    switch (pu__) {
        case CPU: {
        #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_bands__; i++) {
                for (int ig = 0; ig < res__.pw_coeffs().num_rows_loc(); ig++) {
                    res__.pw_coeffs().prime(ig, i) *= p_norm__[i];
                }
                if (res__.has_mt() && res__.mt_coeffs().num_rows_loc()) {
                    for (int j = 0; j < res__.mt_coeffs().num_rows_loc(); j++) {
                        res__.mt_coeffs().prime(j, i) *= p_norm__[i];
                    }
                }
            }
            break;
        }
        case GPU: {
            #ifdef __GPU
            scale_matrix_columns_gpu(res__.pw_coeffs().num_rows_loc(),
                                     num_bands__,
                                     res__.pw_coeffs().prime().at<GPU>(),
                                     p_norm__.at<GPU>());

            if (res__.has_mt() && res__.mt_coeffs().num_rows_loc()) {
                scale_matrix_columns_gpu(res__.mt_coeffs().num_rows_loc(),
                                         num_bands__,
                                         res__.mt_coeffs().prime().at<GPU>(),
                                         p_norm__.at<GPU>());
            }
            #endif
            break;
        }
    }
}

inline mdarray<double, 1>
Band::residuals_aux(K_point* kp__,
                    int ispn__,
                    int num_bands__,
                    std::vector<double>& eval__,
                    wave_functions& hpsi__,
                    wave_functions& opsi__,
                    wave_functions& res__,
                    mdarray<double, 2>& h_diag__,
                    mdarray<double, 1>& o_diag__) const
{
    PROFILE("sirius::Band::residuals_aux");

    assert(kp__->num_gkvec_loc() == res__.pw_coeffs().num_rows_loc());
    assert(kp__->num_gkvec_loc() == hpsi__.pw_coeffs().num_rows_loc());
    assert(kp__->num_gkvec_loc() == opsi__.pw_coeffs().num_rows_loc());
    if (res__.has_mt()) {
        assert(res__.mt_coeffs().num_rows_loc() == hpsi__.mt_coeffs().num_rows_loc());
        assert(res__.mt_coeffs().num_rows_loc() == opsi__.mt_coeffs().num_rows_loc());
    }
    assert(num_bands__ != 0);

    auto pu = ctx_.processing_unit();

    mdarray<double, 1> eval(eval__.data(), num_bands__, "residuals_aux::eval");
    if (pu == GPU) {
        eval.allocate(memory_t::device);
        eval.copy<memory_t::host, memory_t::device>();
    }

    compute_res(pu, num_bands__, eval, hpsi__, opsi__, res__);

    /* compute norm */
    auto res_norm = res__.l2norm(num_bands__);
    
    /* apply preconditioner */
    apply_p(pu, num_bands__, ispn__, res__, h_diag__, o_diag__, eval);

    auto p_norm = res__.l2norm(num_bands__);
    for (int i = 0; i < num_bands__; i++) {
        p_norm[i] = 1.0 / p_norm[i];
    }
    if (pu == GPU) {
        p_norm.copy<memory_t::host, memory_t::device>();
    }
    /* normalize preconditioned residuals */
    normalize_res(pu, num_bands__, res__, p_norm);

    return std::move(res_norm);
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

    int num_sc = (ctx_.num_mag_dims() == 3) ? 2 : 1;

    /* compute residuals */
    for (int ispn = 0; ispn < num_sc; ispn++) {
        compute_res(pu, num_bands__, eval, hpsi__.component(ispn), opsi__.component(ispn), res__.component(ispn));
    }

    /* compute norm */
    auto res_norm = res__.l2norm(pu, num_bands__);

    for (int ispn = 0; ispn < num_sc; ispn++) {
        apply_p(pu, num_bands__, ctx_.num_mag_dims() == 3 ? ispn : ispn__, res__.component(ispn), h_diag__, o_diag__, eval);
    }

    auto p_norm = res__.l2norm(pu, num_bands__);
    for (int i = 0; i < num_bands__; i++) {
        p_norm[i] = 1.0 / p_norm[i];
    }
    if (pu == GPU) {
        p_norm.copy<memory_t::host, memory_t::device>();
    }

    /* normalize preconditioned residuals */
    for (int ispn = 0; ispn < num_sc; ispn++) {
        normalize_res(pu, num_bands__, res__.component(ispn), p_norm);
    }
    
    if (ctx_.control().verbosity_ >= 5) {
        auto n_norm = res__.l2norm(pu, num_bands__);
        if (kp__->comm().rank() == 0) {
            for (int i = 0; i < num_bands__; i++) {
                DUMP("norms of residual %3i: %18.14f %24.14f %18.14f", i, res_norm[i], p_norm[i], n_norm[i]);
            }
        }
    }

    return std::move(res_norm);
}

template <typename T, typename W>
int Band::residuals_common(K_point*             kp__,
                           int                  ispn__,
                           int                  N__,
                           int                  num_bands__,
                           std::vector<double>& eval__,
                           std::vector<double>& eval_old__,
                           dmatrix<T>&          evec__,
                           W&                   hphi__,
                           W&                   ophi__,
                           W&                   hpsi__,
                           W&                   opsi__,
                           W&                   res__,
                           mdarray<double, 2>&  h_diag__,
                           mdarray<double, 1>&  o_diag__) const
{
    assert(N__ != 0);

    auto& itso = ctx_.iterative_solver_input();
    bool converge_by_energy = (itso.converge_by_energy_ == 1);

    int n{0};
    if (converge_by_energy) {

        /* main trick here: first estimate energy difference, and only then compute unconverged residuals */
        auto get_ev_idx = [&](double tol__)
        {
            auto empty_tol = std::max(5 * tol__, itso.empty_states_tolerance_);
            std::vector<int> ev_idx;
            for (int i = 0; i < num_bands__; i++) {
                double tol = tol__ + empty_tol * std::abs(kp__->band_occupancy(i + ispn__ * ctx_.num_fv_states()) / ctx_.max_occupancy() - 1);
                if (std::abs(eval__[i] - eval_old__[i]) > tol) {
                    ev_idx.push_back(i);
                }
            }
            return std::move(ev_idx);
        };

        auto ev_idx = get_ev_idx(itso.energy_tolerance_);

        if ((n = static_cast<int>(ev_idx.size())) == 0) {
            return 0;
        }

        std::vector<double> eval_tmp(n);

        int bs = ctx_.cyclic_block_size();
        dmatrix<T> evec_tmp(N__, n, ctx_.blacs_grid(), bs, bs);
        int num_rows_local = evec_tmp.num_rows_local();
        for (int j = 0; j < n; j++) {
            eval_tmp[j] = eval__[ev_idx[j]];
            auto pos_src = evec__.spl_col().location(ev_idx[j]);
            auto pos_dest = evec_tmp.spl_col().location(j);

            if (pos_src.rank == kp__->comm_col().rank()) {
                kp__->comm_col().isend(&evec__(0, pos_src.local_index), num_rows_local, pos_dest.rank, ev_idx[j]);
            }
            if (pos_dest.rank == kp__->comm_col().rank()) {
               kp__->comm_col().recv(&evec_tmp(0, pos_dest.local_index), num_rows_local, pos_src.rank, ev_idx[j]);
            }
        }
        if (ctx_.processing_unit() == GPU && evec__.blacs_grid().comm().size() == 1) {
            evec_tmp.allocate(memory_t::device);
        }
        /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} and O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
        transform<T>(ctx_.processing_unit(), 1.0, std::vector<W*>({&hphi__, &ophi__}), 0, N__, evec_tmp, 0, 0, 0.0, {&hpsi__, &opsi__}, 0, n);

        auto res_norm = residuals_aux(kp__, ispn__, n, eval_tmp, hpsi__, opsi__, res__, h_diag__, o_diag__);

        int nmax = n;
        n = 0;
        for (int i = 0; i < nmax; i++) {
            /* take the residual if it's norm is above the threshold */
            if (res_norm[i] > itso.residual_tolerance_) {
                /* shift unconverged residuals to the beginning of array */
                if (n != i) {
                    res__.copy_from(res__, i, 1, n, ctx_.processing_unit());
                }
                n++;
            }
        }
        if (ctx_.control().verbosity_ >= 3 && kp__->comm().rank() == 0) {
            DUMP("initial and final number of residuals : %i %i", nmax, n);
        }
    } else {
        /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} and O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
        transform<T>(ctx_.processing_unit(), 1.0, std::vector<W*>({&hphi__, &ophi__}), 0, N__, evec__, 0, 0, 0.0, {&hpsi__, &opsi__}, 0, num_bands__);

        auto res_norm = residuals_aux(kp__, ispn__, num_bands__, eval__, hpsi__, opsi__, res__, h_diag__, o_diag__);

        for (int i = 0; i < num_bands__; i++) {
            double tol = itso.residual_tolerance_ + 1e-3 * std::abs(kp__->band_occupancy(i + ispn__ * ctx_.num_fv_states()) / ctx_.max_occupancy() - 1);
            /* take the residual if it's norm is above the threshold */
            if (res_norm[i] > tol) {
                /* shift unconverged residuals to the beginning of array */
                if (n != i) {
                    res__.copy_from(res__, i, 1, n, ctx_.processing_unit());
                }
                n++;
            }
        }
        if (ctx_.control().verbosity_ >= 3 && kp__->comm().rank() == 0) {
            DUMP("number of residuals : %i", n);
        }
    }
    return n;
}

template <typename T>
inline int Band::residuals(K_point* kp__,
                           int ispn__,
                           int N__,
                           int num_bands__,
                           std::vector<double>& eval__,
                           std::vector<double>& eval_old__,
                           dmatrix<T>& evec__,
                           wave_functions& hphi__,
                           wave_functions& ophi__,
                           wave_functions& hpsi__,
                           wave_functions& opsi__,
                           wave_functions& res__,
                           mdarray<double, 2>& h_diag__,
                           mdarray<double, 1>& o_diag__) const
{
    PROFILE("sirius::Band::residuals");
    int n = residuals_common<T, wave_functions>(kp__, ispn__, N__, num_bands__, eval__, eval_old__, evec__,
                                                hphi__, ophi__, hpsi__, opsi__, res__, h_diag__, o_diag__);
                                        
    /* prevent numerical noise */
    if (std::is_same<T, double>::value && kp__->comm().rank() == 0) {
        switch (ctx_.processing_unit()) {
            case CPU: {
                for (int i = 0; i < n; i++) {
                    res__.pw_coeffs().prime(0, i) = res__.pw_coeffs().prime(0, i).real();
                }
                break;
            }
            case GPU: {
                #ifdef __GPU
                make_real_g0_gpu(res__.pw_coeffs().prime().at<GPU>(), res__.pw_coeffs().prime().ld(), n);
                #endif
                break;
            }
        }
    }

    if (ctx_.control().print_checksum_ && n != 0) {
        auto cs = res__.checksum(0, n);
        auto cs1 = hpsi__.checksum(0, n);
        auto cs2 = opsi__.checksum(0, n);
        if (kp__->comm().rank() == 0) {
            DUMP("checksum(res): %18.10f %18.10f", cs.real(), cs.imag());
            DUMP("checksum(hpsi): %18.10f %18.10f", cs1.real(), cs1.imag());
            DUMP("checksum(opsi): %18.10f %18.10f", cs2.real(), cs2.imag());
        }
    }

    return n;
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

    int n = residuals_common<T, Wave_functions>(kp__, ispn__, N__, num_bands__, eval__, eval_old__, evec__,
                                                hphi__, ophi__, hpsi__, opsi__, res__, h_diag__, o_diag__);
                                        
    int num_sc = ctx_.num_mag_dims() == 3 ? 2 : 1;

    /* prevent numerical noise */
    /* this only happens for real wave-functions (Gamma-point case), non-magnetic or collinear magnetic */
    if (std::is_same<T, double>::value && kp__->comm().rank() == 0) {
        switch (ctx_.processing_unit()) {
            case CPU: {
                for (int i = 0; i < n; i++) {
                    res__.component(0).pw_coeffs().prime(0, i) = res__.component(0).pw_coeffs().prime(0, i).real();
                }
                break;
            }
            case GPU: {
                #ifdef __GPU
                if (n != 0) {
                    make_real_g0_gpu(res__.component(0).pw_coeffs().prime().at<GPU>(), res__.component(0).pw_coeffs().prime().ld(), n);
                }
                #endif
                break;
            }
        }
    }

    if (ctx_.control().print_checksum_ && n != 0) {
        for (int ispn = 0; ispn < num_sc; ispn++) {
            auto cs = res__.component(ispn).checksum(0, n);
            auto cs1 = hpsi__.component(ispn).checksum(0, n);
            auto cs2 = opsi__.component(ispn).checksum(0, n);
            if (kp__->comm().rank() == 0) {
                DUMP("checksum(res%i): %18.10f %18.10f", ispn, cs.real(), cs.imag());
                DUMP("checksum(hpsi%i): %18.10f %18.10f", ispn, cs1.real(), cs1.imag());
                DUMP("checksum(opsi%i): %18.10f %18.10f", ispn, cs2.real(), cs2.imag());
            }
        }
    }

    return n;
}

