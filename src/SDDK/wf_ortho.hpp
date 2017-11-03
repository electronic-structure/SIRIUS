#include "eigenproblem.h"

/// Orthogonalize n new wave-functions to the N old wave-functions
template <typename T>
inline void orthogonalize(int N__,
                          int n__,
                          std::vector<wave_functions*> wfs__,
                          int idx_bra__,
                          int idx_ket__,
                          dmatrix<T>& o__,
                          wave_functions& tmp__)
{
    PROFILE("sddk::wave_functions::orthogonalize");

    auto pu = wfs__[0]->pu();

    const char* sddk_pp_raw = std::getenv("SDDK_PRINT_PERFORMANCE");
    int sddk_pp = (sddk_pp_raw == NULL) ? 0 : std::atoi(sddk_pp_raw);

    auto& comm = wfs__[idx_bra__]->comm();

    int K{0};
    if (sddk_pp) {
        K = wfs__[idx_bra__]->pw_coeffs().num_rows_loc();
        if (wfs__[idx_bra__]->has_mt()) {
            K += wfs__[idx_bra__]->mt_coeffs().num_rows_loc();
        }
        comm.allreduce(&K, 1);
        if (std::is_same<T, double>::value) {
            K *= 2;
        }
    }

    double ngop{0};
    if (std::is_same<T, double>::value) {
        ngop = 2e-9;
    }
    if (std::is_same<T, double_complex>::value) {
        ngop = 8e-9;
    }

    if (sddk_pp) {
        comm.barrier();
    }
    //double time = -omp_get_wtime();

    double gflops{0};

    /* project out the old subspace:
     * |\tilda phi_new> = |phi_new> - |phi_old><phi_old|phi_new> */
    if (N__ > 0) {
        inner(*wfs__[idx_bra__], 0, N__, *wfs__[idx_ket__], N__, n__, 0.0, o__, 0, 0);
        transform(pu, -1.0, wfs__, 0, N__, o__, 0, 0, 1.0, wfs__, N__, n__);

        if (sddk_pp) {
            gflops += static_cast<int>(1 + wfs__.size()) * ngop * N__ * n__ * K; // inner and transfrom have the same number of flops
        }
    }

    /* orthogonalize new n__ x n__ block */
    inner(*wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, 0.0, o__, 0, 0);

    if (sddk_pp) {
        gflops += ngop * n__ * n__ * K;
    }

    /* single MPI rank */
    if (o__.blacs_grid().comm().size() == 1) {
        bool use_magma{false};
        #if defined(__GPU) && defined(__MAGMA)
        if (pu == GPU) {
            use_magma = true;
        }
        #endif

        if (use_magma) {
            #ifdef __GPU
            /* Cholesky factorization */
            if (int info = linalg<GPU>::potrf(n__, o__.template at<GPU>(), o__.ld())) {
                std::stringstream s;
                s << "error in GPU factorization, info = " << info;
                TERMINATE(s);
            }
            /* inversion of triangular matrix */
            if (linalg<GPU>::trtri(n__, o__.template at<GPU>(), o__.ld())) {
                TERMINATE("error in inversion");
            }
            #endif
        } else { /* CPU version */
            //check_hermitian("OVLP", o__, n__);
            //o__.serialize("overlap.dat", n__);
            /* Cholesky factorization */
            if (int info = linalg<CPU>::potrf(n__, &o__(0, 0), o__.ld())) {
                std::stringstream s;
                s << "error in factorization, info = " << info << std::endl
                  << "number of existing states: " << N__ << std::endl
                  << "number of new states: " << n__ << std::endl
                  << "number of wave_functions: " << wfs__.size() << std::endl
                  << "idx_bra: " << idx_bra__ << " " << "idx_ket:" << idx_ket__;
                TERMINATE(s);
            }
            /* inversion of triangular matrix */
            if (linalg<CPU>::trtri(n__, &o__(0, 0), o__.ld())) {
                TERMINATE("error in inversion");
            }
            if (pu == GPU) {
                #ifdef __GPU
                acc::copyin(o__.template at<GPU>(), o__.ld(), o__.template at<CPU>(), o__.ld(), n__, n__);
                #endif
            }
        }

        /* CPU version */
        if (pu == CPU) {
            /* multiplication by triangular matrix */
            for (auto& e: wfs__) {
                /* wave functions are complex, transformation matrix is complex */
                if (std::is_same<T, double_complex>::value) {
                    linalg<CPU>::trmm('R', 'U', 'N', e->pw_coeffs().num_rows_loc(), n__, double_complex(1, 0),
                                      reinterpret_cast<double_complex*>(o__.template at<CPU>()), o__.ld(),
                                      e->pw_coeffs().prime().at<CPU>(0, N__), e->pw_coeffs().prime().ld());

                    if (e->has_mt() && e->mt_coeffs().num_rows_loc()) {
                        linalg<CPU>::trmm('R', 'U', 'N', e->mt_coeffs().num_rows_loc(), n__, double_complex(1, 0),
                                          reinterpret_cast<double_complex*>(o__.template at<CPU>()), o__.ld(),
                                          e->mt_coeffs().prime().at<CPU>(0, N__), e->mt_coeffs().prime().ld());
                    }
                }
                /* wave functions are real (psi(G) = psi^{*}(-G)), transformation matrix is real */
                if (std::is_same<T, double>::value) {
                    linalg<CPU>::trmm('R', 'U', 'N', 2 * e->pw_coeffs().num_rows_loc(), n__, 1.0,
                                      reinterpret_cast<double*>(o__.template at<CPU>()), o__.ld(),
                                      reinterpret_cast<double*>(e->pw_coeffs().prime().at<CPU>(0, N__)), 2 * e->pw_coeffs().prime().ld());

                    if (e->has_mt() && e->mt_coeffs().num_rows_loc()) {
                        linalg<CPU>::trmm('R', 'U', 'N', 2 * e->mt_coeffs().num_rows_loc(), n__, 1.0,
                                          reinterpret_cast<double*>(o__.template at<CPU>()), o__.ld(),
                                          reinterpret_cast<double*>(e->mt_coeffs().prime().at<CPU>(0, N__)), 2 * e->mt_coeffs().prime().ld());
                    }
                }
            }
        }
        #ifdef __GPU
        if (pu == GPU) {
            /* multiplication by triangular matrix */
            for (auto& e: wfs__) {
                if (std::is_same<T, double_complex>::value) {
                    double_complex alpha(1, 0);

                    linalg<GPU>::trmm('R', 'U', 'N', e->pw_coeffs().num_rows_loc(), n__, &alpha,
                                      reinterpret_cast<double_complex*>(o__.template at<GPU>()), o__.ld(),
                                      e->pw_coeffs().prime().at<GPU>(0, N__), e->pw_coeffs().prime().ld());

                    if (e->has_mt() && e->mt_coeffs().num_rows_loc()) {
                        linalg<GPU>::trmm('R', 'U', 'N', e->mt_coeffs().num_rows_loc(), n__, &alpha,
                                          reinterpret_cast<double_complex*>(o__.template at<GPU>()), o__.ld(),
                                          e->mt_coeffs().prime().at<GPU>(0, N__), e->mt_coeffs().prime().ld());
                    }
                    /* alpha should not go out of the scope, so wait */
                    acc::sync_stream(-1);
                }
                if (std::is_same<T, double>::value) {
                    double alpha{1};

                    linalg<GPU>::trmm('R', 'U', 'N', 2 * e->pw_coeffs().num_rows_loc(), n__, &alpha,
                                      reinterpret_cast<double*>(o__.template at<GPU>()), o__.ld(),
                                      reinterpret_cast<double*>(e->pw_coeffs().prime().at<GPU>(0, N__)), 2 * e->pw_coeffs().prime().ld());

                    if (e->has_mt() && e->mt_coeffs().num_rows_loc()) {
                        linalg<GPU>::trmm('R', 'U', 'N', 2 * e->mt_coeffs().num_rows_loc(), n__, &alpha,
                                          reinterpret_cast<double*>(o__.template at<GPU>()), o__.ld(),
                                          reinterpret_cast<double*>(e->mt_coeffs().prime().at<GPU>(0, N__)), 2 * e->mt_coeffs().prime().ld());
                    }
                    acc::sync_stream(-1);
                }
            }
            acc::sync_stream(-1);
        }
        #endif
    } else { /* parallel transformation */
        sddk::timer t1("sddk::wave_functions::orthogonalize|potrf");
        if (int info = linalg<CPU>::potrf(n__, o__)) {
            std::stringstream s;
            s << "error in factorization, info = " << info;
            TERMINATE(s);
        }
        t1.stop();

        sddk::timer t2("sddk::wave_functions::orthogonalize|trtri");
        if (linalg<CPU>::trtri(n__, o__)) {
            TERMINATE("error in inversion");
        }
        t2.stop();

        /* o is upper triangular matrix */
        for (int i = 0; i < n__; i++) {
            for (int j = i + 1; j < n__; j++) {
                o__.set(j, i, 0);
            }
        }

        /* phi is transformed into phi, so we can't use it as the output buffer; use tmp instead and then overwrite phi */
        for (auto& e: wfs__) {
            transform(pu, *e, N__, n__, o__, 0, 0, tmp__, 0, n__);
            e->copy_from(tmp__, 0, n__, N__, pu);
        }
    }
}

template <typename T>
inline void orthogonalize(int             N__,
                          int             n__,
                          wave_functions& phi__,
                          wave_functions& hphi__,
                          wave_functions& ophi__,
                          dmatrix<T>&     o__,
                          wave_functions& tmp__)
{
    static_assert(std::is_same<T, double>::value || std::is_same<T, double_complex>::value, "wrong type");

    auto wfs = {&phi__, &hphi__, &ophi__};

    orthogonalize(N__, n__, wfs, 0, 2, o__, tmp__);
}

/// Orthogonalize n new wave-functions to the N old wave-functions
template <typename T>
inline void orthogonalize(device_t                     pu__,
                          int                          num_sc__,
                          int                          N__,
                          int                          n__,
                          std::vector<Wave_functions*> wfs__,
                          int                          idx_bra__,
                          int                          idx_ket__,
                          dmatrix<T>&                  o__,
                          wave_functions&              tmp__)
{
    PROFILE("sddk::wave_functions::orthogonalize");

    /* project out the old subspace:
     * |\tilda phi_new> = |phi_new> - |phi_old><phi_old|phi_new> */
    if (N__ > 0) {
        inner(*wfs__[idx_bra__], 0, N__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);
        transform(pu__, -1.0, wfs__, 0, N__, o__, 0, 0, 1.0, wfs__, N__, n__);
    }

    const char* sddk_debug_raw = std::getenv("SDDK_DEBUG");
    int sddk_debug = (sddk_debug_raw == NULL) ? 0 : std::atoi(sddk_debug_raw);

    if (sddk_debug >= 2) {
        if (o__.blacs_grid().comm().rank() == 0) {
            printf("check QR decomposition, matrix size : %i\n", n__);
        }
        inner(*wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);

        linalg<CPU>::geqrf(n__, n__, o__, 0, 0);
        auto diag = o__.get_diag(n__);
        if (o__.blacs_grid().comm().rank() == 0) {
            for (int i = 0; i < n__; i++) {
                if (std::abs(diag[i]) < 1e-6) {
                    std::cout << "small norm: " << i << " " << diag[i] << std::endl;
                }
            }
        }

        if (o__.blacs_grid().comm().rank() == 0) {
            printf("check eigen-values, matrix size : %i\n", n__);
        }
        inner(*wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);
        
        std::vector<double> eo(n__);
        dmatrix<T> evec(o__.num_rows(), o__.num_cols(), o__.blacs_grid(), o__.bs_row(), o__.bs_col());

        auto solver = experimental::Eigensolver_factory<T>(experimental::ev_solver_t::scalapack);
        solver->solve(n__, o__, eo.data(), evec);

        if (o__.blacs_grid().comm().rank() == 0) {
            for (int i = 0; i < n__; i++) {
                if (eo[i] < 1e-6) {
                    std::cout << "small eigen-value " << i << " " << eo[i] << std::endl;
                }
            }
        }

    //    //if (o__.blacs_grid().comm().rank() == 0) { 
    //    //    std::cout << "smallest ev of the new n x x block: " << eo[0] << std::endl;
    //    //}
    }

    /* orthogonalize new n__ x n__ block */
    inner(*wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);
    o__.make_real_diag(n__);

    if (sddk_debug >= 1) {
        if (o__.blacs_grid().comm().rank() == 0) {
            printf("check diagonal\n");
        }
        auto diag = o__.get_diag(n__);
        for (int i = 0; i < n__; i++) {
            if (std::real(diag[i]) <= 0 || std::imag(diag[i]) > 1e-12) {
                std::cout << "wrong diagonal: " << i << " " << diag[i] << std::endl;
            }
        }
        if (o__.blacs_grid().comm().rank() == 0) {
            printf("check hermitian\n");
        }
        double d = check_hermitian(o__, n__);
        if (d > 1e-12 && o__.blacs_grid().comm().rank() == 0) {
            std::stringstream s;
            s << "matrix is not hermitian, max diff = " << d;
            WARNING(s);
        }
    }

    /* single MPI rank */
    if (o__.blacs_grid().comm().size() == 1) {
        bool use_magma{false};
        #if defined(__GPU) && defined(__MAGMA)
        if (pu__ == GPU) {
            use_magma = true;
        }
        #endif

        if (use_magma) {
            #ifdef __GPU
            /* Cholesky factorization */
            if (int info = linalg<GPU>::potrf(n__, o__.template at<GPU>(), o__.ld())) {
                std::stringstream s;
                s << "error in GPU factorization, info = " << info;
                TERMINATE(s);
            }
            /* inversion of triangular matrix */
            if (linalg<GPU>::trtri(n__, o__.template at<GPU>(), o__.ld())) {
                TERMINATE("error in inversion");
            }
            #endif
        } else { /* CPU version */
            //check_hermitian("OVLP", o__, n__);
            //o__.serialize("overlap.dat", n__);
            /* Cholesky factorization */
            if (int info = linalg<CPU>::potrf(n__, &o__(0, 0), o__.ld())) {
                std::stringstream s;
                s << "error in factorization, info = " << info << std::endl
                  << "number of existing states: " << N__ << std::endl
                  << "number of new states: " << n__ << std::endl
                  << "number of wave_functions: " << wfs__.size() << std::endl
                  << "idx_bra: " << idx_bra__ << " " << "idx_ket:" << idx_ket__;
                TERMINATE(s);
            }
            /* inversion of triangular matrix */
            if (linalg<CPU>::trtri(n__, &o__(0, 0), o__.ld())) {
                TERMINATE("error in inversion");
            }
            if (pu__ == GPU) {
                #ifdef __GPU
                acc::copyin(o__.template at<GPU>(), o__.ld(), o__.template at<CPU>(), o__.ld(), n__, n__);
                #endif
            }
        }

        for (int isc = 0; isc < num_sc__; isc++) {
            /* CPU version */
            if (pu__ == CPU) {
                /* multiplication by triangular matrix */
                for (auto& e: wfs__) {
                    /* alias for spin component of wave-functions */
                    auto& wfsc = e->component(isc);
                    /* wave functions are complex, transformation matrix is complex */
                    if (std::is_same<T, double_complex>::value) {
                        linalg<CPU>::trmm('R', 'U', 'N', wfsc.pw_coeffs().num_rows_loc(), n__, double_complex(1, 0),
                                          reinterpret_cast<double_complex*>(o__.template at<CPU>()), o__.ld(),
                                          wfsc.pw_coeffs().prime().at<CPU>(0, N__), e->component(isc).pw_coeffs().prime().ld());

                        if (wfsc.has_mt() && wfsc.mt_coeffs().num_rows_loc()) {
                            linalg<CPU>::trmm('R', 'U', 'N', wfsc.mt_coeffs().num_rows_loc(), n__, double_complex(1, 0),
                                              reinterpret_cast<double_complex*>(o__.template at<CPU>()), o__.ld(),
                                              wfsc.mt_coeffs().prime().at<CPU>(0, N__), wfsc.mt_coeffs().prime().ld());
                        }
                    }
                    /* wave functions are real (psi(G) = psi^{*}(-G)), transformation matrix is real */
                    if (std::is_same<T, double>::value) {
                        linalg<CPU>::trmm('R', 'U', 'N', 2 * wfsc.pw_coeffs().num_rows_loc(), n__, 1.0,
                                          reinterpret_cast<double*>(o__.template at<CPU>()), o__.ld(),
                                          reinterpret_cast<double*>(wfsc.pw_coeffs().prime().at<CPU>(0, N__)), 2 * wfsc.pw_coeffs().prime().ld());

                        if (wfsc.has_mt() && wfsc.mt_coeffs().num_rows_loc()) {
                            linalg<CPU>::trmm('R', 'U', 'N', 2 * wfsc.mt_coeffs().num_rows_loc(), n__, 1.0,
                                              reinterpret_cast<double*>(o__.template at<CPU>()), o__.ld(),
                                              reinterpret_cast<double*>(wfsc.mt_coeffs().prime().at<CPU>(0, N__)), 2 * wfsc.mt_coeffs().prime().ld());
                        }
                    }
                }
            }
            #ifdef __GPU
            if (pu__ == GPU) {
                /* multiplication by triangular matrix */
                for (auto& e: wfs__) {
                    auto& wfsc = e->component(isc);
                    if (std::is_same<T, double_complex>::value) {
                        double_complex alpha(1, 0);

                        linalg<GPU>::trmm('R', 'U', 'N', wfsc.pw_coeffs().num_rows_loc(), n__, &alpha,
                                          reinterpret_cast<double_complex*>(o__.template at<GPU>()), o__.ld(),
                                          wfsc.pw_coeffs().prime().at<GPU>(0, N__), wfsc.pw_coeffs().prime().ld());

                        if (wfsc.has_mt() && wfsc.mt_coeffs().num_rows_loc()) {
                            linalg<GPU>::trmm('R', 'U', 'N', wfsc.mt_coeffs().num_rows_loc(), n__, &alpha,
                                              reinterpret_cast<double_complex*>(o__.template at<GPU>()), o__.ld(),
                                              wfsc.mt_coeffs().prime().at<GPU>(0, N__), wfsc.mt_coeffs().prime().ld());
                        }
                        /* alpha should not go out of the scope, so wait */
                        acc::sync_stream(-1);
                    }
                    if (std::is_same<T, double>::value) {
                        double alpha{1};

                        linalg<GPU>::trmm('R', 'U', 'N', 2 * wfsc.pw_coeffs().num_rows_loc(), n__, &alpha,
                                          reinterpret_cast<double*>(o__.template at<GPU>()), o__.ld(),
                                          reinterpret_cast<double*>(wfsc.pw_coeffs().prime().at<GPU>(0, N__)), 2 * wfsc.pw_coeffs().prime().ld());

                        if (wfsc.has_mt() && wfsc.mt_coeffs().num_rows_loc()) {
                            linalg<GPU>::trmm('R', 'U', 'N', 2 * wfsc.mt_coeffs().num_rows_loc(), n__, &alpha,
                                              reinterpret_cast<double*>(o__.template at<GPU>()), o__.ld(),
                                              reinterpret_cast<double*>(wfsc.mt_coeffs().prime().at<GPU>(0, N__)), 2 * wfsc.mt_coeffs().prime().ld());
                        }
                        acc::sync_stream(-1);
                    }
                }
                acc::sync_stream(-1);
            }
            #endif
        }
    } else { /* parallel transformation */
        sddk::timer t1("sddk::wave_functions::orthogonalize|potrf");
        mdarray<T, 1> diag;
        if (sddk_debug >= 1) {
            diag = o__.get_diag(n__);
        }
        o__.make_real_diag(n__);
        if (int info = linalg<CPU>::potrf(n__, o__)) {
            std::stringstream s;
            s << "error in Cholesky factorization, info = " << info << ", matrix size = " << n__;
            if (sddk_debug >= 1) {
                s << std::endl << "  diag = " << diag[info - 1];
            }
            TERMINATE(s);
        }
        t1.stop();

        sddk::timer t2("sddk::wave_functions::orthogonalize|trtri");
        if (linalg<CPU>::trtri(n__, o__)) {
            TERMINATE("error in inversion");
        }
        t2.stop();

        /* o is upper triangular matrix */
        for (int i = 0; i < n__; i++) {
            for (int j = i + 1; j < n__; j++) {
                o__.set(j, i, 0);
            }
        }

        /* phi is transformed into phi, so we can't use it as the output buffer; use tmp instead and then overwrite phi */
        for (auto& e: wfs__) {
            for (int isc = 0; isc < num_sc__; isc++) {
                transform(pu__, e->component(isc), N__, n__, o__, 0, 0, tmp__, 0, n__);
                e->component(isc).copy_from(tmp__, 0, n__, N__, pu__);
            }
        }
    }
}


template <typename T>
inline void orthogonalize(device_t        pu__,
                          int             num_sc__,
                          int             N__,
                          int             n__,
                          Wave_functions& phi__,
                          Wave_functions& hphi__,
                          Wave_functions& ophi__,
                          dmatrix<T>&     o__,
                          wave_functions& tmp__)
{
    static_assert(std::is_same<T, double>::value || std::is_same<T, double_complex>::value, "wrong type");

    auto wfs = {&phi__, &hphi__, &ophi__};

    orthogonalize(pu__, num_sc__, N__, n__, wfs, 0, 2, o__, tmp__);
}

template <typename T>
inline void orthogonalize(int             N__,
                          int             n__,
                          wave_functions& phi__,
                          wave_functions& hphi__,
                          dmatrix<T>&     o__,
                          wave_functions& tmp__)
{
    static_assert(std::is_same<T, double>::value || std::is_same<T, double_complex>::value, "wrong type");

    auto wfs = {&phi__, &hphi__};

    orthogonalize(N__, n__, wfs, 0, 0, o__, tmp__);
}

template <typename T>
inline void orthogonalize(int             N__,
                          int             n__,
                          wave_functions& phi__,
                          dmatrix<T>&     o__,
                          wave_functions& tmp__)
{
    static_assert(std::is_same<T, double>::value || std::is_same<T, double_complex>::value, "wrong type");

    auto wfs = {&phi__};

    orthogonalize(N__, n__, wfs, 0, 0, o__, tmp__);
}


template <typename T>
inline void orthogonalize(device_t        pu__,
                          int             num_sc__,
                          int             N__,
                          int             n__,
                          Wave_functions& phi__,
                          Wave_functions& hphi__,
                          dmatrix<T>&     o__,
                          wave_functions& tmp__)
{
    static_assert(std::is_same<T, double>::value || std::is_same<T, double_complex>::value, "wrong type");

    auto wfs = {&phi__, &hphi__};

    orthogonalize(pu__, num_sc__, N__, n__, wfs, 0, 0, o__, tmp__);
}

template <typename T>
inline void orthogonalize(device_t        pu__,
                          int             num_sc__,
                          int             N__,
                          int             n__,
                          Wave_functions& phi__,
                          dmatrix<T>&     o__,
                          wave_functions& tmp__)
{
    static_assert(std::is_same<T, double>::value || std::is_same<T, double_complex>::value, "wrong type");

    auto wfs = {&phi__};

    orthogonalize(pu__, num_sc__, N__, n__, wfs, 0, 0, o__, tmp__);
}


