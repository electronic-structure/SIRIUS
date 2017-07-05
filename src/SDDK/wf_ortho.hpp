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
        
    /* project out the old subspace:
     * |\tilda phi_new> = |phi_new> - |phi_old><phi_old|phi_new> */
    if (N__ > 0) {
        inner(*wfs__[idx_bra__], 0, N__, *wfs__[idx_ket__], N__, n__, 0.0, o__, 0, 0);
        transform(-1.0, wfs__, 0, N__, o__, 0, 0, 1.0, wfs__, N__, n__);
    }

    /* orthogonalize new n__ x n__ block */
    inner(*wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, 0.0, o__, 0, 0);

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
            transform(*e, N__, n__, o__, 0, 0, tmp__, 0, n__);
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
        inner(num_sc__, *wfs__[idx_bra__], 0, N__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);
        transform(num_sc__, -1.0, wfs__, 0, N__, o__, 0, 0, 1.0, wfs__, N__, n__);
    }

    /* orthogonalize new n__ x n__ block */
    inner(num_sc__, *wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);

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
            for (int isc = 0; isc < num_sc__; isc++) {
                transform(e->component(isc), N__, n__, o__, 0, 0, tmp__, 0, n__);
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

