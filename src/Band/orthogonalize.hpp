template <>
inline void 
Band::orthogonalize<double_complex>(K_point*        kp__,
                                    int             N__,
                                    int             n__,
                                    wave_functions& phi__,
                                    wave_functions& hphi__,
                                    wave_functions& ophi__,
                                    matrix<double_complex>& o__) const
{
    auto wfs = {&phi__, &hphi__, &ophi__};

    /* project out the old subspace:
     * |\tilda phi_new> = |phi_new> - |phi_old><phi_old|phi_new> */
    if (N__ > 0) {
        STOP();
        //inner(phi__, 0, N__, ophi__, N__, n__, o__, 0, 0);

        if (ctx_.processing_unit() == CPU) {
            for (auto& e: wfs) {
                /* transform PW part */
                linalg<CPU>::gemm(0, 0, e->pw_coeffs().num_rows_loc(), n__, N__,
                                  double_complex(-1, 0), 
                                  e->pw_coeffs().prime().at<CPU>(0, 0), e->pw_coeffs().prime().ld(),
                                  o__.at<CPU>(0, 0), o__.ld(),
                                  double_complex(1, 0),
                                  e->pw_coeffs().prime().at<CPU>(0, N__), e->pw_coeffs().prime().ld());
               if (ctx_.full_potential() && e->mt_coeffs().num_rows_loc()) {
                    /* transform muffin-tin part */
                    linalg<CPU>::gemm(0, 0, e->mt_coeffs().num_rows_loc(), n__, N__,
                                      double_complex(-1, 0), 
                                      e->mt_coeffs().prime().at<CPU>(0, 0), e->mt_coeffs().prime().ld(),
                                      o__.at<CPU>(0, 0), o__.ld(),
                                      double_complex(1, 0),
                                      e->mt_coeffs().prime().at<CPU>(0, N__), e->mt_coeffs().prime().ld());
               }
            }
        }
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            STOP();
            //==/* copy overlap matrix <phi_old|phi_new> to GPU */
            //==acc::copyin(o__.at<GPU>(0, 0), o__.ld(), o__.at<CPU>(0, 0), o__.ld(), N__, n__);

            //==double alpha = 1;
            //==double m_alpha = -1;

            //==for (int i = 0; i < 3; i++)
            //=={
            //==    linalg<GPU>::gemm(0, 0, 2 * kp__->num_gkvec_loc(), n__, N__,
            //==                      &m_alpha, 
            //==                      (double*)wfs[i]->coeffs().at<GPU>(0, 0), 2 * kp__->num_gkvec_loc(),
            //==                      o__.at<GPU>(0, 0), o__.ld(),
            //==                      &alpha,
            //==                      (double*)wfs[i]->coeffs().at<GPU>(0, N__), 2 * kp__->num_gkvec_loc());
            //==}

            //==acc::sync_stream(-1);
        }
        #endif
    }

    /* orthogonalize new n__ x n__ block */
    //inner(phi__, N__, n__, ophi__, N__, n__, o__, 0, 0);
    STOP();
    
    if (ctx_.processing_unit() == CPU) {
        int info;
        if ((info = linalg<CPU>::potrf(n__, &o__(0, 0), o__.ld()))) {
            std::stringstream s;
            s << "error in factorization, info = " << info;
            TERMINATE(s);
        }

        if (linalg<CPU>::trtri(n__, &o__(0, 0), o__.ld())) {
            TERMINATE("error in inversion");
        }

        for (auto& e: wfs) {
            linalg<CPU>::trmm('R', 'U', 'N', e->pw_coeffs().num_rows_loc(), n__, double_complex(1, 0),
                              o__.at<CPU>(0, 0), o__.ld(),
                              e->pw_coeffs().prime().at<CPU>(0, N__), e->pw_coeffs().prime().ld());
            if (ctx_.full_potential() && e->mt_coeffs().num_rows_loc()) {
                linalg<CPU>::trmm('R', 'U', 'N', e->mt_coeffs().num_rows_loc(), n__, double_complex(1, 0),
                                  o__.at<CPU>(0, 0), o__.ld(),
                                  e->mt_coeffs().prime().at<CPU>(0, N__), e->mt_coeffs().prime().ld());
            }
        }
    }

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU)
    {
        STOP();
        //== acc::copyin(o__.at<GPU>(0, 0), o__.ld(), o__.at<CPU>(0, 0), o__.ld(), n__, n__);

        //== int info;
        //== if ((info = linalg<GPU>::potrf(n__, o__.at<GPU>(0, 0), o__.ld())))
        //== {
        //==     std::stringstream s;
        //==     s << "error in factorization, info = " << info;
        //==     TERMINATE(s);
        //== }

        //== if (linalg<GPU>::trtri(n__, o__.at<GPU>(0, 0), o__.ld()))
        //==     TERMINATE("error in inversion");

        //== double alpha = 1;

        //== for (int i = 0; i < 3; i++)
        //== {
        //==     linalg<GPU>::trmm('R', 'U', 'N', 2 * kp__->num_gkvec_loc(), n__, &alpha,
        //==                       o__.at<GPU>(0, 0), o__.ld(),
        //==                       (double*)wfs[i]->coeffs().at<GPU>(0, N__), 2 * kp__->num_gkvec_loc());
        //== }
        //== acc::sync_stream(-1);
    }
    #endif
}

template <>
inline void 
Band::orthogonalize<double>(K_point*        kp__,
                            int             N__,
                            int             n__,
                            wave_functions& phi__,
                            wave_functions& hphi__,
                            wave_functions& ophi__,
                            matrix<double>& o__) const
{
    PROFILE_WITH_TIMER("sirius::Band::orthogonalize");

    auto wfs = {&phi__, &hphi__, &ophi__};

    /* project out the old subspace:
     * |\tilda phi_new> = |phi_new> - |phi_old><phi_old|phi_new> */
    if (N__ > 0) {
        //inner(phi__, 0, N__, ophi__, N__, n__, o__, 0, 0);
        STOP();

        if (ctx_.processing_unit() == CPU) {
            for (auto& e: wfs) {
                linalg<CPU>::gemm(0, 0, 2 * e->pw_coeffs().num_rows_loc(), n__, N__,
                                  -1.0, 
                                  (double*)e->pw_coeffs().prime().at<CPU>(0, 0), 2 * e->pw_coeffs().prime().ld(),
                                  o__.at<CPU>(0, 0), o__.ld(),
                                  1.0,
                                  (double*)e->pw_coeffs().prime().at<CPU>(0, N__), 2 * e->pw_coeffs().prime().ld());
            }
        }
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            /* copy overlap matrix <phi_old|phi_new> to GPU */
            acc::copyin(o__.at<GPU>(0, 0), o__.ld(), o__.at<CPU>(0, 0), o__.ld(), N__, n__);

            double alpha = 1;
            double m_alpha = -1;

            for (int i = 0; i < 3; i++) {
                linalg<GPU>::gemm(0, 0, 2 * kp__->num_gkvec_loc(), n__, N__,
                                  &m_alpha, 
                                  (double*)wfs[i]->coeffs().at<GPU>(0, 0), 2 * kp__->num_gkvec_loc(),
                                  o__.at<GPU>(0, 0), o__.ld(),
                                  &alpha,
                                  (double*)wfs[i]->coeffs().at<GPU>(0, N__), 2 * kp__->num_gkvec_loc());
            }

            acc::sync_stream(-1);
        }
        #endif
    }

    /* orthogonalize new n__ x n__ block */
    //inner(phi__, N__, n__, ophi__, N__, n__, o__, 0, 0);
    STOP();

    if (ctx_.processing_unit() == CPU) {
        int info;
        if ((info = linalg<CPU>::potrf(n__, &o__(0, 0), o__.ld()))) {
            std::stringstream s;
            s << "error in factorization, info = " << info;
            TERMINATE(s);
        }

        if (linalg<CPU>::trtri(n__, &o__(0, 0), o__.ld())) {
            TERMINATE("error in inversion");
        }

        for (auto& e: wfs) {
            linalg<CPU>::trmm('R', 'U', 'N', 2 * e->pw_coeffs().num_rows_loc(), n__, 1.0,
                              o__.at<CPU>(0, 0), o__.ld(),
                              (double*)e->pw_coeffs().prime().at<CPU>(0, N__), 2 * e->pw_coeffs().prime().ld());
        }
    }

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        acc::copyin(o__.at<GPU>(0, 0), o__.ld(), o__.at<CPU>(0, 0), o__.ld(), n__, n__);

        int info;
        if ((info = linalg<GPU>::potrf(n__, o__.at<GPU>(0, 0), o__.ld()))) {
            std::stringstream s;
            s << "error in factorization, info = " << info;
            TERMINATE(s);
        }

        if (linalg<GPU>::trtri(n__, o__.at<GPU>(0, 0), o__.ld())) {
            TERMINATE("error in inversion");
        }

        double alpha = 1;

        for (int i = 0; i < 3; i++) {
            linalg<GPU>::trmm('R', 'U', 'N', 2 * kp__->num_gkvec_loc(), n__, &alpha,
                              o__.at<GPU>(0, 0), o__.ld(),
                              (double*)wfs[i]->coeffs().at<GPU>(0, N__), 2 * kp__->num_gkvec_loc());
        }
        acc::sync_stream(-1);
    }
    #endif

    //// --== DEBUG ==--
    //phi__.inner<double>(0, N__ + n__, ophi__, 0, N__ + n__, o__, 0, 0);
    //for (int i = 0; i < N__ + n__; i++)
    //{
    //    for (int j = 0; j < N__ + n__; j++)
    //    {
    //        double a = o__(j, i);
    //        if (i == j) a -= 1;

    //        if (std::abs(a) > 1e-10)
    //        {
    //            printf("wrong overlap");
    //            std::stringstream s;
    //            s << "wrong overlap, diff=" << a;
    //            TERMINATE(s);
    //        }
    //    }
    //}
}

