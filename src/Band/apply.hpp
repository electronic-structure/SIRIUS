/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 */
template <typename T>
void Band::apply_h(K_point* kp__, 
                   int ispn__,
                   int N__,
                   int n__,
                   wave_functions& phi__,
                   wave_functions& hphi__,
                   Hloc_operator& h_op,
                   D_operator<T>& d_op) const
{
    PROFILE("sirius::Band::apply_h");
    #ifdef __GPU
    STOP();
    #endif

    STOP();

    //==/* set initial hphi */
    //==hphi__.copy_from(phi__, N__, n__);

    //==#ifdef __GPU
    //==if (ctx_.processing_unit() == GPU) {
    //==    hphi__.copy_to_host(N__, n__);
    //==}
    //==#endif
    //==/* apply local part of Hamiltonian */
    //==h_op.apply(ispn__, hphi__, N__, n__);
    //==#ifdef __GPU
    //==if (ctx_.processing_unit() == GPU) {
    //==    hphi__.copy_to_device(N__, n__);
    //==}
    //==#endif

    //==#ifdef __PRINT_OBJECT_CHECKSUM
    //=={
    //==    #ifdef __GPU
    //==    if (ctx_.processing_unit() == GPU) phi__.copy_to_host(N__, n__);
    //==    #endif
    //==    auto cs1 = mdarray<double_complex, 1>(&phi__(0, N__), kp__->num_gkvec_loc() * n__).checksum();
    //==    auto cs2 = mdarray<double_complex, 1>(&hphi__(0, N__), kp__->num_gkvec_loc() * n__).checksum();
    //==    kp__->comm().allreduce(&cs1, 1);
    //==    kp__->comm().allreduce(&cs2, 1);
    //==    DUMP("checksum(phi): %18.10f %18.10f", cs1.real(), cs1.imag());
    //==    DUMP("checksum(hloc_phi): %18.10f %18.10f", cs2.real(), cs2.imag());
    //==}
    //==#endif

    //==if (!ctx_.unit_cell().mt_lo_basis_size()) {
    //==    return;
    //==}

    //==for (int i = 0; i < kp__->beta_projectors().num_beta_chunks(); i++)
    //=={
    //==    kp__->beta_projectors().generate(i);

    //==    kp__->beta_projectors().inner<T>(i, phi__, N__, n__);

    //==    if (!ctx_.iterative_solver_input_section().real_space_prj_)
    //==    {
    //==        d_op.apply(i, ispn__, hphi__, N__, n__);
    //==    }
    //==    else
    //==    {
    //==        STOP();
    //==        //add_nl_h_o_rs(kp__, n__, phi, hphi, ophi, packed_mtrx_offset__, d_mtrx_packed__, q_mtrx_packed__, kappa__);
    //==    }
    //==}
}

/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 *  \param [out] ophi Overlap operator, applied to wave-functions [storage: CPU || GPU].
 */
template <typename T>
void Band::apply_h_o(K_point* kp__, 
                     int ispn__,
                     int N__,
                     int n__,
                     wave_functions& phi__,
                     wave_functions& hphi__,
                     wave_functions& ophi__,
                     Hloc_operator& h_op,
                     D_operator<T>& d_op,
                     Q_operator<T>& q_op) const
{
    PROFILE("sirius::Band::apply_h_o");
    
    double t1 = -omp_get_wtime();
    /* set initial hphi */
    hphi__.copy_from(phi__, N__, n__);

    #ifdef __GPU
    /* if we run on GPU, but the FFT driver is hybrid (it expects a CPU pointer),
     * copy wave-functions to CPU */
    if (ctx_.processing_unit() == GPU && !ctx_.fft_coarse().gpu_only()) {
        hphi__.pw_coeffs().copy_to_host(N__, n__);
    }
    #endif
    /* apply local part of Hamiltonian */
    h_op.apply(ispn__, hphi__, N__, n__);
    #ifdef __GPU
    /* if we run on GPU, but the FFT driver is CPU-GPU hybrid, the result of h_op is stored on CPU
     * and has to be copied back to GPU */
    if (ctx_.processing_unit() == GPU && !ctx_.fft_coarse().gpu_only()) {
        hphi__.pw_coeffs().copy_to_device(N__, n__);
    }
    #endif
    t1 += omp_get_wtime();

    if (kp__->comm().rank() == 0 && ctx_.control().print_performance_) {
        DUMP("hloc performace: %12.6f bands/sec", n__ / t1);
    }

    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        auto cs1 = phi__.checksum(N__, n__);
        auto cs2 = hphi__.checksum(N__, n__);
        DUMP("checksum(phi): %18.10f %18.10f", cs1.real(), cs1.imag());
        DUMP("checksum(hloc_phi): %18.10f %18.10f", cs2.real(), cs2.imag());
    }
    #endif

    /* set intial ophi */
    ophi__.copy_from(phi__, N__, n__);

    if (!ctx_.unit_cell().mt_lo_basis_size()) {
        return;
    }

    for (int i = 0; i < kp__->beta_projectors().num_beta_chunks(); i++) {
        kp__->beta_projectors().generate(i);

        kp__->beta_projectors().inner<T>(i, phi__, N__, n__);

        if (!ctx_.iterative_solver_input_section().real_space_prj_) {
            d_op.apply(i, ispn__, hphi__, N__, n__);
            q_op.apply(i, 0, ophi__, N__, n__);
        } else {
            STOP();
            //add_nl_h_o_rs(kp__, n__, phi, hphi, ophi, packed_mtrx_offset__, d_mtrx_packed__, q_mtrx_packed__, kappa__);
        }
    }

    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        auto cs1 = hphi__.checksum(N__, n__);
        auto cs2 = ophi__.checksum(N__, n__);
        DUMP("checksum(hphi): %18.10f %18.10f", cs1.real(), cs1.imag());
        DUMP("checksum(ophi): %18.10f %18.10f", cs2.real(), cs2.imag());
    }
    #endif

    //== if (!kp__->gkvec().reduced())
    //== {
    //==     // --== DEBUG ==--
    //==     printf("check in apply_h_o\n");
    //==     for (int i = N__; i < N__ + n__; i++)
    //==     {
    //==         bool f1 = false;
    //==         bool f2 = false;
    //==         bool f3 = false;
    //==         double e1 = 0;
    //==         double e2 = 0;
    //==         double e3 = 0;
    //==         for (int igk = 0; igk < kp__->num_gkvec(); igk++)
    //==         {
    //==             auto G = kp__->gkvec()[igk] * (-1);
    //==             int igk1 = kp__->gkvec().index_by_gvec(G);
    //==             if (std::abs(phi__(igk, i) - std::conj(phi__(igk1, i))) > 1e-12)
    //==             {
    //==                 f1 = true;
    //==                 e1 = std::max(e1, std::abs(phi__(igk, i) - std::conj(phi__(igk1, i))));
    //==             }
    //==             if (std::abs(hphi__(igk, i) - std::conj(hphi__(igk1, i))) > 1e-12)
    //==             {
    //==                 f2 = true;
    //==                 e2 = std::max(e2, std::abs(hphi__(igk, i) - std::conj(hphi__(igk1, i))));
    //==             }
    //==             if (std::abs(ophi__(igk, i) - std::conj(ophi__(igk1, i))) > 1e-12)
    //==             {
    //==                 f3 = true;
    //==                 e3 = std::max(e3, std::abs(ophi__(igk, i) - std::conj(ophi__(igk1, i))));
    //==             }
    //==         }
    //==         if (f1) printf("phi[%i] is not real, %20.16f\n", i, e1);
    //==         if (f2) printf("hphi[%i] is not real, %20.16f\n", i, e2);
    //==         if (f3) printf("ophi[%i] is not real, %20.16f\n", i, e3);
    //==     }
    //==     printf("done.\n");
    //== }
}

inline void Band::apply_fv_o(K_point* kp__,
                             bool apw_only__,
                             bool add_o1__,
                             Interstitial_operator& istl_op__, 
                             int N__,
                             int n__,
                             wave_functions& phi__,
                             wave_functions& ophi__) const
{
    PROFILE("sirius::Band::apply_fv_o");
    
    if (!apw_only__) {
        /* zero the local-orbital part */
        for (int ibnd = N__; ibnd < N__ + n__; ibnd++) {
            for (int ilo = 0; ilo < ophi__.mt_coeffs().num_rows_loc(); ilo++) {
                ophi__.mt_coeffs().prime(ilo, ibnd) = 0;
            }
        }
    }

    /* interstitial part */
    istl_op__.apply_o(kp__->gkvec_vloc(), N__, n__, phi__, ophi__);

    matrix<double_complex> alm(kp__->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());
    matrix<double_complex> oalm(kp__->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());
    matrix<double_complex> tmp(unit_cell_.max_mt_aw_basis_size(), n__);

    mdarray<double_complex, 1> v(n__);

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        if (!apw_only__) {
            phi__.mt_coeffs().copy_to_host(N__, n__);
        }
        alm.allocate(memory_t::device);
        tmp.allocate(memory_t::device);
    }
    #endif

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& atom = unit_cell_.atom(ia);
        auto& type = atom.type(); 
        /* number of AW for this atom */
        int naw = atom.mt_aw_basis_size();
        /* number of lo for this atom */
        int nlo = atom.mt_lo_basis_size();
        kp__->alm_coeffs_loc().generate(ia, alm);

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            alm.copy_to_device();
        }
        #endif

        sddk::timer t1("sirius::Band::apply_fv_o|apw-apw");

        if (ctx_.processing_unit() == CPU) {
            /* tmp(lm, i) = A(G, lm)^{T} * C(G, i) */
            linalg<CPU>::gemm(1, 0, naw, n__, kp__->num_gkvec_loc(),
                              alm.at<CPU>(), alm.ld(),
                              phi__.pw_coeffs().prime().at<CPU>(0, N__), phi__.pw_coeffs().prime().ld(),
                              tmp.at<CPU>(), tmp.ld());
        }
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            /* tmp(lm, i) = A(G, lm)^{T} * C(G, i) */
            linalg<GPU>::gemm(1, 0, naw, n__, kp__->num_gkvec_loc(),
                              alm.at<GPU>(), alm.ld(),
                              phi__.pw_coeffs().prime().at<GPU>(0, N__), phi__.pw_coeffs().prime().ld(),
                              tmp.at<GPU>(), tmp.ld());
            tmp.copy_to_host(naw * n__);
        }
        #endif

        kp__->comm().allreduce(tmp.at<CPU>(), static_cast<int>(tmp.size()));

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            tmp.copy_to_device(naw * n__);
        }
        #endif

        for (int xi = 0; xi < naw; xi++) {
            for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++) {
                alm(ig, xi) = std::conj(alm(ig, xi));
            }
        }
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            alm.copy_to_device();
        }
        #endif

        if (ctx_.processing_unit() == CPU) {
            /* APW-APW contribution to overlap */
            linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, naw,
                              linalg_const<double_complex>::one(),
                              alm.at<CPU>(), alm.ld(),
                              tmp.at<CPU>(), tmp.ld(),
                              linalg_const<double_complex>::one(),
                              ophi__.pw_coeffs().prime().at<CPU>(0, N__),
                              ophi__.pw_coeffs().prime().ld());
        }
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            /* APW-APW contribution to overlap */
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, naw,
                              &linalg_const<double_complex>::one(),
                              alm.at<GPU>(), alm.ld(),
                              tmp.at<GPU>(), tmp.ld(),
                              &linalg_const<double_complex>::one(),
                              ophi__.pw_coeffs().prime().at<GPU>(0, N__),
                              ophi__.pw_coeffs().prime().ld());
        }
        #endif
        t1.stop();

        if (!nlo || apw_only__) {
            continue;
        }
            
        /* local orbital coefficients of atom ia for all states */
        matrix<double_complex> phi_lo_ia(nlo, n__);
        auto ia_location = phi__.spl_num_atoms().location(ia);
        if (ia_location.rank == kp__->comm().rank()) {
            for (int i = 0; i < n__; i++) {
                std::memcpy(&phi_lo_ia(0, i),
                            phi__.mt_coeffs().prime().at<CPU>(phi__.offset_mt_coeffs(ia_location.local_index), N__ + i),
                            nlo * sizeof(double_complex));
            }
        }
        kp__->comm().bcast(phi_lo_ia.at<CPU>(), static_cast<int>(phi_lo_ia.size()), ia_location.rank);

        sddk::timer t2("sirius::Band::apply_fv_o|apw-lo");
        oalm.zero();
        #pragma omp parallel for
        for (int ilo = 0; ilo < nlo; ilo++) {
            int xi_lo = type.mt_aw_basis_size() + ilo;
            /* local orbital indices */
            int l_lo     = type.indexb(xi_lo).l;
            int lm_lo    = type.indexb(xi_lo).lm;
            int idxrf_lo = type.indexb(xi_lo).idxrf;
            int order_lo = type.indexb(xi_lo).order;
            /* use oalm as temporary buffer to compute alm*o */
            for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
                for (int igloc = 0; igloc < kp__->num_gkvec_loc(); igloc++) {
                    oalm(igloc, ilo) += alm(igloc, type.indexb_by_lm_order(lm_lo, order_aw)) *
                                        atom.symmetry_class().o_radial_integral(l_lo, order_aw, order_lo);
                }
            }
        }
        linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, nlo,
                          linalg_const<double_complex>::one(),
                          oalm.at<CPU>(), oalm.ld(),
                          phi_lo_ia.at<CPU>(), phi_lo_ia.ld(),
                          linalg_const<double_complex>::one(),
                          ophi__.pw_coeffs().prime().at<CPU>(0, N__),
                          ophi__.pw_coeffs().prime().ld());
        t2.stop();

        sddk::timer t3("sirius::Band::apply_fv_o|lo-apw");
        /* lo-APW contribution */
        for (int i = 0; i < n__; i++) {
            for (int ilo = 0; ilo < type.mt_lo_basis_size(); ilo++) {
                int xi_lo = type.mt_aw_basis_size() + ilo;
                /* local orbital indices */
                int l_lo     = type.indexb(xi_lo).l;
                int lm_lo    = type.indexb(xi_lo).lm;
                int order_lo = type.indexb(xi_lo).order;
                int idxrf_lo = type.indexb(xi_lo).idxrf;
                
                if (ia_location.rank == kp__->comm().rank()) {
                    /* lo-lo contribution */
                    for (int jlo = 0; jlo < type.mt_lo_basis_size(); jlo++) {
                        int xi_lo1 = type.mt_aw_basis_size() + jlo;
                        int lm1    = type.indexb(xi_lo1).lm;
                        int order1 = type.indexb(xi_lo1).order;
                        int idxrf1 = type.indexb(xi_lo1).idxrf;
                        if (lm_lo == lm1) {
                            ophi__.mt_coeffs().prime(ophi__.offset_mt_coeffs(ia_location.local_index) + ilo, N__ + i) +=
                                phi_lo_ia(jlo, i) * atom.symmetry_class().o_radial_integral(l_lo, order_lo, order1);
                        }
                    }

                    for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
                        /* lo-APW contribution */
                        ophi__.mt_coeffs().prime(ophi__.offset_mt_coeffs(ia_location.local_index) + ilo, N__ + i) += 
                            atom.symmetry_class().o_radial_integral(l_lo, order_lo, order_aw) *
                            tmp(type.indexb_by_lm_order(lm_lo, order_aw), i);
                    }
                }
            }
        }
        t3.stop();
    }

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU && !apw_only__) {
        ophi__.mt_coeffs().copy_to_device(N__, n__);
    }
    #endif

    //#ifdef __PRINT_OBJECT_CHECKSUM
    //{
    //    auto cs2 = ophi__.checksum(N__, n__);
    //    DUMP("checksum(ophi): %18.10f %18.10f", cs2.real(), cs2.imag());
    //}
    //#endif
}

inline void Band::apply_fv_h_o(K_point* kp__,
                               Interstitial_operator& istl_op__, 
                               int nlo__,
                               int N__,
                               int n__,
                               wave_functions& phi__,
                               wave_functions& hphi__,
                               wave_functions& ophi__) const
{
    PROFILE("sirius::Band::apply_fv_h_o");

    assert(ophi__.mt_coeffs().num_rows_loc() == hphi__.mt_coeffs().num_rows_loc());
    
    if (N__ == 0) {
        for (int ibnd = 0; ibnd < nlo__; ibnd++) {
            for (int j = 0; j < hphi__.pw_coeffs().num_rows_loc(); j++) {
                hphi__.pw_coeffs().prime(j, ibnd) = 0;
                ophi__.pw_coeffs().prime(j, ibnd) = 0;
            }
        }
    }

    /* zero the local-orbital part */
    for (int ibnd = N__; ibnd < N__ + n__; ibnd++) {
        for (int ilo = 0; ilo < hphi__.mt_coeffs().num_rows_loc(); ilo++) {
            hphi__.mt_coeffs().prime(ilo, ibnd) = 0;
            ophi__.mt_coeffs().prime(ilo, ibnd) = 0;
        }
    }

    /* interstitial part */
    if (N__ == 0) {
        /* don't apply to the pure local orbital basis functions */
        istl_op__.apply(kp__->gkvec_vloc(), nlo__, n__ - nlo__, phi__, hphi__, ophi__);
    } else {
        istl_op__.apply(kp__->gkvec_vloc(), N__, n__, phi__, hphi__, ophi__);
    }

    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        auto cs1 = hphi__.checksum(N__, n__);
        auto cs2 = ophi__.checksum(N__, n__);
        DUMP("checksum(hphi_istl): %18.10f %18.10f", cs1.real(), cs1.imag());
        DUMP("checksum(ophi_istl): %18.10f %18.10f", cs2.real(), cs2.imag());
    }
    #endif

    matrix<double_complex> alm(kp__->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());
    matrix<double_complex> halm(kp__->num_gkvec_loc(), std::max(unit_cell_.max_mt_aw_basis_size(),
                                                                unit_cell_.max_mt_lo_basis_size()));
    matrix<double_complex> tmp(unit_cell_.max_mt_aw_basis_size(), n__);
    matrix<double_complex> htmp(unit_cell_.max_mt_aw_basis_size(), n__);

    double_complex zone(1, 0);

    mdarray<double_complex, 1> v(n__);

    matrix<double_complex> hmt(unit_cell_.max_mt_aw_basis_size(), unit_cell_.max_mt_lo_basis_size());

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        phi__.mt_coeffs().copy_to_host(N__, n__);
        alm.allocate(memory_t::device);
        tmp.allocate(memory_t::device);
        halm.allocate(memory_t::device);
        htmp.allocate(memory_t::device);
        v.allocate(memory_t::device);
    }
    #endif

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& atom = unit_cell_.atom(ia);
        auto& type = atom.type(); 
        /* number of AW for this atom */
        int naw = atom.mt_aw_basis_size();
        /* number of lo for this atom */
        int nlo = atom.mt_lo_basis_size();
        kp__->alm_coeffs_loc().generate(ia, alm);
        apply_hmt_to_apw<spin_block_t::nm>(atom, kp__->num_gkvec_loc(), alm, halm);

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            alm.copy_to_device();
            halm.copy_to_device();
        }
        #endif

        sddk::timer t1("sirius::Band::apply_fv_h_o|apw-apw");

        if (ctx_.processing_unit() == CPU) {
            /* tmp(lm, i) = A(G, lm)^{T} * C(G, i) */
            linalg<CPU>::gemm(1, 0, naw, n__, kp__->num_gkvec_loc(),
                              alm.at<CPU>(), alm.ld(),
                              phi__.pw_coeffs().prime().at<CPU>(0, N__), phi__.pw_coeffs().prime().ld(),
                              tmp.at<CPU>(), tmp.ld());
            /* htmp(lm, i) = H_{mt}A(G, lm)^{T} * C(G, i) */
            linalg<CPU>::gemm(1, 0, naw, n__, kp__->num_gkvec_loc(),
                              halm.at<CPU>(), halm.ld(),
                              phi__.pw_coeffs().prime().at<CPU>(0, N__), phi__.pw_coeffs().prime().ld(),
                              htmp.at<CPU>(), htmp.ld());
        }
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            /* tmp(lm, i) = A(G, lm)^{T} * C(G, i) */
            linalg<GPU>::gemm(1, 0, naw, n__, kp__->num_gkvec_loc(),
                              alm.at<GPU>(), alm.ld(),
                              phi__.pw_coeffs().prime().at<GPU>(0, N__), phi__.pw_coeffs().prime().ld(),
                              tmp.at<GPU>(), tmp.ld());
            /* htmp(lm, i) = H_{mt}A(G, lm)^{T} * C(G, i) */
            linalg<GPU>::gemm(1, 0, naw, n__, kp__->num_gkvec_loc(),
                              halm.at<GPU>(), halm.ld(),
                              phi__.pw_coeffs().prime().at<GPU>(0, N__), phi__.pw_coeffs().prime().ld(),
                              htmp.at<GPU>(), htmp.ld());
            tmp.copy_to_host(naw * n__);
            htmp.copy_to_host(naw * n__);
        }
        #endif

        kp__->comm().allreduce(tmp.at<CPU>(), static_cast<int>(tmp.size()));
        kp__->comm().allreduce(htmp.at<CPU>(), static_cast<int>(tmp.size()));

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            tmp.copy_to_device(naw * n__);
            htmp.copy_to_device(naw * n__);
        }
        #endif

        for (int xi = 0; xi < naw; xi++) {
            for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++) {
                alm(ig, xi) = std::conj(alm(ig, xi));
            }
        }
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            alm.copy_to_device();
        }
        #endif

        if (ctx_.processing_unit() == CPU) {
            /* APW-APW contribution to overlap */
            linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, naw,
                              zone,
                              alm.at<CPU>(), alm.ld(),
                              tmp.at<CPU>(), tmp.ld(),
                              zone,
                              ophi__.pw_coeffs().prime().at<CPU>(0, N__),
                              ophi__.pw_coeffs().prime().ld());
            /* APW-APW contribution to Hamiltonian */
            linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, naw,
                              zone,
                              alm.at<CPU>(), alm.ld(),
                              htmp.at<CPU>(), htmp.ld(),
                              zone,
                              hphi__.pw_coeffs().prime().at<CPU>(0, N__),
                              hphi__.pw_coeffs().prime().ld());
        }
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            /* APW-APW contribution to overlap */
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, naw,
                              &zone,
                              alm.at<GPU>(), alm.ld(),
                              tmp.at<GPU>(), tmp.ld(),
                              &zone,
                              ophi__.pw_coeffs().prime().at<GPU>(0, N__),
                              ophi__.pw_coeffs().prime().ld());
            /* APW-APW contribution to Hamiltonian */
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, naw, 
                              &zone,
                              alm.at<GPU>(), alm.ld(),
                              htmp.at<GPU>(), htmp.ld(),
                              &zone,
                              hphi__.pw_coeffs().prime().at<GPU>(0, N__),
                              hphi__.pw_coeffs().prime().ld());
        }
        #endif
        t1.stop();

        if (!nlo) {
            continue;
        }
            
        /* local orbital coefficients of atom ia for all states */
        matrix<double_complex> phi_lo_ia(nlo, n__);
        auto ia_location = phi__.spl_num_atoms().location(ia);
        if (ia_location.rank == kp__->comm().rank()) {
            for (int i = 0; i < n__; i++) {
                std::memcpy(&phi_lo_ia(0, i),
                            phi__.mt_coeffs().prime().at<CPU>(phi__.offset_mt_coeffs(ia_location.local_index), N__ + i),
                            nlo * sizeof(double_complex));
            }
        }
        kp__->comm().bcast(phi_lo_ia.at<CPU>(), static_cast<int>(phi_lo_ia.size()), ia_location.rank);

        sddk::timer t2("sirius::Band::apply_fv_h_o|apw-lo");
        halm.zero();
        #pragma omp parallel for
        for (int ilo = 0; ilo < nlo; ilo++) {
            int xi_lo = type.mt_aw_basis_size() + ilo;
            /* local orbital indices */
            int l_lo     = type.indexb(xi_lo).l;
            int lm_lo    = type.indexb(xi_lo).lm;
            int idxrf_lo = type.indexb(xi_lo).idxrf;
            int order_lo = type.indexb(xi_lo).order;
            /* use halm as temporary buffer to compute alm*o */
            for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
                for (int igloc = 0; igloc < kp__->num_gkvec_loc(); igloc++) {
                    halm(igloc, ilo) += alm(igloc, type.indexb_by_lm_order(lm_lo, order_aw)) *
                                        atom.symmetry_class().o_radial_integral(l_lo, order_aw, order_lo);
                }
            }
            for (int xi = 0; xi < naw; xi++) {
                int lm_aw    = type.indexb(xi).lm;
                int idxrf_aw = type.indexb(xi).idxrf;
                auto& gc = gaunt_coefs_->gaunt_vector(lm_aw, lm_lo);
                hmt(xi, ilo) = atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_aw, idxrf_lo, gc);
            }
        }
        linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, nlo,
                          zone,
                          halm.at<CPU>(), halm.ld(),
                          phi_lo_ia.at<CPU>(), phi_lo_ia.ld(),
                          zone,
                          ophi__.pw_coeffs().prime().at<CPU>(0, N__),
                          ophi__.pw_coeffs().prime().ld());

        linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), nlo, naw,
                          alm.at<CPU>(), alm.ld(),
                          hmt.at<CPU>(), hmt.ld(),
                          halm.at<CPU>(), halm.ld());

        linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, nlo,
                          zone,
                          halm.at<CPU>(), halm.ld(),
                          phi_lo_ia.at<CPU>(), phi_lo_ia.ld(),
                          zone,
                          hphi__.pw_coeffs().prime().at<CPU>(0, N__),
                          hphi__.pw_coeffs().prime().ld());
        t2.stop();

        sddk::timer t3("sirius::Band::apply_fv_h_o|lo-apw");
        /* lo-APW contribution */
        for (int i = 0; i < n__; i++) {
            for (int ilo = 0; ilo < type.mt_lo_basis_size(); ilo++) {
                int xi_lo = type.mt_aw_basis_size() + ilo;
                /* local orbital indices */
                int l_lo     = type.indexb(xi_lo).l;
                int lm_lo    = type.indexb(xi_lo).lm;
                int order_lo = type.indexb(xi_lo).order;
                int idxrf_lo = type.indexb(xi_lo).idxrf;
                
                if (ia_location.rank == kp__->comm().rank()) {
                    /* lo-lo contribution */
                    for (int jlo = 0; jlo < type.mt_lo_basis_size(); jlo++) {
                        int xi_lo1 = type.mt_aw_basis_size() + jlo;
                        int lm1    = type.indexb(xi_lo1).lm;
                        int order1 = type.indexb(xi_lo1).order;
                        int idxrf1 = type.indexb(xi_lo1).idxrf;
                        auto& gc = gaunt_coefs_->gaunt_vector(lm_lo, lm1);
                        if (lm_lo == lm1) {
                            ophi__.mt_coeffs().prime(ophi__.offset_mt_coeffs(ia_location.local_index) + ilo, N__ + i) +=
                                phi_lo_ia(jlo, i) * atom.symmetry_class().o_radial_integral(l_lo, order_lo, order1);
                        }
                        hphi__.mt_coeffs().prime(hphi__.offset_mt_coeffs(ia_location.local_index) + ilo, N__ + i) +=
                            phi_lo_ia(jlo, i) * atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf1, gc);
                    }

                    for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
                        /* lo-APW contribution */
                        ophi__.mt_coeffs().prime(ophi__.offset_mt_coeffs(ia_location.local_index) + ilo, N__ + i) += 
                            atom.symmetry_class().o_radial_integral(l_lo, order_lo, order_aw) *
                            tmp(type.indexb_by_lm_order(lm_lo, order_aw), i);
                    }

                    double_complex z(0, 0);
                    for (int xi = 0; xi < naw; xi++) {
                        int lm_aw    = type.indexb(xi).lm;
                        int idxrf_aw = type.indexb(xi).idxrf;
                        auto& gc = gaunt_coefs_->gaunt_vector(lm_lo, lm_aw);
                        z += atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf_aw, gc) * tmp(xi, i);
                    }
                    /* lo-APW contribution */
                    hphi__.mt_coeffs().prime(hphi__.offset_mt_coeffs(ia_location.local_index) + ilo, N__ + i) += z;
                }
            }
        }
        t3.stop();
    }

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        hphi__.mt_coeffs().copy_to_device(N__, n__);
        ophi__.mt_coeffs().copy_to_device(N__, n__);
    }
    #endif

    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        auto cs1 = hphi__.checksum(N__, n__);
        auto cs2 = ophi__.checksum(N__, n__);
        DUMP("checksum(hphi): %18.10f %18.10f", cs1.real(), cs1.imag());
        DUMP("checksum(ophi): %18.10f %18.10f", cs2.real(), cs2.imag());
    }
    #endif
}

//TODO: port to GPU
inline void Band::apply_magnetic_field(wave_functions& fv_states__,
                                       Gvec const& gkvec__,
                                       Periodic_function<double>* effective_magnetic_field__[3],
                                       std::vector<wave_functions>& hpsi__) const
{
    PROFILE("sirius::Band::apply_magnetic_field");

    assert(hpsi__.size() == 2 || hpsi__.size() == 3);

    fv_states__.pw_coeffs().remap_forward(gkvec__.partition().gvec_fft_slab(),
                                          ctx_.mpi_grid_fft().communicator(1 << 1),
                                          ctx_.num_fv_states());
    
    /* components of H|psi> to with H is applied */
    std::vector<int> iv(1, 0);
    if (hpsi__.size() == 3) {
        iv.push_back(2);
    }
    for (int i: iv) {
        hpsi__[i].pw_coeffs().set_num_extra(gkvec__.partition().gvec_count_fft(),
                                            ctx_.mpi_grid_fft().communicator(1 << 1),
                                            ctx_.num_fv_states());
        assert(fv_states__.pw_coeffs().spl_num_col().local_size() == hpsi__[i].pw_coeffs().spl_num_col().local_size());
    }

    std::vector<double_complex> psi_r;
    if (hpsi__.size() == 3) {
        psi_r.resize(ctx_.fft().local_size());
    }

    ctx_.fft().prepare(gkvec__.partition());

    for (int i = 0; i < fv_states__.pw_coeffs().spl_num_col().local_size(); i++) {
        /* transform first-variational state to real space */
        ctx_.fft().transform<1>(gkvec__.partition(), fv_states__.pw_coeffs().extra().at<CPU>(0, i));
        /* save for a reuse */
        if (hpsi__.size() == 3) {
            STOP(); // fix output of fft driver
            ctx_.fft().output(&psi_r[0]);
        }
        #ifdef __GPU
        if (ctx_.fft().hybrid()) {
            ctx_.fft().buffer().copy_to_host();
        }
        #endif

        for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
            /* hpsi(r) = psi(r) * B_z(r) * Theta(r) */
            ctx_.fft().buffer(ir) *= (effective_magnetic_field__[0]->f_rg(ir) * ctx_.step_function().theta_r(ir));
        }

        #ifdef __GPU
        if (ctx_.fft().hybrid()) {
            ctx_.fft().buffer().copy_to_device();
        }
        #endif
        ctx_.fft().transform<-1>(gkvec__.partition(), hpsi__[0].pw_coeffs().extra().at<CPU>(0, i));

        if (hpsi__.size() == 3) {
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                /* hpsi(r) = psi(r) * (B_x(r) - iB_y(r)) * Theta(r) */
                ctx_.fft().buffer(ir) = psi_r[ir] * ctx_.step_function().theta_r(ir) * 
                                        double_complex(effective_magnetic_field__[1]->f_rg(ir),
                                                      -effective_magnetic_field__[2]->f_rg(ir));
            }
            #ifdef __GPU
            if (ctx_.fft().hybrid()) {
                ctx_.fft().buffer().copy_to_device();
            }
            #endif
            ctx_.fft().transform<-1>(gkvec__.partition(), hpsi__[2].pw_coeffs().extra().at<CPU>(0, i));
        }
    }

    ctx_.fft().dismiss();

    for (int i: iv) {
        hpsi__[i].pw_coeffs().remap_backward(gkvec__.partition().gvec_fft_slab(),
                                             ctx_.mpi_grid_fft().communicator(1 << 1),
                                             ctx_.num_fv_states());
    }

    mdarray<double_complex, 3> zm(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(), ctx_.num_mag_dims());

    for (int ialoc = 0; ialoc < fv_states__.spl_num_atoms().local_size(); ialoc++) {
        int ia = fv_states__.spl_num_atoms()[ialoc];
        auto& atom = unit_cell_.atom(ia);
        int offset = fv_states__.offset_mt_coeffs(ialoc);
        int mt_basis_size = atom.type().mt_basis_size();
        
        zm.zero();
        
        /* only upper triangular part of zm is computed because it is a hermitian matrix */
        #pragma omp parallel for default(shared)
        for (int xi2 = 0; xi2 < mt_basis_size; xi2++) {
            int lm2    = atom.type().indexb(xi2).lm;
            int idxrf2 = atom.type().indexb(xi2).idxrf;
            
            for (int i = 0; i < ctx_.num_mag_dims(); i++) {
                for (int xi1 = 0; xi1 <= xi2; xi1++) {
                    int lm1    = atom.type().indexb(xi1).lm;
                    int idxrf1 = atom.type().indexb(xi1).idxrf;

                    zm(xi1, xi2, i) = gaunt_coefs_->sum_L3_gaunt(lm1, lm2, atom.b_radial_integrals(idxrf1, idxrf2, i)); 
                }
            }
        }
        /* compute bwf = B_z*|wf_j> */
        linalg<CPU>::hemm(0, 0, mt_basis_size, ctx_.num_fv_states(),
                          linalg_const<double_complex>::one(),
                          zm.at<CPU>(), zm.ld(), 
                          fv_states__.mt_coeffs().prime().at<CPU>(offset, 0), fv_states__.mt_coeffs().prime().ld(),
                          linalg_const<double_complex>::zero(),
                          hpsi__[0].mt_coeffs().prime().at<CPU>(offset, 0), hpsi__[0].mt_coeffs().prime().ld());
        
        /* compute bwf = (B_x - iB_y)|wf_j> */
        if (hpsi__.size() == 3) {
            /* reuse first (z) component of zm matrix to store (B_x - iB_y) */
            for (int xi2 = 0; xi2 < mt_basis_size; xi2++) {
                for (int xi1 = 0; xi1 <= xi2; xi1++) {
                    zm(xi1, xi2, 0) = zm(xi1, xi2, 1) - double_complex(0, 1) * zm(xi1, xi2, 2);
                }
                
                /* remember: zm for x,y,z, components of magnetic field is hermitian and we computed 
                 * only the upper triangular part */
                for (int xi1 = xi2 + 1; xi1 < mt_basis_size; xi1++) {
                    zm(xi1, xi2, 0) = std::conj(zm(xi2, xi1, 1)) - double_complex(0, 1) * std::conj(zm(xi2, xi1, 2));
                }
            }
              
            linalg<CPU>::gemm(0, 0, mt_basis_size, ctx_.num_fv_states(), mt_basis_size,
                              zm.at<CPU>(), zm.ld(), 
                              fv_states__.mt_coeffs().prime().at<CPU>(offset, 0), fv_states__.mt_coeffs().prime().ld(),
                              hpsi__[2].mt_coeffs().prime().at<CPU>(offset, 0), hpsi__[2].mt_coeffs().prime().ld());
        }
    }

   /* copy Bz|\psi> to -Bz|\psi> */
    for (int i = 0; i < ctx_.num_fv_states(); i++) {
        for (int j = 0; j < fv_states__.pw_coeffs().num_rows_loc(); j++) {
            hpsi__[1].pw_coeffs().prime(j, i) = -hpsi__[0].pw_coeffs().prime(j, i);
        }
        for (int j = 0; j < fv_states__.mt_coeffs().num_rows_loc(); j++) {
            hpsi__[1].mt_coeffs().prime(j, i) = -hpsi__[0].mt_coeffs().prime(j, i);
        }
    }
}

//== template <spin_block_t sblock>
//== void Band::apply_uj_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi)
//== {
//==     Timer t("sirius::Band::apply_uj_correction");
//== 
//==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==     {
//==         if (unit_cell_.atom(ia)->apply_uj_correction())
//==         {
//==             Atom_type* type = unit_cell_.atom(ia)->type();
//== 
//==             int offset = unit_cell_.atom(ia)->offset_wf();
//== 
//==             int l = unit_cell_.atom(ia)->uj_correction_l();
//== 
//==             int nrf = type->indexr().num_rf(l);
//== 
//==             for (int order2 = 0; order2 < nrf; order2++)
//==             {
//==                 for (int lm2 = Utils::lm_by_l_m(l, -l); lm2 <= Utils::lm_by_l_m(l, l); lm2++)
//==                 {
//==                     int idx2 = type->indexb_by_lm_order(lm2, order2);
//==                     for (int order1 = 0; order1 < nrf; order1++)
//==                     {
//==                         double ori = unit_cell_.atom(ia)->symmetry_class()->o_radial_integral(l, order2, order1);
//==                         
//==                         for (int ist = 0; ist < parameters_.spl_fv_states().local_size(); ist++)
//==                         {
//==                             for (int lm1 = Utils::lm_by_l_m(l, -l); lm1 <= Utils::lm_by_l_m(l, l); lm1++)
//==                             {
//==                                 int idx1 = type->indexb_by_lm_order(lm1, order1);
//==                                 double_complex z1 = fv_states(offset + idx1, ist) * ori;
//== 
//==                                 if (sblock == uu)
//==                                 {
//==                                     hpsi(offset + idx2, ist, 0) += z1 * 
//==                                         unit_cell_.atom(ia)->uj_correction_matrix(lm2, lm1, 0, 0);
//==                                 }
//== 
//==                                 if (sblock == dd)
//==                                 {
//==                                     hpsi(offset + idx2, ist, 1) += z1 *
//==                                         unit_cell_.atom(ia)->uj_correction_matrix(lm2, lm1, 1, 1);
//==                                 }
//== 
//==                                 if (sblock == ud)
//==                                 {
//==                                     hpsi(offset + idx2, ist, 2) += z1 *
//==                                         unit_cell_.atom(ia)->uj_correction_matrix(lm2, lm1, 0, 1);
//==                                 }
//==                                 
//==                                 if (sblock == du)
//==                                 {
//==                                     hpsi(offset + idx2, ist, 3) += z1 *
//==                                         unit_cell_.atom(ia)->uj_correction_matrix(lm2, lm1, 1, 0);
//==                                 }
//==                             }
//==                         }
//==                     }
//==                 }
//==             }
//==         }
//==     }
//== }

//== void Band::apply_so_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi)
//== {
//==     Timer t("sirius::Band::apply_so_correction");
//== 
//==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==     {
//==         Atom_type* type = unit_cell_.atom(ia)->type();
//== 
//==         int offset = unit_cell_.atom(ia)->offset_wf();
//== 
//==         for (int l = 0; l <= parameters_.lmax_apw(); l++)
//==         {
//==             int nrf = type->indexr().num_rf(l);
//== 
//==             for (int order1 = 0; order1 < nrf; order1++)
//==             {
//==                 for (int order2 = 0; order2 < nrf; order2++)
//==                 {
//==                     double sori = unit_cell_.atom(ia)->symmetry_class()->so_radial_integral(l, order1, order2);
//==                     
//==                     for (int m = -l; m <= l; m++)
//==                     {
//==                         int idx1 = type->indexb_by_l_m_order(l, m, order1);
//==                         int idx2 = type->indexb_by_l_m_order(l, m, order2);
//==                         int idx3 = (m + l != 0) ? type->indexb_by_l_m_order(l, m - 1, order2) : 0;
//==                         int idx4 = (m - l != 0) ? type->indexb_by_l_m_order(l, m + 1, order2) : 0;
//== 
//==                         for (int ist = 0; ist < (int)parameters_.spl_fv_states().local_size(); ist++)
//==                         {
//==                             double_complex z1 = fv_states(offset + idx2, ist) * double(m) * sori;
//==                             hpsi(offset + idx1, ist, 0) += z1;
//==                             hpsi(offset + idx1, ist, 1) -= z1;
//==                             // apply L_{-} operator
//==                             if (m + l) hpsi(offset + idx1, ist, 2) += fv_states(offset + idx3, ist) * sori * 
//==                                                                       sqrt(double(l * (l + 1) - m * (m - 1)));
//==                             // apply L_{+} operator
//==                             if (m - l) hpsi(offset + idx1, ist, 3) += fv_states(offset + idx4, ist) * sori * 
//==                                                                       sqrt(double(l * (l + 1) - m * (m + 1)));
//==                         }
//==                     }
//==                 }
//==             }
//==         }
//==     }
//== }


