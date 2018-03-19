/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 */
template <typename T>
void Hamiltonian::apply_h(K_point* kp__,
                          int ispn__,
                          int N__,
                          int n__,
                          Wave_functions& phi__,
                          Wave_functions& hphi__) const
{
    PROFILE("sirius::Hamiltonian::apply_h");

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
    //==        //add_nl_h_o_rs(kp__, n__, phi, hphi, ophi, packed_mtrx_offset__, d_mtrx_packed__, q_mtrx_packed__,
    //kappa__);
    //==    }
    //==}
}

/** \param [in]  kp   Pointer to k-point.
 *  \param [in]  ispn Index of spin.
 *  \param [in]  N    Starting index of wave-functions.
 *  \param [in]  n    Number of wave-functions to which H and S are applied.
 *  \param [in]  phi  Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 *  \param [out] ophi Overlap operator, applied to wave-functions [storage: CPU || GPU].
 *  \param [in]  d_op D-operator representation.
 *  \param [in]  q_op Q-operator representation.
 *
 *  In non-collinear case (ispn = 2) the Hamiltonian and S operator are applied to both components of spinor
 *  wave-functions. Otherwise they are applied to a single component.
 */
template <typename T>
void Hamiltonian::apply_h_s(K_point* kp__,
                            int ispn__,
                            int N__,
                            int n__,
                            Wave_functions& phi__,
                            Wave_functions& hphi__,
                            Wave_functions& sphi__) const
{
    PROFILE("sirius::Hamiltonian::apply_h_s");

    if ((phi__.num_sc() != hphi__.num_sc()) || (phi__.num_sc() != sphi__.num_sc())) {
        TERMINATE("wrong number of spin components");
    }

    double t1 = -omp_get_wtime();

/* for the data remapping we need phi on CPU */
#ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        for (int ispn = 0; ispn < phi__.num_sc(); ispn++) {
            if (phi__.pw_coeffs(ispn).is_remapped() || ctx_.fft_coarse().pu() == CPU) {
                phi__.pw_coeffs(ispn).copy_to_host(N__, n__);
            }
        }
    }
#endif
    /* apply local part of Hamiltonian */
    local_op_->apply_h(ispn__, phi__, hphi__, N__, n__);
#ifdef __GPU
    if (ctx_.processing_unit() == GPU && ctx_.fft_coarse().pu() == CPU) {
        for (int ispn = 0; ispn < phi__.num_sc(); ispn++) {
            hphi__.pw_coeffs(ispn).copy_to_device(N__, n__);
        }
    }
#endif
    t1 += omp_get_wtime();

    if (kp__->comm().rank() == 0 && ctx_.control().print_performance_) {
        DUMP("hloc performace: %12.6f bands/sec", n__ / t1);
    }

    int nsc = (ispn__ == 2) ? 2 : 1;

    if (ctx_.control().print_checksum_) {
        for (int ispn = 0; ispn < nsc; ispn++) {
            auto cs1 = phi__.checksum(ctx_.processing_unit(), ispn, N__, n__);
            auto cs2 = hphi__.checksum(ctx_.processing_unit(), ispn, N__, n__);
            if (kp__->comm().rank() == 0) {
                std::stringstream s;
                s << "phi_" << ispn;
                print_checksum(s.str(), cs1);
                s.str("");
                s << "hphi_" << ispn;
                print_checksum(s.str(), cs2);
            }
        }
    }

    /* set intial sphi */
    for (int ispn = 0; ispn < nsc; ispn++) {
        sphi__.copy_from(ctx_.processing_unit(), n__, phi__, ispn, N__, ispn, N__);
    }
    /* return if there are no beta-projectors */
    if (!ctx_.unit_cell().mt_lo_basis_size()) {
        return;
    }

    for (int i = 0; i < kp__->beta_projectors().num_chunks(); i++) {
        /* generate beta-projectors for a block of atoms */
        kp__->beta_projectors().generate(i);
        /* non-collinear case */
        if (ispn__ == 2) {
            for (int ispn = 0; ispn < 2; ispn++) {

                auto beta_phi = kp__->beta_projectors().inner<T>(i, phi__, ispn, N__, n__);

                /* apply diagonal spin blocks */
                D<T>().apply(i, ispn, hphi__, N__, n__, kp__->beta_projectors(), beta_phi);
                /* apply non-diagonal spin blocks */
                /* xor 3 operator will map 0 to 3 and 1 to 2 */
                D<T>().apply(i, ispn ^ 3, hphi__, N__, n__, kp__->beta_projectors(), beta_phi);

                /* apply Q operator (diagonal in spin) */
                Q<T>().apply(i, ispn, sphi__, N__, n__, kp__->beta_projectors(), beta_phi);
                /* apply non-diagonal spin blocks */
                if (ctx_.so_correction()) {
                    Q<T>().apply(i, ispn ^ 3, sphi__, N__, n__, kp__->beta_projectors(), beta_phi);
                }
            }
        } else { /* non-magnetic or collinear case */

            auto beta_phi = kp__->beta_projectors().inner<T>(i, phi__, ispn__, N__, n__);

            D<T>().apply(i, ispn__, hphi__, N__, n__, kp__->beta_projectors(), beta_phi);
            Q<T>().apply(i, ispn__, sphi__, N__, n__, kp__->beta_projectors(), beta_phi);
        }
    }

    /* apply the hubbard potential if relevant */
    if (ctx_.hubbard_correction() && !ctx_.gamma_point()) {

       // copy the hubbard wave functions on GPU (if needed) and
       // return afterwards, or if they are not already calculated
       // compute the wave functions and copy them on GPU (if needed)

        this->U().generate_atomic_orbitals(*kp__, Q<T>());

	// Apply the hubbard potential and deallocate the hubbard wave
	// functions on GPU (if needed)
        this->U().apply_hubbard_potential(*kp__, N__, n__, phi__, hphi__);

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                kp__->hubbard_wave_functions().deallocate_on_device(ispn);
            }
        }
        #endif
    }

    if (ctx_.control().print_checksum_) {
        for (int ispn = 0; ispn < nsc; ispn++) {
            auto cs1 = hphi__.checksum(ctx_.processing_unit(), ispn, N__, n__);
            auto cs2 = sphi__.checksum(ctx_.processing_unit(), ispn, N__, n__);
            if (kp__->comm().rank() == 0) {
                std::stringstream s;
                s << "hphi_" << ispn;
                print_checksum(s.str(), cs1);
                s.str("");
                s << "sphi_" << ispn;
                print_checksum(s.str(), cs2);
            }
        }
    }
}

inline void Hamiltonian::apply_fv_o(K_point* kp__,
                                    bool apw_only__,
                                    bool add_o1__,
                                    int N__,
                                    int n__,
                                    Wave_functions& phi__,
                                    Wave_functions& ophi__) const
{
    PROFILE("sirius::Hamiltonian::apply_fv_o");

    if (!apw_only__) {
        /* zero the local-orbital part */
        ophi__.mt_coeffs(0).zero<memory_t::host>(N__, n__);
    }

    /* interstitial part */
    local_op_->apply_o(N__, n__, phi__, ophi__);

    matrix<double_complex> alm(kp__->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());
    matrix<double_complex> oalm(kp__->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());
    matrix<double_complex> tmp(unit_cell_.max_mt_aw_basis_size(), n__);

#ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        if (!apw_only__) {
            phi__.mt_coeffs(0).copy_to_host(N__, n__);
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

        sddk::timer t1("sirius::Hamiltonian::apply_fv_o|apw-apw");

        switch (ctx_.processing_unit()) {
            case CPU: {
                /* tmp(lm, i) = A(G, lm)^{T} * C(G, i) */
                linalg<CPU>::gemm(1, 0, naw, n__, kp__->num_gkvec_loc(), alm.at<CPU>(), alm.ld(),
                                  phi__.pw_coeffs(0).prime().at<CPU>(0, N__), phi__.pw_coeffs(0).prime().ld(), tmp.at<CPU>(),
                                  tmp.ld());
                break;
            }
            case GPU: {
#ifdef __GPU
                alm.copy<memory_t::host, memory_t::device>();
                /* tmp(lm, i) = A(G, lm)^{T} * C(G, i) */
                linalg<GPU>::gemm(1, 0, naw, n__, kp__->num_gkvec_loc(), alm.at<GPU>(), alm.ld(),
                                  phi__.pw_coeffs(0).prime().at<GPU>(0, N__), phi__.pw_coeffs(0).prime().ld(), tmp.at<GPU>(),
                                  tmp.ld());
                tmp.copy<memory_t::device, memory_t::host>(naw * n__);
#endif
                break;
            }
        }

        kp__->comm().allreduce(tmp.at<CPU>(), static_cast<int>(tmp.size()));

        for (int xi = 0; xi < naw; xi++) {
            for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++) {
                alm(ig, xi) = std::conj(alm(ig, xi));
            }
        }

        switch (ctx_.processing_unit()) {
            case CPU: {
                /* APW-APW contribution to overlap */
                linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, naw, linalg_const<double_complex>::one(), alm.at<CPU>(),
                                  alm.ld(), tmp.at<CPU>(), tmp.ld(), linalg_const<double_complex>::one(),
                                  ophi__.pw_coeffs(0).prime().at<CPU>(0, N__), ophi__.pw_coeffs(0).prime().ld());
                break;
            }
            case GPU: {
#ifdef __GPU
                alm.copy<memory_t::host, memory_t::device>();
                tmp.copy<memory_t::host, memory_t::device>(naw * n__);
                /* APW-APW contribution to overlap */
                linalg<GPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, naw, &linalg_const<double_complex>::one(),
                                  alm.at<GPU>(), alm.ld(), tmp.at<GPU>(), tmp.ld(), &linalg_const<double_complex>::one(),
                                  ophi__.pw_coeffs(0).prime().at<GPU>(0, N__), ophi__.pw_coeffs(0).prime().ld());
#endif
                break;
            }
        }

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
                            phi__.mt_coeffs(0).prime().at<CPU>(phi__.offset_mt_coeffs(ia_location.local_index), N__ + i),
                            nlo * sizeof(double_complex));
            }
        }
        kp__->comm().bcast(phi_lo_ia.at<CPU>(), static_cast<int>(phi_lo_ia.size()), ia_location.rank);

        sddk::timer t2("sirius::Hamiltonian::apply_fv_o|apw-lo");
        oalm.zero();
        #pragma omp parallel for
        for (int ilo = 0; ilo < nlo; ilo++) {
            int xi_lo = type.mt_aw_basis_size() + ilo;
            /* local orbital indices */
            int l_lo     = type.indexb(xi_lo).l;
            int lm_lo    = type.indexb(xi_lo).lm;
            int order_lo = type.indexb(xi_lo).order;
            /* use oalm as temporary buffer to compute alm*o */
            for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
                for (int igloc = 0; igloc < kp__->num_gkvec_loc(); igloc++) {
                    oalm(igloc, ilo) += alm(igloc, type.indexb_by_lm_order(lm_lo, order_aw)) *
                                        atom.symmetry_class().o_radial_integral(l_lo, order_aw, order_lo);
                }
            }
        }
        linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, nlo, linalg_const<double_complex>::one(), oalm.at<CPU>(),
                          oalm.ld(), phi_lo_ia.at<CPU>(), phi_lo_ia.ld(), linalg_const<double_complex>::one(),
                          ophi__.pw_coeffs(0).prime().at<CPU>(0, N__), ophi__.pw_coeffs(0).prime().ld());
        t2.stop();

        sddk::timer t3("sirius::Hamiltonian::apply_fv_o|lo-apw");
        /* lo-APW contribution */
        for (int i = 0; i < n__; i++) {
            for (int ilo = 0; ilo < type.mt_lo_basis_size(); ilo++) {
                int xi_lo = type.mt_aw_basis_size() + ilo;
                /* local orbital indices */
                int l_lo     = type.indexb(xi_lo).l;
                int lm_lo    = type.indexb(xi_lo).lm;
                int order_lo = type.indexb(xi_lo).order;

                if (ia_location.rank == kp__->comm().rank()) {
                    /* lo-lo contribution */
                    for (int jlo = 0; jlo < type.mt_lo_basis_size(); jlo++) {
                        int xi_lo1 = type.mt_aw_basis_size() + jlo;
                        int lm1    = type.indexb(xi_lo1).lm;
                        int order1 = type.indexb(xi_lo1).order;
                        if (lm_lo == lm1) {
                            ophi__.mt_coeffs(0).prime(ophi__.offset_mt_coeffs(ia_location.local_index) + ilo, N__ + i) +=
                                phi_lo_ia(jlo, i) * atom.symmetry_class().o_radial_integral(l_lo, order_lo, order1);
                        }
                    }

                    for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
                        /* lo-APW contribution */
                        ophi__.mt_coeffs(0).prime(ophi__.offset_mt_coeffs(ia_location.local_index) + ilo, N__ + i) +=
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
        ophi__.mt_coeffs(0).copy_to_device(N__, n__);
    }
#endif

    if (ctx_.control().print_checksum_) {
        auto cs = ophi__.checksum_pw(ctx_.processing_unit(), 0, 0, N__ + n__);
        if (ophi__.comm().rank() == 0) {
            print_checksum("ophi", cs);
        }
    }
}

/* first come the local orbitals, then the singular components, then the auxiliary basis functions */
inline void Hamiltonian::apply_fv_h_o(K_point* kp__,
                                      int nlo__,
                                      int N__,
                                      int n__,
                                      Wave_functions& phi__,
                                      Wave_functions& hphi__,
                                      Wave_functions& ophi__) const
{
    PROFILE("sirius::Hamiltonian::apply_fv_h_o");

    if (N__ == 0) {
        /* zero plane-wave part of pure local orbital basis functions */
        hphi__.pw_coeffs(0).zero<memory_t::host | memory_t::device>(0, nlo__);
        ophi__.pw_coeffs(0).zero<memory_t::host | memory_t::device>(0, nlo__);
    }
    /* zero the local-orbital part */
    hphi__.mt_coeffs(0).zero<memory_t::host | memory_t::device>(N__, n__);
    ophi__.mt_coeffs(0).zero<memory_t::host | memory_t::device>(N__, n__);

    /* interstitial part */
    if (N__ == 0) {
        /* don't apply to the pure local orbital basis functions */
        local_op_->apply_h_o(nlo__, n__ - nlo__, phi__, hphi__, ophi__);
    } else {
        local_op_->apply_h_o(N__, n__, phi__, hphi__, ophi__);
    }

    //if (ctx_.control().print_checksum_) {
    //    auto cs1 = hphi__.checksum_pw(N__, n__, ctx_.processing_unit());
    //    auto cs2 = ophi__.checksum_pw(N__, n__, ctx_.processing_unit());
    //    if (kp__->comm().rank() == 0) {
    //        DUMP("checksum(hphi_istl): %18.10f %18.10f", cs1.real(), cs1.imag());
    //        DUMP("checksum(ophi_istl): %18.10f %18.10f", cs2.real(), cs2.imag());
    //    }
    //}

    matrix<double_complex> alm(kp__->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size(), ctx_.dual_memory_t());
    matrix<double_complex> halm(kp__->num_gkvec_loc(),
                                std::max(unit_cell_.max_mt_aw_basis_size(), unit_cell_.max_mt_lo_basis_size()),
                                ctx_.dual_memory_t());

    matrix<double_complex> tmp1(unit_cell_.max_mt_aw_basis_size(), n__, ctx_.dual_memory_t());
    matrix<double_complex> tmp2(unit_cell_.max_mt_aw_basis_size(), n__, ctx_.dual_memory_t());

// matrix<double_complex> hmt(unit_cell_.max_mt_aw_basis_size(), unit_cell_.max_mt_lo_basis_size());

#ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        phi__.mt_coeffs(0).copy_to_host(N__, n__);
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

        if (ctx_.processing_unit() == GPU) {
            alm.copy<memory_t::host, memory_t::device>();
            halm.copy<memory_t::host, memory_t::device>();
        }

        /* create arrays with proper dimensions from the already allocated chunk of memory */
        matrix<double_complex> alm_phi;
        matrix<double_complex> halm_phi;

        switch (ctx_.processing_unit()) {
            case CPU: {
                alm_phi  = matrix<double_complex>(tmp1.at<CPU>(), naw, n__);
                halm_phi = matrix<double_complex>(tmp2.at<CPU>(), naw, n__);
                break;
            }
            case GPU: {
                alm_phi  = matrix<double_complex>(tmp1.at<CPU>(), tmp1.at<GPU>(), naw, n__);
                halm_phi = matrix<double_complex>(tmp2.at<CPU>(), tmp2.at<GPU>(), naw, n__);
                break;
            }
        }

        sddk::timer t1("sirius::Hamiltonian::apply_fv_h_o|apw-apw");

        if (ctx_.processing_unit() == CPU) {
            /* tmp(lm, i) = A(G, lm)^{T} * C(G, i) */
            linalg<CPU>::gemm(1, 0, naw, n__, kp__->num_gkvec_loc(), alm.at<CPU>(), alm.ld(),
                              phi__.pw_coeffs(0).prime().at<CPU>(0, N__), phi__.pw_coeffs(0).prime().ld(),
                              alm_phi.at<CPU>(), alm_phi.ld());
            /* htmp(lm, i) = H_{mt}A(G, lm)^{T} * C(G, i) */
            linalg<CPU>::gemm(1, 0, naw, n__, kp__->num_gkvec_loc(), halm.at<CPU>(), halm.ld(),
                              phi__.pw_coeffs(0).prime().at<CPU>(0, N__), phi__.pw_coeffs(0).prime().ld(),
                              halm_phi.at<CPU>(), halm_phi.ld());
        }
#ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            /* tmp(lm, i) = A(G, lm)^{T} * C(G, i) */
            linalg<GPU>::gemm(1, 0, naw, n__, kp__->num_gkvec_loc(), alm.at<GPU>(), alm.ld(),
                              phi__.pw_coeffs(0).prime().at<GPU>(0, N__), phi__.pw_coeffs(0).prime().ld(),
                              alm_phi.at<GPU>(), alm_phi.ld());
            /* htmp(lm, i) = H_{mt}A(G, lm)^{T} * C(G, i) */
            linalg<GPU>::gemm(1, 0, naw, n__, kp__->num_gkvec_loc(), halm.at<GPU>(), halm.ld(),
                              phi__.pw_coeffs(0).prime().at<GPU>(0, N__), phi__.pw_coeffs(0).prime().ld(),
                              halm_phi.at<GPU>(), halm_phi.ld());
            alm_phi.copy<memory_t::device, memory_t::host>();
            halm_phi.copy<memory_t::device, memory_t::host>();
        }
#endif

        kp__->comm().allreduce(alm_phi.at<CPU>(), static_cast<int>(alm_phi.size()));
        kp__->comm().allreduce(halm_phi.at<CPU>(), static_cast<int>(halm_phi.size()));

        if (ctx_.processing_unit() == GPU) {
            alm_phi.copy<memory_t::host, memory_t::device>();
            halm_phi.copy<memory_t::host, memory_t::device>();
        }

        for (int xi = 0; xi < naw; xi++) {
            for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++) {
                alm(ig, xi) = std::conj(alm(ig, xi));
            }
        }
        if (ctx_.processing_unit() == GPU) {
            alm.copy<memory_t::host, memory_t::device>();
        }

        if (ctx_.processing_unit() == CPU) {
            /* APW-APW contribution to overlap */
            linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, naw, linalg_const<double_complex>::one(), alm.at<CPU>(),
                              alm.ld(), alm_phi.at<CPU>(), alm_phi.ld(), linalg_const<double_complex>::one(),
                              ophi__.pw_coeffs(0).prime().at<CPU>(0, N__), ophi__.pw_coeffs(0).prime().ld());
            /* APW-APW contribution to Hamiltonian */
            linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, naw, linalg_const<double_complex>::one(), alm.at<CPU>(),
                              alm.ld(), halm_phi.at<CPU>(), halm_phi.ld(), linalg_const<double_complex>::one(),
                              hphi__.pw_coeffs(0).prime().at<CPU>(0, N__), hphi__.pw_coeffs(0).prime().ld());
        }
#ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            /* APW-APW contribution to overlap */
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, naw, &linalg_const<double_complex>::one(),
                              alm.at<GPU>(), alm.ld(), alm_phi.at<GPU>(), alm_phi.ld(),
                              &linalg_const<double_complex>::one(), ophi__.pw_coeffs(0).prime().at<GPU>(0, N__),
                              ophi__.pw_coeffs(0).prime().ld());
            /* APW-APW contribution to Hamiltonian */
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, naw, &linalg_const<double_complex>::one(),
                              alm.at<GPU>(), alm.ld(), halm_phi.at<GPU>(), halm_phi.ld(),
                              &linalg_const<double_complex>::one(), hphi__.pw_coeffs(0).prime().at<GPU>(0, N__),
                              hphi__.pw_coeffs(0).prime().ld());
        }
#endif
        t1.stop();

        if (!nlo) {
            continue;
        }

        /* local orbital coefficients of atom ia for all states */
        matrix<double_complex> phi_lo_ia(nlo, n__, ctx_.dual_memory_t());
        auto ia_location = phi__.spl_num_atoms().location(ia);
        if (ia_location.rank == kp__->comm().rank()) {
            for (int i = 0; i < n__; i++) {
                std::memcpy(&phi_lo_ia(0, i),
                            phi__.mt_coeffs(0).prime().at<CPU>(phi__.offset_mt_coeffs(ia_location.local_index), N__ + i),
                            nlo * sizeof(double_complex));
            }
        }
        kp__->comm().bcast(phi_lo_ia.at<CPU>(), static_cast<int>(phi_lo_ia.size()), ia_location.rank);
        if (ctx_.processing_unit() == GPU) {
            phi_lo_ia.copy<memory_t::host, memory_t::device>();
        }

        matrix<double_complex> hmt(naw, nlo, ctx_.dual_memory_t());

        sddk::timer t2("sirius::Hamiltonian::apply_fv_h_o|apw-lo");
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
                auto& gc     = gaunt_coefs_->gaunt_vector(lm_aw, lm_lo);
                hmt(xi, ilo) = atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_aw, idxrf_lo, gc);
            }
        }
        if (ctx_.processing_unit() == CPU) {
            linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, nlo, linalg_const<double_complex>::one(),
                              halm.at<CPU>(), halm.ld(), phi_lo_ia.at<CPU>(), phi_lo_ia.ld(),
                              linalg_const<double_complex>::one(), ophi__.pw_coeffs(0).prime().at<CPU>(0, N__),
                              ophi__.pw_coeffs(0).prime().ld());

            linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), nlo, naw, alm.at<CPU>(), alm.ld(), hmt.at<CPU>(), hmt.ld(),
                              halm.at<CPU>(), halm.ld());

            linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, nlo, linalg_const<double_complex>::one(),
                              halm.at<CPU>(), halm.ld(), phi_lo_ia.at<CPU>(), phi_lo_ia.ld(),
                              linalg_const<double_complex>::one(), hphi__.pw_coeffs(0).prime().at<CPU>(0, N__),
                              hphi__.pw_coeffs(0).prime().ld());
        }
#ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            halm.copy<memory_t::host, memory_t::device>();
            hmt.copy<memory_t::host, memory_t::device>();
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, nlo, &linalg_const<double_complex>::one(),
                              halm.at<GPU>(), halm.ld(), phi_lo_ia.at<GPU>(), phi_lo_ia.ld(),
                              &linalg_const<double_complex>::one(), ophi__.pw_coeffs(0).prime().at<GPU>(0, N__),
                              ophi__.pw_coeffs(0).prime().ld());

            linalg<GPU>::gemm(0, 0, kp__->num_gkvec_loc(), nlo, naw, alm.at<GPU>(), alm.ld(), hmt.at<GPU>(), hmt.ld(),
                              halm.at<GPU>(), halm.ld());
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, nlo, &linalg_const<double_complex>::one(),
                              halm.at<GPU>(), halm.ld(), phi_lo_ia.at<GPU>(), phi_lo_ia.ld(),
                              &linalg_const<double_complex>::one(), hphi__.pw_coeffs(0).prime().at<GPU>(0, N__),
                              hphi__.pw_coeffs(0).prime().ld());
        }
#endif
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
                        auto& gc   = gaunt_coefs_->gaunt_vector(lm_lo, lm1);
                        if (lm_lo == lm1) {
                            ophi__.mt_coeffs(0).prime(ophi__.offset_mt_coeffs(ia_location.local_index) + ilo, N__ + i) +=
                                phi_lo_ia(jlo, i) * atom.symmetry_class().o_radial_integral(l_lo, order_lo, order1);
                        }
                        hphi__.mt_coeffs(0).prime(hphi__.offset_mt_coeffs(ia_location.local_index) + ilo, N__ + i) +=
                            phi_lo_ia(jlo, i) * atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf1, gc);
                    }

                    for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
                        /* lo-APW contribution */
                        ophi__.mt_coeffs(0).prime(ophi__.offset_mt_coeffs(ia_location.local_index) + ilo, N__ + i) +=
                            atom.symmetry_class().o_radial_integral(l_lo, order_lo, order_aw) *
                            alm_phi(type.indexb_by_lm_order(lm_lo, order_aw), i);
                    }

                    double_complex z(0, 0);
                    for (int xi = 0; xi < naw; xi++) {
                        int lm_aw    = type.indexb(xi).lm;
                        int idxrf_aw = type.indexb(xi).idxrf;
                        auto& gc     = gaunt_coefs_->gaunt_vector(lm_lo, lm_aw);
                        z += atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf_aw, gc) * alm_phi(xi, i);
                    }
                    /* lo-APW contribution */
                    hphi__.mt_coeffs(0).prime(hphi__.offset_mt_coeffs(ia_location.local_index) + ilo, N__ + i) += z;
                }
            }
        }
        t3.stop();
    }

#ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        hphi__.mt_coeffs(0).copy_to_device(N__, n__);
        ophi__.mt_coeffs(0).copy_to_device(N__, n__);
    }
#endif

    //if (ctx_.control().print_checksum_) {
    //    auto cs1 = hphi__.checksum_pw(N__, n__, ctx_.processing_unit());
    //    auto cs2 = ophi__.checksum_pw(N__, n__, ctx_.processing_unit());
    //    if (kp__->comm().rank() == 0) {
    //        DUMP("checksum(hphi_pw): %18.10f %18.10f", cs1.real(), cs1.imag());
    //        DUMP("checksum(ophi_pw): %18.10f %18.10f", cs2.real(), cs2.imag());
    //    }
    //}

    //if (ctx_.control().print_checksum_) {
    //    auto cs1 = hphi__.checksum(N__, n__);
    //    auto cs2 = ophi__.checksum(N__, n__);
    //    if (kp__->comm().rank() == 0) {
    //        DUMP("checksum(hphi): %18.10f %18.10f", cs1.real(), cs1.imag());
    //        DUMP("checksum(ophi): %18.10f %18.10f", cs2.real(), cs2.imag());
    //    }
    //}
}

// TODO: port to GPU
inline void Hamiltonian::apply_magnetic_field(K_point*                     kp__,
                                              Wave_functions&              fv_states__,
                                              std::vector<Wave_functions>& hpsi__) const
{
    PROFILE("sirius::Hamiltonian::apply_magnetic_field");

    assert(hpsi__.size() == 2 || hpsi__.size() == 3);

    local_op_->apply_b(0, ctx_.num_fv_states(), fv_states__, hpsi__);

    mdarray<double_complex, 3> zm(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(), ctx_.num_mag_dims());

    for (int ialoc = 0; ialoc < fv_states__.spl_num_atoms().local_size(); ialoc++) {
        int ia            = fv_states__.spl_num_atoms()[ialoc];
        auto& atom        = unit_cell_.atom(ia);
        int offset        = fv_states__.offset_mt_coeffs(ialoc);
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
                          zm.at<CPU>(),
                          zm.ld(),
                          fv_states__.mt_coeffs(0).prime().at<CPU>(offset, 0),
                          fv_states__.mt_coeffs(0).prime().ld(),
                          linalg_const<double_complex>::zero(),
                          hpsi__[0].mt_coeffs(0).prime().at<CPU>(offset, 0),
                          hpsi__[0].mt_coeffs(0).prime().ld());

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
                              zm.at<CPU>(),
                              zm.ld(),
                              fv_states__.mt_coeffs(0).prime().at<CPU>(offset, 0),
                              fv_states__.mt_coeffs(0).prime().ld(),
                              hpsi__[2].mt_coeffs(0).prime().at<CPU>(offset, 0),
                              hpsi__[2].mt_coeffs(0).prime().ld());
        }
    }

    /* copy Bz|\psi> to -Bz|\psi> */
    for (int i = 0; i < ctx_.num_fv_states(); i++) {
        for (int j = 0; j < fv_states__.pw_coeffs(0).num_rows_loc(); j++) {
            hpsi__[1].pw_coeffs(0).prime(j, i) = -hpsi__[0].pw_coeffs(0).prime(j, i);
        }
        for (int j = 0; j < fv_states__.mt_coeffs(0).num_rows_loc(); j++) {
            hpsi__[1].mt_coeffs(0).prime(j, i) = -hpsi__[0].mt_coeffs(0).prime(j, i);
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
