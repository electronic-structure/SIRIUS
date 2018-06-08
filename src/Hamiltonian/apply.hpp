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
        this->U().apply_hubbard_potential(*kp__, ispn__, N__, n__, phi__, hphi__);

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

inline void Hamiltonian::apply_fv_h_o(K_point*        kp__,
                                      bool            apw_only__,
                                      bool            phi_is_lo__,
                                      int             N__,
                                      int             n__,
                                      Wave_functions& phi__,
                                      Wave_functions* hphi__,
                                      Wave_functions* ophi__) const
{
    PROFILE("sirius::Hamiltonian::apply_fv_h_o");

    /* trivial case */
    if (hphi__ == nullptr && ophi__ == nullptr) {
        return;
    }

    if (!apw_only__) {
        if (hphi__ != nullptr) {
            /* zero the local-orbital part */
            hphi__->mt_coeffs(0).zero<memory_t::host>(N__, n__);
        }
        if (ophi__ != nullptr) {
            /* zero the local-orbital part */
            ophi__->mt_coeffs(0).zero<memory_t::host>(N__, n__);
        }
    }

    if (!phi_is_lo__) {
        /* interstitial part */
        local_op_->apply_h_o(N__, n__, phi__, hphi__, ophi__);
    } else {
        /* zero the APW part */
        switch (ctx_.processing_unit()) {
            case CPU: {
                if (hphi__ != nullptr) {
                    hphi__->pw_coeffs(0).zero<memory_t::host>(N__, n__);
                }
                if (ophi__ != nullptr) {
                    ophi__->pw_coeffs(0).zero<memory_t::host>(N__, n__);
                }
                break;
            }
            case GPU: {
                if (hphi__ != nullptr) {
                    hphi__->pw_coeffs(0).zero<memory_t::device>(N__, n__);
                }
                if (ophi__ != nullptr) {
                    ophi__->pw_coeffs(0).zero<memory_t::device>(N__, n__);
                }
                break;
            }
        }
    }

#if defined(__GPU)
    if (ctx_.processing_unit() == GPU && !apw_only__) {
        phi__.mt_coeffs(0).copy_to_host(N__, n__);
    }
#endif

    /* short name for local number of G+k vectors */
    int ngv = kp__->num_gkvec_loc();

    /* split atoms in blocks */
    int num_atoms_in_block = 2 * omp_get_max_threads();
    int nblk = utils::num_blocks(unit_cell_.num_atoms(), num_atoms_in_block);

    /* maximum number of AW radial functions in a block of atoms */
    int max_mt_aw = num_atoms_in_block * unit_cell_.max_mt_aw_basis_size();
    /* maximum number of LO radial functions in a block of atoms */
    int max_mt_lo = num_atoms_in_block * unit_cell_.max_mt_lo_basis_size();

    /* matching coefficients for a block of atoms */
    matrix<double_complex> alm_block;
    matrix<double_complex> halm_block;

    switch (ctx_.processing_unit()) {
        case CPU: {
            alm_block = matrix<double_complex>(ngv, max_mt_aw, memory_t::host);
            if (hphi__ != nullptr) {
                halm_block = matrix<double_complex>(ngv, std::max(max_mt_aw, max_mt_lo), memory_t::host);
            }
            break;
        }
        case GPU: {
            alm_block = matrix<double_complex>(ngv, max_mt_aw, memory_t::host_pinned | memory_t::device);
            if (hphi__ != nullptr) {
                halm_block = matrix<double_complex>(ngv, std::max(max_mt_aw, max_mt_lo),
                                                    memory_t::host_pinned | memory_t::device);
            }
            break;
        }
    }
    /* buffers for alm_phi and halm_phi */
    mdarray<double_complex, 1> alm_phi_buf;
    if (ophi__ != nullptr) {
        alm_phi_buf = mdarray<double_complex, 1>(max_mt_aw * n__, ctx_.dual_memory_t());
    }
    mdarray<double_complex, 1> halm_phi_buf;
    if (hphi__ != nullptr) {
        halm_phi_buf = mdarray<double_complex, 1>(max_mt_aw * n__, ctx_.dual_memory_t());
    }

    auto generate_alm = [&](int atom_begin, int atom_end, std::vector<int>& offsets_aw)
    {
        utils::timer t1("sirius::Hamiltonian::apply_fv_o|alm");
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for (int ia = atom_begin; ia < atom_end; ia++) {
                if (ia % omp_get_num_threads() == tid) {
                    int ialoc = ia - atom_begin;
                    auto& atom = unit_cell_.atom(ia);
                    auto& type = atom.type();

                    /* wrapper for matching coefficients for a given atom */
                    mdarray<double_complex, 2> alm_tmp;
                    mdarray<double_complex, 2> halm_tmp;
                    switch (ctx_.processing_unit()) {
                        case CPU: {
                            alm_tmp = mdarray<double_complex, 2>(alm_block.at<CPU>(0, offsets_aw[ialoc]),
                                                                 ngv, type.mt_aw_basis_size());
                            if (hphi__ != nullptr) {
                                halm_tmp = mdarray<double_complex, 2>(halm_block.at<CPU>(0, offsets_aw[ialoc]),
                                                                      ngv, type.mt_aw_basis_size());
                            }
                            break;
                        }
                        case GPU: {
                            alm_tmp = mdarray<double_complex, 2>(alm_block.at<CPU>(0, offsets_aw[ialoc]),
                                                                 alm_block.at<GPU>(0, offsets_aw[ialoc]),
                                                                 ngv, type.mt_aw_basis_size());
                            if (hphi__ != nullptr) {
                                halm_tmp = mdarray<double_complex, 2>(halm_block.at<CPU>(0, offsets_aw[ialoc]),
                                                                      halm_block.at<GPU>(0, offsets_aw[ialoc]),
                                                                      ngv, type.mt_aw_basis_size());
                            }
                            break;
                        }
                    }

                    /* generate LAPW matching coefficients on the CPU */
                    kp__->alm_coeffs_loc().generate(ia, alm_tmp);
                    /* conjugate alm */
                    for (int xi = 0; xi < type.mt_aw_basis_size(); xi++) {
                        for (int igk = 0; igk < ngv; igk++) {
                            alm_tmp(igk, xi) = std::conj(alm_tmp(igk, xi));
                        }
                    }
#if defined(__GPU)
                    if (ctx_.processing_unit() == GPU) {
                        alm_tmp.async_copy<memory_t::host, memory_t::device>(tid);
                    }
#endif
                    if (hphi__ != nullptr) {
                        apply_hmt_to_apw<spin_block_t::nm>(atom, ngv, alm_tmp, halm_tmp);
#if defined(__GPU)
                        if (ctx_.processing_unit() == GPU) {
                            halm_tmp.async_copy<memory_t::host, memory_t::device>(tid);
                        }
#endif
                    }
                }
            }
#if defined(__GPU)
            if (ctx_.processing_unit() == GPU) {
                acc::sync_stream(tid);
            }
#endif
        }
    };

    auto compute_alm_phi = [&](matrix<double_complex>& alm_phi, matrix<double_complex>& halm_phi, int num_mt_aw)
    {
        utils::timer t1("sirius::Hamiltonian::apply_fv_o|alm_phi");

        /* first zgemm: A(G, lm)^{T} * C(G, i) and  hA(G, lm)^{T} * C(G, i) */
        switch (ctx_.processing_unit()) {
            case CPU: {
                if (ophi__ != nullptr) {
                    /* create resulting array with proper dimensions from the already allocated chunk of memory */
                    alm_phi = matrix<double_complex>(alm_phi_buf.at<CPU>(), num_mt_aw, n__);
                    /* alm_phi(lm, i) = A(G, lm)^{T} * C(G, i), remember that Alm was conjugated */
                    linalg<CPU>::gemm(2, 0, num_mt_aw, n__, ngv,
                                      alm_block.at<CPU>(), alm_block.ld(),
                                      phi__.pw_coeffs(0).prime().at<CPU>(0, N__), phi__.pw_coeffs(0).prime().ld(),
                                      alm_phi.at<CPU>(), alm_phi.ld());
                }
                if (hphi__ != nullptr) {
                    /* create resulting array with proper dimensions from the already allocated chunk of memory */
                    halm_phi = matrix<double_complex>(halm_phi_buf.at<CPU>(), num_mt_aw, n__);
                    /* halm_phi(lm, i) = H_{mt}A(G, lm)^{T} * C(G, i) */
                    linalg<CPU>::gemm(2, 0, num_mt_aw, n__, ngv,
                                      halm_block.at<CPU>(), halm_block.ld(),
                                      phi__.pw_coeffs(0).prime().at<CPU>(0, N__), phi__.pw_coeffs(0).prime().ld(),
                                      halm_phi.at<CPU>(), halm_phi.ld());
                }
                break;
            }
            case GPU: {
#if defined(__GPU)
                if (ophi__ != nullptr) {
                    /* create resulting array with proper dimensions from the already allocated chunk of memory */
                    alm_phi = matrix<double_complex>(alm_phi_buf.at<CPU>(), alm_phi_buf.at<GPU>(), num_mt_aw, n__);
                    /* alm_phi(lm, i) = A(G, lm)^{T} * C(G, i) */
                    linalg<GPU>::gemm(2, 0, num_mt_aw, n__, ngv,
                                      alm_block.at<GPU>(), alm_block.ld(),
                                      phi__.pw_coeffs(0).prime().at<GPU>(0, N__), phi__.pw_coeffs(0).prime().ld(),
                                      alm_phi.at<GPU>(), alm_phi.ld());
                    alm_phi.copy<memory_t::device, memory_t::host>();
                }
                if (hphi__ != nullptr) {
                    /* create resulting array with proper dimensions from the already allocated chunk of memory */
                    halm_phi = matrix<double_complex>(halm_phi_buf.at<CPU>(), halm_phi_buf.at<GPU>(), num_mt_aw, n__);
                    /* halm_phi(lm, i) = H_{mt}A(G, lm)^{T} * C(G, i) */
                    linalg<GPU>::gemm(2, 0, num_mt_aw, n__, ngv,
                                      halm_block.at<GPU>(), halm_block.ld(),
                                      phi__.pw_coeffs(0).prime().at<GPU>(0, N__), phi__.pw_coeffs(0).prime().ld(),
                                      halm_phi.at<GPU>(), halm_phi.ld());

                    halm_phi.copy<memory_t::device, memory_t::host>();
                }
#endif
                break;
            }
        }

        if (hphi__ != nullptr) {
            kp__->comm().allreduce(halm_phi.at<CPU>(), num_mt_aw * n__);
            if (ctx_.processing_unit() == GPU) {
                halm_phi.copy<memory_t::host, memory_t::device>();
            }
        }

        if (ophi__ != nullptr) {
            kp__->comm().allreduce(alm_phi.at<CPU>(), num_mt_aw * n__);
            if (ctx_.processing_unit() == GPU) {
                alm_phi.copy<memory_t::host, memory_t::device>();
            }
        }
    };

    auto compute_apw_apw = [&](matrix<double_complex>& alm_phi, matrix<double_complex>& halm_phi, int num_mt_aw)
    {
        utils::timer t1("sirius::Hamiltonian::apply_fv_o|apw-apw");
        /* second zgemm: Alm^{*} (Alm * C) */
        switch (ctx_.processing_unit()) {
            case CPU: {
                if (ophi__ != nullptr) {
                    /* APW-APW contribution to overlap */
                    linalg<CPU>::gemm(0, 0, ngv, n__, num_mt_aw,
                                      linalg_const<double_complex>::one(),
                                      alm_block.at<CPU>(), alm_block.ld(),
                                      alm_phi.at<CPU>(), alm_phi.ld(),
                                      linalg_const<double_complex>::one(),
                                      ophi__->pw_coeffs(0).prime().at<CPU>(0, N__), ophi__->pw_coeffs(0).prime().ld());

                }
                if (hphi__ != nullptr) {
                    /* APW-APW contribution to Hamiltonian */
                    linalg<CPU>::gemm(0, 0, ngv, n__, num_mt_aw,
                                      linalg_const<double_complex>::one(),
                                      alm_block.at<CPU>(), alm_block.ld(),
                                      halm_phi.at<CPU>(), halm_phi.ld(),
                                      linalg_const<double_complex>::one(),
                                      hphi__->pw_coeffs(0).prime().at<CPU>(0, N__), hphi__->pw_coeffs(0).prime().ld());
                }
                break;
            }
            case GPU: {
#if defined(__GPU)
                if (ophi__ != nullptr) {
                    /* APW-APW contribution to overlap */
                    linalg<GPU>::gemm(0, 0, ngv, n__, num_mt_aw,
                                      &linalg_const<double_complex>::one(),
                                      alm_block.at<GPU>(), alm_block.ld(),
                                      alm_phi.at<GPU>(), alm_phi.ld(),
                                      &linalg_const<double_complex>::one(),
                                      ophi__->pw_coeffs(0).prime().at<GPU>(0, N__), ophi__->pw_coeffs(0).prime().ld());

                }
                if (hphi__ != nullptr) {
                    /* APW-APW contribution to Hamiltonian */
                    linalg<GPU>::gemm(0, 0, ngv, n__, num_mt_aw,
                                      &linalg_const<double_complex>::one(),
                                      alm_block.at<GPU>(), alm_block.ld(),
                                      halm_phi.at<GPU>(), halm_phi.ld(),
                                      &linalg_const<double_complex>::one(),
                                      hphi__->pw_coeffs(0).prime().at<GPU>(0, N__), hphi__->pw_coeffs(0).prime().ld());
                }
#endif
                break;
            }
        }
    };

    auto collect_lo = [&](int atom_begin, int atom_end, std::vector<int>& offsets_lo, matrix<double_complex>& phi_lo_block)
    {
        utils::timer t1("sirius::Hamiltonian::apply_fv_o|phi_lo");
        /* broadcast local orbital coefficients */
        for (int ia = atom_begin; ia < atom_end; ia++) {
            int ialoc = ia - atom_begin;
            auto& atom = unit_cell_.atom(ia);
            auto& type = atom.type();
            auto ia_location = phi__.spl_num_atoms().location(ia);
            
            /* lo coefficients for a given atom and all bands */
            matrix<double_complex> phi_lo_ia(type.mt_lo_basis_size(), n__);

            if (ia_location.rank == kp__->comm().rank()) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n__; i++) {
                    std::memcpy(&phi_lo_ia(0, i),
                                phi__.mt_coeffs(0).prime().at<CPU>(phi__.offset_mt_coeffs(ia_location.local_index), N__ + i),
                                type.mt_lo_basis_size() * sizeof(double_complex));
                }
            }
            /* broadcast from a rank */
            kp__->comm().bcast(phi_lo_ia.at<CPU>(), static_cast<int>(phi_lo_ia.size()), ia_location.rank);
            /* wrtite into a proper position in a block */
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n__; i++) {
                std::memcpy(&phi_lo_block(offsets_lo[ialoc], i), &phi_lo_ia(0, i),
                            type.mt_lo_basis_size() * sizeof(double_complex));
            }
        } // ia

        if (ctx_.processing_unit() == GPU) {
            phi_lo_block.copy<memory_t::host, memory_t::device>();
        }
    };

    auto compute_apw_lo = [&](int atom_begin, int atom_end, int num_mt_lo,
                              std::vector<int>& offsets_aw, std::vector<int> offsets_lo,
                              matrix<double_complex>& phi_lo_block)
    {
        utils::timer t1("sirius::Hamiltonian::apply_fv_o|apw-lo");
        /* apw-lo block for hphi */
        if (hphi__ != nullptr) {
            for (int ia = atom_begin; ia < atom_end; ia++) {
                int ialoc = ia - atom_begin;
                auto& atom = unit_cell_.atom(ia);
                auto& type = atom.type();
                int naw = type.mt_aw_basis_size();
                int nlo = type.mt_lo_basis_size();

                matrix<double_complex> hmt(naw, nlo, ctx_.dual_memory_t());
                #pragma omp parallel for schedule(static)
                for (int ilo = 0; ilo < nlo; ilo++) {
                    int xi_lo = naw + ilo;
                    /* local orbital indices */
                    int lm_lo    = type.indexb(xi_lo).lm;
                    int idxrf_lo = type.indexb(xi_lo).idxrf;
                    for (int xi = 0; xi < naw; xi++) {
                        int lm_aw    = type.indexb(xi).lm;
                        int idxrf_aw = type.indexb(xi).idxrf;
                        auto& gc     = gaunt_coefs_->gaunt_vector(lm_aw, lm_lo);
                        hmt(xi, ilo) = atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_aw, idxrf_lo, gc);
                    }
                }
                switch (ctx_.processing_unit()) {
                    case CPU: {
                        linalg<CPU>::gemm(0, 0, ngv, nlo, naw,
                                          alm_block.at<CPU>(0, offsets_aw[ialoc]), alm_block.ld(),
                                          hmt.at<CPU>(), hmt.ld(),
                                          halm_block.at<CPU>(0, offsets_lo[ialoc]), halm_block.ld());
                        break;
                    }
                    case GPU: {
#if defined(__GPU)
                        hmt.copy<memory_t::host, memory_t::device>();
                        linalg<GPU>::gemm(0, 0, ngv, nlo, naw,
                                          alm_block.at<GPU>(0, offsets_aw[ialoc]), alm_block.ld(),
                                          hmt.at<GPU>(), hmt.ld(),
                                          halm_block.at<GPU>(0, offsets_lo[ialoc]), halm_block.ld());
#endif
                        break;

                    }
                }
            } //ia
            switch (ctx_.processing_unit()) {
                case CPU: {
                    linalg<CPU>::gemm(0, 0, ngv, n__, num_mt_lo,
                                      linalg_const<double_complex>::one(),
                                      halm_block.at<CPU>(), halm_block.ld(),
                                      phi_lo_block.at<CPU>(), phi_lo_block.ld(),
                                      linalg_const<double_complex>::one(),
                                      hphi__->pw_coeffs(0).prime().at<CPU>(0, N__), hphi__->pw_coeffs(0).prime().ld());
                    break;

                }
                case GPU: {
#if defined(__GPU)
                    linalg<GPU>::gemm(0, 0, ngv, n__, num_mt_lo,
                                      &linalg_const<double_complex>::one(),
                                      halm_block.at<GPU>(), halm_block.ld(),
                                      phi_lo_block.at<GPU>(), phi_lo_block.ld(),
                                      &linalg_const<double_complex>::one(),
                                      hphi__->pw_coeffs(0).prime().at<GPU>(0, N__), hphi__->pw_coeffs(0).prime().ld());
#endif
                    break;
                }
            }
        }

        /* apw-lo block for ophi */
        if (ophi__ != nullptr) {
            halm_block.zero();
            for (int ia = atom_begin; ia < atom_end; ia++) {
                int ialoc = ia - atom_begin;
                auto& atom = unit_cell_.atom(ia);
                auto& type = atom.type();
                int naw = type.mt_aw_basis_size();
                int nlo = type.mt_lo_basis_size();

                matrix<double_complex> hmt(naw, nlo, ctx_.dual_memory_t());
                #pragma omp parallel for schedule(static)
                for (int ilo = 0; ilo < nlo; ilo++) {
                    int xi_lo = naw + ilo;
                    /* local orbital indices */
                    int l_lo     = type.indexb(xi_lo).l;
                    int lm_lo    = type.indexb(xi_lo).lm;
                    int order_lo = type.indexb(xi_lo).order;
                    /* use halm as temporary buffer to compute alm*o */
                    for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
                        for (int igloc = 0; igloc < ngv; igloc++) {
                            halm_block(igloc, offsets_lo[ialoc] + ilo) +=
                                alm_block(igloc, offsets_aw[ialoc] + type.indexb_by_lm_order(lm_lo, order_aw)) *
                                atom.symmetry_class().o_radial_integral(l_lo, order_aw, order_lo);
                        } // TODO: block copy to GPU
                    }
                }
            } //ia
            switch (ctx_.processing_unit()) {
                case CPU: {
                    linalg<CPU>::gemm(0, 0, ngv, n__, num_mt_lo,
                                      linalg_const<double_complex>::one(),
                                      halm_block.at<CPU>(), halm_block.ld(),
                                      phi_lo_block.at<CPU>(), phi_lo_block.ld(),
                                      linalg_const<double_complex>::one(),
                                      ophi__->pw_coeffs(0).prime().at<CPU>(0, N__), ophi__->pw_coeffs(0).prime().ld());
                    break;

                }
                case GPU: {
#if defined(__GPU)
                    halm_block.copy<memory_t::host, memory_t::device>(ngv * num_mt_lo);
                    linalg<GPU>::gemm(0, 0, ngv, n__, num_mt_lo,
                                      &linalg_const<double_complex>::one(),
                                      halm_block.at<GPU>(), halm_block.ld(),
                                      phi_lo_block.at<GPU>(), phi_lo_block.ld(),
                                      &linalg_const<double_complex>::one(),
                                      ophi__->pw_coeffs(0).prime().at<GPU>(0, N__), ophi__->pw_coeffs(0).prime().ld());
#endif
                    break;
                }
            }
        }
    };

    /* loop over blocks of atoms */
    for (int iblk = 0; iblk < nblk; iblk++) {
        int atom_begin = iblk * num_atoms_in_block;
        int atom_end = std::min(unit_cell_.num_atoms(), (iblk + 1) * num_atoms_in_block);
        /* actual number of AW radial functions in a block of atoms */
        int num_mt_aw{0};
        /* actual number of local orbitals in a block of atoms */
        int num_mt_lo{0};
        std::vector<int> offsets_aw(num_atoms_in_block);
        std::vector<int> offsets_lo(num_atoms_in_block);
        for (int ia = atom_begin; ia < atom_end; ia++) {
            int ialoc = ia - atom_begin;
            auto& atom = unit_cell_.atom(ia);
            auto& type = atom.type();
            offsets_aw[ialoc] = num_mt_aw;
            offsets_lo[ialoc] = num_mt_lo;
            num_mt_aw += type.mt_aw_basis_size();
            num_mt_lo += type.mt_lo_basis_size();
        }

        /* created alm and halm for a block of atoms */
        generate_alm(atom_begin, atom_end, offsets_aw);

        matrix<double_complex> alm_phi;
        matrix<double_complex> halm_phi;

        compute_alm_phi(alm_phi, halm_phi, num_mt_aw);

        if (!phi_is_lo__) {
            compute_apw_apw(alm_phi, halm_phi, num_mt_aw);
        }
 
        if (!apw_only__) {
            /* local orbital coefficients for a block of atoms and all states */
            matrix<double_complex> phi_lo_block(num_mt_lo, n__, ctx_.dual_memory_t());
            collect_lo(atom_begin, atom_end, offsets_lo, phi_lo_block);

            compute_apw_lo(atom_begin, atom_end, num_mt_lo, offsets_aw, offsets_lo, phi_lo_block);

            utils::timer t3("sirius::Hamiltonian::apply_fv_o|lo-lo-apw");
            /* lo-APW contribution */
            for (int ia = atom_begin; ia < atom_end; ia++) {
                int ialoc = ia - atom_begin;
                auto& atom = unit_cell_.atom(ia);
                auto& type = atom.type();
                
                auto ia_location = phi__.spl_num_atoms().location(ia);

                if (ia_location.rank == kp__->comm().rank()) {
                    int offset_mt_coeffs = phi__.offset_mt_coeffs(ia_location.local_index);

                    #pragma omp parallel for schedule(static)
                    for (int ilo = 0; ilo < type.mt_lo_basis_size(); ilo++) {
                        int xi_lo = type.mt_aw_basis_size() + ilo;
                        /* local orbital indices */
                        int l_lo     = type.indexb(xi_lo).l;
                        int lm_lo    = type.indexb(xi_lo).lm;
                        int order_lo = type.indexb(xi_lo).order;
                        int idxrf_lo = type.indexb(xi_lo).idxrf;

                        /* lo-lo contribution */
                        for (int jlo = 0; jlo < type.mt_lo_basis_size(); jlo++) {
                            int xi_lo1 = type.mt_aw_basis_size() + jlo;
                            int lm1    = type.indexb(xi_lo1).lm;
                            int order1 = type.indexb(xi_lo1).order;
                            int idxrf1 = type.indexb(xi_lo1).idxrf;
                            if (lm_lo == lm1 && ophi__ != nullptr) {
                                for (int i = 0; i < n__; i++) {
                                    ophi__->mt_coeffs(0).prime(offset_mt_coeffs + ilo, N__ + i) +=
                                        phi_lo_block(offsets_lo[ialoc] + jlo, i) *
                                        atom.symmetry_class().o_radial_integral(l_lo, order_lo, order1);
                                }
                            }
                            if (hphi__ != nullptr) {
                                auto& gc = gaunt_coefs_->gaunt_vector(lm_lo, lm1);
                                for (int i = 0; i < n__; i++) {
                                    hphi__->mt_coeffs(0).prime(offset_mt_coeffs + ilo, N__ + i) +=
                                        phi_lo_block(offsets_lo[ialoc] + jlo, i) *
                                        atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf1, gc);
                                }
                            }
                        }

                        /* lo-APW contribution */
                        if (!phi_is_lo__) {
                            if (ophi__ != nullptr) {
                                for (int i = 0; i < n__; i++) {
                                    /* lo-APW contribution to ophi */
                                    for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
                                        ophi__->mt_coeffs(0).prime(offset_mt_coeffs + ilo, N__ + i) +=
                                            atom.symmetry_class().o_radial_integral(l_lo, order_lo, order_aw) *
                                            alm_phi(offsets_aw[ialoc] + type.indexb_by_lm_order(lm_lo, order_aw), i);
                                    }
                                }
                            }

                            if (hphi__ != nullptr) {
                                for (int i = 0; i < n__; i++) {
                                    double_complex z(0, 0);
                                    for (int xi = 0; xi < type.mt_aw_basis_size(); xi++) {
                                        int lm_aw    = type.indexb(xi).lm;
                                        int idxrf_aw = type.indexb(xi).idxrf;
                                        auto& gc     = gaunt_coefs_->gaunt_vector(lm_lo, lm_aw);
                                        z += atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf_aw, gc) *
                                             alm_phi(offsets_aw[ialoc] + xi, i);
                                    }
                                    /* lo-APW contribution to hphi */
                                    hphi__->mt_coeffs(0).prime(offset_mt_coeffs + ilo, N__ + i) += z;
                                }
                            }
                        }
                    }
                }
            }
            t3.stop();
        }
    }
#if defined(__GPU)
    if (ctx_.processing_unit() == GPU && !apw_only__) {
        if (hphi__ != nullptr) {
            hphi__->mt_coeffs(0).copy_to_device(N__, n__);
        }
        if (ophi__ != nullptr) {
            ophi__->mt_coeffs(0).copy_to_device(N__, n__);
        }
    }
#endif
    if (ctx_.control().print_checksum_) {
        if (hphi__) {
            auto cs1 = hphi__->checksum_pw(ctx_.processing_unit(), 0, N__, n__);
            auto cs2 = hphi__->checksum(ctx_.processing_unit(), 0, N__, n__);
            if (kp__->comm().rank() == 0) {
                print_checksum("hphi_pw", cs1);
                print_checksum("hphi", cs2);
            }
        }
        if (ophi__) {
            auto cs1 = ophi__->checksum_pw(ctx_.processing_unit(), 0, N__, n__);
            auto cs2 = ophi__->checksum(ctx_.processing_unit(), 0, N__, n__);
            if (kp__->comm().rank() == 0) {
                print_checksum("ophi_pw", cs1);
                print_checksum("ophi", cs2);
            }
        }
    }
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
