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
    PROFILE_WITH_TIMER("sirius::Band::apply_h");

    /* set initial hphi */
    hphi__.copy_from(phi__, N__, n__);

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) hphi__.copy_to_host(N__, n__);
    #endif
    /* apply local part of Hamiltonian */
    h_op.apply(ispn__, hphi__, N__, n__);
    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) hphi__.copy_to_device(N__, n__);
    #endif

    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) phi__.copy_to_host(N__, n__);
        #endif
        auto cs1 = mdarray<double_complex, 1>(&phi__(0, N__), kp__->num_gkvec_loc() * n__).checksum();
        auto cs2 = mdarray<double_complex, 1>(&hphi__(0, N__), kp__->num_gkvec_loc() * n__).checksum();
        kp__->comm().allreduce(&cs1, 1);
        kp__->comm().allreduce(&cs2, 1);
        DUMP("checksum(phi): %18.10f %18.10f", cs1.real(), cs1.imag());
        DUMP("checksum(hloc_phi): %18.10f %18.10f", cs2.real(), cs2.imag());
    }
    #endif

    if (!ctx_.unit_cell().mt_lo_basis_size()) {
        return;
    }

    for (int i = 0; i < kp__->beta_projectors().num_beta_chunks(); i++)
    {
        kp__->beta_projectors().generate(i);

        kp__->beta_projectors().inner<T>(i, phi__, N__, n__);

        if (!ctx_.iterative_solver_input_section().real_space_prj_)
        {
            d_op.apply(i, ispn__, hphi__, N__, n__);
        }
        else
        {
            STOP();
            //add_nl_h_o_rs(kp__, n__, phi, hphi, ophi, packed_mtrx_offset__, d_mtrx_packed__, q_mtrx_packed__, kappa__);
        }
    }
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
    PROFILE_WITH_TIMER("sirius::Band::apply_h_o");
    
    double t1 = -runtime::wtime();
    /* set initial hphi */
    hphi__.copy_from(phi__, N__, n__);

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU && !ctx_.fft_coarse().gpu_only()) {
        hphi__.copy_to_host(N__, n__);
    }
    #endif
    /* apply local part of Hamiltonian */
    h_op.apply(ispn__, hphi__, N__, n__);
    #ifdef __GPU
    if (ctx_.processing_unit() == GPU && !ctx_.fft_coarse().gpu_only()) {
        hphi__.copy_to_device(N__, n__);
    }
    #endif
    t1 += runtime::wtime();

    if (kp__->comm().rank() == 0) {
        DUMP("hloc performace: %12.6f bands/sec", n__ / t1);
    }

    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            phi__.copy_to_host(N__, n__);
            if (ctx_.fft_coarse().gpu_only()) {
                hphi__.copy_to_host(N__, n__);
            }
        }
        #endif
        auto cs1 = mdarray<double_complex, 1>(&phi__(0, N__), kp__->num_gkvec_loc() * n__).checksum();
        auto cs2 = mdarray<double_complex, 1>(&hphi__(0, N__), kp__->num_gkvec_loc() * n__).checksum();
        kp__->comm().allreduce(&cs1, 1);
        kp__->comm().allreduce(&cs2, 1);
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

inline void Band::apply_fv_h_o(K_point* kp__,
                               Periodic_function<double>* effective_potential__,
                               int N__,
                               int n__,
                               wave_functions& phi__,
                               wave_functions& hphi__,
                               wave_functions& ophi__) const
{
    PROFILE_WITH_TIMER("sirius::Band::apply_fv_h_o");

    ctx_.fft().prepare(kp__->gkvec().partition());

    mdarray<double_complex, 1> buf_rg(ctx_.fft().local_size());
    mdarray<double_complex, 1> buf_pw(kp__->gkvec().partition().gvec_count_fft());

    hphi__.copy_from(phi__, N__, n__);
    ophi__.copy_from(phi__, N__, n__);

     phi__.pw_coeffs().remap_forward(N__, n__, kp__->gkvec().partition(), ctx_.mpi_grid_fft().communicator(1 << 1));
    hphi__.pw_coeffs().set_num_extra(n__, kp__->gkvec().partition(), ctx_.mpi_grid_fft().communicator(1 << 1), N__);
    ophi__.pw_coeffs().set_num_extra(n__, kp__->gkvec().partition(), ctx_.mpi_grid_fft().communicator(1 << 1), N__);
    
    for (int j = 0; j < phi__.pw_coeffs().spl_num_col().local_size(); j++) {
        /* phi(G) -> phi(r) */
        ctx_.fft().transform<1>(kp__->gkvec().partition(), phi__.pw_coeffs().extra().at<CPU>(0, j));
        #pragma omp parallel for
        for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
            /* multiply by step function */
            ctx_.fft().buffer(ir) *= ctx_.step_function().theta_r(ir);
            /* save phi(r) * Theta(r) */
            buf_rg[ir] = ctx_.fft().buffer(ir);
        }
        /* phi(r) * Theta(r) -> ophi(G) */
        ctx_.fft().transform<-1>(kp__->gkvec().partition(), ophi__.pw_coeffs().extra().at<CPU>(0, j));
        #pragma omp parallel for
        for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
            /* multiply be effective potential */
            ctx_.fft().buffer(ir) = buf_rg[ir] * effective_potential__->f_rg(ir);
        }
        /* phi(r) * Theta(r) * V(r) -> ophi(G) */
        ctx_.fft().transform<-1>(kp__->gkvec().partition(), hphi__.pw_coeffs().extra().at<CPU>(0, j));
        
        for (int x: {0, 1, 2}) {
            for (int igloc = 0; igloc < kp__->gkvec().partition().gvec_count_fft(); igloc++) {
                /* global index of G-vector */
                int ig = kp__->gkvec().partition().gvec_offset_fft() + igloc;
                buf_pw[igloc] = phi__.pw_coeffs().extra()(igloc, j) * kp__->gkvec().gkvec_cart(ig)[x];
            }
            /* transform Cartesian component of wave-function gradient to real space */
            ctx_.fft().transform<1>(kp__->gkvec().partition(), &buf_pw[0]);
            #pragma omp parallel for
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                /* multiply be step function */
                ctx_.fft().buffer(ir) *= ctx_.step_function().theta_r(ir);
            }
            /* transform back to pw domain */
            ctx_.fft().transform<-1>(kp__->gkvec().partition(), &buf_pw[0]);
            for (int igloc = 0; igloc < kp__->gkvec().partition().gvec_count_fft(); igloc++) {
                int ig = kp__->gkvec().partition().gvec_offset_fft() + igloc;
                hphi__.pw_coeffs().extra()(igloc, j) += 0.5 * buf_pw[igloc] * kp__->gkvec().gkvec_cart(ig)[x];
            }
        }
    }

    hphi__.pw_coeffs().remap_backward(N__, n__, kp__->gkvec().partition(), ctx_.mpi_grid_fft().communicator(1 << 1));
    ophi__.pw_coeffs().remap_backward(N__, n__, kp__->gkvec().partition(), ctx_.mpi_grid_fft().communicator(1 << 1));

    ctx_.fft().dismiss();

    matrix<double_complex> alm(kp__->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());
    matrix<double_complex> halm(kp__->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());
    matrix<double_complex> tmp(unit_cell_.max_mt_aw_basis_size(), n__);
    matrix<double_complex> htmp(unit_cell_.max_mt_aw_basis_size(), n__);

    double_complex zone(1, 0);

    assert(ophi__.mt_coeffs().num_rows_loc() == hphi__.mt_coeffs().num_rows_loc());
    /* zero the local-orbital part */
    for (int ibnd = N__; ibnd < N__ + n__; ibnd++) {
        for (int ilo = 0; ilo < hphi__.mt_coeffs().num_rows_loc(); ilo++) {
            ophi__.mt_coeffs().prime(ilo, ibnd) = 0;
            hphi__.mt_coeffs().prime(ilo, ibnd) = 0;
        }
    }
    
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& atom = unit_cell_.atom(ia);
        auto& type = atom.type(); 
        int nmt = atom.mt_aw_basis_size();
        kp__->alm_coeffs_loc().generate(ia, alm);
        apply_hmt_to_apw<spin_block_t::nm>(atom, kp__->num_gkvec_loc(), alm, halm);
        
        /* tmp(lm, i) = A(G, lm)^{T} * C(G, i) */
        linalg<CPU>::gemm(1, 0, nmt, n__, kp__->num_gkvec_loc(),
                          alm.at<CPU>(), alm.ld(),
                          phi__.pw_coeffs().prime().at<CPU>(0, N__), phi__.pw_coeffs().prime().ld(),
                          tmp.at<CPU>(), tmp.ld());
        kp__->comm().allreduce(tmp.at<CPU>(), static_cast<int>(tmp.size()));

        /* htmp(lm, i) = H_{mt}A(G, lm)^{T} * C(G, i) */
        linalg<CPU>::gemm(1, 0, nmt, n__, kp__->num_gkvec_loc(),
                          halm.at<CPU>(), halm.ld(),
                          phi__.pw_coeffs().prime().at<CPU>(0, N__), phi__.pw_coeffs().prime().ld(),
                          htmp.at<CPU>(), htmp.ld());
        kp__->comm().allreduce(htmp.at<CPU>(), static_cast<int>(tmp.size()));

        for (int xi = 0; xi < nmt; xi++) {
            for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++) {
                alm(ig, xi) = std::conj(alm(ig, xi));
            }
        }
        /* APW-APW contribution to overlap */
        linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, nmt, zone,
                          alm.at<CPU>(), alm.ld(),
                          tmp.at<CPU>(), tmp.ld(),
                          zone,
                          ophi__.pw_coeffs().prime().at<CPU>(0, N__),
                          ophi__.pw_coeffs().prime().ld());
        /* APW-APW contribution to Hamiltonian */
        linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, nmt, zone,
                          alm.at<CPU>(), alm.ld(),
                          htmp.at<CPU>(), htmp.ld(),
                          zone,
                          hphi__.pw_coeffs().prime().at<CPU>(0, N__),
                          hphi__.pw_coeffs().prime().ld());
        
        /* local orbital coefficients of atom ia for all states */
        matrix<double_complex> phi_lo_ia(atom.mt_lo_basis_size(), n__);
        auto ia_location = phi__.spl_num_atoms().location(ia);
        if (ia_location.second == kp__->comm().rank()) {
            for (int i = 0; i < n__; i++) {
                std::memcpy(&phi_lo_ia(0, i),
                            phi__.mt_coeffs().prime().at<CPU>(phi__.offset_mt_coeffs(ia_location.first), N__ + i),
                            atom.mt_lo_basis_size() * sizeof(double_complex));
            }
        }
        kp__->comm().bcast(phi_lo_ia.at<CPU>(), static_cast<int>(phi_lo_ia.size()), ia_location.second);

        /* sum over local obritals (this are the APW-lo and lo-lo contributions) */
        for (int i = 0; i < n__; i++) {
            for (int ilo = 0; ilo < type.mt_lo_basis_size(); ilo++) {
                int xi_lo = type.mt_aw_basis_size() + ilo;
                /* local orbital indices */
                int l_lo     = type.indexb(xi_lo).l;
                int lm_lo    = type.indexb(xi_lo).lm;
                int order_lo = type.indexb(xi_lo).order;
                int idxrf_lo = type.indexb(xi_lo).idxrf;

                for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
                    /* APW-lo contribution */
                    auto z = phi_lo_ia(ilo, i) * atom.symmetry_class().o_radial_integral(l_lo, order_aw, order_lo);
                    for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++) {
                        ophi__.pw_coeffs().prime(ig, N__ + i) += alm(ig, type.indexb_by_lm_order(lm_lo, order_aw)) * z;
                    }
                }

                for (int xi = 0; xi < nmt; xi++) {
                    int lm_aw    = type.indexb(xi).lm;
                    int idxrf_aw = type.indexb(xi).idxrf;
                    auto& gc = gaunt_coefs_->gaunt_vector(lm_aw, lm_lo);

                    auto z = phi_lo_ia(ilo, i) * atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_aw, idxrf_lo, gc);
                    /* APW-lo contribution */
                    for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++) {
                        hphi__.pw_coeffs().prime(ig, N__ + i) += alm(ig, xi) * z; 
                    }
                }
                
                if (ia_location.second == kp__->comm().rank()) {
                    /* lo-lo contribution */
                    for (int jlo = 0; jlo < type.mt_lo_basis_size(); jlo++) {
                        int xi_lo1 = type.mt_aw_basis_size() + jlo;
                        int lm1    = type.indexb(xi_lo1).lm;
                        int order1 = type.indexb(xi_lo1).order;
                        int idxrf1 = type.indexb(xi_lo1).idxrf;
                        auto& gc = gaunt_coefs_->gaunt_vector(lm_lo, lm1);
                        if (lm_lo == lm1) {
                            ophi__.mt_coeffs().prime(ophi__.offset_mt_coeffs(ia_location.first) + ilo, N__ + i) +=
                                phi_lo_ia(jlo, i) * atom.symmetry_class().o_radial_integral(l_lo, order_lo, order1);
                        }
                        hphi__.mt_coeffs().prime(hphi__.offset_mt_coeffs(ia_location.first) + ilo, N__ + i) +=
                            phi_lo_ia(jlo, i) * atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf1, gc);
                    }

                    for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
                        /* lo-APW contribution */
                        ophi__.mt_coeffs().prime(ophi__.offset_mt_coeffs(ia_location.first) + ilo, N__ + i) += 
                            atom.symmetry_class().o_radial_integral(l_lo, order_lo, order_aw) *
                            tmp(type.indexb_by_lm_order(lm_lo, order_aw), i);
                    }

                    double_complex z(0, 0);
                    for (int xi = 0; xi < nmt; xi++) {
                        int lm_aw    = type.indexb(xi).lm;
                        int idxrf_aw = type.indexb(xi).idxrf;
                        auto& gc = gaunt_coefs_->gaunt_vector(lm_lo, lm_aw);
                        z += atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf_aw, gc) * tmp(xi, i);
                    }
                    /* lo-APW contribution */
                    hphi__.mt_coeffs().prime(hphi__.offset_mt_coeffs(ia_location.first) + ilo, N__ + i) += z;
                }
            }
        }
    }
}

