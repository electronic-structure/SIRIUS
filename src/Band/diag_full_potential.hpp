inline void Band::diag_fv_exact(K_point* kp, Hamiltonian& hamiltonian__) const
{
    PROFILE("sirius::Band::diag_fv_exact");

    //if (kp->num_ranks() > 1 && !gen_evp_solver().parallel()) {
    //    TERMINATE("eigen-value solver is not parallel");
    //}

    auto mem_type = (ctx_.gen_evp_solver_type() == ev_solver_t::magma) ? memory_t::host_pinned : memory_t::host;
    int ngklo = kp->gklo_basis_size();
    int bs = ctx_.cyclic_block_size();
    dmatrix<double_complex> h(ngklo, ngklo, ctx_.blacs_grid(), bs, bs, mem_type);
    dmatrix<double_complex> o(ngklo, ngklo, ctx_.blacs_grid(), bs, bs, mem_type);

    if (ctx_.comm().rank() == 0 && ctx_.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    /* setup Hamiltonian and overlap */
    switch (ctx_.processing_unit()) {
        case CPU: {
            hamiltonian__.set_fv_h_o<CPU, electronic_structure_method_t::full_potential_lapwlo>(kp, h, o);
            break;
        }
        #ifdef __GPU
        case GPU: {
            hamiltonian__.set_fv_h_o<GPU, electronic_structure_method_t::full_potential_lapwlo>(kp, h, o);
            break;
        }
        #endif
        default: {
            TERMINATE("wrong processing unit");
        }
    }

    if (ctx_.control().verification_ >= 1) {
        double max_diff = check_hermitian(h, ngklo);
        if (max_diff > 1e-12) {
            std::stringstream s;
            s << "H matrix is not hermitian" << std::endl
              << "max error: " << max_diff;
            TERMINATE(s);
        }
        max_diff = check_hermitian(o, ngklo);
        if (max_diff > 1e-12) {
            std::stringstream s;
            s << "O matrix is not hermitian" << std::endl
              << "max error: " << max_diff;
            TERMINATE(s);
        }
    }

    if (ctx_.control().print_checksum_) {
        auto z1 = h.checksum();
        auto z2 = o.checksum();
        kp->comm().allreduce(&z1, 1);
        kp->comm().allreduce(&z2, 1);
        if (kp->comm().rank() == 0) {
            print_checksum("h_lapw", z1);
            print_checksum("o_lapw", z2);
        }
    }

    assert(kp->gklo_basis_size() > ctx_.num_fv_states());

    std::vector<double> eval(ctx_.num_fv_states());

    sddk::timer t("sirius::Band::diag_fv_exact|genevp");
    auto solver = ctx_.gen_evp_solver<double_complex>();

    if (solver->solve(kp->gklo_basis_size(), ctx_.num_fv_states(), h, o, eval.data(), kp->fv_eigen_vectors())) {
        TERMINATE("error in generalized eigen-value problem");
    }
    t.stop();
    kp->set_fv_eigen_values(&eval[0]);

    if (ctx_.control().verbosity_ >= 4 && kp->comm().rank() == 0) {
        for (int i = 0; i < ctx_.num_fv_states(); i++) {
            DUMP("eval[%i]=%20.16f", i, eval[i]);
        }
    }

    if (ctx_.control().print_checksum_) {
        auto z1 = kp->fv_eigen_vectors().checksum();
        kp->comm().allreduce(&z1, 1);
        if (kp->comm().rank() == 0) {
            DUMP("checksum(fv_eigen_vectors): %18.10f %18.10f", std::real(z1), std::imag(z1));
        }
    }

    /* remap to slab */
    kp->fv_eigen_vectors_slab().pw_coeffs(0).remap_from(kp->fv_eigen_vectors(), 0);
    kp->fv_eigen_vectors_slab().mt_coeffs(0).remap_from(kp->fv_eigen_vectors(), kp->num_gkvec());
    
    /* renormalize wave-functions */
    if (ctx_.valence_relativity() == relativity_t::iora) {
        Wave_functions ofv(kp->gkvec_partition(), unit_cell_.num_atoms(),
                           [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, ctx_.num_fv_states(), 1);
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            kp->fv_eigen_vectors_slab().allocate_on_device(0);
            kp->fv_eigen_vectors_slab().copy_to_device(0, 0, ctx_.num_fv_states());
            ofv.allocate_on_device(0);
        }
        #endif

        hamiltonian__.apply_fv_o(kp, false, false, 0, ctx_.num_fv_states(), kp->fv_eigen_vectors_slab(), ofv);

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            kp->fv_eigen_vectors_slab().deallocate_on_device(0);
            ofv.deallocate_on_device(0);
        }
        #endif

        std::vector<double> norm(ctx_.num_fv_states(), 0);
        for (int i = 0; i < ctx_.num_fv_states(); i++) {
            for (int j = 0; j < ofv.pw_coeffs(0).num_rows_loc(); j++) {
                norm[i] += std::real(std::conj(kp->fv_eigen_vectors_slab().pw_coeffs(0).prime(j, i)) * ofv.pw_coeffs(0).prime(j, i));
            }
            for (int j = 0; j < ofv.mt_coeffs(0).num_rows_loc(); j++) {
                norm[i] += std::real(std::conj(kp->fv_eigen_vectors_slab().mt_coeffs(0).prime(j, i)) * ofv.mt_coeffs(0).prime(j, i));
            }
        }
        kp->comm().allreduce(norm);
        for (int i = 0; i < ctx_.num_fv_states(); i++) {
            norm[i] = 1 / std::sqrt(norm[i]);
            for (int j = 0; j < ofv.pw_coeffs(0).num_rows_loc(); j++) {
                kp->fv_eigen_vectors_slab().pw_coeffs(0).prime(j, i) *= norm[i];
            }
            for (int j = 0; j < ofv.mt_coeffs(0).num_rows_loc(); j++) {
                kp->fv_eigen_vectors_slab().mt_coeffs(0).prime(j, i) *= norm[i];
            }
        }
    }

    if (ctx_.control().verification_ >= 2) {
        STOP();
        ///* check application of H and O */
        //wave_functions phi(ctx_.processing_unit(), kp->gkvec(), unit_cell_.num_atoms(),
        //                   [this](int ia) {return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states());
        //wave_functions hphi(ctx_.processing_unit(), kp->gkvec(), unit_cell_.num_atoms(),
        //                   [this](int ia) {return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states());
        //wave_functions ophi(ctx_.processing_unit(), kp->gkvec(), unit_cell_.num_atoms(),
        //                   [this](int ia) {return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states());
        //
        //for (int i = 0; i < ctx_.num_fv_states(); i++) {
        //    std::memcpy(phi.pw_coeffs().prime().at<CPU>(0, i),
        //                kp->fv_eigen_vectors().at<CPU>(0, i),
        //                kp->num_gkvec() * sizeof(double_complex));
        //    if (unit_cell_.mt_lo_basis_size()) {
        //        std::memcpy(phi.mt_coeffs().prime().at<CPU>(0, i),
        //                    kp->fv_eigen_vectors().at<CPU>(kp->num_gkvec(), i),
        //                    unit_cell_.mt_lo_basis_size() * sizeof(double_complex));
        //    }
        //}

        //apply_fv_h_o(kp, 0, 0, ctx_.num_fv_states(), phi, hphi, ophi);

        //dmatrix<double_complex> ovlp(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(), ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
        //dmatrix<double_complex> hmlt(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(), ctx_.cyclic_block_size(), ctx_.cyclic_block_size());

        //inner(phi, 0, ctx_.num_fv_states(), hphi, 0, ctx_.num_fv_states(), 0.0, hmlt, 0, 0);
        //inner(phi, 0, ctx_.num_fv_states(), ophi, 0, ctx_.num_fv_states(), 0.0, ovlp, 0, 0);

        //for (int i = 0; i < ctx_.num_fv_states(); i++) {
        //    for (int j = 0; j < ctx_.num_fv_states(); j++) {
        //        double_complex z = (i == j) ? ovlp(i, j) - 1.0 : ovlp(i, j);
        //        double_complex z1 = (i == j) ? hmlt(i, j) - eval[i] : hmlt(i, j);
        //        if (std::abs(z) > 1e-10) {
        //            printf("ovlp(%i, %i) = %f %f\n", i, j, z.real(), z.imag());
        //        }
        //        if (std::abs(z1) > 1e-10) {
        //            printf("hmlt(%i, %i) = %f %f\n", i, j, z1.real(), z1.imag());
        //        }
        //    }
        //}
    }
}

inline void Band::get_singular_components(K_point* kp__, Hamiltonian& H__) const
{
    PROFILE("sirius::Band::get_singular_components");

    auto o_diag_tmp = H__.get_o_diag(kp__, ctx_.step_function().theta_pw(0).real());

    mdarray<double, 2> o_diag(kp__->num_gkvec_loc(), 1, memory_t::host, "o_diag");
    mdarray<double, 1> diag1(kp__->num_gkvec_loc(), memory_t::host, "diag1");
    for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++) {
        o_diag[ig] = o_diag_tmp[ig];
        diag1[ig] = 1;
    }

    if (ctx_.processing_unit() == GPU) {
        o_diag.allocate(memory_t::device);
        o_diag.copy<memory_t::host, memory_t::device>();
        diag1.allocate(memory_t::device);
        diag1.copy<memory_t::host, memory_t::device>();
    }

    auto& psi = kp__->singular_components();

    int ncomp = psi.num_wf();

    if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 3) {
        printf("number of singular components: %i\n", ncomp);
    }

    auto& itso = ctx_.iterative_solver_input();

    int num_phi = itso.subspace_size_ * ncomp;

    Wave_functions  phi(kp__->gkvec_partition(), num_phi);
    Wave_functions ophi(kp__->gkvec_partition(), num_phi);
    Wave_functions opsi(kp__->gkvec_partition(), ncomp);
    Wave_functions  res(kp__->gkvec_partition(), ncomp);

    int bs = ctx_.cyclic_block_size();

    dmatrix<double_complex> ovlp(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> ovlp_old(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> evec(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        psi.pw_coeffs(0).allocate_on_device();
        psi.pw_coeffs(0).copy_to_device(0, ncomp);
        phi.pw_coeffs(0).allocate_on_device();
        res.pw_coeffs(0).allocate_on_device();
        ophi.pw_coeffs(0).allocate_on_device();
        opsi.pw_coeffs(0).allocate_on_device();
        if (kp__->comm().size() == 1) {
            evec.allocate(memory_t::device);
            ovlp.allocate(memory_t::device);
        }
    }
    #endif

    std::vector<double> eval(ncomp, 1e10);
    std::vector<double> eval_old(ncomp);

    if (ctx_.control().print_checksum_) {
        auto cs2 = phi.checksum(ctx_.processing_unit(), 0, 0, ncomp);
        if (kp__->comm().rank() == 0) {
            print_checksum("phi", cs2);
        }
    }

    phi.copy_from(ctx_.processing_unit(), ncomp, psi, 0, 0, 0, 0);

    /* current subspace size */
    int N{0};

    /* number of newly added basis functions */
    int n = ncomp;

    if (ctx_.control().verbosity_ >= 3 && kp__->comm().rank() == 0) {
        printf("iterative solver tolerance: %18.12f\n", ctx_.iterative_solver_tolerance());
    }

    if (kp__->comm().rank() == 0 && ctx_.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    auto std_solver = ctx_.std_evp_solver<double_complex>();

    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++) {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        H__.apply_fv_o(kp__, true, true, N, n, phi, ophi);

        if (ctx_.control().verification_ >= 1) {
            dmatrix<double_complex> tmp;
            set_subspace_mtrx(0, N + n, phi, ophi, ovlp, tmp);

            if (ctx_.control().verification_ >= 2) {
                ovlp.serialize("overlap", N + n);
            }

            double max_diff = check_hermitian(ovlp, N + n);
            if (max_diff > 1e-12) {
                std::stringstream s;
                s << "overlap matrix is not hermitian, max_err = " << max_diff;
                TERMINATE(s);
            }
        }

        orthogonalize(ctx_.processing_unit(), 0, phi, ophi, N, n, ovlp, res);
        
        /* setup eigen-value problem
         * N is the number of previous basis functions
         * n is the number of new basis functions */
        set_subspace_mtrx(N, n, phi, ophi, ovlp, ovlp_old);

        if (ctx_.control().verification_ >= 1) {

            if (ctx_.control().verification_ >= 2) {
                ovlp.serialize("overlap_ortho", N + n);
            }

            double max_diff = check_hermitian(ovlp, N + n);
            if (max_diff > 1e-12) {
                std::stringstream s;
                s << "overlap matrix is not hermitian, max_err = " << max_diff;
                TERMINATE(s);
            }
        }

        /* increase size of the variation space */
        N += n;

        eval_old = eval;

        /* solve standard eigen-value problem with the size N */
        if (std_solver->solve(N, ncomp, ovlp, eval.data(), evec)) {
            std::stringstream s;
            s << "error in diagonalziation";
            TERMINATE(s);
        }

        for (auto e: eval) {
            if (e < 0) {
                std::stringstream s;
                s << "overlap matrix is not positively defined";
                TERMINATE(s);
            }
        }

        if (ctx_.control().verbosity_ >= 3 && kp__->comm().rank() == 0) {
            printf("step: %i, current subspace size: %i, maximum subspace size: %i\n", k, N, num_phi);
            if (ctx_.control().verbosity_ >= 4) {
                for (int i = 0; i < ncomp; i++) {
                    printf("eval[%i]=%20.16f, diff=%20.16f\n", i, eval[i], std::abs(eval[i] - eval_old[i]));
                }
            }
        }

        /* don't compute residuals on last iteration */
        if (k != itso.num_steps_ - 1) {
            /* get new preconditionined residuals, and also opsi and psi as a by-product */
            n = residuals(kp__, 0, N, ncomp, eval, eval_old, evec, ophi, phi, opsi, psi, res, o_diag, diag1);
        }

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (N + n > num_phi || n <= itso.min_num_res_ || k == (itso.num_steps_ - 1)) {
            sddk::timer t1("sirius::Band::diag_fv_full_potential_davidson|update_phi");
            /* recompute wave-functions */
            /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
            transform(ctx_.processing_unit(), 0, phi, 0, N, evec, 0, 0, psi, 0, ncomp);

            /* exit the loop if the eigen-vectors are converged or this is a last iteration */
            if (n <= itso.min_num_res_ || k == (itso.num_steps_ - 1)) {
                break;
            }
            else { /* otherwise, set Psi as a new trial basis */
                if (ctx_.control().verbosity_ >= 3 && kp__->comm().rank() == 0) {
                    printf("subspace size limit reached\n");
                }

                if (itso.converge_by_energy_) {
                    transform(ctx_.processing_unit(), 0, ophi, 0, N, evec, 0, 0, opsi, 0, ncomp);
                }

                ovlp_old.zero();
                for (int i = 0; i < ncomp; i++) {
                    ovlp_old.set(i, i, eval[i]);
                }
                /* update basis functions */
                phi.copy_from(ctx_.processing_unit(), ncomp, psi, 0, 0, 0, 0);
                ophi.copy_from(ctx_.processing_unit(), ncomp, opsi, 0, 0, 0, 0);
                /* number of basis functions that we already have */
                N = ncomp;
            }
        }
        /* expand variational subspace with new basis vectors obtatined from residuals */
        phi.copy_from(ctx_.processing_unit(), n, res, 0, 0, 0, N);
    }

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        psi.pw_coeffs(0).copy_to_host(0, ncomp);
        psi.pw_coeffs(0).deallocate_on_device();
    }
    #endif

    if (ctx_.control().verbosity_ >= 2 && kp__->comm().rank() == 0) {
        printf("lowest and highest eigen-values of the singluar components: %20.16f %20.16f\n", eval.front(), eval.back());
    }

    kp__->comm().barrier();
}

inline void Band::diag_fv_davidson(K_point* kp, Hamiltonian& H__) const
{
    PROFILE("sirius::Band::diag_fv_davidson");

    H__.local_op().prepare(kp->gkvec_partition());

    get_singular_components(kp, H__);

    auto h_diag = H__.get_h_diag(kp, H__.local_op().v0(0), ctx_.step_function().theta_pw(0).real());
    auto o_diag = H__.get_o_diag(kp, ctx_.step_function().theta_pw(0).real());

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_fv_states();

    auto& itso = ctx_.iterative_solver_input();

    /* short notation for target wave-functions */
    auto& psi = kp->fv_eigen_vectors_slab();

    //bool converge_by_energy = (itso.converge_by_energy_ == 1);

    int nlo = ctx_.unit_cell().mt_lo_basis_size();

    int ncomp = kp->singular_components().num_wf();

    /* number of auxiliary basis functions */
    int num_phi = nlo + ncomp + itso.subspace_size_ * num_bands;
    if (num_phi >= kp->num_gkvec()) {
        TERMINATE("subspace is too big");
    }

    /* allocate wave-functions */
    Wave_functions  phi(kp->gkvec_partition(), unit_cell_.num_atoms(),
                        [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, num_phi);
    Wave_functions hphi(kp->gkvec_partition(), unit_cell_.num_atoms(),
                        [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, num_phi);
    Wave_functions ophi(kp->gkvec_partition(), unit_cell_.num_atoms(),
                        [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, num_phi);
    Wave_functions hpsi(kp->gkvec_partition(), unit_cell_.num_atoms(),
                        [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, num_bands);
    Wave_functions opsi(kp->gkvec_partition(), unit_cell_.num_atoms(),
                        [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, num_bands);

    /* residuals */
    /* res is also used as a temporary array in orthogonalize() and the first time nlo + ncomp + num_bands
     * states will be orthogonalized */
    Wave_functions res(kp->gkvec_partition(), unit_cell_.num_atoms(),
                       [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, nlo + ncomp + num_bands);

    //auto mem_type = (gen_evp_solver_->type() == ev_magma) ? memory_t::host_pinned : memory_t::host;

    int bs = ctx_.cyclic_block_size();

    dmatrix<double_complex> hmlt(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> ovlp(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> evec(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> hmlt_old(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);

    if (nlo) {
        phi.pw_coeffs(0).zero<memory_t::host>(0, nlo);
        phi.mt_coeffs(0).zero<memory_t::host>(0, nlo);
        for (int ialoc = 0; ialoc < phi.spl_num_atoms().local_size(); ialoc++) {
            int ia = phi.spl_num_atoms()[ialoc];
            for (int xi = 0; xi < unit_cell_.atom(ia).mt_lo_basis_size(); xi++) {
                phi.mt_coeffs(0).prime(phi.offset_mt_coeffs(ialoc) + xi, unit_cell_.atom(ia).offset_lo() + xi) = 1.0;
            }
        }
    }

    if (ncomp != 0) {
        phi.mt_coeffs(0).zero<memory_t::host>(nlo, ncomp);
        for (int j = 0; j < ncomp; j++) {
            std::memcpy(phi.pw_coeffs(0).prime().at<CPU>(0, nlo + j),
                        kp->singular_components().pw_coeffs(0).prime().at<CPU>(0, j),
                        phi.pw_coeffs(0).num_rows_loc() * sizeof(double_complex));
        }
    }

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        psi.allocate_on_device(0);
        psi.copy_to_device(0, 0, num_bands);

        phi.allocate_on_device(0);
        phi.copy_to_device(0, 0, nlo + ncomp);

        res.allocate_on_device(0);

        hphi.allocate_on_device(0);
        ophi.allocate_on_device(0);

        hpsi.allocate_on_device(0);
        opsi.allocate_on_device(0);
    
        if (kp->comm().size() == 1) {
            evec.allocate(memory_t::device);
            ovlp.allocate(memory_t::device);
            hmlt.allocate(memory_t::device);
        }
    }
    #endif

    std::vector<double> eval(num_bands);
    for (int i = 0; i < num_bands; i++) {
        eval[i] = kp->band_energy(i);
    }
    std::vector<double> eval_old(num_bands);

    /* trial basis functions */
    phi.copy_from(ctx_.processing_unit(), num_bands, psi, 0, 0, 0, nlo + ncomp);

    if (ctx_.control().print_checksum_) {
        auto cs1 = psi.checksum(ctx_.processing_unit(), 0, 0, num_bands);
        auto cs2 = phi.checksum(ctx_.processing_unit(), 0, 0, nlo + ncomp + num_bands);
        if (kp->comm().rank() == 0) {
            print_checksum("psi", cs1);
            print_checksum("phi", cs2);
        }
    }

    /* current subspace size */
    int N = 0;

    /* number of newly added basis functions */
    int n = nlo + ncomp + num_bands;

    if (ctx_.control().verbosity_ >= 3 && kp->comm().rank() == 0) {
        DUMP("iterative solver tolerance: %18.12f", ctx_.iterative_solver_tolerance());
    }

    if (ctx_.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    auto std_solver = ctx_.std_evp_solver<double_complex>();

    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++) {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        H__.apply_fv_h_o(kp, nlo, N, n, phi, hphi, ophi);
        
        orthogonalize(ctx_.processing_unit(), 0, phi, hphi, ophi, N, n, ovlp, res);

        /* setup eigen-value problem
         * N is the number of previous basis functions
         * n is the number of new basis functions */
        set_subspace_mtrx(N, n, phi, hphi, hmlt, hmlt_old);

        /* increase size of the variation space */
        N += n;

        eval_old = eval;

        /* solve standard eigen-value problem with the size N */
        if (std_solver->solve(N, num_bands, hmlt, eval.data(), evec)) {
            std::stringstream s;
            s << "error in diagonalziation";
            TERMINATE(s);
        }

        if (ctx_.control().verbosity_ >= 3 && kp->comm().rank() == 0) {
            DUMP("step: %i, current subspace size: %i, maximum subspace size: %i", k, N, num_phi);
            if (ctx_.control().verbosity_ >= 4) {
                for (int i = 0; i < num_bands; i++) {
                    DUMP("eval[%i]=%20.16f, diff=%20.16f", i, eval[i], std::abs(eval[i] - eval_old[i]));
                }
            }
        }

        /* don't compute residuals on last iteration */
        if (k != itso.num_steps_ - 1) {
            /* get new preconditionined residuals, and also hpsi and opsi as a by-product */
            n = residuals(kp, 0, N, num_bands, eval, eval_old, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag);
        }

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (N + n > num_phi || n <= itso.min_num_res_ || k == (itso.num_steps_ - 1)) {
            sddk::timer t1("sirius::Band::diag_fv_davidson|update_phi");
            /* recompute wave-functions */
            /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
            transform(ctx_.processing_unit(), 0, phi, 0, N, evec, 0, 0, psi, 0, num_bands);

            /* exit the loop if the eigen-vectors are converged or this is a last iteration */
            if (n <= itso.min_num_res_ || k == (itso.num_steps_ - 1)) {
                break;
            }
            else { /* otherwise, set Psi as a new trial basis */
                if (ctx_.control().verbosity_ >= 3 && kp->comm().rank() == 0) {
                    DUMP("subspace size limit reached");
                }

                /* update basis functions */
                phi.copy_from(ctx_.processing_unit(), num_bands, psi, 0, 0, 0, nlo + ncomp);
                phi.copy_from(ctx_.processing_unit(), n, res, 0, 0, 0, nlo + ncomp + num_bands);
                /* number of basis functions that we already have */
                N = nlo + ncomp;
                n += num_bands;
            }
        } else {
            /* expand variational subspace with new basis vectors obtatined from residuals */
            phi.copy_from(ctx_.processing_unit(), n, res, 0, 0, 0, N);
        }
    }

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        psi.pw_coeffs(0).copy_to_host(0, num_bands);
        psi.mt_coeffs(0).copy_to_host(0, num_bands);
        psi.deallocate_on_device(0);
    }
    #endif

    kp->set_fv_eigen_values(&eval[0]);
    kp->comm().barrier();
}

inline void Band::diag_sv(K_point*     kp__,
                          Hamiltonian& hamiltonian__) const
{
    PROFILE("sirius::Band::diag_sv");

    if (!ctx_.need_sv()) {
        kp__->bypass_sv();
        return;
    }

    std::vector<double> band_energies(ctx_.num_bands());

    /* product of the second-variational Hamiltonian and a first-variational wave-function */
    std::vector<Wave_functions> hpsi;
    for (int i = 0; i < ctx_.num_mag_comp(); i++) {
        hpsi.push_back(std::move(Wave_functions(kp__->gkvec_partition(),
                                                unit_cell_.num_atoms(),
                                                [this](int ia)
                                                {
                                                    return unit_cell_.atom(ia).mt_basis_size();
                                                },
                                                ctx_.num_fv_states())));
    }

    /* compute product of magnetic field and wave-function */
    if (ctx_.num_spins() == 2) {
        hamiltonian__.apply_magnetic_field(kp__, kp__->fv_states(), hpsi);
    }
    else {
        hpsi[0].pw_coeffs(0).prime().zero();
        hpsi[0].mt_coeffs(0).prime().zero();
    }

    if (ctx_.comm().rank() == 0 && ctx_.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    //== if (ctx_.uj_correction())
    //== {
    //==     apply_uj_correction<uu>(kp->fv_states_col(), hpsi);
    //==     if (ctx_.num_mag_dims() != 0) apply_uj_correction<dd>(kp->fv_states_col(), hpsi);
    //==     if (ctx_.num_mag_dims() == 3)
    //==     {
    //==         apply_uj_correction<ud>(kp->fv_states_col(), hpsi);
    //==         if (ctx_.std_evp_solver()->parallel()) apply_uj_correction<du>(kp->fv_states_col(), hpsi);
    //==     }
    //== }

    //== if (ctx_.so_correction()) apply_so_correction(kp->fv_states_col(), hpsi);

    int nfv = ctx_.num_fv_states();
    int bs  = ctx_.cyclic_block_size();

#ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        kp__->fv_states().allocate_on_device(0);
        kp__->fv_states().copy_to_device(0, 0, nfv);
        for (int i = 0; i < ctx_.num_mag_comp(); i++) {
            hpsi[i].allocate_on_device(0);
            hpsi[i].copy_to_device(0, 0, nfv);
        }
    }
#endif

    if (ctx_.comm().rank() == 0 && ctx_.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    //#ifdef __PRINT_OBJECT_CHECKSUM
    //auto z1 = kp->fv_states().checksum(0, nfv);
    //DUMP("checksum(fv_states): %18.10f %18.10f", std::real(z1), std::imag(z1));
    //for (int i = 0; i < ctx_.num_mag_comp(); i++) {
    //    z1 = hpsi[i].checksum(0, nfv);
    //    DUMP("checksum(hpsi[i]): %18.10f %18.10f", std::real(z1), std::imag(z1));
    //}
    //#endif

    auto std_solver = ctx_.std_evp_solver<double_complex>();

    if (ctx_.num_mag_dims() != 3) {
        dmatrix<double_complex> h(nfv, nfv, ctx_.blacs_grid(), bs, bs);
        if (kp__->num_ranks() == 1 && ctx_.processing_unit() == GPU) {
            h.allocate(memory_t::device);
        }
        /* perform one or two consecutive diagonalizations */
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {

            /* compute <wf_i | h * wf_j> */
            inner(ctx_.processing_unit(), 0, kp__->fv_states(), 0, nfv, hpsi[ispn], 0, nfv, h, 0, 0);
            
            for (int i = 0; i < nfv; i++) {
                h.add(i, i, kp__->fv_eigen_value(i));
            }
            //#ifdef __PRINT_OBJECT_CHECKSUM
            //auto z1 = h.checksum();
            //DUMP("checksum(h): %18.10f %18.10f", std::real(z1), std::imag(z1));
            //#endif
            sddk::timer t1("sirius::Band::diag_sv|stdevp");
            std_solver->solve(nfv, nfv, h, &band_energies[ispn * nfv], kp__->sv_eigen_vectors(ispn));
        }
    } else {
        int nb = ctx_.num_bands();
        dmatrix<double_complex> h(nb, nb, ctx_.blacs_grid(), bs, bs);
        if (kp__->num_ranks() == 1 && ctx_.processing_unit() == GPU) {
            h.allocate(memory_t::device);
        }
        /* compute <wf_i | h * wf_j> for up-up block */
        inner(ctx_.processing_unit(), 0, kp__->fv_states(), 0, nfv, hpsi[0], 0, nfv, h, 0, 0);
        /* compute <wf_i | h * wf_j> for dn-dn block */
        inner(ctx_.processing_unit(), 0, kp__->fv_states(), 0, nfv, hpsi[1], 0, nfv, h, nfv, nfv);
        /* compute <wf_i | h * wf_j> for up-dn block */
        inner(ctx_.processing_unit(), 0, kp__->fv_states(), 0, nfv, hpsi[2], 0, nfv, h, 0, nfv);

        if (kp__->comm().size() == 1) {
            for (int i = 0; i < nfv; i++) {
                for (int j = 0; j < nfv; j++) {
                    h(nfv + j, i) = std::conj(h(i, nfv + j));
                }
            }
        } else {
            linalg<CPU>::tranc(nfv, nfv, h, 0, nfv, h, nfv, 0);
        }

        for (int i = 0; i < nfv; i++) {
            h.add(i,       i,       kp__->fv_eigen_value(i));
            h.add(i + nfv, i + nfv, kp__->fv_eigen_value(i));
        }
        //#ifdef __PRINT_OBJECT_CHECKSUM
        //auto z1 = h.checksum();
        //DUMP("checksum(h): %18.10f %18.10f", std::real(z1), std::imag(z1));
        //#endif
        sddk::timer t1("sirius::Band::diag_sv|stdevp");
        std_solver->solve(nb, nb, h, &band_energies[0], kp__->sv_eigen_vectors(0));
    }

#ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        kp__->fv_states().deallocate_on_device(0);
        for (int i = 0; i < ctx_.num_mag_comp(); i++) {
            hpsi[i].deallocate_on_device(0);
        }
    }
#endif
 
    kp__->set_band_energies(&band_energies[0]);
}
