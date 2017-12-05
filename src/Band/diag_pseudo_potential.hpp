#ifdef __GPU
extern "C" void compute_chebyshev_polynomial_gpu(int num_gkvec,
                                                 int n,
                                                 double c,
                                                 double r,
                                                 cuDoubleComplex* phi0,
                                                 cuDoubleComplex* phi1,
                                                 cuDoubleComplex* phi2);
#endif

template <typename T>
inline void Band::diag_pseudo_potential_exact(K_point* kp__,
                                              int ispn__,
                                              D_operator<T>& d_op__,
                                              Q_operator<T>& q_op__) const
{
    PROFILE("sirius::Band::diag_pseudo_potential_exact");

    STOP();

//    /* short notation for target wave-functions */
//    auto& psi = kp__->spinor_wave_functions(ispn__);
//
//    /* short notation for number of target wave-functions */
//    int num_bands = ctx_.num_fv_states();     
//
//    int ngk = kp__->num_gkvec();
//
//    /* number of spin components, treated simultaneously */
//    int nsc = (ctx_.num_mag_dims() == 3) ? 2 : 1;
//
//    Wave_functions  phi(ctx_.processing_unit(), kp__->gkvec(), ngk, nsc);
//    Wave_functions hphi(ctx_.processing_unit(), kp__->gkvec(), ngk, nsc);
//    Wave_functions ophi(ctx_.processing_unit(), kp__->gkvec(), ngk, nsc);
//    
//    std::vector<double> eval(ngk);
//
//    phi.component(0).pw_coeffs().prime().zero();
//    for (int i = 0; i < ngk; i++) {
//        phi.component(0).pw_coeffs().prime(i, i) = 1;
//    }
//
//    apply_h_o(kp__, ispn__, 0, ngk, phi, hphi, ophi, d_op__, q_op__);
//        
//    //Utils::check_hermitian("h", hphi.coeffs(), ngk);
//    //Utils::check_hermitian("o", ophi.coeffs(), ngk);
//
//    #ifdef __PRINT_OBJECT_CHECKSUM
//    auto z1 = hphi.pw_coeffs().prime().checksum();
//    auto z2 = ophi.pw_coeffs().prime().checksum();
//    printf("checksum(h): %18.10f %18.10f\n", z1.real(), z1.imag());
//    printf("checksum(o): %18.10f %18.10f\n", z2.real(), z2.imag());
//    #endif
//
//    auto gen_solver = ctx_.gen_evp_solver<double_complex>();
//    
//    TERMINATE("fix this later");
//    dmatrix<double_complex> hmlt(hphi[0].pw_coeffs().prime().template at<CPU>(), ngk, ngk);
//    dmatrix<double_complex> ovlp(ophi[0].pw_coeffs().prime().template at<CPU>(), ngk, ngk);
//    dmatrix<double_complex> Z(psi.pw_coeffs().prime().template at<CPU>(), ngk, ngk);
//
//    if (gen_solver->solve(ngk, num_bands, hmlt, ovlp, &eval[0], Z)) {
//        TERMINATE("error in evp solve");
//    }
//
//    for (int j = 0; j < ctx_.num_fv_states(); j++) {
//        kp__->band_energy(j + ispn__ * ctx_.num_fv_states()) = eval[j];
//    }
}

template <typename T>
inline int Band::diag_pseudo_potential_davidson(K_point*       kp__,
                                                D_operator<T>& d_op__,
                                                Q_operator<T>& q_op__) const
{
    PROFILE("sirius::Band::diag_pseudo_potential_davidson");

    if (kp__->comm().rank() == 0 && ctx_.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    auto& itso = ctx_.iterative_solver_input();

    bool converge_by_energy = (itso.converge_by_energy_ == 1);

    if (ctx_.control().verbosity_ >= 2 && kp__->comm().rank() == 0) {
        printf("iterative solver tolerance: %18.12f\n", ctx_.iterative_solver_tolerance());
    }

    /* true if this is a non-collinear case */
    bool nc_mag = (ctx_.num_mag_dims() == 3);

    /* number of spin components, treated simultaneously 
     *   1 - in case of non-magnetic or collinear calculation
     *   2 - in case of non-collinear calculation
     */
    int num_sc = nc_mag ? 2 : 1;
    
    /* number of steps in spin index
     * 1 - in case of non-magnetic calculation
     * 2 - in case of collinear calculation (up, dn) 
     * 1 - in case of non-collinear calculation (two spin components are treated simultaneously)
     */
    int num_spin_steps = nc_mag ? 1 : ctx_.num_spins();

    /* short notation for number of target wave-functions */
    int num_bands = nc_mag ? ctx_.num_bands() : ctx_.num_fv_states();
    
    /* short notation for target wave-functions */
    auto& psi = kp__->spinor_wave_functions();

    sddk::timer t1("sirius::Band::diag_pseudo_potential_davidson|wf");

    /* maximum subspace size */
    int num_phi = itso.subspace_size_ * num_bands;

    if (num_phi > kp__->num_gkvec()) {
        std::stringstream s;
        s << "subspace size is too large!";
        TERMINATE(s);
    }
    
    /* total memory size of all wave-functions */
    size_t size = sizeof(double_complex) * num_sc * kp__->num_gkvec_loc() * (3 * num_phi + 3 * num_bands);
    /* get preallocatd memory buffer */
    double_complex* mem_buf_ptr = static_cast<double_complex*>(ctx_.memory_buffer(size));

    /* allocate wave-functions */

    /* auxiliary wave-functions */
    Wave_functions phi(mem_buf_ptr, kp__->gkvec(), num_phi, num_sc);
    mem_buf_ptr += kp__->num_gkvec_loc() * num_phi * num_sc;

    /* Hamiltonian, applied to auxiliary wave-functions */
    Wave_functions hphi(mem_buf_ptr, kp__->gkvec(), num_phi, num_sc);
    mem_buf_ptr += kp__->num_gkvec_loc() * num_phi * num_sc;

    /* S operator, applied to auxiliary wave-functions */
    Wave_functions sphi(mem_buf_ptr, kp__->gkvec(), num_phi, num_sc);
    mem_buf_ptr += kp__->num_gkvec_loc() * num_phi * num_sc;

    /* Hamiltonain, applied to new Psi wave-functions */
    Wave_functions hpsi(mem_buf_ptr, kp__->gkvec(), num_bands, num_sc);
    mem_buf_ptr += kp__->num_gkvec_loc() * num_bands * num_sc;

    /* S operator, applied to new Psi wave-functions */
    Wave_functions spsi(mem_buf_ptr, kp__->gkvec(), num_bands, num_sc);
    mem_buf_ptr += kp__->num_gkvec_loc() * num_bands * num_sc;

    /* residuals */
    Wave_functions res(mem_buf_ptr, kp__->gkvec(), num_bands, num_sc);
    t1.stop();

    sddk::timer t2("sirius::Band::diag_pseudo_potential_davidson|alloc");
    auto mem_type = (ctx_.std_evp_solver_type() == ev_solver_t::magma) ? memory_t::host_pinned : memory_t::host;

    int bs = ctx_.cyclic_block_size();

    dmatrix<T> hmlt(num_phi, num_phi, ctx_.blacs_grid(), bs, bs, mem_type);
    dmatrix<T> ovlp(num_phi, num_phi, ctx_.blacs_grid(), bs, bs, mem_type);
    dmatrix<T> evec(num_phi, num_phi, ctx_.blacs_grid(), bs, bs, mem_type);
    dmatrix<T> hmlt_old(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<T> ovlp_old(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);

    kp__->beta_projectors().prepare();

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        if (!keep_wf_on_gpu) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                psi.pw_coeffs(ispn).allocate_on_device();
                psi.pw_coeffs(ispn).copy_to_device(0, num_bands);
            }
        }
        for (int i = 0; i < num_sc; i++) {
            phi.pw_coeffs(i).allocate_on_device();
            res.pw_coeffs(i).allocate_on_device();

            hphi.pw_coeffs(i).allocate_on_device();
            sphi.pw_coeffs(i).allocate_on_device();

            hpsi.pw_coeffs(i).allocate_on_device();
            spsi.pw_coeffs(i).allocate_on_device();
        }
    
        if (ctx_.blacs_grid().comm().size() == 1) {
            evec.allocate(memory_t::device);
            ovlp.allocate(memory_t::device);
            hmlt.allocate(memory_t::device);
        }
    }
    #endif

    if (kp__->comm().rank() == 0 && ctx_.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }
    t2.stop();

    /* get diagonal elements for preconditioning */
    auto h_diag = get_h_diag(kp__, *local_op_, d_op__);
    auto o_diag = get_o_diag(kp__, q_op__);

    if (ctx_.control().print_checksum_) {
        auto cs1 = h_diag.checksum();
        auto cs2 = o_diag.checksum();
        kp__->comm().allreduce(&cs1, 1);
        kp__->comm().allreduce(&cs2, 1);
        if (kp__->comm().rank() == 0) {
            print_checksum("h_diag", cs1);
            print_checksum("o_diag", cs2);
        }
    }

    auto std_solver = ctx_.std_evp_solver<T>();
    auto gen_solver = ctx_.gen_evp_solver<T>();

    int niter{0};
    
    sddk::timer t3("sirius::Band::diag_pseudo_potential_davidson|iter");
    for (int ispin_step = 0; ispin_step < num_spin_steps; ispin_step++) {

        std::vector<double> eval(num_bands);
        std::vector<double> eval_old(num_bands, 1e100);

        /* trial basis functions */
        for (int ispn = 0; ispn < num_sc; ispn++) {
            phi.copy_from(ctx_.processing_unit(), num_bands, psi, ctx_.num_mag_dims() == 3 ? ispn : ispin_step, 0, ispn, 0);
        }

        /* fisrt phase: setup and diagonalize reduced Hamiltonian and get eigen-values;
         * this is done before the main itertive loop */

        /* apply Hamiltonian and S operators to the basis functions */
        apply_h_s<T>(kp__, ispin_step, 0, num_bands, phi, hphi, sphi, d_op__, q_op__);
        
        /* setup eigen-value problem
         * N is the number of previous basis functions
         * n is the number of new basis functions */
        set_subspace_mtrx(0, num_bands, phi, hphi, hmlt, hmlt_old);
        /* setup overlap matrix */
        set_subspace_mtrx(0, num_bands, phi, sphi, ovlp, ovlp_old);

        /* current subspace size */
        int N = num_bands;

        sddk::timer t1("sirius::Band::diag_pseudo_potential_davidson|evp");
        /* solve generalized eigen-value problem with the size N and get lowest num_bands eigen-vectors */
        if (gen_solver->solve(N, num_bands, hmlt, ovlp, eval.data(), evec)) {
            std::stringstream s;
            s << "error in diagonalziation";
            TERMINATE(s);
        }
        t1.stop();

        if (ctx_.control().verbosity_ >= 4 && kp__->comm().rank() == 0) {
            for (int i = 0; i < num_bands; i++) {
                printf("eval[%i]=%20.16f\n", i, eval[i]);
            }
        }

        /* number of newly added basis functions */
        int n{0};

        /* second phase: start iterative diagonalization */
        for (int k = 0; k < itso.num_steps_; k++) {

            /* don't compute residuals on last iteration */
            if (k != itso.num_steps_ - 1) {
                /* get new preconditionined residuals, and also hpsi and opsi as a by-product */
                n = residuals<T>(kp__, nc_mag ? 2 : ispin_step, N, num_bands, eval, eval_old, evec, hphi,
                                 sphi, hpsi, spsi, res, h_diag, o_diag);
            }

            /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
            if (N + n > num_phi || n <= itso.min_num_res_ || k == (itso.num_steps_ - 1)) {
                sddk::timer t1("sirius::Band::diag_pseudo_potential_davidson|update_phi");
                /* recompute wave-functions */
                /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
                if (ctx_.settings().always_update_wf_ || k + n > 0) {
                    /* in case of non-collinear magnetism transform two components */ 
                    transform<T>(ctx_.processing_unit(), nc_mag ? 2 : ispin_step, {&phi}, 0, N, evec, 0, 0, {&psi}, 0, num_bands);
                    /* update eigen-values */
                    for (int j = 0; j < num_bands; j++) {
                        kp__->band_energy(j + ispin_step * ctx_.num_fv_states()) = eval[j];
                    }
                } else {
                    if (ctx_.control().verbosity_ >= 2 && kp__->comm().rank() == 0) {
                        printf("wave-functions are not recomputed\n");
                    }
                }

                /* exit the loop if the eigen-vectors are converged or this is a last iteration */
                if (n <= itso.min_num_res_ || k == (itso.num_steps_ - 1)) {
                    break;
                } else { /* otherwise, set Psi as a new trial basis */
                    if (ctx_.control().verbosity_ >= 3 && kp__->comm().rank() == 0) {
                        printf("subspace size limit reached\n");
                    }
                    hmlt_old.zero();
                    for (int i = 0; i < num_bands; i++) {
                        hmlt_old.set(i, i, eval[i]);
                    }
                    if (!itso.orthogonalize_) {
                        ovlp_old.zero();
                        for (int i = 0; i < num_bands; i++) {
                            ovlp_old.set(i, i, 1);
                        }
                    }

                    /* need to compute all hpsi and opsi states (not only unconverged) */
                    if (converge_by_energy) {
                        transform<T>(ctx_.processing_unit(), nc_mag ? 2 : ispin_step, 1.0, std::vector<Wave_functions*>({&hphi, &sphi}), 0, N, evec, 0, 0, 0.0, {&hpsi, &spsi}, 0, num_bands);
                    }

                    /* update basis functions, hphi and ophi */
                    for (int ispn = 0; ispn < num_sc; ispn++) {
                        phi.copy_from(ctx_.processing_unit(), num_bands, psi, nc_mag ? ispn : ispin_step, 0, nc_mag ? ispn : 0, 0);
                        hphi.copy_from(ctx_.processing_unit(), num_bands, hpsi, ispn, 0, ispn, 0);
                        sphi.copy_from(ctx_.processing_unit(), num_bands, spsi, ispn, 0, ispn, 0);
                    }
                    /* number of basis functions that we already have */
                    N = num_bands;
                }
            }

            /* expand variational subspace with new basis vectors obtatined from residuals */
            for (int ispn = 0; ispn < num_sc; ispn++) {
                phi.copy_from(ctx_.processing_unit(), n, res, ispn, 0, ispn, N);
            }

            /* apply Hamiltonian and overlap operators to the new basis functions */
            apply_h_s<T>(kp__, ispin_step, N, n, phi, hphi, sphi, d_op__, q_op__);

            if (itso.orthogonalize_) {
                orthogonalize<T>(ctx_.processing_unit(), nc_mag ? 2 : 0, phi, hphi, sphi, N, n, ovlp, res);
            }

            /* setup eigen-value problem
             * N is the number of previous basis functions
             * n is the number of new basis functions */
            set_subspace_mtrx(N, n, phi, hphi, hmlt, hmlt_old);

            if (ctx_.control().verification_ >= 1) {
                double max_diff = check_hermitian(hmlt, N + n);
                if (max_diff > 1e-12) {
                    std::stringstream s;
                    s << "H matrix is not hermitian, max_err = " << max_diff;
                    WARNING(s);
                }
            }

            if (!itso.orthogonalize_) {
                /* setup overlap matrix */
                set_subspace_mtrx(N, n, phi, sphi, ovlp, ovlp_old);

                if (ctx_.control().verification_ >= 1) {
                    double max_diff = check_hermitian(ovlp, N + n);
                    if (max_diff > 1e-12) {
                        std::stringstream s;
                        s << "S matrix is not hermitian, max_err = " << max_diff;
                        WARNING(s);
                    }
                }
            }

            /* increase size of the variation space */
            N += n;

            eval_old = eval;

            sddk::timer t1("sirius::Band::diag_pseudo_potential_davidson|evp");
            if (itso.orthogonalize_) {
                /* solve standard eigen-value problem with the size N */
                if (std_solver->solve(N, num_bands, hmlt, eval.data(), evec)) {
                    std::stringstream s;
                    s << "error in diagonalziation";
                    TERMINATE(s);
                }
            } else {
                /* solve generalized eigen-value problem with the size N */
                if (gen_solver->solve(N, num_bands, hmlt, ovlp, eval.data(), evec)) {
                    std::stringstream s;
                    s << "error in diagonalziation";
                    TERMINATE(s);
                }
            }
            t1.stop();

            if (ctx_.control().verbosity_ >= 2 && kp__->comm().rank() == 0) {
                DUMP("step: %i, current subspace size: %i, maximum subspace size: %i", k, N, num_phi);
                if (ctx_.control().verbosity_ >= 4) {
                    for (int i = 0; i < num_bands; i++) {
                        DUMP("eval[%i]=%20.16f, diff=%20.16f, occ=%20.16f", i, eval[i], std::abs(eval[i] - eval_old[i]),
                             kp__->band_occupancy(i + ispin_step * ctx_.num_fv_states()));
                    }
                }
            }
            niter++;
        }
    } /* loop over ispin_step */
    t3.stop();

    //phi.component(0).copy_from(psi.component(0), 0, num_bands, ctx_.processing_unit());
    //apply_h_o<T>(kp__, 0, 0, num_bands, phi, hphi, ophi, d_op__, q_op__);
    //for (int i = 0; i < num_bands; i++) {
    //    double rnorm = 0;
    //    for (int ig = 0; ig < kp__->num_gkvec(); ig++) {
    //        rnorm += std::pow(std::abs(hphi.component(0).pw_coeffs().prime(ig, i) - kp__->band_energy(i) * ophi.component(0).pw_coeffs().prime(ig, i)), 2);
    //    }
    //    std::cout << "band: " << i << ", l2norm: " << std::sqrt(rnorm) << std::endl;
    //}

    kp__->beta_projectors().dismiss();

    //if (ctx_.control().print_checksum_) {
    //    auto cs = psi.checksum(0, ctx_.num_fv_states());
    //    if (kp__->comm().rank() == 0) {
    //        DUMP("checksum(psi): %18.10f %18.10f", cs.real(), cs.imag());
    //    }
    //}

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            psi.pw_coeffs(ispn).copy_to_host(0, num_bands);
            if (!keep_wf_on_gpu) {
                psi.pw_coeffs(ispn).deallocate_on_device();
            }
        }
    }
    #endif

    return niter;
}

template <typename T>
inline void Band::diag_pseudo_potential_chebyshev(K_point* kp__,
                                                  int ispn__,
                                                  D_operator<T>& d_op__,
                                                  Q_operator<T>& q_op__,
                                                  P_operator<T>& p_op__) const
{
    PROFILE("sirius::Band::diag_pseudo_potential_chebyshev");

//==     auto pu = ctx_.processing_unit();
//== 
//==     /* short notation for number of target wave-functions */
//==     int num_bands = ctx_.num_fv_states();
//== 
//==     auto& itso = ctx_.iterative_solver_input_section();
//== 
//==     /* short notation for target wave-functions */
//==     auto& psi = kp__->spinor_wave_functions<false>(ispn__);
//== 
//== //== 
//== //==     //auto& beta_pw_panel = kp__->beta_pw_panel();
//== //==     //dmatrix<double_complex> S(unit_cell_.mt_basis_size(), unit_cell_.mt_basis_size(), kp__->blacs_grid());
//== //==     //linalg<CPU>::gemm(2, 0, unit_cell_.mt_basis_size(), unit_cell_.mt_basis_size(), kp__->num_gkvec(), complex_one,
//== //==     //                  beta_pw_panel, beta_pw_panel, complex_zero, S);
//== //==     //for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//== //==     //{
//== //==     //    auto type = unit_cell_.atom(ia)->type();
//== //==     //    int nbf = type->mt_basis_size();
//== //==     //    int ofs = unit_cell_.atom(ia)->offset_lo();
//== //==     //    matrix<double_complex> qinv(nbf, nbf);
//== //==     //    type->uspp().q_mtrx >> qinv;
//== //==     //    linalg<CPU>::geinv(nbf, qinv);
//== //==     //    for (int i = 0; i < nbf; i++)
//== //==     //    {
//== //==     //        for (int j = 0; j < nbf; j++) S.add(ofs + j, ofs + i, qinv(j, i));
//== //==     //    }
//== //==     //}
//== //==     //linalg<CPU>::geinv(unit_cell_.mt_basis_size(), S);
//== //== 
//== //== 
//==     /* maximum order of Chebyshev polynomial*/
//==     int order = itso.num_steps_ + 2;
//== 
//==     std::vector< Wave_functions<false>* > phi(order);
//==     for (int i = 0; i < order; i++) {
//==         phi[i] = new Wave_functions<false>(kp__->num_gkvec_loc(), num_bands, pu);
//==     }
//== 
//==     Wave_functions<false> hphi(kp__->num_gkvec_loc(), num_bands, pu);
//== 
//==     /* trial basis functions */
//==     phi[0]->copy_from(psi, 0, num_bands);
//== 
//==     /* apply Hamiltonian to the basis functions */
//==     apply_h<T>(kp__, ispn__, 0, num_bands, *phi[0], hphi, h_op__, d_op__);
//== 
//==     /* compute Rayleight quotients */
//==     std::vector<double> e0(num_bands, 0.0);
//==     if (pu == CPU) {
//==         #pragma omp parallel for schedule(static)
//==         for (int i = 0; i < num_bands; i++) {
//==             for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
//==                 e0[i] += std::real(std::conj((*phi[0])(igk, i)) * hphi(igk, i));
//==             }
//==         }
//==     }
//==     kp__->comm().allreduce(e0);
//== 
//==     //== if (parameters_.processing_unit() == GPU)
//==     //== {
//==     //==     #ifdef __GPU
//==     //==     mdarray<double, 1> e0_loc(kp__->spl_fv_states().local_size());
//==     //==     e0_loc.allocate_on_device();
//==     //==     e0_loc.zero_on_device();
//== 
//==     //==     compute_inner_product_gpu(kp__->num_gkvec_row(),
//==     //==                               (int)kp__->spl_fv_states().local_size(),
//==     //==                               phi[0].at<GPU>(),
//==     //==                               hphi.at<GPU>(),
//==     //==                               e0_loc.at<GPU>());
//==     //==     e0_loc.copy_to_host();
//==     //==     for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
//==     //==     {
//==     //==         int i = kp__->spl_fv_states(iloc);
//==     //==         e0[i] = e0_loc(iloc);
//==     //==     }
//==     //==     #endif
//==     //== }
//==     //== 
//== 
//==     /* estimate low and upper bounds of the Chebyshev filter */
//==     double lambda0 = -1e10;
//==     //double emin = 1e100;
//==     for (int i = 0; i < num_bands; i++)
//==     {
//==         lambda0 = std::max(lambda0, e0[i]);
//==         //emin = std::min(emin, e0[i]);
//==     }
//==     double lambda1 = 0.5 * std::pow(ctx_.gk_cutoff(), 2);
//== 
//==     double r = (lambda1 - lambda0) / 2.0;
//==     double c = (lambda1 + lambda0) / 2.0;
//== 
//==     auto apply_p = [kp__, &p_op__, num_bands](Wave_functions<false>& phi, Wave_functions<false>& op_phi) {
//==         op_phi.copy_from(phi, 0, num_bands);
//==         //for (int i = 0; i < kp__->beta_projectors().num_beta_chunks(); i++) {
//==         //    kp__->beta_projectors().generate(i);
//== 
//==         //    kp__->beta_projectors().inner<T>(i, phi, 0, num_bands);
//== 
//==         //    p_op__.apply(i, 0, op_phi, 0, num_bands);
//==         //}
//==     };
//== 
//==     apply_p(hphi, *phi[1]);
//==     
//==     /* compute \psi_1 = (S^{-1}H\psi_0 - c\psi_0) / r */
//==     if (pu == CPU) {
//==         #pragma omp parallel for schedule(static)
//==         for (int i = 0; i < num_bands; i++) {
//==             for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
//==                 (*phi[1])(igk, i) = ((*phi[1])(igk, i) - (*phi[0])(igk, i) * c) / r;
//==             }
//==         }
//==     }
//== //==     //if (parameters_.processing_unit() == GPU)
//== //==     //{
//== //==     //    #ifdef __GPU
//== //==     //    compute_chebyshev_polynomial_gpu(kp__->num_gkvec_row(), (int)kp__->spl_fv_states().local_size(), c, r,
//== //==     //                                     phi[0].at<GPU>(), phi[1].at<GPU>(), NULL);
//== //==     //    phi[1].panel().copy_to_host();
//== //==     //    #endif
//== //==     //}
//== //== 
//== 
//==     /* compute higher polynomial orders */
//==     for (int k = 2; k < order; k++) {
//== 
//==         apply_h<T>(kp__, ispn__, 0, num_bands, *phi[k - 1], hphi, h_op__, d_op__);
//== 
//==         apply_p(hphi, *phi[k]);
//== 
//==         if (pu == CPU) {
//==             #pragma omp parallel for schedule(static)
//==             for (int i = 0; i < num_bands; i++) {
//==                 for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
//==                     (*phi[k])(igk, i) = ((*phi[k])(igk, i) - c * (*phi[k - 1])(igk, i)) * 2.0 / r - (*phi[k - 2])(igk, i);
//==                 }
//==             }
//==         }
//==         //== if (parameters_.processing_unit() == GPU)
//==         //== {
//==         //==     #ifdef __GPU
//==         //==     compute_chebyshev_polynomial_gpu(kp__->num_gkvec(), num_bands, c, r,
//==         //==                                      phi[k - 2].at<GPU>(), phi[k - 1].at<GPU>(), phi[k].at<GPU>());
//==         //==     phi[k].copy_to_host();
//==         //==     #endif
//==         //== }
//==     }
//== 
//==     /* allocate Hamiltonian and overlap */
//==     matrix<T> hmlt(num_bands, num_bands);
//==     matrix<T> ovlp(num_bands, num_bands);
//==     matrix<T> evec(num_bands, num_bands);
//==     matrix<T> hmlt_old;
//==     matrix<T> ovlp_old;
//== 
//==     int bs = ctx_.cyclic_block_size();
//== 
//==     dmatrix<T> hmlt_dist;
//==     dmatrix<T> ovlp_dist;
//==     dmatrix<T> evec_dist;
//==     if (kp__->comm().size() == 1) {
//==         hmlt_dist = dmatrix<T>(&hmlt(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
//==         ovlp_dist = dmatrix<T>(&ovlp(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
//==         evec_dist = dmatrix<T>(&evec(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
//==     } else {
//==         hmlt_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
//==         ovlp_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
//==         evec_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
//==     }
//== 
//==     std::vector<double> eval(num_bands);
//== 
//==     /* apply Hamiltonian and overlap operators to the new basis functions */
//==     apply_h_o<T>(kp__, ispn__, 0, num_bands, *phi[order - 1], hphi, *phi[0], h_op__, d_op__, q_op__);
//==     
//==     //orthogonalize<T>(kp__, N, n, phi, hphi, ophi, ovlp);
//== 
//==     /* setup eigen-value problem */
//==     set_h_o<T>(kp__, 0, num_bands, *phi[order - 1], hphi, *phi[0], hmlt, ovlp, hmlt_old, ovlp_old);
//== 
//==     /* solve generalized eigen-value problem with the size N */
//==     diag_h_o<T>(kp__, num_bands, num_bands, hmlt, ovlp, evec, hmlt_dist, ovlp_dist, evec_dist, eval);
//== 
//==     /* recompute wave-functions */
//==     /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
//==     psi.transform_from<T>(*phi[order - 1], num_bands, evec, num_bands);
//== 
//==     for (int j = 0; j < ctx_.num_fv_states(); j++) {
//==         kp__->band_energy(j + ispn__ * ctx_.num_fv_states()) = eval[j];
//==     }
//== 
//==     for (int i = 0; i < order; i++) {
//==         delete phi[i];
//==     }
}

//template <typename T>
//inline T 
//inner_local(K_point* kp__,
//            wave_functions& a,
//            int ia,
//            wave_functions& b,
//            int ib);
//
//template<>
//inline double 
//inner_local<double>(K_point* kp__,
//                    wave_functions& a,
//                    int ia,
//                    wave_functions& b,
//                    int ib)
//{
//    double result{0};
//    double* a_tmp = reinterpret_cast<double*>(&a.pw_coeffs().prime(0, ia));
//    double* b_tmp = reinterpret_cast<double*>(&b.pw_coeffs().prime(0, ib));
//    for (int igk = 0; igk < 2 * kp__->num_gkvec_loc(); igk++) {
//        result += a_tmp[igk] * b_tmp[igk];
//    }
//
//    if (kp__->comm().rank() == 0) {
//        result = 2 * result - a_tmp[0] * b_tmp[0];
//    } else {
//        result *= 2;
//    }
//
//    return result;
//}
//
//template<>
//inline double_complex 
//inner_local<double_complex>(K_point* kp__,
//                            wave_functions& a,
//                            int ia,
//                            wave_functions& b,
//                            int ib)
//{
//    double_complex result{0, 0};
//    for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
//        result += std::conj(a.pw_coeffs().prime(igk, ia)) * b.pw_coeffs().prime(igk, ib);
//    }
//    return result;
//}

template <typename T>
inline void Band::diag_pseudo_potential_rmm_diis(K_point* kp__,
                                                 int ispn__,
                                                 D_operator<T>& d_op__,
                                                 Q_operator<T>& q_op__) const

{
    //auto& itso = ctx_.iterative_solver_input();
    double tol = ctx_.iterative_solver_tolerance();

    if (tol > 1e-4) {
        diag_pseudo_potential_davidson(kp__, d_op__, q_op__);
        return;
    }

    PROFILE("sirius::Band::diag_pseudo_potential_rmm_diis");

    /* get diagonal elements for preconditioning */
    //auto h_diag = get_h_diag(kp__, ispn__, local_op_->v0(ispn__), d_op__);
    //auto o_diag = get_o_diag(kp__, q_op__);
    STOP();

//    /* short notation for number of target wave-functions */
//    int num_bands = ctx_.num_fv_states();
//
//    //auto pu = ctx_.processing_unit();
//
//    /* short notation for target wave-functions */
//    auto& psi = kp__->spinor_wave_functions(ispn__);
//
//    int niter = itso.num_steps_;
//
//    Eigenproblem_lapack evp_solver(2 * linalg_base::dlamch('S'));
//
//    std::vector<wave_functions*> phi(niter);
//    std::vector<wave_functions*> res(niter);
//    std::vector<wave_functions*> ophi(niter);
//    std::vector<wave_functions*> hphi(niter);
//
//    for (int i = 0; i < niter; i++) {
//        phi[i]  = new wave_functions(ctx_.processing_unit(), kp__->gkvec(), num_bands);
//        res[i]  = new wave_functions(ctx_.processing_unit(), kp__->gkvec(), num_bands);
//        hphi[i] = new wave_functions(ctx_.processing_unit(), kp__->gkvec(), num_bands);
//        ophi[i] = new wave_functions(ctx_.processing_unit(), kp__->gkvec(), num_bands);
//    }
//
//    wave_functions  phi_tmp(ctx_.processing_unit(), kp__->gkvec(), num_bands);
//    wave_functions hphi_tmp(ctx_.processing_unit(), kp__->gkvec(), num_bands);
//    wave_functions ophi_tmp(ctx_.processing_unit(), kp__->gkvec(), num_bands);
//
//    auto mem_type = (gen_evp_solver_->type() == ev_magma) ? memory_t::host_pinned : memory_t::host;
//
//    int bs = ctx_.cyclic_block_size();
//
//    dmatrix<T> hmlt(num_bands, num_bands, ctx_.blacs_grid(), bs, bs, mem_type);
//    dmatrix<T> ovlp(num_bands, num_bands, ctx_.blacs_grid(), bs, bs, mem_type);
//    dmatrix<T> evec(num_bands, num_bands, ctx_.blacs_grid(), bs, bs, mem_type);
//    dmatrix<T> hmlt_old;
//    dmatrix<T> ovlp_old;
//
//    std::vector<double> eval(num_bands);
//    for (int i = 0; i < num_bands; i++) {
//        eval[i] = kp__->band_energy(i);
//    }
//    std::vector<double> eval_old(num_bands);
//
//    /* trial basis functions */
//    phi[0]->copy_from(psi, 0, num_bands);
//
//    std::vector<int> last(num_bands, 0);
//    std::vector<bool> conv_band(num_bands, false);
//    std::vector<double> res_norm(num_bands);
//    std::vector<double> res_norm_start(num_bands);
//    std::vector<double> lambda(num_bands, 0);
//    
//    auto update_res = [kp__, num_bands, &phi, &res, &hphi, &ophi, &last, &conv_band]
//                      (std::vector<double>& res_norm__, std::vector<double>& eval__) -> void
//    {
//        sddk::timer t("sirius::Band::diag_pseudo_potential_rmm_diis|res");
//        std::vector<double> e_tmp(num_bands, 0), d_tmp(num_bands, 0);
//
//        #pragma omp parallel for
//        for (int i = 0; i < num_bands; i++) {
//            if (!conv_band[i]) {
//                e_tmp[i] = std::real(inner_local<T>(kp__, *phi[last[i]], i, *hphi[last[i]], i));
//                d_tmp[i] = std::real(inner_local<T>(kp__, *phi[last[i]], i, *ophi[last[i]], i));
//            }
//        }
//        kp__->comm().allreduce(e_tmp);
//        kp__->comm().allreduce(d_tmp);
//        
//        res_norm__ = std::vector<double>(num_bands, 0);
//        #pragma omp parallel for
//        for (int i = 0; i < num_bands; i++) {
//            if (!conv_band[i]) {
//                eval__[i] = e_tmp[i] / d_tmp[i];
//
//                /* compute residual r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
//                for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
//                    (*res[last[i]]).pw_coeffs().prime(igk, i) = (*hphi[last[i]]).pw_coeffs().prime(igk, i) - eval__[i] * (*ophi[last[i]]).pw_coeffs().prime(igk, i);
//                }
//                res_norm__[i] = std::real(inner_local<T>(kp__, *res[last[i]], i, *res[last[i]], i));
//            }
//        }
//        kp__->comm().allreduce(res_norm__);
//        for (int i = 0; i < num_bands; i++) {
//            if (!conv_band[i]) {
//                res_norm__[i] = std::sqrt(res_norm__[i]);
//            }
//        }
//    };
//
//    auto apply_h_o = [this, kp__, num_bands, &phi, &phi_tmp, &hphi, &hphi_tmp, &ophi, &ophi_tmp, &conv_band, &last,
//                      &d_op__, &q_op__, ispn__]() -> int
//    {
//        sddk::timer t("sirius::Band::diag_pseudo_potential_rmm_diis|h_o");
//        int n{0};
//        for (int i = 0; i < num_bands; i++) {
//            if (!conv_band[i]) {
//                std::memcpy(&phi_tmp.pw_coeffs().prime(0, n), &(*phi[last[i]]).pw_coeffs().prime(0, i), kp__->num_gkvec_loc() * sizeof(double_complex));
//                n++;
//            }
//        }
//
//        if (n == 0) {
//            return 0;
//        }
//        
//        /* apply Hamiltonian and overlap operators to the initial basis functions */
//        this->apply_h_o<T>(kp__, ispn__, 0, n, phi_tmp, hphi_tmp, ophi_tmp, d_op__, q_op__);
//
//        n = 0;
//        for (int i = 0; i < num_bands; i++) {
//            if (!conv_band[i]) {
//                std::memcpy(&(*hphi[last[i]]).pw_coeffs().prime(0, i), &hphi_tmp.pw_coeffs().prime(0, n), kp__->num_gkvec_loc() * sizeof(double_complex));
//                std::memcpy(&(*ophi[last[i]]).pw_coeffs().prime(0, i), &ophi_tmp.pw_coeffs().prime(0, n), kp__->num_gkvec_loc() * sizeof(double_complex));
//                n++;
//            }
//        }
//        return n;
//    };
//
//    STOP();
//
//    //auto apply_preconditioner = [kp__, num_bands, &h_diag, &o_diag, &eval, &conv_band]
//    //                            (std::vector<double> lambda,
//    //                             wave_functions& res__,
//    //                             double alpha,
//    //                             wave_functions& kres__) -> void
//    //{
//    //    sddk::timer t("sirius::Band::diag_pseudo_potential_rmm_diis|pre");
//    //    #pragma omp parallel for
//    //    for (int i = 0; i < num_bands; i++) {
//    //        if (!conv_band[i]) {
//    //            for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
//    //                double p = h_diag[igk] - eval[i] * o_diag[igk];
//
//    //                p *= 2; // QE formula is in Ry; here we convert to Ha
//    //                p = 0.25 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
//    //                kres__.pw_coeffs().prime(igk, i) = alpha * kres__.pw_coeffs().prime(igk, i) + lambda[i] * res__.pw_coeffs().prime(igk, i) / p;
//    //            }
//    //        }
//
//    //        //== double Ekin = 0;
//    //        //== double norm = 0;
//    //        //== for (int igk = 0; igk < kp__->num_gkvec(); igk++)
//    //        //== {
//    //        //==     Ekin += 0.5 * std::pow(std::abs(res__(igk, i)), 2) * std::pow(kp__->gkvec_cart(igk).length(), 2);
//    //        //==     norm += std::pow(std::abs(res__(igk, i)), 2);
//    //        //== }
//    //        //== Ekin /= norm;
//    //        //== for (int igk = 0; igk < kp__->num_gkvec(); igk++)
//    //        //== {
//    //        //==     double x = std::pow(kp__->gkvec_cart(igk).length(), 2) / 3 / Ekin;
//    //        //==     kres__(igk, i) = alpha * kres__(igk, i) + lambda[i] * res__(igk, i) * 
//    //        //==         (4.0 / 3 / Ekin) * (27 + 18 * x + 12 * x * x + 8 * x * x * x) / (27 + 18 * x + 12 * x * x + 8 * x * x * x + 16 * x * x * x * x);
//    //        //== }
//    //    }
//    //};
//
//    /* apply Hamiltonian and overlap operators to the initial basis functions */
//    this->apply_h_o<T>(kp__, ispn__, 0, num_bands, *phi[0], *hphi[0], *ophi[0], d_op__, q_op__);
//    
//    /* compute initial residuals */
//    update_res(res_norm_start, eval);
//
//    bool conv{true};
//    for (int i = 0; i < num_bands; i++) {
//        if (res_norm_start[i] > itso.residual_tolerance_) {
//            conv = false;
//        }
//    }
//    if (conv) {
//        DUMP("all bands are converged at stage#0");
//        return;
//    }
//
//    last = std::vector<int>(num_bands, 1);
//    
//    phi[1]->pw_coeffs().prime().zero();
//    /* apply preconditioner to the initial residuals */
//    //apply_preconditioner(std::vector<double>(num_bands, 1), *res[0], 0.0, *phi[1]);
//    STOP();
//    
//    /* apply H and O to the preconditioned residuals */
//    apply_h_o();
//
//    /* estimate lambda */
//    std::vector<double> f1(num_bands, 0);
//    std::vector<double> f2(num_bands, 0);
//    std::vector<double> f3(num_bands, 0);
//    std::vector<double> f4(num_bands, 0);
//
//    #pragma omp parallel for
//    for (int i = 0; i < num_bands; i++) {
//        if (!conv_band[i]) {
//            f1[i] = std::real(inner_local<T>(kp__, *phi[1], i, *ophi[1], i));     //  <KR_i | OKR_i>
//            f2[i] = std::real(inner_local<T>(kp__, *phi[0], i, *ophi[1], i)) * 2; // <phi_i | OKR_i>
//            f3[i] = std::real(inner_local<T>(kp__, *phi[1], i, *hphi[1], i));     //  <KR_i | HKR_i>
//            f4[i] = std::real(inner_local<T>(kp__, *phi[0], i, *hphi[1], i)) * 2; // <phi_i | HKR_i>
//        }
//    }
//    kp__->comm().allreduce(f1);
//    kp__->comm().allreduce(f2);
//    kp__->comm().allreduce(f3);
//    kp__->comm().allreduce(f4);
//
//    #pragma omp parallel for
//    for (int i = 0; i < num_bands; i++) {
//        if (!conv_band[i]) {
//            double a = f1[i] * f4[i] - f2[i] * f3[i];
//            double b = f3[i] - eval[i] * f1[i];
//            double c = eval[i] * f2[i] - f4[i];
//
//            lambda[i] = (b - std::sqrt(b * b - 4.0 * a * c)) / 2.0 / a;
//            if (std::abs(lambda[i]) > 2.0) {
//                lambda[i] = 2.0 * Utils::sign(lambda[i]);
//            }
//            if (std::abs(lambda[i]) < 0.5) {
//                lambda[i] = 0.5 * Utils::sign(lambda[i]);
//            }
//            
//            /* construct new basis functions */
//            for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
//                 (*phi[1]).pw_coeffs().prime(igk, i) =  (*phi[0]).pw_coeffs().prime(igk, i) + lambda[i] *  (*phi[1]).pw_coeffs().prime(igk, i);
//                (*hphi[1]).pw_coeffs().prime(igk, i) = (*hphi[0]).pw_coeffs().prime(igk, i) + lambda[i] * (*hphi[1]).pw_coeffs().prime(igk, i);
//                (*ophi[1]).pw_coeffs().prime(igk, i) = (*ophi[0]).pw_coeffs().prime(igk, i) + lambda[i] * (*ophi[1]).pw_coeffs().prime(igk, i);
//            }
//        }
//    }
//    /* compute new residuals */
//    update_res(res_norm, eval);
//    /* check which bands have converged */
//    for (int i = 0; i < num_bands; i++) {
//        if (res_norm[i] < itso.residual_tolerance_) {
//            conv_band[i] = true;
//        }
//    }
//
//    mdarray<T, 3> A(niter, niter, num_bands);
//    mdarray<T, 3> B(niter, niter, num_bands);
//    mdarray<T, 2> V(niter, num_bands);
//    std::vector<double> ev(niter);
//    
//    /* start adjusting residuals */
//    for (int iter = 2; iter < niter; iter++) {
//        sddk::timer t1("sirius::Band::diag_pseudo_potential_rmm_diis|AB");
//        A.zero();
//        B.zero();
//        for (int i = 0; i < num_bands; i++) {
//            if (!conv_band[i]) {
//                for (int i1 = 0; i1 < iter; i1++) {
//                    for (int i2 = 0; i2 < iter; i2++) {
//                        A(i1, i2, i) = inner_local<T>(kp__, *res[i1], i, *res[i2], i);
//                        B(i1, i2, i) = inner_local<T>(kp__, *phi[i1], i, *ophi[i2], i);
//                    }
//                }
//            }
//        }
//        kp__->comm().allreduce(A.template at<CPU>(), (int)A.size());
//        kp__->comm().allreduce(B.template at<CPU>(), (int)B.size());
//        t1.stop();
//
//        sddk::timer t2("sirius::Band::diag_pseudo_potential_rmm_diis|phi");
//        for (int i = 0; i < num_bands; i++) {
//            if (!conv_band[i]) {
//                if (evp_solver.solve(iter, 1, &A(0, 0, i), A.ld(), &B(0, 0, i), B.ld(), &ev[0], &V(0, i), V.ld()) == 0) {
//                    /* zero phi */
//                    std::memset(&(*phi[iter]).pw_coeffs().prime(0, i), 0, kp__->num_gkvec_loc() * sizeof(double_complex));
//                    /* zero residual */
//                    std::memset(&(*res[iter]).pw_coeffs().prime(0, i), 0, kp__->num_gkvec_loc() * sizeof(double_complex));
//                    /* make linear combinations */
//                    for (int i1 = 0; i1 < iter; i1++) {
//                        for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
//                            (*phi[iter]).pw_coeffs().prime(igk, i) += (*phi[i1]).pw_coeffs().prime(igk, i) * V(i1, i);
//                            (*res[iter]).pw_coeffs().prime(igk, i) += (*res[i1]).pw_coeffs().prime(igk, i) * V(i1, i);
//                        }
//                    }
//                    last[i] = iter;
//                } else {
//                    conv_band[i] = true;
//                }
//            }
//        }
//        t2.stop();
//        
//        //apply_preconditioner(lambda, *res[iter], 1.0, *phi[iter]);
//        STOP();
//
//        apply_h_o();
//
//        eval_old = eval;
//
//        update_res(res_norm, eval);
//        
//        for (int i = 0; i < num_bands; i++) {
//            if (!conv_band[i]) {
//                if (res_norm[i] < itso.residual_tolerance_) {
//                    conv_band[i] = true;
//                }
//                //if (kp__->band_occupancy(i) <= 1e-2) {
//                //    conv_band[i] = true;
//                //}
//                //if (kp__->band_occupancy(i) > 1e-2 && std::abs(eval[i] - eval_old[i]) < tol) {
//                //    conv_band[i] = true;
//                //}
//                //if (kp__->band_occupancy(i) > 1e-2 && res_norm[i] < itso.residual_tolerance_) {
//                //    conv_band[i] = true;
//                //}
//                //if (kp__->band_occupancy(i) <= 1e-2 ||
//                //    res_norm[i] / res_norm_start[i] < 0.7 ||
//                //    (kp__->band_occupancy(i) > 1e-2 && std::abs(eval[i] - eval_old[i]) < tol)) {
//                //    conv_band[i] = true;
//                //}
//            }
//        }
//        if (std::all_of(conv_band.begin(), conv_band.end(), [](bool e){return e;})) {
//            std::cout << "early exit from the diis loop" << std::endl;
//            break;
//        }
//    }
//
//    #pragma omp parallel for
//    for (int i = 0; i < num_bands; i++) {
//        std::memcpy(&phi_tmp.pw_coeffs().prime(0, i),  &(*phi[last[i]]).pw_coeffs().prime(0, i),  kp__->num_gkvec_loc() * sizeof(double_complex));
//        std::memcpy(&hphi_tmp.pw_coeffs().prime(0, i), &(*hphi[last[i]]).pw_coeffs().prime(0, i), kp__->num_gkvec_loc() * sizeof(double_complex));
//        std::memcpy(&ophi_tmp.pw_coeffs().prime(0, i), &(*ophi[last[i]]).pw_coeffs().prime(0, i), kp__->num_gkvec_loc() * sizeof(double_complex));
//    }
//    orthogonalize<T>(0, num_bands, phi_tmp, hphi_tmp, ophi_tmp, ovlp, *res[0]);
//
//    /* setup eigen-value problem
//     * N is the number of previous basis functions
//     * n is the number of new basis functions */
//    set_subspace_mtrx(0, num_bands, phi_tmp, hphi_tmp, hmlt, hmlt_old);
//
//    if (std_evp_solver().solve(num_bands, num_bands, hmlt.template at<CPU>(), hmlt.ld(),
//                               eval.data(), evec.template at<CPU>(), evec.ld(),
//                               hmlt.num_rows_local(), hmlt.num_cols_local())) {
//        std::stringstream s;
//        s << "error in diagonalziation";
//        TERMINATE(s);
//    }
//
//    /* recompute wave-functions */
//    /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
//    transform<T>(phi_tmp, 0, num_bands, evec, 0, 0, psi, 0, num_bands);
//
//    for (int j = 0; j < ctx_.num_fv_states(); j++) {
//        kp__->band_energy(j + ispn__ * ctx_.num_fv_states()) = eval[j];
//    }
//
//    for (int i = 0; i < niter; i++) {
//        delete phi[i];
//        delete res[i];
//        delete hphi[i];
//        delete ophi[i];
//    }
}

