inline void K_point::initialize()
{
    PROFILE("sirius::K_point::initialize");
    
    zil_.resize(ctx_.lmax_apw() + 1);
    for (int l = 0; l <= ctx_.lmax_apw(); l++) {
        zil_[l] = std::pow(double_complex(0, 1), l);
    }
   
    l_by_lm_ = Utils::l_by_lm(ctx_.lmax_apw());

    int bs = ctx_.cyclic_block_size();

    /* In case of collinear magnetism we store only non-zero spinor components.
     *
     * non magnetic case: 
     * +---+
     * |   |
     * +---+
     *
     * collinear case:
     * +---+
     * |uu |
     * +---+---+
     *     |dd |
     *     +---+
     *
     * non collinear case:
     * +-------+
     * |       |
     * +-------+
     * |       |
     * +-------+
     */
    int nst = (ctx_.num_mag_dims() == 3) ? ctx_.num_bands() : ctx_.num_fv_states();

    auto mem_type_evp = (ctx_.std_evp_solver_type() == ev_magma) ? memory_t::host_pinned : memory_t::host;
    auto mem_type_gevp = (ctx_.gen_evp_solver_type() == ev_magma) ? memory_t::host_pinned : memory_t::host;

    if (use_second_variation && ctx_.need_sv()) {
        /* in case of collinear magnetism store pure up and pure dn components, otherwise store the full matrix */
        sv_eigen_vectors_[0] = dmatrix<double_complex>(nst, nst, ctx_.blacs_grid(), bs, bs, mem_type_evp);
        if (ctx_.num_mag_dims() == 1) {
            sv_eigen_vectors_[1] = dmatrix<double_complex>(nst, nst, ctx_.blacs_grid(), bs, bs, mem_type_evp);
        }
    }

    if (use_second_variation) {
        fv_eigen_values_.resize(ctx_.num_fv_states());
    }

    /* Build a full list of G+k vectors for all MPI ranks */
    generate_gkvec(ctx_.gk_cutoff());
    /* build a list of basis functions */
    generate_gklo_basis();

    if (ctx_.esm_type() == electronic_structure_method_t::full_potential_lapwlo) {
        if (ctx_.iterative_solver_input().type_ == "exact") {
            alm_coeffs_row_ = std::unique_ptr<Matching_coefficients>(
                new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec_row(), igk_row_, gkvec_));
            alm_coeffs_col_ = std::unique_ptr<Matching_coefficients>(
                new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec_col(), igk_col_, gkvec_));
        }
        alm_coeffs_loc_ = std::unique_ptr<Matching_coefficients>(
            new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec_loc(), igk_loc_, gkvec_));
    }

    if (!ctx_.full_potential()) {
        /* compute |beta> projectors for atom types */
        beta_projectors_ = std::unique_ptr<Beta_projectors>(new Beta_projectors(ctx_, gkvec_));

        //if (false) {
        //    p_mtrx_ = mdarray<double_complex, 3>(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(), unit_cell_.num_atom_types());
        //    p_mtrx_.zero();

        //    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        //        auto& atom_type = unit_cell_.atom_type(iat);

        //        if (!atom_type.pp_desc().augment) {
        //            continue;
        //        }
        //        int nbf = atom_type.mt_basis_size();
        //        int ofs = atom_type.offset_lo();

        //        matrix<double_complex> qinv(nbf, nbf);
        //        for (int xi1 = 0; xi1 < nbf; xi1++) {
        //            for (int xi2 = 0; xi2 < nbf; xi2++) {
        //                qinv(xi2, xi1) = ctx_.augmentation_op(iat).q_mtrx(xi2, xi1);
        //            }
        //        }
        //        linalg<CPU>::geinv(nbf, qinv);
        //        
        //        /* compute P^{+}*P */
        //        linalg<CPU>::gemm(2, 0, nbf, nbf, num_gkvec_loc(),
        //                          beta_projectors_->beta_gk_t().at<CPU>(0, ofs), beta_projectors_->beta_gk_t().ld(), 
        //                          beta_projectors_->beta_gk_t().at<CPU>(0, ofs), beta_projectors_->beta_gk_t().ld(), 
        //                          &p_mtrx_(0, 0, iat), p_mtrx_.ld());
        //        comm().allreduce(&p_mtrx_(0, 0, iat), unit_cell_.max_mt_basis_size() * unit_cell_.max_mt_basis_size());

        //        for (int xi1 = 0; xi1 < nbf; xi1++) {
        //            for (int xi2 = 0; xi2 < nbf; xi2++) {
        //                qinv(xi2, xi1) += p_mtrx_(xi2, xi1, iat);
        //            }
        //        }
        //        /* compute (Q^{-1} + P^{+}*P)^{-1} */
        //        linalg<CPU>::geinv(nbf, qinv);
        //        for (int xi1 = 0; xi1 < nbf; xi1++) {
        //            for (int xi2 = 0; xi2 < nbf; xi2++) {
        //                p_mtrx_(xi2, xi1, iat) = qinv(xi2, xi1);
        //            }
        //        }
        //    }
        //}
    }

    if (ctx_.full_potential()) {
        if (use_second_variation) {
            /* allocate fv eien vectors */
            fv_eigen_vectors_slab_ = std::unique_ptr<wave_functions>(
                new wave_functions(ctx_.processing_unit(), gkvec(), unit_cell_.num_atoms(),
                    [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, ctx_.num_fv_states()));

            fv_eigen_vectors_slab_->pw_coeffs().prime().zero();
            fv_eigen_vectors_slab_->mt_coeffs().prime().zero();
            /* starting guess for wave-functions */
            for (int i = 0; i < ctx_.num_fv_states(); i++) {
                for (int igloc = 0; igloc < gkvec().gvec_count(comm().rank()); igloc++) {
                    int ig = igloc + gkvec().gvec_offset(comm().rank());
                    if (ig == i) {
                        fv_eigen_vectors_slab_->pw_coeffs().prime(igloc, i) = 1.0;
                    }
                    if (ig == i + 1) {
                        fv_eigen_vectors_slab_->pw_coeffs().prime(igloc, i) = 0.5;
                    }
                    if (ig == i + 2) {
                        fv_eigen_vectors_slab_->pw_coeffs().prime(igloc, i) = 0.125;
                    }
                }
            }
            if (ctx_.iterative_solver_input().type_ == "exact") {
                fv_eigen_vectors_ = dmatrix<double_complex>(gklo_basis_size(), ctx_.num_fv_states(), ctx_.blacs_grid(), bs, bs, mem_type_gevp);
            } else {
                int ncomp = ctx_.iterative_solver_input().num_singular_;
                if (ncomp < 0) {
                    ncomp = ctx_.num_fv_states();
                }

                singular_components_ = std::unique_ptr<wave_functions>(new wave_functions(ctx_.processing_unit(), gkvec(), ncomp));
                singular_components_->pw_coeffs().prime().zero();
                /* starting guess for wave-functions */
                for (int i = 0; i < ncomp; i++) {
                    for (int igloc = 0; igloc < gkvec().gvec_count(comm().rank()); igloc++) {
                        int ig = igloc + gkvec().gvec_offset(comm().rank());
                        if (ig == i) {
                            singular_components_->pw_coeffs().prime(igloc, i) = 1.0;
                        }
                        if (ig == i + 1) {
                            singular_components_->pw_coeffs().prime(igloc, i) = 0.5;
                        }
                        if (ig == i + 2) {
                            singular_components_->pw_coeffs().prime(igloc, i) = 0.125;
                        }
                        singular_components_->pw_coeffs().prime(igloc, i) += 0.01 * type_wrapper<double_complex>::random();
                    }
                }
            }

            fv_states_ = std::unique_ptr<wave_functions>(new wave_functions(ctx_.processing_unit(),
                                                                            gkvec(),
                                                                            unit_cell_.num_atoms(),
                                                                            [this](int ia)
                                                                            {
                                                                                return unit_cell_.atom(ia).mt_basis_size();
                                                                            },
                                                                            ctx_.num_fv_states()));

            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                spinor_wave_functions_[ispn] = std::unique_ptr<wave_functions>(new wave_functions(ctx_.processing_unit(),
                                                                                                  gkvec(),
                                                                                                  unit_cell_.num_atoms(),
                                                                                                  [this](int ia)
                                                                                                  {
                                                                                                      return unit_cell_.atom(ia).mt_basis_size();
                                                                                                  },
                                                                                                  nst));
            }
        } else {
            TERMINATE_NOT_IMPLEMENTED
        }
    } else {
        assert(ctx_.num_fv_states() < num_gkvec());

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            spinor_wave_functions_[ispn] = std::unique_ptr<wave_functions>(new wave_functions(ctx_.processing_unit(), gkvec(), nst));

            if (ctx_.processing_unit() == GPU) {
                /* allocate GPU memory */
                spinor_wave_functions_[ispn]->pw_coeffs().prime().allocate(memory_t::device);
            }
        }
    }
}
