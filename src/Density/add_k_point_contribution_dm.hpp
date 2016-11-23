template <typename T> 
inline void Density::add_k_point_contribution_dm(K_point* kp__,
                                                 mdarray<double_complex, 4>& density_matrix__)
{
    PROFILE_WITH_TIMER("sirius::Density::add_k_point_contribution_dm");
    
    if (ctx_.esm_type() == electronic_structure_method_t::full_potential_lapwlo) {
        /* non-magnetic or spin-collinear case */
        if (ctx_.num_mag_dims() != 3) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                int nbnd = kp__->num_occupied_bands(ispn);

                mdarray<double_complex, 2> wf1(unit_cell_.max_mt_basis_size(), nbnd);
                mdarray<double_complex, 2> wf2(unit_cell_.max_mt_basis_size(), nbnd);

                for (int ialoc = 0; ialoc < kp__->spinor_wave_functions(ispn).spl_num_atoms().local_size(); ialoc++) {
                    int ia = kp__->spinor_wave_functions(ispn).spl_num_atoms()[ialoc];
                    int mt_basis_size = unit_cell_.atom(ia).type().mt_basis_size();
                    int offset_wf = kp__->spinor_wave_functions(ispn).offset_mt_coeffs(ialoc);

                    for (int i = 0; i < nbnd; i++) {
                        for (int xi = 0; xi < mt_basis_size; xi++) {
                            auto c = kp__->spinor_wave_functions(ispn).mt_coeffs().prime(offset_wf + xi, i);
                            wf1(xi, i) = std::conj(c);
                            wf2(xi, i) = c * kp__->band_occupancy(i + ispn * ctx_.num_fv_states()) * kp__->weight();
                        }
                    }
                    /* add |psi_j> n_j <psi_j| to density matrix */
                    linalg<CPU>::gemm(0, 1, mt_basis_size, mt_basis_size, nbnd, complex_one, 
                                          &wf1(0, 0), wf1.ld(), &wf2(0, 0), wf2.ld(), complex_one, 
                                          density_matrix__.at<CPU>(0, 0, ispn, ia), density_matrix__.ld());
                }
            }
        } else {
            int nbnd = kp__->num_occupied_bands();

            mdarray<double_complex, 3> wf1(unit_cell_.max_mt_basis_size(), nbnd, ctx_.num_spins());
            mdarray<double_complex, 3> wf2(unit_cell_.max_mt_basis_size(), nbnd, ctx_.num_spins());

            assert(kp__->spinor_wave_functions(0).spl_num_atoms().local_size() ==
                   kp__->spinor_wave_functions(1).spl_num_atoms().local_size());

            for (int ialoc = 0; ialoc < kp__->spinor_wave_functions(0).spl_num_atoms().local_size(); ialoc++) {
                int ia = kp__->spinor_wave_functions(0).spl_num_atoms()[ialoc];
                int mt_basis_size = unit_cell_.atom(ia).type().mt_basis_size();
                int offset_wf = kp__->spinor_wave_functions(0).offset_mt_coeffs(ialoc);

                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    for (int i = 0; i < nbnd; i++) {

                        for (int xi = 0; xi < mt_basis_size; xi++) {
                            auto c = kp__->spinor_wave_functions(ispn).mt_coeffs().prime(offset_wf + xi, i);
                            wf1(xi, i, ispn) = std::conj(c);
                            wf2(xi, i, ispn) = c * kp__->band_occupancy(i) * kp__->weight();
                        }
                    }
                }
                /* compute diagonal terms */
                for (int ispn = 0; ispn < 2; ispn++) {
                    linalg<CPU>::gemm(0, 1, mt_basis_size, mt_basis_size, nbnd, complex_one, 
                                      &wf1(0, 0, ispn), wf1.ld(), &wf2(0, 0, ispn), wf2.ld(), complex_one, 
                                      density_matrix__.at<CPU>(0, 0, ispn, ia), density_matrix__.ld());
                }
                /* offdiagonal term */
                linalg<CPU>::gemm(0, 1, mt_basis_size, mt_basis_size, nbnd, complex_one, 
                                  &wf1(0, 0, 1), wf1.ld(), &wf2(0, 0, 0), wf2.ld(), complex_one, 
                                  density_matrix__.at<CPU>(0, 0, 2, ia), density_matrix__.ld());
            }
        }
    }

    if (ctx_.esm_type() == electronic_structure_method_t::pseudopotential) {
        if (!ctx_.unit_cell().mt_lo_basis_size()) {
            return;
        }

        kp__->beta_projectors().prepare();

        //== #ifdef __GPU
        //== bool allocate_on_gpu[] = {false, false};
        //== if (ctx_.processing_unit() == GPU) {
        //==     for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        //==         int nbnd = kp__->num_occupied_bands(ispn);
        //==         if (!kp__->spinor_wave_functions(ispn).pw_coeffs().prime().on_device()) {
        //==             allocate_on_gpu[ispn] = true;
        //==             kp__->spinor_wave_functions(ispn).copy_to_device(nbnd);
        //==         }
        //==     }
        //== }
        //== #endif

        if (ctx_.num_mag_dims() != 3) {
            for (int chunk = 0; chunk < kp__->beta_projectors().num_beta_chunks(); chunk++) {
                kp__->beta_projectors().generate(chunk);

                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    /* total number of occupied bands for this spin */
                    int nbnd = kp__->num_occupied_bands(ispn);
                    /* compute <beta|psi> */
                    kp__->beta_projectors().inner<T>(chunk, kp__->spinor_wave_functions(ispn), 0, nbnd);

                    /* number of beta projectors */
                    int nbeta = kp__->beta_projectors().beta_chunk(chunk).num_beta_;

                    auto beta_psi = kp__->beta_projectors().beta_phi<T>(chunk, nbnd);

                    splindex<block> spl_nbnd(nbnd, kp__->comm().size(), kp__->comm().rank());

                    int nbnd_loc = spl_nbnd.local_size();
                    if (nbnd_loc) { // TODO: this part can also be moved to GPU
                        #pragma omp parallel
                        {
                            /* auxiliary arrays */
                            mdarray<double_complex, 2> bp1(nbeta, nbnd_loc);
                            mdarray<double_complex, 2> bp2(nbeta, nbnd_loc);
                            #pragma omp for
                            for (int ia = 0; ia < kp__->beta_projectors().beta_chunk(chunk).num_atoms_; ia++) {
                                int nbf = kp__->beta_projectors().beta_chunk(chunk).desc_(0, ia);
                                int offs = kp__->beta_projectors().beta_chunk(chunk).desc_(1, ia);
                                int ja = kp__->beta_projectors().beta_chunk(chunk).desc_(3, ia);

                                for (int i = 0; i < nbnd_loc; i++) {
                                    int j = spl_nbnd[i];

                                    for (int xi = 0; xi < nbf; xi++) {
                                        bp1(xi, i) = beta_psi(offs + xi, j);
                                        bp2(xi, i) = std::conj(bp1(xi, i)) * kp__->weight() *
                                                     kp__->band_occupancy(j + ispn * ctx_.num_fv_states());
                                    }
                                }

                                linalg<CPU>::gemm(0, 1, nbf, nbf, nbnd_loc, complex_one, &bp1(0, 0), bp1.ld(),
                                                  &bp2(0, 0), bp2.ld(), complex_one, &density_matrix__(0, 0, ispn, ja), 
                                                  density_matrix__.ld());
                            }
                        }
                    }
                }
            }
        } else {
            STOP();
        }

        //#ifdef __GPU
        //if (ctx_.processing_unit() == GPU) {
        //    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        //        if (allocate_on_gpu[ispn]) {
        //            kp__->spinor_wave_functions<false>(ispn).deallocate_on_device();
        //        }
        //    }
        //}
        //#endif

        kp__->beta_projectors().dismiss();
    }
}

