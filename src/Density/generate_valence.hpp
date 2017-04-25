inline void Density::generate_valence(K_point_set& ks__)
{
    PROFILE("sirius::Density::generate_valence");

    double wt{0};
    double occ_val{0};
    for (int ik = 0; ik < ks__.num_kpoints(); ik++) {
        wt += ks__[ik]->weight();
        for (int j = 0; j < ctx_.num_bands(); j++) {
            occ_val += ks__[ik]->weight() * ks__[ik]->band_occupancy(j);
        }
    }

    if (std::abs(wt - 1.0) > 1e-12) {
        TERMINATE("K_point weights don't sum to one");
    }

    if (std::abs(occ_val - unit_cell_.num_valence_electrons()) > 1e-8) {
        std::stringstream s;
        s << "wrong occupancies" << std::endl
          << "  computed : " << occ_val << std::endl
          << "  required : " << unit_cell_.num_valence_electrons() << std::endl
          << "  difference : " << std::abs(occ_val - unit_cell_.num_valence_electrons());
        WARNING(s);
    }
    
    density_matrix_.zero();

    /* zero density and magnetization */
    zero();
    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        rho_mag_coarse_[i]->zero();
    }
    
    /* start the main loop over k-points */
    for (int ikloc = 0; ikloc < ks__.spl_num_kpoints().local_size(); ikloc++) {
        int ik = ks__.spl_num_kpoints(ikloc);
        auto kp = ks__[ik];

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            int nbnd = kp->num_occupied_bands(ispn);
            /* copy wave-functions to GPU */
            #ifdef __GPU
            if (ctx_.processing_unit() == GPU) {
                kp->spinor_wave_functions(ispn).pw_coeffs().copy_to_device(0, nbnd);
            }
            #endif
            /* swap wave functions */
            switch (ctx_.processing_unit()) {
                case CPU: {
                    kp->spinor_wave_functions(ispn).pw_coeffs().remap_forward(kp->gkvec().partition().gvec_fft_slab(),
                                                                              kp->gkvec().comm_ortho_fft(),
                                                                              nbnd);
                    break;
                }
                case GPU: {
                    kp->spinor_wave_functions(ispn).pw_coeffs().remap_forward<memory_t::host | memory_t::device>(kp->gkvec().partition().gvec_fft_slab(),
                                                                                                                 kp->gkvec().comm_ortho_fft(),
                                                                                                                 nbnd);
                    break;
                }
            }
        }
        
        if (ctx_.esm_type() == electronic_structure_method_t::full_potential_lapwlo) {
            add_k_point_contribution_dm<double_complex>(kp, density_matrix_);
        }
        
        if (ctx_.esm_type() == electronic_structure_method_t::pseudopotential) {
            if (ctx_.gamma_point()) {
                add_k_point_contribution_dm<double>(kp, density_matrix_);
            } else {
                add_k_point_contribution_dm<double_complex>(kp, density_matrix_);
            }
        }

        /* add contribution from regular space grid */
        add_k_point_contribution_rg(kp);
    }

    if (density_matrix_.size()) {
        ctx_.comm().allreduce(density_matrix_.at<CPU>(), static_cast<int>(density_matrix_.size()));
    }

    ctx_.fft_coarse().prepare(ctx_.gvec_coarse().partition());
    auto& comm = ctx_.gvec_coarse().comm_ortho_fft();
    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        /* reduce arrays; assume that each rank did its own fraction of the density */
        comm.allreduce(&rho_mag_coarse_[j]->f_rg(0), ctx_.fft_coarse().local_size()); 
        /* transform to PW domain */
        rho_mag_coarse_[j]->fft_transform(-1);
        /* get the whole vector of PW coefficients */
        auto fpw = rho_mag_coarse_[j]->gather_f_pw(); // TODO: reuse FFT G-vec arrays
        /* map to fine G-vector grid */
        if (j == 0) {
            for (int ig = 0; ig < ctx_.gvec_coarse().num_gvec(); ig++) {
                auto G = ctx_.gvec_coarse().gvec(ig);
                rho_->f_pw(G) = fpw[ig];
            }
        } else {
            for (int ig = 0; ig < ctx_.gvec_coarse().num_gvec(); ig++) {
                auto G = ctx_.gvec_coarse().gvec(ig);
                magnetization_[j - 1]->f_pw(G) = fpw[ig];
            }
        }
    }
    ctx_.fft_coarse().dismiss();

    if (!ctx_.full_potential()) {
        augment(ks__);

        double nel = rho_->f_pw(0).real() * unit_cell_.omega();
        /* check the number of electrons */
        if (std::abs(nel - unit_cell_.num_electrons()) > 1e-8) {
            std::stringstream s;
            s << "wrong unsymmetrized density" << std::endl
              << "  obtained value : " << nel << std::endl 
              << "  target value : " << unit_cell_.num_electrons() << std::endl
              << "  difference : " << std::abs(nel - unit_cell_.num_electrons()) << std::endl;
            WARNING(s);
        }
    }

    if (ctx_.esm_type() == electronic_structure_method_t::pseudopotential) {
        symmetrize_density_matrix();
    }

    /* for muffin-tin part */
    if (ctx_.full_potential()) {
        generate_valence_mt(ks__);
    }
}

