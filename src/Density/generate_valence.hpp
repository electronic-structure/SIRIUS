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
        std::stringstream s;
        s << "K_point weights don't sum to one" << std::endl
          << "  obtained sum: " << wt; 
        TERMINATE(s);
    }

    if (std::abs(occ_val - unit_cell_.num_valence_electrons()) > 1e-8) {
        std::stringstream s;
        s << "wrong band occupancies" << std::endl
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
            
            #ifdef __GPU
            if (ctx_.processing_unit() == GPU && !keep_wf_on_gpu) {
                /* allocate GPU memory */
                kp->spinor_wave_functions(ispn).pw_coeffs().prime().allocate(memory_t::device);
                kp->spinor_wave_functions(ispn).pw_coeffs().copy_to_device(0, nbnd);
            }
            #endif
            /* swap wave functions */
            //kp->spinor_wave_functions(ispn).pw_coeffs().remap_forward(ctx_.processing_unit(), kp->gkvec().partition().gvec_fft_slab(), nbnd);
            kp->spinor_wave_functions(ispn).pw_coeffs().remap_forward(CPU, kp->gkvec().partition().gvec_fft_slab(), nbnd);
        }
        
        if (ctx_.esm_type() == electronic_structure_method_t::full_potential_lapwlo) {
            add_k_point_contribution_dm<double_complex>(kp, density_matrix_);
        }
        
        if (ctx_.esm_type() == electronic_structure_method_t::pseudopotential) {
	  if (ctx_.gamma_point()&&(!ctx_.so_correction())) {
                add_k_point_contribution_dm<double>(kp, density_matrix_);
            } else {
                add_k_point_contribution_dm<double_complex>(kp, density_matrix_);
            }
        }

        /* add contribution from regular space grid */
        add_k_point_contribution_rg(kp);

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU && !keep_wf_on_gpu) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                /* deallocate GPU memory */
                kp->spinor_wave_functions(ispn).pw_coeffs().deallocate_on_device();
            }
        }
        #endif
    }

    if (density_matrix_.size()) {
        ctx_.comm().allreduce(density_matrix_.at<CPU>(), static_cast<int>(density_matrix_.size()));
    }

    ctx_.fft_coarse().prepare(ctx_.gvec_coarse().partition());
    auto& comm = ctx_.gvec_coarse().comm_ortho_fft();
    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        /* reduce arrays; assume that each rank did its own fraction of the density */
        comm.allreduce(&rho_mag_coarse_[j]->f_rg(0), ctx_.fft_coarse().local_size()); 
        if (ctx_.control().print_checksum_) {
            auto cs = mdarray<double, 1>(&rho_mag_coarse_[j]->f_rg(0), ctx_.fft_coarse().local_size()).checksum();
            ctx_.fft_coarse().comm().allreduce(&cs, 1);
            if (ctx_.comm().rank() == 0) {
                DUMP("checksum(rho_mag_coarse_rg) : %18.10f", cs);
            }
        }
        /* transform to PW domain */
        rho_mag_coarse_[j]->fft_transform(-1);
        /* get the whole vector of PW coefficients */
        auto fpw = rho_mag_coarse_[j]->gather_f_pw(); // TODO: reuse FFT G-vec arrays
        if (ctx_.control().print_checksum_ && ctx_.comm().rank() == 0) {
            auto z1 = mdarray<double_complex, 1>(&fpw[0], ctx_.gvec_coarse().num_gvec()).checksum();
            DUMP("checksum(rho_mag_coarse_pw) : %18.10f %18.10f", z1.real(), z1.imag());
        }
        /* map to fine G-vector grid */
        for (int i = 0; i < static_cast<int>(lf_gvec_.size()); i++) {
            int igloc = lf_gvec_[i];
            int ig = ctx_.gvec_coarse().index_by_gvec(ctx_.gvec().gvec(ctx_.gvec().offset() + igloc));
            rho_vec_[j]->f_pw_local(igloc) = fpw[ig];
        }
    }
    ctx_.fft_coarse().dismiss();

    if (!ctx_.full_potential()) {
        augment(ks__);

        double nel = rho_->f_0().real() * unit_cell_.omega();
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

    if (ctx_.esm_type() == electronic_structure_method_t::pseudopotential && ctx_.use_symmetry()) {
        symmetrize_density_matrix();
    }

    /* for muffin-tin part */
    if (ctx_.full_potential()) {
        generate_valence_mt(ks__);
    }
}

