inline void Density::generate_valence(K_point_set& ks__)
{
    PROFILE("sirius::Density::generate_valence");

    /* check weights */
    double wt{0};
    double occ_val{0};
    for (int ik = 0; ik < ks__.num_kpoints(); ik++) {
        wt += ks__[ik]->weight();
        for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) { 
            for (int j = 0; j < ctx_.num_bands(); j++) {
                occ_val += ks__[ik]->weight() * ks__[ik]->band_occupancy(j, ispn);
            }
        }
    }

    if (std::abs(wt - 1.0) > 1e-12) {
        std::stringstream s;
        s << "K_point weights don't sum to one" << std::endl
          << "  obtained sum: " << wt; 
        TERMINATE(s);
    }

    if (std::abs(occ_val - unit_cell_.num_valence_electrons()) > 1e-8 && ctx_.comm().rank() == 0) {
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
                kp->spinor_wave_functions().pw_coeffs(ispn).prime().allocate(memory_t::device);
                kp->spinor_wave_functions().pw_coeffs(ispn).copy_to_device(0, nbnd); // TODO: copy this asynchronously
            }
#endif
            /* swap wave functions for the FFT transformation */
            //kp->spinor_wave_functions(ispn).pw_coeffs().remap_forward(ctx_.processing_unit(), kp->gkvec().partition().gvec_fft_slab(), nbnd);
            kp->spinor_wave_functions().pw_coeffs(ispn).remap_forward(CPU, nbnd);
        }
        
        if (ctx_.electronic_structure_method() == electronic_structure_method_t::full_potential_lapwlo) {
            add_k_point_contribution_dm<double_complex>(kp, density_matrix_);
        }
        
        if (ctx_.electronic_structure_method() == electronic_structure_method_t::pseudopotential) {
            if (ctx_.gamma_point() && (ctx_.so_correction() == false)) {
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
                kp->spinor_wave_functions().pw_coeffs(ispn).deallocate_on_device();
            }
        }
#endif
    }

    if (density_matrix_.size()) {
        ctx_.comm().allreduce(density_matrix_.at<CPU>(), static_cast<int>(density_matrix_.size()));
    }

    ctx_.fft_coarse().prepare(ctx_.gvec_coarse_partition());
    auto& comm = ctx_.gvec_coarse_partition().comm_ortho_fft();
    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        /* reduce arrays; assume that each rank did its own fraction of the density */
        /* comm_ortho_fft is idential to a product of column communicator inside k-point with k-point communicator */
        comm.allreduce(&rho_mag_coarse_[j]->f_rg(0), ctx_.fft_coarse().local_size()); 
        if (ctx_.control().print_checksum_) {
            auto cs = mdarray<double, 1>(&rho_mag_coarse_[j]->f_rg(0), ctx_.fft_coarse().local_size()).checksum();
            ctx_.fft_coarse().comm().allreduce(&cs, 1);
            if (ctx_.comm().rank() == 0) {
                print_checksum("rho_mag_coarse_rg", cs);
            }
        }
        /* transform to PW domain */
        rho_mag_coarse_[j]->fft_transform(-1);
        /* map to fine G-vector grid */
        for (int igloc = 0; igloc < ctx_.gvec_coarse().count(); igloc++) {
            rho_vec_[j]->f_pw_local(ctx_.gvec().gvec_base_mapping(igloc)) = rho_mag_coarse_[j]->f_pw_local(igloc);
        }
    }
    ctx_.fft_coarse().dismiss();

    if (!ctx_.full_potential()) {
        augment(ks__);
        
        if (ctx_.control().print_hash_ && ctx_.comm().rank() == 0) {
            auto h = mdarray<double_complex, 1>(&rho_->f_pw_local(0), ctx_.gvec().count()).hash();
            print_hash("rho", h);
        }

        double nel = rho_->f_0().real() * unit_cell_.omega();
        /* check the number of electrons */
        if (std::abs(nel - unit_cell_.num_electrons()) > 1e-8 && ctx_.comm().rank() == 0) {
            std::stringstream s;
            s << "wrong unsymmetrized density" << std::endl
              << "  obtained value : " << std::scientific << nel << std::endl 
              << "  target value : " << std::scientific << unit_cell_.num_electrons() << std::endl
              << "  difference : " << std::scientific << std::abs(nel - unit_cell_.num_electrons()) << std::endl;
            WARNING(s);
        }
    }

    /* for muffin-tin part */
    if (ctx_.full_potential()) {
        generate_valence_mt(ks__);
    }
}

