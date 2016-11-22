inline void Density::generate_valence(K_set& ks__)
{
    PROFILE_WITH_TIMER("sirius::Density::generate_valence");

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
    
    /* start the main loop over k-points */
    for (int ikloc = 0; ikloc < ks__.spl_num_kpoints().local_size(); ikloc++) {
        int ik = ks__.spl_num_kpoints(ikloc);
        auto kp = ks__[ik];

        /* swap wave functions */
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            int nbnd = kp->num_occupied_bands(ispn);
            kp->spinor_wave_functions(ispn).pw_coeffs().remap_forward(kp->gkvec().partition().gvec_fft_slab(),
                                                                      ctx_.mpi_grid_fft().communicator(1 << 1),
                                                                      nbnd);

            /* copy wave-functions to GPU */
            if (!ctx_.full_potential()) {
                #ifdef __GPU
                if (ctx_.processing_unit() == GPU) {
                    kp->spinor_wave_functions(ispn).pw_coeffs().allocate_on_device();
                    kp->spinor_wave_functions(ispn).pw_coeffs().copy_to_device(0, nbnd);
                }
                #endif
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

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            #ifdef __GPU
            if (ctx_.processing_unit() == GPU) {
                kp->spinor_wave_functions(ispn).pw_coeffs().deallocate_on_device();
            }
            #endif
        }
    }

    if (density_matrix_.size()) {
        ctx_.comm().allreduce(density_matrix_.at<CPU>(), static_cast<int>(density_matrix_.size()));
    }

    /* reduce arrays; assume that each rank did its own fraction of the density */
    auto& comm = (ctx_.fft().parallel()) ? ctx_.mpi_grid().communicator(1 << _mpi_dim_k_ | 1 << _mpi_dim_k_col_)
                                         : ctx_.comm();

    comm.allreduce(&rho_->f_rg(0), ctx_.fft().local_size()); 
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        comm.allreduce(&magnetization_[j]->f_rg(0), ctx_.fft().local_size()); 
    }

    /* for muffin-tin part */
    switch (ctx_.esm_type()) {
        case electronic_structure_method_t::full_potential_lapwlo: {
            generate_valence_density_mt(ks__);
            break;
        }
        case electronic_structure_method_t::full_potential_pwlo: {
            STOP();
        }
        default: {
            break;
        }
    }

    ctx_.fft().prepare(ctx_.gvec().partition());
    /* get rho(G) and mag(G)
     * they are required to symmetrize density and magnetization */
    rho_->fft_transform(-1);
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        magnetization_[j]->fft_transform(-1);
    }
    ctx_.fft().dismiss();

    //== printf("number of electrons: %f\n", rho_->f_pw(0).real() * unit_cell_.omega());
    //== STOP();

    if (!ctx_.full_potential()) {
        augment(ks__);
    }

    if (ctx_.esm_type() == electronic_structure_method_t::pseudopotential) {
        symmetrize_density_matrix();
    }
}

inline void Density::generate(K_set& ks__)
{
    PROFILE_WITH_TIMER("sirius::Density::generate");

    generate_valence(ks__);

    if (ctx_.full_potential()) {
        /* find the core states */
        generate_core_charge_density();
        /* add core contribution */
        for (int ialoc = 0; ialoc < (int)unit_cell_.spl_num_atoms().local_size(); ialoc++) {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            for (int ir = 0; ir < unit_cell_.atom(ia).num_mt_points(); ir++) {
                rho_->f_mt<index_domain_t::local>(0, ir, ialoc) += unit_cell_.atom(ia).symmetry_class().core_charge_density(ir) / y00;
            }
        }
        /* synchronize muffin-tin part */
        rho_->sync_mt();
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            magnetization_[j]->sync_mt();
        }
    }
    
    double nel{0};
    if (ctx_.full_potential()) {
        std::vector<double> nel_mt;
        double nel_it;
        nel = rho_->integrate(nel_mt, nel_it);
    } else {
        nel = rho_->f_pw(0).real() * unit_cell_.omega();
    }

    if (std::abs(nel - unit_cell_.num_electrons()) > 1e-5) {
        std::stringstream s;
        s << "wrong charge density after k-point summation" << std::endl
          << "obtained value : " << nel << std::endl 
          << "target value : " << unit_cell_.num_electrons() << std::endl
          << "difference : " << fabs(nel - unit_cell_.num_electrons()) << std::endl;
        if (ctx_.full_potential()) {
            s << "total core leakage : " << core_leakage();
            for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
                s << std::endl << "  atom class : " << ic << ", core leakage : " << core_leakage(ic);
            }
        }
        WARNING(s);
    }

    #ifdef __PRINT_OBJECT_HASH
    DUMP("hash(rhomt): %16llX", rho_->f_mt().hash());
    DUMP("hash(rhoit): %16llX", rho_->f_it().hash());
    #endif

    //if (debug_level > 1) check_density_continuity_at_mt();
}

