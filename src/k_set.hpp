void K_set::initialize()
{
    // ============================================================
    // distribute k-points along the 1-st dimension of the MPI grid
    // ============================================================
    spl_num_kpoints_.split(num_kpoints(), parameters_.mpi_grid().dimension_size(_dim_k_), 
                           parameters_.mpi_grid().coordinate(_dim_k_));

    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
        kpoints_[spl_num_kpoints_[ikloc]]->initialize();

    if (verbosity_level >= 2) print_info();
}

void K_set::update()
{
    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
        kpoints_[spl_num_kpoints_[ikloc]]->update();
}

void K_set::sync_band_energies()
{
    mdarray<double, 2> band_energies(parameters_.num_bands(), num_kpoints());
    band_energies.zero();
    
    // assume that side processors store the full e(k) array
    if (parameters_.mpi_grid().side(1 << 0))
    {
        for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
        {
            int ik = spl_num_kpoints_[ikloc];
            kpoints_[ik]->get_band_energies(&band_energies(0, ik));
        }
    }

    Platform::allreduce(&band_energies(0, 0), parameters_.num_bands() * num_kpoints());

    for (int ik = 0; ik < num_kpoints(); ik++) kpoints_[ik]->set_band_energies(&band_energies(0, ik));
}

void K_set::find_eigen_states(Potential* potential, bool precompute)
{
    Timer t("sirius::K_set::find_eigen_states");
    
    if (precompute)
    {
        potential->generate_pw_coefs();
        potential->update_atomic_potential();
        parameters_.generate_radial_functions();
        parameters_.generate_radial_integrals();
    }
    
    // solve secular equation and generate wave functions
    for (int ikloc = 0; ikloc < spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = spl_num_kpoints(ikloc);
        band_->solve_fv(kpoints_[ik], potential->effective_potential());
        kpoints_[ik]->generate_fv_states();
        kpoints_[ik]->distribute_fv_states_row();
        band_->solve_sv(kpoints_[ik], potential->effective_magnetic_field());
        kpoints_[ik]->generate_spinor_wave_functions();
    }

    // synchronize eigen-values
    sync_band_energies();

    if (Platform::mpi_rank() == 0 && verbosity_level >= 5)
    {
        printf("Lowest band energies\n");
        for (int ik = 0; ik < num_kpoints(); ik++)
        {
            printf("ik : %2i, ", ik); 
            if (parameters_.num_mag_dims() != 1)
            {
                for (int j = 0; j < std::min(10, parameters_.num_bands()); j++) 
                    printf("%12.6f", kpoints_[ik]->band_energy(j));
            }
            else
            {
                for (int j = 0; j < std::min(10, parameters_.num_fv_states()); j++) 
                    printf("%12.6f", kpoints_[ik]->band_energy(j));
                printf("\n         ");
                for (int j = 0; j < std::min(10, parameters_.num_fv_states()); j++) 
                    printf("%12.6f", kpoints_[ik]->band_energy(parameters_.num_fv_states() + j));
            }
            printf("\n");
        }
    }
    
    // compute eigen-value sums
    valence_eval_sum();
}

double K_set::valence_eval_sum()
{
    double eval_sum = 0.0;

    for (int ik = 0; ik < num_kpoints(); ik++)
    {
        double wk = kpoints_[ik]->weight();
        for (int j = 0; j < parameters_.num_bands(); j++)
            eval_sum += wk * kpoints_[ik]->band_energy(j) * kpoints_[ik]->band_occupancy(j);
    }

    //** parameters_.rti().valence_eval_sum = eval_sum;
    return eval_sum;
}

void K_set::find_band_occupancies()
{
    Timer t("sirius::Density::find_band_occupancies");

    double ef = 0.15;

    double de = 0.1;

    int s = 1;
    int sp;

    double ne = 0.0;

    mdarray<double, 2> bnd_occ(parameters_.num_bands(), num_kpoints());
    
    // TODO: safe way not to get stuck here
    while (true)
    {
        ne = 0.0;
        for (int ik = 0; ik < num_kpoints(); ik++)
        {
            for (int j = 0; j < parameters_.num_bands(); j++)
            {
                bnd_occ(j, ik) = Utils::gaussian_smearing(kpoints_[ik]->band_energy(j) - ef, parameters_.smearing_width()) * 
                                 parameters_.max_occupancy();
                ne += bnd_occ(j, ik) * kpoints_[ik]->weight();
            }
        }

        if (fabs(ne - parameters_.num_valence_electrons()) < 1e-11) break;

        sp = s;
        s = (ne > parameters_.num_valence_electrons()) ? -1 : 1;

        de = s * fabs(de);

        (s != sp) ? de *= 0.5 : de *= 1.25; 
        
        ef += de;
    } 
    energy_fermi_ = ef;

    //parameters_.rti().energy_fermi = ef;
    
    for (int ik = 0; ik < num_kpoints(); ik++) kpoints_[ik]->set_band_occupancies(&bnd_occ(0, ik));

    double gap = 0.0;
    
    int nve = int(parameters_.num_valence_electrons() + 1e-12);
    if ((parameters_.num_spins() == 2) || 
        ((fabs(nve - parameters_.num_valence_electrons()) < 1e-12) && nve % 2 == 0))
    {
        // find band gap
        std::vector< std::pair<double, double> > eband;
        std::pair<double, double> eminmax;

        for (int j = 0; j < parameters_.num_bands(); j++)
        {
            eminmax.first = 1e10;
            eminmax.second = -1e10;

            for (int ik = 0; ik < num_kpoints(); ik++)
            {
                eminmax.first = std::min(eminmax.first, kpoints_[ik]->band_energy(j));
                eminmax.second = std::max(eminmax.second, kpoints_[ik]->band_energy(j));
            }

            eband.push_back(eminmax);
        }
        
        std::sort(eband.begin(), eband.end());

        int ist = nve;
        if (parameters_.num_spins() == 1) ist /= 2; 

        if (eband[ist].first > eband[ist - 1].second) gap = eband[ist].first - eband[ist - 1].second;

        band_gap_ = gap;

        //parameters_.rti().band_gap = gap;
    }
}

void K_set::print_info()
{
    pstdout pout(100 * num_kpoints());

    if (parameters_.mpi_grid().side(1 << 0))
    {
        for (int ikloc = 0; ikloc < spl_num_kpoints().local_size(); ikloc++)
        {
            int ik = spl_num_kpoints(ikloc);
            pout.printf("%4i   %8.4f %8.4f %8.4f   %12.6f     %6i            %6i\n", 
                        ik, kpoints_[ik]->vk()[0], kpoints_[ik]->vk()[1], kpoints_[ik]->vk()[2], 
                        kpoints_[ik]->weight(), kpoints_[ik]->num_gkvec(), kpoints_[ik]->apwlo_basis_size());
        }
    }
    if (Platform::mpi_rank() == 0)
    {
        printf("\n");
        printf("total number of k-points : %i\n", num_kpoints());
        for (int i = 0; i < 80; i++) printf("-");
        printf("\n");
        printf("  ik                vk                    weight  num_gkvec  apwlo_basis_size\n");
        for (int i = 0; i < 80; i++) printf("-");
        printf("\n");
    }
    pout.flush(0);
}

void K_set::save_wave_functions()
{
    if (Platform::mpi_rank() == 0)
    {
        HDF5_tree fout(storage_file_name, false);
        fout["parameters"].write("num_kpoints", num_kpoints());
        fout["parameters"].write("num_bands", parameters_.num_bands());
        fout["parameters"].write("num_spins", parameters_.num_spins());
    }

    if (parameters_.mpi_grid().side(1 << _dim_k_ | 1 << _dim_col_))
    {
        for (int ik = 0; ik < num_kpoints(); ik++)
        {
            int rank = spl_num_kpoints_.location(_splindex_rank_, ik);
            
            if (parameters_.mpi_grid().coordinate(_dim_k_) == rank) kpoints_[ik]->save_wave_functions(ik);
            
            parameters_.mpi_grid().barrier(1 << _dim_k_ | 1 << _dim_col_);
        }
    }
}

void K_set::load_wave_functions()
{
    HDF5_tree fin(storage_file_name, false);
    int num_spins;
    fin["parameters"].read("num_spins", &num_spins);
    if (num_spins != parameters_.num_spins()) error_local(__FILE__, __LINE__, "wrong number of spins");

    int num_bands;
    fin["parameters"].read("num_bands", &num_bands);
    if (num_bands != parameters_.num_bands()) error_local(__FILE__, __LINE__, "wrong number of bands");
    
    int num_kpoints_in;
    fin["parameters"].read("num_kpoints", &num_kpoints_in);

    // ==================================================================
    // index of current k-points in the hdf5 file, which (in general) may 
    // contain a different set of k-points 
    // ==================================================================
    std::vector<int> ikidx(num_kpoints(), -1); 
    // read available k-points
    double vk_in[3];
    for (int jk = 0; jk < num_kpoints_in; jk++)
    {
        fin["kpoints"][jk].read("coordinates", vk_in, 3);
        for (int ik = 0; ik < num_kpoints(); ik++)
        {
            vector3d<double> dvk; 
            for (int x = 0; x < 3; x++) dvk[x] = vk_in[x] - kpoints_[ik]->vk(x);
            if (dvk.length() < 1e-12)
            {
                ikidx[ik] = jk;
                break;
            }
        }
    }

    for (int ik = 0; ik < num_kpoints(); ik++)
    {
        int rank = spl_num_kpoints_.location(_splindex_rank_, ik);
        
        if (parameters_.mpi_grid().coordinate(0) == rank) kpoints_[ik]->load_wave_functions(ikidx[ik]);
    }
}

int K_set::max_num_gkvec()
{
    int max_num_gkvec_ = 0;
    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
    {
        int ik = spl_num_kpoints_[ikloc];
        max_num_gkvec_ = std::max(max_num_gkvec_, kpoints_[ik]->num_gkvec());
    }
    Platform::allreduce<op_max>(&max_num_gkvec_, 1);
    return max_num_gkvec_;
}

//** void K_set::force(mdarray<double, 2>& forcek)
//** {
//**     mdarray<double, 2> ffac(parameters_.num_gvec_shells(), parameters_.num_atom_types());
//**     parameters_.get_step_function_form_factors(ffac);
//** 
//**     forcek.zero();
//** 
//**     for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
//**     {
//**          //kpoints_[spl_num_kpoints_[ikloc]]->ibs_force<cpu, apwlo>(ffac, forcek);
//**     }
//**     Platform::allreduce(&forcek(0, 0), (int)forcek.size(), parameters_.mpi_grid().communicator(1 << _dim_k_));
//** }

