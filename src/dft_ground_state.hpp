inline double DFT_ground_state::energy_enuc()
{
    double enuc = 0.0;
    for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
    {
        int ia = parameters_.spl_num_atoms(ialoc);
        int zn = parameters_.atom(ia)->type()->zn();
        double r0 = parameters_.atom(ia)->type()->radial_grid(0);
        enuc -= 0.5 * zn * (potential_->coulomb_potential()->f_mt<local>(0, 0, ialoc) * y00 + zn / r0);
    }
    Platform::allreduce(&enuc, 1);
    
    return enuc;
}

inline double DFT_ground_state::core_eval_sum()
{
    double sum = 0.0;
    for (int ic = 0; ic < parameters_.num_atom_symmetry_classes(); ic++)
    {
        sum += parameters_.atom_symmetry_class(ic)->core_eval_sum() * 
               parameters_.atom_symmetry_class(ic)->num_atoms();
    }
    return sum;
}

void DFT_ground_state::move_atoms(int istep)
{
    mdarray<double, 2> atom_force(3, parameters_.num_atoms());
    forces(atom_force);

    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        vector3d<double> pos = parameters_.atom(ia)->position();

        vector3d<double> forcef = parameters_.get_coordinates<fractional, direct>(vector3d<double>(&atom_force(0, ia)));

        for (int x = 0; x < 3; x++) pos[x] += 1.0 * forcef[x];
        
        parameters_.atom(ia)->set_position(pos);
    }
}

void DFT_ground_state::update()
{
    parameters_.update();
    potential_->update();
    kset_->update();
}

void DFT_ground_state::forces(mdarray<double, 2>& forces)
{
    Force::total_force(parameters_, potential_, density_, kset_, forces);
}

void DFT_ground_state::scf_loop(double charge_tol, double energy_tol, int num_dft_iter)
{
    Timer t("sirius::DFT_ground_state::scf_loop");

    density_mixer* mx = new density_mixer(density_->rho(), density_->magnetization(), parameters_.num_mag_dims());
    mx->load();
    
    double eold = 0.0;

    for (int iter = 0; iter < num_dft_iter; iter++)
    {
        Timer t1("sirius::DFT_ground_state::scf_loop|iteration");

        kset_->find_eigen_states(potential_, true);
        kset_->find_band_occupancies();
        kset_->valence_eval_sum();
        density_->generate(*kset_);

        double rms = mx->mix();

        Platform::bcast(&rms, 1, 0);

        potential_->generate_effective_potential(density_->rho(), density_->magnetization());
        
        double etot = total_energy();
        
        print_info();
        
        if (Platform::mpi_rank() == 0)
        {
            printf("iteration : %3i, charge RMS %12.6f, energy difference : %12.6f, beta : %12.6f\n", 
                    iter, rms, fabs(eold - etot), mx->beta());
        }
        
        if (fabs(eold - etot) < energy_tol && rms < charge_tol) break;

        eold = etot;
    }
    
    parameters_.create_storage_file();
    potential_->save();
    density_->save();

    delete mx;
}

void DFT_ground_state::relax_atom_positions()
{
    for (int i = 0; i < 5; i++)
    {
        scf_loop(1e-4, 1e-4, 100);
        move_atoms(i);
        update();
    }
}
