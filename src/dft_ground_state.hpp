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
        double pos[3];
        parameters_.atom(ia)->get_position(pos);

        double forcef[3];
        parameters_.get_coordinates<fractional, direct>(&atom_force(0, ia), &forcef[0]);

        for (int x = 0; x < 3; x++) pos[x] += 1.0 * forcef[x];
        
        parameters_.atom(ia)->set_position(pos);
    }
}

void DFT_ground_state::update()
{
    parameters_.update();
    potential_->update();
    kset_.update();
}

void DFT_ground_state::forces(mdarray<double, 2>& atom_force)
{
    mdarray<double, 2> forcehf(3, parameters_.num_atoms());
    mdarray<double, 2> forcerho(3, parameters_.num_atoms());

    pstdout pout;

    kset_.force(atom_force);
    
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        pout.printf("atom : %i  forcek : %f %f %f\n", ia, atom_force(0, ia), atom_force(1, ia), atom_force(2, ia));
    }
    
    MT_function<double>* g[3];
    for (int x = 0; x < 3; x++) 
    {
        g[x] = new MT_function<double>(Argument(arg_lm, parameters_.lmmax_pot()), 
                                       Argument(arg_radial, parameters_.max_num_mt_points()));
    }
    
    forcehf.zero();
    for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
    {
        int ia = parameters_.spl_num_atoms(ialoc);
        gradient(parameters_.atom(ia)->type()->radial_grid(), potential_->coulomb_potential_mt(ialoc), g[0], g[1], g[2]);
        for (int x = 0; x < 3; x++) forcehf(x, ia) = parameters_.atom(ia)->type()->zn() * (*g[x])(0, 0) * y00;
    }
    Platform::allreduce(&forcehf(0, 0), (int)forcehf.size());
    
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        pout.printf("atom : %i  forcehf : %f %f %f\n", ia, forcehf(0, ia), forcehf(1, ia), forcehf(2, ia));
    }
    
    for (int x = 0; x < 3; x++) 
    {
        delete g[x];
        g[x] = new MT_function<double>(Argument(arg_lm, parameters_.lmmax_rho()), 
                                       Argument(arg_radial, parameters_.max_num_mt_points()));
    }

    forcerho.zero();
    for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
    {
        int ia = parameters_.spl_num_atoms(ialoc);
        gradient(parameters_.atom(ia)->type()->radial_grid(), density_->density_mt(ialoc), g[0], g[1], g[2]);
        for (int x = 0; x < 3; x++)
        {
            forcerho(x, ia) = inner(parameters_.atom(ia)->type()->radial_grid(), 
                                    potential_->effective_potential_mt(ialoc), g[x]);
        }
    }
    Platform::allreduce(&forcerho(0, 0), (int)forcerho.size());
    
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        pout.printf("atom : %i  forcerho : %f %f %f\n", ia, forcerho(0, ia), forcerho(1, ia), forcerho(2, ia));
    }
    
    
    for (int x = 0; x < 3; x++) delete g[x];

    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        for (int x = 0; x < 3; x++) atom_force(x, ia) += (forcehf(x, ia) + forcerho(x, ia));
    }
    
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        pout.printf("atom : %i  force : %f %f %f\n", ia, atom_force(0, ia), atom_force(1, ia), atom_force(2, ia));
    }
    pout.printf("===\n");

    pout.flush(0);
    
    stop_here
}

void DFT_ground_state::scf_loop()
{
    parameters_.print_info();

    mixer<double>* mx = new periodic_function_mixer<double>(density_->rho(), 0.1);
    mixer<double>* mxmag[3];
    for (int i = 0; i < parameters_.num_mag_dims(); i++) 
        mxmag[i] = new periodic_function_mixer<double>(density_->magnetization(i), 0.1); 
    
    mx->load();
    for (int i = 0; i < parameters_.num_mag_dims(); i++) mxmag[i]->load();
    
    double eold = 1e100;

    for (int iter = 0; iter < 100; iter++)
    {
        kset_.find_eigen_states(potential_, true);
        kset_.find_band_occupancies();
        kset_.valence_eval_sum();
        density_->generate(kset_);
        density_->integrate();

        double rms = mx->mix();
        for (int i = 0; i < parameters_.num_mag_dims(); i++) rms += mxmag[i]->mix();
        
        potential_->generate_effective_potential(density_->rho(), density_->magnetization());
        
        parameters_.print_rti();
        
        std::cout << "charge RMS : " << rms << " energy difference : " << fabs(eold - parameters_.total_energy()) << std::endl;

        if (fabs(eold - parameters_.total_energy()) < 1e-4 && rms < 1e-4)
        {   
            std::cout << "Done after " << iter << " iterations!" << std::endl;
            break;
        }

        eold = parameters_.total_energy();
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
        scf_loop();
        move_atoms(i);
        update();
    }
}
