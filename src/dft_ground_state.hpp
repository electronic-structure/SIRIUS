void DFT_ground_state::move_atoms(int istep)
{
    mdarray<double, 2> atom_force(3, parameters_->num_atoms());
    forces(atom_force);

    for (int ia = 0; ia < parameters_->num_atoms(); ia++)
    {
        double pos[3];
        parameters_->atom(ia)->get_position(pos);

        double forcef[3];
        parameters_->get_coordinates<fractional, direct>(&atom_force(0, ia), &forcef[0]);

        for (int x = 0; x < 3; x++) pos[x] += 1.0 * forcef[x];
        
        parameters_->atom(ia)->set_position(pos);
    }
    parameters_->update();
    kset_->update();
}

void DFT_ground_state::forces(mdarray<double, 2>& atom_force)
{
    mdarray<double, 2> forcehf(3, parameters_->num_atoms());
    mdarray<double, 2> forcerho(3, parameters_->num_atoms());

    kset_->force(atom_force);
    
    mt_function<double>* g[3];
    for (int x = 0; x < 3; x++) 
    {
        g[x] = new mt_function<double>(Argument(arg_lm, parameters_->lmmax_pot()), 
                                       Argument(arg_radial, parameters_->max_num_mt_points()));
    }
    
    for (int ialoc = 0; ialoc < parameters_->spl_num_atoms().local_size(); ialoc++)
    {
        int ia = parameters_->spl_num_atoms(ialoc);
        gradient(parameters_->atom(ia)->type()->radial_grid(), potential_->coulomb_potential_mt(ialoc), g[0], g[1], g[2]);
        for (int x = 0; x < 3; x++) forcehf(x, ia) = parameters_->atom(ia)->type()->zn() * (*g[x])(0, 0) * y00;
    }
    
    for (int x = 0; x < 3; x++) 
    {
        delete g[x];
        g[x] = new mt_function<double>(Argument(arg_lm, parameters_->lmmax_rho()), 
                                       Argument(arg_radial, parameters_->max_num_mt_points()));
    }

    for (int ialoc = 0; ialoc < parameters_->spl_num_atoms().local_size(); ialoc++)
    {
        int ia = parameters_->spl_num_atoms(ialoc);
        gradient(parameters_->atom(ia)->type()->radial_grid(), density_->density_mt(ialoc), g[0], g[1], g[2]);
        for (int x = 0; x < 3; x++)
        {
            forcerho(x, ia) = inner(parameters_->atom(ia)->type()->radial_grid(), 
                                    potential_->effective_potential_mt(ialoc), g[x]);
        }
    }

    for (int x = 0; x < 3; x++) delete g[x];

    for (int ia = 0; ia < parameters_->num_atoms(); ia++)
    {
        for (int x = 0; x < 3; x++) atom_force(x, ia) += (forcehf(x, ia) + forcerho(x, ia));
    }
    
    for (int ia = 0; ia < parameters_->num_atoms(); ia++)
    {
        printf("atom : %i  force : %f %f %f\n", ia, atom_force(0, ia), atom_force(1, ia), atom_force(2, ia));
    }
}

void DFT_ground_state::scf_loop()
{
    parameters_->print_info();

    mixer<double>* mx = new periodic_function_mixer<double>(density_->rho());
    
    mx->load();
    
    double eold = 1e100;

    for (int iter = 0; iter < 100; iter++)
    {
        kset_->find_eigen_states(potential_, true);
        kset_->find_band_occupancies();
        kset_->valence_eval_sum();
        density_->generate(*kset_);
        density_->integrate();

        mx->load();
        double rms = mx->mix();
    
        potential_->generate_effective_potential(density_->rho(), density_->magnetization());
        
        parameters_->print_rti();
        
        std::cout << "charge RMS : " << rms << " energy difference : " << fabs(eold - parameters_->total_energy()) << std::endl;

        if (fabs(eold - parameters_->total_energy()) < 1e-4 && rms < 1e-4)
        {   
            std::cout << "Done after " << iter << " iterations!" << std::endl;
            break;
        }

        eold = parameters_->total_energy();
    }
    
    parameters_->create_storage_file();
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
    }
}
