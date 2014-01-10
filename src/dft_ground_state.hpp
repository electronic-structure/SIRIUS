/** \file dft_ground_state.hpp

    \brief Contains remaining implementation of sirius::DFT_ground_state class.
*/

inline double DFT_ground_state::energy_enuc()
{
    double enuc = 0.0;
    if (parameters_.unit_cell()->potential_type() == full_potential)
    {
        for (int ialoc = 0; ialoc < parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);
            int zn = parameters_.unit_cell()->atom(ia)->type()->zn();
            double r0 = parameters_.unit_cell()->atom(ia)->type()->radial_grid(0);
            enuc -= 0.5 * zn * (potential_->coulomb_potential()->f_mt<local>(0, 0, ialoc) * y00 + zn / r0);
        }
        Platform::allreduce(&enuc, 1);
    }
    
    return enuc;
}

inline double DFT_ground_state::core_eval_sum()
{
    double sum = 0.0;
    for (int ic = 0; ic < parameters_.unit_cell()->num_atom_symmetry_classes(); ic++)
    {
        sum += parameters_.unit_cell()->atom_symmetry_class(ic)->core_eval_sum() * 
               parameters_.unit_cell()->atom_symmetry_class(ic)->num_atoms();
    }
    return sum;
}

void DFT_ground_state::move_atoms(int istep)
{
    mdarray<double, 2> atom_force(3, parameters_.unit_cell()->num_atoms());
    forces(atom_force);

    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        vector3d<double> pos = parameters_.unit_cell()->atom(ia)->position();

        vector3d<double> forcef = parameters_.unit_cell()->get_fractional_coordinates(vector3d<double>(&atom_force(0, ia)));

        for (int x = 0; x < 3; x++) pos[x] += 1.0 * forcef[x];
        
        parameters_.unit_cell()->atom(ia)->set_position(pos);
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

void DFT_ground_state::scf_loop(double potential_tol, double energy_tol, int num_dft_iter)
{
    Timer t("sirius::DFT_ground_state::scf_loop");
    
    Mixer* mx = NULL;
    if (parameters_.mixer_input_section_.type_ == "broyden")
    {
        mx = new Broyden_mixer(potential_->size(), parameters_.mixer_input_section_.max_history_, 
                               parameters_.mixer_input_section_.beta_);
    }
    else
    {
        stop_here
    }

    double eold = 0.0;
    double rms = 1.0;

    for (int iter = 0; iter < num_dft_iter; iter++)
    {
        Timer t1("sirius::DFT_ground_state::scf_loop|iteration");

        switch(parameters_.potential_type())
        {
            case full_potential:
            {
                potential_->generate_effective_potential(density_->rho(), density_->magnetization());
                break;
            }
            case ultrasoft_pseudopotential:
            {
                potential_->generate_effective_potential(density_->rho(), density_->rho_pseudo_core(), density_->magnetization());
                break;
            }
            default:
            {
                stop_here
            }
        }

        potential_->pack(mx->input_buffer());

        if (iter == 0)
        {
            mx->initialize();
        }
        else
        {
            rms = mx->mix();
            potential_->unpack(mx->output_buffer());
        }
        Platform::bcast(&rms, 1, 0);
        
        kset_->find_eigen_states(potential_, true);
        kset_->find_band_occupancies();
        density_->generate(*kset_);
        
        double etot = total_energy();
        
        print_info();
        
        if (Platform::mpi_rank() == 0)
        {
            printf("iteration : %3i, potential RMS %12.6f, energy difference : %12.6f, beta : %12.6f\n", 
                    iter, rms, fabs(eold - etot), mx->beta());
        }
        
        if (fabs(eold - etot) < energy_tol && rms < potential_tol) break;

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
