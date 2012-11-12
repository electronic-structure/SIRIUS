#include "sirius.h"

extern "C" 
{

/*
    primitive set functions
*/
void FORTRAN(sirius_set_lattice_vectors)(real8* a1, real8* a2, real8* a3)
{
    sirius::static_global().set_lattice_vectors(a1, a2, a3);
}

void FORTRAN(sirius_set_lmax_apw)(int4* lmax_apw)
{
    sirius::static_global().set_lmax_apw(*lmax_apw);
}

void FORTRAN(sirius_set_lmax_rho)(int4* lmax_rho)
{
    sirius::static_global().set_lmax_rho(*lmax_rho);
}

void FORTRAN(sirius_set_lmax_pot)(int4* lmax_pot)
{
    sirius::static_global().set_lmax_pot(*lmax_pot);
}

void FORTRAN(sirius_set_pw_cutoff)(real8* pw_cutoff)
{
    sirius::static_global().set_pw_cutoff(*pw_cutoff);
}

void FORTRAN(sirius_set_aw_cutoff)(real8* aw_cutoff)
{
    sirius::static_global().set_aw_cutoff(*aw_cutoff);
}

void FORTRAN(sirius_set_charge_density_ptr)(real8* rhomt, real8* rhoit)
{
    sirius::density->set_charge_density_ptr(rhomt, rhoit);
}

void FORTRAN(sirius_set_magnetization_ptr)(real8* magmt, real8* magit)
{
    sirius::density->set_magnetization_ptr(magmt, magit);
}

void FORTRAN(sirius_set_effective_potential_ptr)(real8* veffmt, real8* veffir)
{
    sirius::potential->set_effective_potential_ptr(veffmt, veffir);
}

void FORTRAN(sirius_set_effective_magnetic_field_ptr)(real8* beffmt, real8* beffir)
{
    sirius::potential->set_effective_magnetic_field_ptr(beffmt, beffir);
}

void FORTRAN(sirius_set_equivalent_atoms)(int4* equivalent_atoms)
{
    sirius::static_global().set_equivalent_atoms(equivalent_atoms);
}

void FORTRAN(sirius_set_num_spins)(int4* num_spins)
{
    sirius::static_global().set_num_spins(*num_spins);
}

void FORTRAN(sirius_set_num_mag_dims)(int4* num_mag_dims)
{
    sirius::static_global().set_num_mag_dims(*num_mag_dims);
}

/*
    primitive get functions
*/
void FORTRAN(sirius_get_max_num_mt_points)(int4* max_num_mt_points)
{
    *max_num_mt_points = sirius::static_global().max_num_mt_points();
}

void FORTRAN(sirius_get_num_mt_points)(int4* atom_type_id, int4* num_mt_points)
{
    *num_mt_points = sirius::static_global().atom_type_by_id(*atom_type_id)->num_mt_points();
}

void FORTRAN(sirius_get_mt_points)(int4* atom_type_id, real8* mt_points)
{
    memcpy(mt_points, sirius::static_global().atom_type_by_id(*atom_type_id)->radial_grid().get_ptr(),
        sirius::static_global().atom_type_by_id(*atom_type_id)->num_mt_points() * sizeof(real8));
}

void FORTRAN(sirius_get_num_grid_points)(int4* num_grid_points)
{
    *num_grid_points = sirius::static_global().fft().size();
}

void FORTRAN(sirius_get_num_bands)(int4* num_bands)
{
    *num_bands = sirius::static_global().num_bands();
}

void FORTRAN(sirius_get_num_gvec)(int4* num_gvec)
{
    *num_gvec = sirius::static_global().num_gvec();
}

void FORTRAN(sirius_get_fft_grid_size)(int4* grid_size)
{
    grid_size[0] = sirius::static_global().fft().size(0);
    grid_size[1] = sirius::static_global().fft().size(1);
    grid_size[2] = sirius::static_global().fft().size(2);
}

void FORTRAN(sirius_get_fft_grid_limits)(int4* d, int4* ul, int4* val)
{
    *val = sirius::static_global().fft().grid_limits(*d, *ul);
}

void FORTRAN(sirius_get_fft_index)(int4* fft_index)
{
    memcpy(fft_index, sirius::static_global().fft_index(),  sirius::static_global().fft().size() * sizeof(int4));
    for (int i = 0; i < sirius::static_global().fft().size(); i++) fft_index[i]++;
}

void FORTRAN(sirius_get_gvec)(int4* gvec)
{
    memcpy(gvec, sirius::static_global().gvec(0), 3 * sirius::static_global().fft().size() * sizeof(int4));
}

void FORTRAN(sirius_get_index_by_gvec)(int4* index_by_gvec)
{
    memcpy(index_by_gvec, sirius::static_global().index_by_gvec(), sirius::static_global().fft().size() * sizeof(int4));
    for (int i = 0; i < sirius::static_global().fft().size(); i++) index_by_gvec[i]++;
}

void FORTRAN(sirius_get_num_electrons)(real8* num_electrons)
{
    *num_electrons = sirius::static_global().num_electrons();
}

void FORTRAN(sirius_get_num_valence_electrons)(real8* num_valence_electrons)
{
    *num_valence_electrons = sirius::static_global().num_valence_electrons();
}

void FORTRAN(sirius_get_num_core_electrons)(real8* num_core_electrons)
{
    *num_core_electrons = sirius::static_global().num_core_electrons();
}

void FORTRAN(sirius_add_atom_type)(int4* atom_type_id, char* _label, int4 label_len)
{
    std::string label(_label, label_len);
    sirius::static_global().add_atom_type(*atom_type_id, label);
}

void FORTRAN(sirius_add_atom)(int4* atom_type_id, real8* position, real8* vector_field)
{
    sirius::static_global().add_atom(*atom_type_id, position, vector_field);
}

/*
    main functions
*/
void FORTRAN(sirius_global_initialize)()
{
    sirius::static_global().initialize();
}

void FORTRAN(sirius_band_initialize)(void)
{
    sirius::band = new sirius::Band(sirius::static_global());
}

void FORTRAN(sirius_potential_initialize)(void)
{
    sirius::potential = new sirius::Potential(sirius::static_global());
}

void FORTRAN(sirius_density_initialize)(void)
{
    sirius::density = new sirius::Density(sirius::static_global(), sirius::potential);
}

void FORTRAN(sirius_clear)(void)
{
    sirius::static_global().clear();
}

void FORTRAN(sirius_initial_density)(void)
{
    sirius::density->initial_density();
}

void FORTRAN(sirius_generate_effective_potential)(void)
{
    sirius::potential->generate_effective_potential(sirius::density->rho(), sirius::density->magnetization());
}

void FORTRAN(sirius_generate_density)(void)
{
    sirius::density->generate();
}

void FORTRAN(sirius_density_find_eigen_states)(void)
{
    sirius::density->find_eigen_states();
}

void FORTRAN(sirius_density_find_band_occupancies)(void)
{
    sirius::density->find_band_occupancies();
}



#if 0
extern "C" void FORTRAN(sirius_get_step_function)(real8* step_function)
{
    //sirius::static_global().get_step_function(step_function);
    memcpy(step_function, global.step_function(), fft().size() * sizeof(real8));
}
#endif

void FORTRAN(sirius_density_add_kpoint)(int4* kpoint_id, real8* vk, real8* weight)
{
    sirius::density->add_kpoint(*kpoint_id, vk, *weight);
}

void FORTRAN(sirius_density_set_band_occupancies)(int4* kpoint_id, real8* band_occupancies)
{
    sirius::density->set_band_occupancies(*kpoint_id, band_occupancies);
}

void FORTRAN(sirius_density_get_band_energies)(int4* kpoint_id, real8* band_energies)
{
    sirius::density->get_band_energies(*kpoint_id, band_energies);
}

void FORTRAN(sirius_density_get_band_occupancies)(int4* kpoint_id, real8* band_occupancies)
{
    sirius::density->get_band_occupancies(*kpoint_id, band_occupancies);
}


void FORTRAN(sirius_density_integrate)(void)
{
    sirius::density->integrate();
}

/*
    print info
*/
void FORTRAN(sirius_print_info)(void)
{
    sirius::static_global().print_info();
    sirius::density->print_info();
}

void FORTRAN(sirius_print_timers)(void)
{
    printf("\n");
    printf("Timers\n");
    for (int i = 0; i < 80; i++) printf("-");
    printf("\n");
    sirius::Timer::print();
}   

void FORTRAN(sirius_timer_start)(char* name_, int4 name_len)
{
    std::string name(name_, name_len);
    sirius::ftimers[name] = new sirius::Timer(name);
}

void FORTRAN(sirius_timer_stop)(char* name_, int4 name_len)
{
    std::string name(name_, name_len);
    if (sirius::ftimers.count(name)) delete sirius::ftimers[name];
}

void FORTRAN(sirius_potential_hdf5_read)()
{
    sirius::potential->hdf5_read();
}

void FORTRAN(sirius_band)(real8* vk, real8* band_energies)
{
    static bool init = false;
    
    if (!init)
    {
        std::vector<double> enu;
        for (int i = 0; i < sirius::static_global().num_atom_types(); i++)
            sirius::static_global().atom_type(i)->solve_free_atom(1e-8, 1e-5, 1e-4, enu);

        sirius::potential->set_spherical_potential();
        sirius::potential->set_nonspherical_potential();
        sirius::static_global().generate_radial_functions();
        sirius::static_global().generate_radial_integrals();
        // generate plane-wave coefficients of the potential in the interstitial region
        
        for (int ir = 0; ir < sirius::static_global().fft().size(); ir++)
             sirius::potential->effective_potential()->f_it(ir) *= sirius::static_global().step_function(ir);

        sirius::static_global().fft().input(sirius::potential->effective_potential()->f_it());
        sirius::static_global().fft().transform(-1);
        sirius::static_global().fft().output(sirius::static_global().num_gvec(), sirius::static_global().fft_index(), 
                                    sirius::potential->effective_potential()->f_pw());
        init = true;
    }
    
    sirius::kpoint kp(sirius::static_global(), vk, 0.0);

    kp.find_eigen_states(sirius::band, sirius::potential->effective_potential(),
                         sirius::potential->effective_magnetic_field());
                         
    kp.get_band_energies(band_energies);
}

void FORTRAN(sirius_print_rti)(void)
{
    sirius::static_global().print_rti();
}

void FORTRAN(sirius_get_occupation_matrix)(int4* atom_id, complex16* occupation_matrix)
{
    int ia = *atom_id - 1;
    sirius::static_global().atom(ia)->get_occupation_matrix(occupation_matrix);
}

void FORTRAN(sirius_set_uj_correction_matrix)(int4* atom_id, int4* l, complex16* uj_correction_matrix)
{
    int ia = *atom_id - 1;
    sirius::static_global().atom(ia)->set_uj_correction_matrix(*l, uj_correction_matrix);
}

void FORTRAN(sirius_set_so_correction)(int4* so_correction)
{
    if (*so_correction != 0) 
        sirius::static_global().set_so_correction(true);
    else
        sirius::static_global().set_so_correction(false);
}

void FORTRAN(sirius_set_uj_correction)(int4* uj_correction)
{
    if (*uj_correction != 0)
        sirius::static_global().set_uj_correction(true);
    else
        sirius::static_global().set_uj_correction(false);
}

} // extern "C"
