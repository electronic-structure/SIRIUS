#include "sirius.h"

/** \file sirius.cpp
    \brief Fortran API
*/

sirius::Density* density = NULL;

sirius::Potential* potential = NULL;

sirius::Global global;

extern "C" 
{

/*
    primitive set functions
*/

/// set lattice vectors

/** Fortran example:
    \code{.F90}
        call sirius_set_lattice_vectors(avec(1,1), avec(1,2), avec(1,3))
    \endcode
*/
void FORTRAN(sirius_set_lattice_vectors)(real8* a1, real8* a2, real8* a3)
{
    global.set_lattice_vectors(a1, a2, a3);
}

void FORTRAN(sirius_set_lmax_apw)(int4* lmax_apw)
{
    global.set_lmax_apw(*lmax_apw);
}

void FORTRAN(sirius_set_lmax_rho)(int4* lmax_rho)
{
    global.set_lmax_rho(*lmax_rho);
}

void FORTRAN(sirius_set_lmax_pot)(int4* lmax_pot)
{
    global.set_lmax_pot(*lmax_pot);
}

void FORTRAN(sirius_set_pw_cutoff)(real8* pw_cutoff)
{
    global.set_pw_cutoff(*pw_cutoff);
}

void FORTRAN(sirius_set_aw_cutoff)(real8* aw_cutoff)
{
    global.set_aw_cutoff(*aw_cutoff);
}

void FORTRAN(sirius_set_charge_density_ptr)(real8* rhomt, real8* rhoit)
{
    density->set_charge_density_ptr(rhomt, rhoit);
}

void FORTRAN(sirius_set_magnetization_ptr)(real8* magmt, real8* magit)
{
    density->set_magnetization_ptr(magmt, magit);
}

void FORTRAN(sirius_set_effective_potential_ptr)(real8* veffmt, real8* veffir)
{
    potential->set_effective_potential_ptr(veffmt, veffir);
}

void FORTRAN(sirius_set_effective_magnetic_field_ptr)(real8* beffmt, real8* beffir)
{
    potential->set_effective_magnetic_field_ptr(beffmt, beffir);
}

void FORTRAN(sirius_set_equivalent_atoms)(int4* equivalent_atoms)
{
    global.set_equivalent_atoms(equivalent_atoms);
}

void FORTRAN(sirius_set_num_spins)(int4* num_spins)
{
    global.set_num_spins(*num_spins);
}

void FORTRAN(sirius_set_num_mag_dims)(int4* num_mag_dims)
{
    global.set_num_mag_dims(*num_mag_dims);
}

/*
    primitive get functions
*/
void FORTRAN(sirius_get_max_num_mt_points)(int4* max_num_mt_points)
{
    *max_num_mt_points = global.max_num_mt_points();
}

void FORTRAN(sirius_get_num_mt_points)(int4* atom_type_id, int4* num_mt_points)
{
    *num_mt_points = global.atom_type_by_id(*atom_type_id)->num_mt_points();
}

void FORTRAN(sirius_get_mt_points)(int4* atom_type_id, real8* mt_points)
{
    memcpy(mt_points, global.atom_type_by_id(*atom_type_id)->radial_grid().get_ptr(),
        global.atom_type_by_id(*atom_type_id)->num_mt_points() * sizeof(real8));
}

void FORTRAN(sirius_get_num_grid_points)(int4* num_grid_points)
{
    *num_grid_points = global.fft().size();
}

void FORTRAN(sirius_get_num_bands)(int4* num_bands)
{
    *num_bands = global.num_bands();
}

void FORTRAN(sirius_get_num_gvec)(int4* num_gvec)
{
    *num_gvec = global.num_gvec();
}

void FORTRAN(sirius_get_fft_grid_size)(int4* grid_size)
{
    grid_size[0] = global.fft().size(0);
    grid_size[1] = global.fft().size(1);
    grid_size[2] = global.fft().size(2);
}

void FORTRAN(sirius_get_fft_grid_limits)(int4* d, int4* ul, int4* val)
{
    *val = global.fft().grid_limits(*d, *ul);
}

void FORTRAN(sirius_get_fft_index)(int4* fft_index)
{
    memcpy(fft_index, global.fft_index(),  global.fft().size() * sizeof(int4));
    for (int i = 0; i < global.fft().size(); i++) fft_index[i]++;
}

void FORTRAN(sirius_get_gvec)(int4* gvec)
{
    memcpy(gvec, global.gvec(0), 3 * global.fft().size() * sizeof(int4));
}

void FORTRAN(sirius_get_index_by_gvec)(int4* index_by_gvec)
{
    memcpy(index_by_gvec, global.index_by_gvec(), global.fft().size() * sizeof(int4));
    for (int i = 0; i < global.fft().size(); i++) index_by_gvec[i]++;
}

void FORTRAN(sirius_get_num_electrons)(real8* num_electrons)
{
    *num_electrons = global.num_electrons();
}

void FORTRAN(sirius_get_num_valence_electrons)(real8* num_valence_electrons)
{
    *num_valence_electrons = global.num_valence_electrons();
}

void FORTRAN(sirius_get_num_core_electrons)(real8* num_core_electrons)
{
    *num_core_electrons = global.num_core_electrons();
}

void FORTRAN(sirius_add_atom_type)(int4* atom_type_id, char* _label, int4 label_len)
{
    std::string label(_label, label_len);
    global.add_atom_type(*atom_type_id, label);
}

void FORTRAN(sirius_add_atom)(int4* atom_type_id, real8* position, real8* vector_field)
{
    global.add_atom(*atom_type_id, position, vector_field);
}

/*
    main functions
*/
void FORTRAN(sirius_platform_initialize)(int4* call_mpi_init_)
{
    bool call_mpi_init = (*call_mpi_init_ != 0) ? true : false; 
    Platform::initialize(call_mpi_init);
}

void FORTRAN(sirius_global_initialize)()
{
    global.initialize();
}

//void FORTRAN(sirius_band_initialize)(void)
//{
//    band = new sirius::Band(global);
//}
//
void FORTRAN(sirius_potential_initialize)(void)
{
    potential = new sirius::Potential(global);
}

void FORTRAN(sirius_density_initialize)(int4* num_kpoints, double* kpoints_, double* kpoint_weights)
{
    mdarray<double, 2> kpoints(kpoints_, 3, *num_kpoints); 
    density = new sirius::Density(global, potential, kpoints, kpoint_weights);
}

void FORTRAN(sirius_clear)(void)
{
    global.clear();
}

void FORTRAN(sirius_initial_density)(void)
{
    density->initial_density();
}

void FORTRAN(sirius_generate_effective_potential)(void)
{
    potential->generate_effective_potential(density->rho(), density->magnetization());
}

void FORTRAN(sirius_generate_density)(void)
{
    density->generate();
}

void FORTRAN(sirius_density_find_eigen_states)(void)
{
    density->find_eigen_states();
}

void FORTRAN(sirius_density_find_band_occupancies)(void)
{
    density->find_band_occupancies();
}



#if 0
extern "C" void FORTRAN(sirius_get_step_function)(real8* step_function)
{
    //global.get_step_function(step_function);
    memcpy(step_function, global.step_function(), fft().size() * sizeof(real8));
}
#endif

void FORTRAN(sirius_density_set_band_occupancies)(int4* ik_, real8* band_occupancies)
{
    int ik = *ik_ - 1;
    density->set_band_occupancies(ik, band_occupancies);
}

void FORTRAN(sirius_density_get_band_energies)(int4* ik_, real8* band_energies)
{
    int ik = *ik_ - 1;
    density->get_band_energies(ik, band_energies);
}

void FORTRAN(sirius_density_get_band_occupancies)(int4* ik_, real8* band_occupancies)
{
    int ik = *ik_ - 1;
    density->get_band_occupancies(ik, band_occupancies);
}

void FORTRAN(sirius_density_integrate)(void)
{
    density->integrate();
}

/*
    print info
*/
void FORTRAN(sirius_print_info)(void)
{
    global.print_info();
    if (density) density->print_info();
}

void FORTRAN(sirius_print_timers)(void)
{
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
    potential->hdf5_read();
}

void FORTRAN(sirius_bands)(int4* num_kpoints, real8* kpoints_, real8* dk_)
{
    mdarray<double, 2> kpoints(kpoints_, 3, *num_kpoints); 

    sirius::kpoint_set kpoint_set_(global.mpi_grid());
    for (int ik = 0; ik < kpoints.size(1); ik++)
        kpoint_set_.add_kpoint(&kpoints(0, ik), 0.0, global);

    // distribute k-points along the 1-st direction of the MPI grid
    splindex<block> spl_num_kpoints_(kpoint_set_.num_kpoints(), global.mpi_grid().dimension_size(0), 
                                     global.mpi_grid().coordinate(0));

    global.solve_free_atoms();

    potential->set_spherical_potential();
    potential->set_nonspherical_potential();
    global.generate_radial_functions();
    global.generate_radial_integrals();

    // generate plane-wave coefficients of the potential in the interstitial region
    for (int ir = 0; ir < global.fft().size(); ir++)
         potential->effective_potential()->f_it(ir) *= global.step_function(ir);

    global.fft().input(potential->effective_potential()->f_it());
    global.fft().transform(-1);
    global.fft().output(global.num_gvec(), global.fft_index(), potential->effective_potential()->f_pw());
    
    sirius::Band* band = new sirius::Band(global);
    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
    {
        int ik = spl_num_kpoints_[ikloc];
        kpoint_set_[ik]->initialize(band);
        kpoint_set_[ik]->find_eigen_states(band, potential->effective_potential(),
                                           potential->effective_magnetic_field());
    } 
    // synchronize eigen-values
    kpoint_set_.sync_band_energies(global.num_bands(), spl_num_kpoints_);

    if (global.mpi_grid().root())
    {
        FILE* fout = fopen("bands.dat", "w");
        for (int i = 0; i < global.num_bands(); i++)
        {
            for (int ik = 0; ik < kpoint_set_.num_kpoints(); ik++)
            {
                fprintf(fout, "%f %f\n", dk_[ik], kpoint_set_[ik]->band_energy(i));
            }
            fprintf(fout, "\n");
        }
        fclose(fout);
    }
}

void FORTRAN(sirius_print_rti)(void)
{
    global.print_rti();
}

void FORTRAN(sirius_get_occupation_matrix)(int4* atom_id, complex16* occupation_matrix)
{
    int ia = *atom_id - 1;
    global.atom(ia)->get_occupation_matrix(occupation_matrix);
}

void FORTRAN(sirius_set_uj_correction_matrix)(int4* atom_id, int4* l, complex16* uj_correction_matrix)
{
    int ia = *atom_id - 1;
    global.atom(ia)->set_uj_correction_matrix(*l, uj_correction_matrix);
}

void FORTRAN(sirius_set_so_correction)(int4* so_correction)
{
    if (*so_correction != 0) 
        global.set_so_correction(true);
    else
        global.set_so_correction(false);
}

void FORTRAN(sirius_set_uj_correction)(int4* uj_correction)
{
    if (*uj_correction != 0)
        global.set_uj_correction(true);
    else
        global.set_uj_correction(false);
}

void FORTRAN(sirius_platform_mpi_rank)(int4* rank)
{
    *rank = Platform::mpi_rank();
}

void FORTRAN(sirius_global_set_sync_flag)(int4* flag)
{
    global.set_sync_flag(*flag);
}

void FORTRAN(sirius_global_get_sync_flag)(int4* flag)
{
    *flag = global.sync_flag();
}

void FORTRAN(sirius_platform_barrier)(void)
{
    Platform::barrier();
}

void FORTRAN(sirius_get_total_energy)(real8* total_energy)
{
    *total_energy = global.total_energy();
}



} // extern "C"
