#include "sirius.h"

extern "C" 
{

/*
    primitive set functions
*/
void FORTRAN(sirius_set_lattice_vectors)(real8* a1, real8* a2, real8* a3)
{
    sirius::global.set_lattice_vectors(a1, a2, a3);
}

void FORTRAN(sirius_set_lmax_apw)(int4* lmax_apw)
{
    sirius::global.set_lmax_apw(*lmax_apw);
}

void FORTRAN(sirius_set_lmax_rho)(int4* lmax_rho)
{
    sirius::global.set_lmax_rho(*lmax_rho);
}

void FORTRAN(sirius_set_lmax_pot)(int4* lmax_pot)
{
    sirius::global.set_lmax_pot(*lmax_pot);
}

void FORTRAN(sirius_set_pw_cutoff)(real8* pw_cutoff)
{
    sirius::global.set_pw_cutoff(*pw_cutoff);
}

void FORTRAN(sirius_set_aw_cutoff)(real8* aw_cutoff)
{
    sirius::global.set_aw_cutoff(*aw_cutoff);
}

void FORTRAN(sirius_set_charge_density_ptr)(real8* rhomt, real8* rhoit)
{
    sirius::density.set_charge_density_ptr(rhomt, rhoit);
}

void FORTRAN(sirius_set_magnetization_ptr)(real8* magmt, real8* magit)
{
    sirius::density.set_magnetization_ptr(magmt, magit);
}

void FORTRAN(sirius_set_effective_potential_ptr)(real8* veffmt, real8* veffir)
{
    sirius::potential.set_effective_potential_ptr(veffmt, veffir);
}

void FORTRAN(sirius_set_equivalent_atoms)(int4* equivalent_atoms)
{
    sirius::global.set_equivalent_atoms(equivalent_atoms);
}

void FORTRAN(sirius_set_num_spins)(int4* num_spins)
{
    sirius::global.set_num_spins(*num_spins);
}

void FORTRAN(sirius_set_num_dmat)(int4* num_dmat)
{
    sirius::global.set_num_dmat(*num_dmat);
}

/*
    primitive get functions
*/
void FORTRAN(sirius_get_max_num_mt_points)(int4* max_num_mt_points)
{
    *max_num_mt_points = sirius::global.max_num_mt_points();
}

void FORTRAN(sirius_get_num_mt_points)(int4* atom_type_id, int4* num_mt_points)
{
    *num_mt_points = sirius::global.atom_type_by_id(*atom_type_id)->num_mt_points();
}

void FORTRAN(sirius_get_mt_points)(int4* atom_type_id, real8* mt_points)
{
    memcpy(mt_points, sirius::global.atom_type_by_id(*atom_type_id)->radial_grid().get_ptr(),
        sirius::global.atom_type_by_id(*atom_type_id)->num_mt_points() * sizeof(real8));
}

void FORTRAN(sirius_get_num_grid_points)(int4* num_grid_points)
{
    *num_grid_points = sirius::global.fft().size();
}

void FORTRAN(sirius_get_num_bands)(int4* num_bands)
{
    *num_bands = sirius::global.num_bands();
}

void FORTRAN(sirius_get_num_gvec)(int4* num_gvec)
{
    *num_gvec = sirius::global.num_gvec();
}

void FORTRAN(sirius_get_fft_grid_size)(int4* grid_size)
{
    grid_size[0] = sirius::global.fft().size(0);
    grid_size[1] = sirius::global.fft().size(1);
    grid_size[2] = sirius::global.fft().size(2);
}

void FORTRAN(sirius_get_fft_grid_limits)(int4* d, int4* ul, int4* val)
{
    *val = sirius::global.fft().grid_limits(*d, *ul);
}

void FORTRAN(sirius_get_fft_index)(int4* fft_index)
{
    memcpy(fft_index, sirius::global.fft_index(),  sirius::global.fft().size() * sizeof(int4));
    for (int i = 0; i < sirius::global.fft().size(); i++) fft_index[i]++;
}

void FORTRAN(sirius_get_gvec)(int4* gvec)
{
    memcpy(gvec, sirius::global.gvec(0), 3 * sirius::global.fft().size() * sizeof(int4));
}

void FORTRAN(sirius_get_index_by_gvec)(int4* index_by_gvec)
{
    memcpy(index_by_gvec, sirius::global.index_by_gvec(), sirius::global.fft().size() * sizeof(int4));
    for (int i = 0; i < sirius::global.fft().size(); i++) index_by_gvec[i]++;
}

void FORTRAN(sirius_get_num_electrons)(real8* num_electrons)
{
    *num_electrons = sirius::global.num_electrons();
}

void FORTRAN(sirius_get_num_valence_electrons)(real8* num_valence_electrons)
{
    *num_valence_electrons = sirius::global.num_valence_electrons();
}

void FORTRAN(sirius_get_num_core_electrons)(real8* num_core_electrons)
{
    *num_core_electrons = sirius::global.num_core_electrons();
}

void FORTRAN(sirius_add_atom_type)(int4* atom_type_id, char* _label, int4 label_len)
{
    std::string label(_label, label_len);
    sirius::global.add_atom_type(*atom_type_id, label);
}

void FORTRAN(sirius_add_atom)(int4* atom_type_id, real8* position, real8* vector_field)
{
    sirius::global.add_atom(*atom_type_id, position, vector_field);
}

/*
    main functions
*/
void FORTRAN(sirius_initialize)(void)
{
    sirius::global.initialize();
    sirius::band.initialize();
    sirius::density.initialize();
    sirius::potential.initialize();
}

void FORTRAN(sirius_clear)(void)
{
    sirius::global.clear();
}

void FORTRAN(sirius_initial_density)(void)
{
    sirius::density.initial_density();
}

void FORTRAN(sirius_generate_effective_potential)(void)
{
    sirius::potential.generate_effective_potential();
}

void FORTRAN(sirius_generate_density)(void)
{
    sirius::density.generate();
}

/*extern "C" void FORTRAN(sirius_get_density)(real8* rhomt, real8* rhoir)
{
    sirius::density.get_density(rhomt, rhoir);
}*/

#if 0
extern "C" void FORTRAN(sirius_get_step_function)(real8* step_function)
{
    //sirius::global.get_step_function(step_function);
    memcpy(step_function, global.step_function(), fft().size() * sizeof(real8));
}
#endif

void FORTRAN(sirius_density_add_kpoint)(int4* kpoint_id, real8* vk, real8* weight)
{
    sirius::density.add_kpoint(*kpoint_id, vk, *weight);
}

void FORTRAN(sirius_density_set_band_occupancies)(int4* kpoint_id, real8* band_occupancies)
{
    sirius::density.set_band_occupancies(*kpoint_id, band_occupancies);
}

void FORTRAN(sirius_density_get_band_energies)(int4* kpoint_id, real8* band_energies)
{
    sirius::density.get_band_energies(*kpoint_id, band_energies);
}

/*
    print info
*/
void FORTRAN(sirius_print_info)(void)
{
    sirius::global.print_info();
    sirius::density.print_info();
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

}
