#include "sirius.h"

extern "C" void FORTRAN(sirius_set_lattice_vectors)(real8* a1, real8* a2, real8* a3)
{
    sirius::global.set_lattice_vectors(a1, a2, a3);
}

extern "C" void FORTRAN(sirius_add_atom_type)(int4* atom_type_id, char* _label, int4 label_len)
{
    std::string label(_label, label_len);
    sirius::global.add_atom_type(*atom_type_id, label);
}

extern "C" void FORTRAN(sirius_add_atom)(int4* atom_type_id, real8* position, real8* vector_field)
{
    sirius::global.add_atom(*atom_type_id, position, vector_field);
}

extern "C" void FORTRAN(sirius_print_info)()
{
    sirius::global.print_info();
}

extern "C" void FORTRAN(sirius_initialize)()
{
    sirius::global.initialize();
    sirius::density.initialize();
    sirius::potential.init();
}

extern "C" void FORTRAN(sirius_clear)()
{
    sirius::global.clear();
}

extern "C" void FORTRAN(sirius_set_lmax_apw)(int4* lmax_apw)
{
    sirius::global.set_lmax_apw(*lmax_apw);
}

extern "C" void FORTRAN(sirius_set_lmax_rho)(int4* lmax_rho)
{
    sirius::global.set_lmax_rho(*lmax_rho);
}

extern "C" void FORTRAN(sirius_set_lmax_pot)(int4* lmax_pot)
{
    sirius::global.set_lmax_pot(*lmax_pot);
}

extern "C" void FORTRAN(sirius_get_max_num_mt_points)(int4* max_num_mt_points)
{
    *max_num_mt_points = sirius::global.max_num_mt_points();
}

extern "C" void FORTRAN(sirius_get_num_mt_points)(int4* atom_type_id, int4* num_mt_points)
{
    *num_mt_points = sirius::global.atom_type_by_id(*atom_type_id)->num_mt_points();
}

extern "C" void FORTRAN(sirius_get_mt_points)(int4* atom_type_id, real8* mt_points)
{
    memcpy(mt_points, sirius::global.atom_type_by_id(*atom_type_id)->radial_grid().get_ptr(),
        sirius::global.atom_type_by_id(*atom_type_id)->num_mt_points() * sizeof(real8));
}

extern "C" void FORTRAN(sirius_get_num_grid_points)(int4* num_grid_points)
{
    *num_grid_points = sirius::global.fft().size();
}

extern "C" void FORTRAN(sirius_initial_density)()
{
    sirius::density.initial_density();
    sirius::potential.generate_effective_potential();
}

extern "C" void FORTRAN(sirius_get_density)(real8* rhomt, real8* rhoir)
{
    sirius::density.get_density(rhomt, rhoir);
}

extern "C" void FORTRAN(sirius_get_step_function)(real8* step_function)
{
    sirius::global.get_step_function(step_function);
}


