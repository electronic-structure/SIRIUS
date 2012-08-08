#include "sirius.h"

extern "C" void FORTRAN(sirius_set_lattice_vectors)(real8* a1, real8* a2, real8* a3)
{
    sirius::sirius_global.set_lattice_vectors(a1, a2, a3);
}

extern "C" void FORTRAN(sirius_add_atom_type)(int4* atom_type_id, char* _label, int4 label_len)
{
    std::string label(_label, label_len);
    sirius::sirius_global.add_atom_type(*atom_type_id, label);
}

extern "C" void FORTRAN(sirius_add_atom)(int4* atom_type_id, real8* position, real8* vector_field)
{
    sirius::sirius_global.add_atom(*atom_type_id, position, vector_field);
}

extern "C" void FORTRAN(sirius_print_info)()
{
    sirius::sirius_global.print_info();
}

extern "C" void FORTRAN(sirius_initialize)()
{
    sirius::sirius_global.initialize();
}

extern "C" void FORTRAN(sirius_clear)()
{
    sirius::sirius_global.clear();
}

