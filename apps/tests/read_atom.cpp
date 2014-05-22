#include <sirius.h>

int main(int argn, char **argv)
{
    sirius::Atom_type atom_type(1, "Si", "Si", full_potential_lapwlo);
    atom_type.init(10, 0);

    atom_type.print_info();
}
