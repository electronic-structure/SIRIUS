#include <sirius.h>

int main(int argn, char **argv)
{
    sirius::Atom_type atom_type(1, "Si");
    atom_type.create_radial_grid();
    atom_type.init(10);

    atom_type.print_info();
}
