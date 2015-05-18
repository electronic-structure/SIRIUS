#include <sirius.h>

int main(int argn, char **argv)
{
    Platform::initialize(1);

    sirius::Atom_type atom_type(1, "Si", "Si.json", full_potential_lapwlo, CPU);
    atom_type.init(10, 8, 0, 0);

    atom_type.print_info();
    Platform::finalize();
}
