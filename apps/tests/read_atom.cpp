#include <sirius.h>

int main(int argn, char **argv)
{
    sirius::initialize(1);

    sirius::Simulation_parameters parameters;

    sirius::Atom_type atom_type(parameters, 0, "Si", "Si.json");
    atom_type.init(0);

    atom_type.print_info();
    sirius::finalize();
}
