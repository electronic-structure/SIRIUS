#include <sirius.h>

using namespace sirius;

void read_atom(std::string fname)
{
    Simulation_parameters params;
    params.electronic_structure_method("pseudopotential");

    Atom_type atype(params, 0, "", fname);
    atype.init(0);
}

int main(int argn, char **argv)
{
    sirius::initialize(1);
    read_atom(argv[1]);
    sirius::finalize();
}
