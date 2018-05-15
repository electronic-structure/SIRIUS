#include <sirius.h>

using namespace sirius;

int main(int argn, char **argv)
{
    sirius::initialize(1);

    Simulation_parameters parameters;

    std::vector< std::unique_ptr<Atom_type> > atom_types;

    atom_types.push_back(std::unique_ptr<Atom_type>(new Atom_type(parameters, 0, "Si", "Si.json")));
    std::cout << atom_types[0].get() << std::endl;
    atom_types.push_back(std::unique_ptr<Atom_type>(new Atom_type(parameters, 1, "C", "C.json")));
    std::cout << atom_types[0].get() << std::endl;

    //std::vector<Atom> atoms;
    //atoms.push_back(std::move(Atom(atom_types[0], vector3d<double>(0, 0, 0), vector3d<double>(0, 0, 0))));
    //atoms.push_back(std::move(Atom(atom_types[1], vector3d<double>(0.5, 0.5, 0.5), vector3d<double>(0, 0, 0))));
    //atoms.push_back(std::move(Atom(atom_types[1], vector3d<double>(0.25, 0.25, 0.25), vector3d<double>(0, 0, 0))));
    //
    //for (size_t iat = 0; iat < atom_types.size(); iat++) {
    //    atom_types[iat].init(0);
    //}

    //for (size_t iat = 0; iat < atom_types.size(); iat++) {
    //    atom_types[iat].print_info();
    //}

    //for (size_t ia = 0; ia < atoms.size(); ia++) {
    //    atoms[ia].type().print_info();
    //}

    //Unit_cell uc(parameters, mpi_comm_world());

    //uc.add_atom_type("Si", "Si.json");
    //uc.add_atom_type("C", "C.json");
    //uc.add_atom("Si", vector3d<double>(0, 0, 0));
    //uc.add_atom("C", vector3d<double>(0.5, 0, 0));
    //uc.add_atom("C", vector3d<double>(0, 0.5, 0));
    //uc.set_lattice_vectors({4,0,0}, {0,4,0}, {0,0,4});

    //uc.initialize();

    {
        Simulation_context ctx(Communicator::world());
        ctx.unit_cell().add_atom_type("Si", "Si.json");
        ctx.unit_cell().add_atom("Si", vector3d<double>(0, 0, 0));

        ctx.unit_cell().add_atom_type("C", "C.json");
        ctx.unit_cell().add_atom("C", vector3d<double>(0.5, 0, 0));
        ctx.unit_cell().add_atom("C", vector3d<double>(0, 0.5, 0));
        ctx.unit_cell().set_lattice_vectors({4,0,0}, {0,4,0}, {0,0,4});

        ctx.initialize();
    }

    sirius::finalize();
}
