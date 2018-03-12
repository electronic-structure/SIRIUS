#include <sirius.h>

using namespace sirius;

void test1()
{
    Simulation_context ctx(mpi_comm_world());
    ctx.set_esm_type("full_potential_lapwlo");

    ctx.unit_cell().add_atom_type("A", "");
    ctx.unit_cell().add_atom_type("B", "");

    ctx.unit_cell().atom_type(0).zn(10);
    ctx.unit_cell().atom_type(1).zn(20);
    
    ctx.initialize();

}

int main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize();

    test1();

    sirius::finalize();
}
