#include <sirius.h>

using namespace sirius;

void test_unit_cell()
{
    Simulation_context ctx("sirius.json", mpi_comm_world());
    ctx.unit_cell().initialize();
    
    vector3d<int> k_grid(2, 2, 1);
    vector3d<int> k_shift(1, 1, 0);

    mdarray<double, 2> kp;
    std::vector<double> wk;
    int nk = ctx.unit_cell().symmetry()->get_irreducible_reciprocal_mesh(k_grid, k_shift, kp, wk);

    printf("number of k-points: %i\n", nk);
    for (int ik = 0; ik < nk; ik++)
    {
        printf("kp: %f %f %f\n", kp(0, ik), kp(1, ik), kp(2, ik));
    }
}

int main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test_unit_cell();
    sirius::finalize();
}
