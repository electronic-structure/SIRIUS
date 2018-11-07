#include <sirius.h>

using namespace sirius;

void test_grid(std::vector<int> grid__)
{
    MPI_grid mpi_grid(grid__, Communicator::world());

    pstdout pout(Communicator::world());

    if (Communicator::world().rank() == 0) {
        pout.printf("dimensions: %i %i %i\n", mpi_grid.communicator(1 << 0).size(), mpi_grid.communicator(1 << 1).size(),
                    mpi_grid.communicator(1 << 2).size());
    }

    pout.printf("rank(flat): %3i, coordinate: %3i %3i %3i, hostname: %s\n", Communicator::world().rank(),
        mpi_grid.communicator(1 << 0).size(), mpi_grid.communicator(1 << 1).size(),
        mpi_grid.communicator(1 << 2).size(), utils::hostname().c_str());
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid=", "{vector3d<int>} MPI grid");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }
    std::vector<int> mpi_grid_dims = args.value<std::vector<int>>("mpi_grid", {1, 1, 1});

    sirius::initialize(1);
    test_grid(mpi_grid_dims);
    sirius::finalize();
}
