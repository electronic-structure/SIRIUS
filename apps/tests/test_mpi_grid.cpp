#include <sirius.h>

using namespace sirius;

void test_grid(std::vector<int> grid__)
{
    MPI_grid mpi_grid(grid__, mpi_comm_world());

    runtime::pstdout pout(mpi_comm_world());

    if (mpi_comm_world().rank() == 0) {
        pout.printf("dimensions: %i %i %i\n", mpi_grid.dimension_size(0), mpi_grid.dimension_size(1), mpi_grid.dimension_size(2));
    }

    pout.printf("rank(flat): %3i, coordinate: %3i %3i %3i, hostname: %s\n", mpi_comm_world().rank(),
        mpi_grid.coordinate(0), mpi_grid.coordinate(1), mpi_grid.coordinate(2), runtime::hostname().c_str());

    //for (int i0 = 0; i0 < mpi_grid.dimension_size(0); i0++)
    //{
    //    for (int i1 = 0; i1 < mpi_grid.dimension_size(1); i1++)
    //    {
    //        if (i0 == mpi_grid.communicator(1 << 0).rank() && i1 == mpi_grid.communicator(1 << 1).rank())
    //        {
    //            printf("rank(flat): %i, rank(cart): %i, i0: %i, i1: %i\n", mpi_comm_world().rank(), mpi_grid.communicator().rank(), i0, i1);
    //        }
    //        mpi_comm_world().barrier();
    //    }
    //}
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
