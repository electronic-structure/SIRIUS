#include <sirius.h>

using namespace sirius;

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid=", "{vector2d<int>} MPI grid");

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }
    std::vector<int> mpi_grid_dims = args.value< std::vector<int> >("mpi_grid", {1, 1});

    sirius::initialize(1);

    {
        MPI_grid mpi_grid(mpi_grid_dims, mpi_comm_world());

        if (mpi_comm_world().rank() == 0)
        {
            printf("dimensions: %i %i\n", mpi_grid_dims[0], mpi_grid_dims[1]);
            printf("dimensions: %i %i\n", mpi_grid.dimension_size(0), mpi_grid.dimension_size(1));
        }

        for (int i0 = 0; i0 < mpi_grid.dimension_size(0); i0++)
        {
            for (int i1 = 0; i1 < mpi_grid.dimension_size(1); i1++)
            {
                if (i0 == mpi_grid.communicator(1 << 0).rank() && i1 == mpi_grid.communicator(1 << 1).rank())
                {
                    printf("rank(flat): %i, rank(cart): %i, i0: %i, i1: %i\n", mpi_comm_world().rank(), mpi_grid.communicator().rank(), i0, i1);
                }
                mpi_comm_world().barrier();
            }
        }
    }

    sirius::finalize();
}
