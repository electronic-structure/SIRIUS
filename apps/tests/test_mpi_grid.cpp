#include <sirius.hpp>

using namespace sirius;

void test_grid(std::vector<int> grid__)
{
    sddk::MPI_grid mpi_grid(grid__, sddk::Communicator::world());

    sddk::pstdout pout(sddk::Communicator::world());

    if (sddk::Communicator::world().rank() == 0) {
        pout << "dimensions: " << mpi_grid.communicator(1 << 0).size() << " " << mpi_grid.communicator(1 << 1).size()
             << " " << mpi_grid.communicator(1 << 2).size() << std::endl;
    }

    pout << "rank(flat): " << sddk::Communicator::world().rank() << ", coordinate: "
         << mpi_grid.communicator(1 << 0).rank() << " "
         << mpi_grid.communicator(1 << 1).rank() << " "
         << mpi_grid.communicator(1 << 2).rank() << ", hostname: " << utils::hostname() << std::endl;
    std::cout << pout.flush(0) << std::endl;
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
    std::vector<int> mpi_grid_dims = args.value("mpi_grid", std::vector<int>({1, 1, 1}));

    sirius::initialize(1);
    test_grid(mpi_grid_dims);
    sirius::finalize();
}
