#include <sirius.h>

using namespace sirius;

void test_comm_split(int comm_size)
{
    if (mpi_comm_world().rank() == 0) {
        printf("sub comm size: %i\n", comm_size);
    }

    auto c1 = mpi_comm_world().split(mpi_comm_world().rank() / comm_size);
    auto c2 = mpi_comm_world().split(mpi_comm_world().rank() % comm_size);

    runtime::pstdout pout(mpi_comm_world());
    
    pout.printf("global rank: %i, c1.rank: %i, c2.rank: %i\n", mpi_comm_world().rank(), c1.rank(), c2.rank());
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--comm_size=", "{int} size of sub-communicator");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    int comm_size = args.value<int>("comm_size", 1);

    sirius::initialize(1);
    test_comm_split(comm_size);
    sirius::finalize();
}
