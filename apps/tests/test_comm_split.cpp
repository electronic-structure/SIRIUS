#include <sirius.hpp>

using namespace sirius;

void test_comm_split(int comm_size)
{
    if (Communicator::world().rank() == 0) {
        printf("sub comm size: %i\n", comm_size);
    }

    auto c1 = Communicator::world().split(Communicator::world().rank() / comm_size);
    auto c2 = Communicator::world().split(Communicator::world().rank() % comm_size);

    auto c3 = c1;
    Communicator c4(c2);

    pstdout pout(Communicator::world());
    
    pout.printf("global rank: %i, c1.rank: %i, c2.rank: %i\n", Communicator::world().rank(), c1.rank(), c2.rank());
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
