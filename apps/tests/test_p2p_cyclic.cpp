#include <sirius.h>

using namespace sirius;

int test1(int size)
{
    double t = -utils::wtime();
    sddk::mdarray<char, 1> buf(size);
    sddk::mdarray<char, 1> buf1(size);
    buf.zero();

    for (int r = 0; r < Communicator::world().size(); r++) {
        int rank = Communicator::world().rank();
        int rank1 = (rank + 1) % Communicator::world().size();
        auto req = Communicator::world().isend(buf.at(memory_t::host), size, rank1, Communicator::get_tag(rank, rank1));
        int rank2 = rank - 1;
        if (rank2 < 0) {
            rank2 = Communicator::world().size() - 1;
        }
        Communicator::world().recv(buf1.at(memory_t::host), size, rank2, Communicator::get_tag(rank, rank2));
        req.wait();
    }
    t += utils::wtime();
    if (Communicator::world().rank() == 0) {
        printf("time : %f sec.\n", t);
    }
    return 0;
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--size=", "buffer size in bytes");

    args.parse_args(argn, argv);
    auto size = args.value("size", (1 << 20));

    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test1(size);
    sirius::finalize();
}
