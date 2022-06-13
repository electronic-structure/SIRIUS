#include <sirius.hpp>

using namespace sirius;

void test_allgather()
{
    int N = 11;
    std::vector<double> vec(N, 0.0);

    splindex<splindex_t::block> spl(N, Communicator::world().size(), Communicator::world().rank());

    for (int i = 0; i < spl.local_size(); i++) {
        vec[spl[i]] = Communicator::world().rank() + 1.0;
    }

    {
        sddk::pstdout pout(Communicator::world());
        if (Communicator::world().rank() == 0) {
            pout << "before" << std::endl;
        }
        pout << "rank : " << Communicator::world().rank() << " array : ";
        for (int i = 0; i < N; i++) {
            pout << vec[i];
        }
        pout << std::endl;
        std::cout << pout.flush(0);

        Communicator::world().allgather(&vec[0], spl.local_size(), spl.global_offset());

        if (Communicator::world().rank() == 0) {
            pout << "after" << std::endl;
        }
        pout << "rank : " << Communicator::world().rank() << " array : ";
        for (int i = 0; i < N; i++) {
            pout << vec[i];
        }
        pout << std::endl;
        std::cout << pout.flush(0);
    }
    Communicator::world().barrier();
}

int main(int argn, char** argv)
{
    sirius::initialize(true);

    test_allgather();

    sirius::finalize();
}
