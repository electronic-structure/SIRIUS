#include <sirius.hpp>

using namespace sirius;

void test_allgather()
{
    int N = 11;
    std::vector<double> vec(N, 0.0);

    sddk::splindex<sddk::splindex_t::block> spl(N, mpi::Communicator::world().size(), mpi::Communicator::world().rank());

    for (int i = 0; i < spl.local_size(); i++) {
        vec[spl[i]] = mpi::Communicator::world().rank() + 1.0;
    }

    {
        mpi::pstdout pout(mpi::Communicator::world());
        if (mpi::Communicator::world().rank() == 0) {
            pout << "before" << std::endl;
        }
        pout << "rank : " << mpi::Communicator::world().rank() << " array : ";
        for (int i = 0; i < N; i++) {
            pout << vec[i];
        }
        pout << std::endl;
        std::cout << pout.flush(0);

        mpi::Communicator::world().allgather(&vec[0], spl.local_size(), spl.global_offset());

        if (mpi::Communicator::world().rank() == 0) {
            pout << "after" << std::endl;
        }
        pout << "rank : " << mpi::Communicator::world().rank() << " array : ";
        for (int i = 0; i < N; i++) {
            pout << vec[i];
        }
        pout << std::endl;
        std::cout << pout.flush(0);
    }
    mpi::Communicator::world().barrier();
}

int main(int argn, char** argv)
{
    sirius::initialize(true);

    test_allgather();

    sirius::finalize();
}
