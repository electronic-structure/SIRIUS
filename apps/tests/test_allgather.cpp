#include <sirius.hpp>

using namespace sirius;

void test_allgather()
{
    int N = 11;
    std::vector<double> vec(N, 0.0);

    sddk::splindex<sddk::splindex_t::block> spl(N, sddk::Communicator::world().size(), sddk::Communicator::world().rank());

    for (int i = 0; i < spl.local_size(); i++) {
        vec[spl[i]] = sddk::Communicator::world().rank() + 1.0;
    }

    {
        sddk::pstdout pout(sddk::Communicator::world());
        if (sddk::Communicator::world().rank() == 0) pout.printf("before\n");
        pout.printf("rank : %i array : ", sddk::Communicator::world().rank());
        for (int i = 0; i < N; i++) {
            pout.printf("%f ", vec[i]);
        }
        pout.printf("\n");
        pout.flush();

        sddk::Communicator::world().allgather(&vec[0], spl.local_size(), spl.global_offset());

        if (sddk::Communicator::world().rank() == 0) pout.printf("after\n");
        pout.printf("rank : %i array : ", sddk::Communicator::world().rank());
        for (int i = 0; i < N; i++) {
            pout.printf("%f ", vec[i]);
        }
        pout.printf("\n");
        pout.flush();
    }
    sddk::Communicator::world().barrier();
}

int main(int argn, char** argv)
{
    sirius::initialize(true);

    test_allgather();

    sirius::finalize();

}
