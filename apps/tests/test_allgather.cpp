#include <sirius.h>

using namespace sirius;

void test_allgather()
{
    int N = 11;
    std::vector<double> vec(N, 0.0);

    splindex<splindex_t::block> spl(N, Communicator::world().size(), Communicator::world().rank());

    for (int i = 0; i < spl.local_size(); i++)
    {
        vec[spl[i]] = Communicator::world().rank() + 1.0;
    }

    {
        sddk::pstdout pout(Communicator::world());
        if (Communicator::world().rank() == 0) pout.printf("before\n");
        pout.printf("rank : %i array : ", Communicator::world().rank());
        for (int i = 0; i < N; i++)
        {
            pout.printf("%f ", vec[i]);
        }
        pout.printf("\n");
        pout.flush();

        Communicator::world().allgather(&vec[0], spl.global_offset(), spl.local_size()); 
 
        if (Communicator::world().rank() == 0) pout.printf("after\n");
        pout.printf("rank : %i array : ", Communicator::world().rank());
        for (int i = 0; i < N; i++)
        {
            pout.printf("%f ", vec[i]);
        }
        pout.printf("\n");
        pout.flush();
    }
    Communicator::world().barrier();
}

int main(int argn, char** argv)
{   
    sirius::initialize(true);

    test_allgather();
    
    sirius::finalize();

}
