#include <sirius.h>

using namespace sirius;

void test_allgather()
{
    int N = 11;
    std::vector<double> vec(N, 0.0);

    splindex<block> spl(N, mpi_comm_world().size(), mpi_comm_world().rank());

    for (int i = 0; i < spl.local_size(); i++)
    {
        vec[spl[i]] = mpi_comm_world().rank() + 1.0;
    }

    {
        pstdout pout(mpi_comm_world());
        if (mpi_comm_world().rank() == 0) pout.printf("before\n");
        pout.printf("rank : %i array : ", mpi_comm_world().rank());
        for (int i = 0; i < N; i++)
        {
            pout.printf("%f ", vec[i]);
        }
        pout.printf("\n");
        pout.flush();

        mpi_comm_world().allgather(&vec[0], spl.global_offset(), spl.local_size()); 
 
        if (mpi_comm_world().rank() == 0) pout.printf("after\n");
        pout.printf("rank : %i array : ", mpi_comm_world().rank());
        for (int i = 0; i < N; i++)
        {
            pout.printf("%f ", vec[i]);
        }
        pout.printf("\n");
        pout.flush();
    }
    mpi_comm_world().barrier();
}

int main(int argn, char** argv)
{   
    sirius::initialize(true);

    test_allgather();
    
    sirius::finalize();

}
