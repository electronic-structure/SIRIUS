#include <sirius.h>

using namespace sirius;

void test_allgather()
{
    int N = 11;
    std::vector<double> vec(N, 0.0);

    Communicator comm(MPI_COMM_WORLD);
    
    splindex<block> spl(N, comm.size(), comm.rank());

    for (int i = 0; i < (int)spl.local_size(); i++)
    {
        vec[spl[i]] = comm.rank() + 1.0;
    }

    {
        pstdout pout(comm);
        if (comm.rank() == 0) pout.printf("before\n");
        pout.printf("rank : %i array : ", comm.rank());
        for (int i = 0; i < N; i++)
        {
            pout.printf("%f ", vec[i]);
        }
        pout.printf("\n");
        pout.flush();

        comm.allgather(&vec[0], (int)spl.global_offset(), (int)spl.local_size()); 
 
        if (comm.rank() == 0) pout.printf("after\n");
        pout.printf("rank : %i array : ", comm.rank());
        for (int i = 0; i < N; i++)
        {
            pout.printf("%f ", vec[i]);
        }
        pout.printf("\n");
        pout.flush();
    }
    comm.barrier();
}

int main(int argn, char** argv)
{   
    Platform::initialize(true);

    test_allgather();
    
    Platform::finalize();

}
