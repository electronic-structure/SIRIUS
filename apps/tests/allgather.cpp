#include <sirius.h>

using namespace sirius;

int main(int argn, char** argv)
{   
    Platform::initialize(true);

    int N = 11;
    std::vector<double> vec(N, 0.0);
    
    splindex<block> spl(N, Platform::num_mpi_ranks(), Platform::mpi_rank());

    for (int i = 0; i < (int)spl.local_size(); i++)
    {
        vec[spl[i]] = Platform::mpi_rank() + 1.0;
    }

    pstdout pout;
    if (Platform::mpi_rank() == 0) pout.printf("before\n");
    pout.printf("rank : %i array : ", Platform::mpi_rank());
    for (int i = 0; i < N; i++)
    {
        pout.printf("%f ", vec[i]);
    }
    pout.printf("\n");
    pout.flush(0);

    Platform::allgather(&vec[0], (int)spl.global_offset(), (int)spl.local_size()); 
 
    if (Platform::mpi_rank() == 0) pout.printf("after\n");
    pout.printf("rank : %i array : ", Platform::mpi_rank());
    for (int i = 0; i < N; i++)
    {
        pout.printf("%f ", vec[i]);
    }
    pout.printf("\n");
    pout.flush(0);

    Platform::barrier();
    
    Platform::finalize();

}
