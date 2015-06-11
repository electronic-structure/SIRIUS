#include <sirius.h>

using namespace sirius;

void test_gemr2d()
{
    Communicator comm(MPI_COMM_WORLD);

    BLACS_grid grid2d(comm, 2, 2, 32);
    BLACS_grid grid_row(comm, 4, 1, 32);

    //int gcontext = linalg_base::create_blacs_handler(comm.mpi_comm());

    int gcontext = grid_row.context();

    int M = 4000;
    int N = 375;


    dmatrix<double_complex> A(M, N, grid2d);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++) A.set(j, i, type_wrapper<double_complex>::random());
    }
    DUMP("hash(A): %llX", A.panel().hash());

    dmatrix<double_complex> B(M, N, grid_row);

    linalg<CPU>::gemr2d(M, N, A, 0, 0, B, 0, 0, gcontext);
    A.zero();

    linalg<CPU>::gemr2d(M, N, B, 0, 0, A, 0, 0, gcontext);
    
    comm.barrier();

    DUMP("hash(A): %llX", A.panel().hash());

}

int main(int argn, char** argv)
{
    Platform::initialize(1);
    
    test_gemr2d();

    Platform::finalize();
}
