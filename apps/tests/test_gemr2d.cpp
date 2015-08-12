#include <sirius.h>

using namespace sirius;

void test_gemr2d()
{
    Communicator comm(MPI_COMM_WORLD);

    BLACS_grid grid_col(comm, 1, comm.size());
    BLACS_grid grid_row(comm, comm.size(), 1);

    int gcontext = grid_row.context();

    int M = 4000;
    int N = 375;

    dmatrix<double_complex> A(M, N, grid_col, 1, 1);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++) A.set(j, i, type_wrapper<double_complex>::random());
    }
    auto h = A.panel().hash();


    dmatrix<double_complex> B(M, N - 1, grid_row, 32, 1);

    linalg<CPU>::gemr2d(M, N - 1, A, 0, 1, B, 0, 0, gcontext);
    //A.zero();

    linalg<CPU>::gemr2d(M, N - 1, B, 0, 0, A, 0, 1, gcontext);
    
    if (A.panel().hash() != h)
    {
        TERMINATE("wrong hash");
    }

}

int main(int argn, char** argv)
{
    Platform::initialize(1);
    
    test_gemr2d();

    Platform::finalize();
}
