#include <sirius.h>

using namespace sirius;

void test_gemr2d()
{
    Communicator comm(MPI_COMM_WORLD);

    BLACS_grid grid_col(comm, 1, comm.size());
    BLACS_grid grid_row(comm, comm.size(), 1);

    int gcontext = grid_row.context();

    int M = 80000;
    int N = 375;

    dmatrix<double_complex> A(M, N, grid_row, splindex_base::block_size(M, comm.size()), 1);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++) A.set(j, i, type_wrapper<double_complex>::random());
    }
    auto h = A.panel().hash();


    dmatrix<double_complex> B(M, N - 1, grid_col, 1, 1);
    B.zero();
    
    double t0 = -Utils::current_time();
    linalg<CPU>::gemr2d(M, N - 1, A, 0, 1, B, 0, 0, gcontext);
    linalg<CPU>::gemr2d(M, N - 1, B, 0, 0, A, 0, 1, gcontext);
    t0 += Utils::current_time();

    if (comm.rank() == 0)
    {
        printf("done in %.4f sec, swap speed: %.4f GB/sec\n", t0, sizeof(double_complex) * 2 * M * N / double(1 << 30) / t0);
    }
    
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
