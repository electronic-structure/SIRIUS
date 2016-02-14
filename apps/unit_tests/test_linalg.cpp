#include <sirius.h>

using namespace sirius;

void test1()
{
    int N = 400;
    matrix<double_complex> A(N, N);
    matrix<double_complex> B(N, N);
    matrix<double_complex> C(N, N);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++) B(j, i) = type_wrapper<double_complex>::random(); 
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++) A(i, j) = B(i, j) + std::conj(B(j, i));
    }
    A >> B;

    linalg<CPU>::heinv(N, A);
    linalg<CPU>::hemm(0, 0, N, N, double_complex(1, 0), A, B, double_complex(0, 0), C);

    int err = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double_complex z = C(i, j);
            if (i == j) z -= 1.0;
            if (std::abs(z) > 1e-10) err++;
        }
    }

    linalg<CPU>::hemm(1, 0, N, N, double_complex(1, 0), A, B, double_complex(0, 0), C);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double_complex z = C(i, j);
            if (i == j) z -= 1.0;
            if (std::abs(z) > 1e-10) err++;
        }
    }

    if (err)
    {
        printf("test1 failed!\n");
        exit(1);
    }
    else
    {
        printf("test1 passed!\n");
    }
}

template <typename T>
void test2()
{
    int N = 400;
    matrix<T> A(N, N);
    matrix<T> B(N, N);
    matrix<T> C(N, N);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++) A(j, i) = type_wrapper<T>::random(); 
    }
    A >> B;

    linalg<CPU>::geinv(N, A);
    linalg<CPU>::gemm(0, 0, N, N, N, A, B, C);

    int err = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            T c = C(i, j);
            if (i == j) c -= 1.0;
            if (std::abs(c) > 1e-10) err++;
        }
    }

    linalg<CPU>::gemm(0, 0, N, N, N, A, B, C);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            T c = C(i, j);
            if (i == j) c -= 1.0;
            if (std::abs(c) > 1e-10) err++;
        }
    }

    if (err)
    {
        printf("test2 failed!\n");
        exit(1);
    }
    else
    {
        printf("test2 passed!\n");
    }
}
#ifdef __SCALAPACK
template <typename T>
void test3()
{
    int bs = 32;

    int num_ranks = mpi_comm_world().size();
    int nrc = (int)std::sqrt(0.1 + num_ranks);
    if (nrc * nrc != num_ranks)
    {
        printf("wrong mpi grid\n");
        exit(-1);
    }

    int N = 400;
    BLACS_grid blacs_grid(mpi_comm_world(), nrc, nrc);

    dmatrix<T> A(N, N, blacs_grid, bs, bs);
    dmatrix<T> B(N, N, blacs_grid, bs, bs);
    dmatrix<T> C(N, N, blacs_grid, bs, bs);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++) A.set(j, i, type_wrapper<T>::random());
    }
    A.panel() >> B.panel();

    T alpha = 1.0;
    T beta = 0.0;

    linalg<CPU>::geinv(N, A);
    
    linalg<CPU>::gemm(0, 0, N, N, N, alpha, A, B, beta, C);

    int err = 0;
    for (int i = 0; i < C.num_cols_local(); i++)
    {
        for (int j = 0; j < C.num_rows_local(); j++)
        {
            T c = C(j, i);
            if (C.icol(i) == C.irow(j)) c -= 1.0;
            if (std::abs(c) > 1e-10) err++;
        }
    }
    
    linalg<CPU>::gemm(0, 0, N, N, N, alpha, B, A, beta, C);

    for (int i = 0; i < C.num_cols_local(); i++)
    {
        for (int j = 0; j < C.num_rows_local(); j++)
        {
            T c = C(j, i);
            if (C.icol(i) == C.irow(j)) c -= 1.0;
            if (std::abs(c) > 1e-10) err++;
        }
    }

    if (err)
    {
        printf("test3 failed!\n");
        exit(1);
    }
    else
    {
        printf("test3 passed!\n");
    }
}
#endif

int main(int argn, char** argv)
{
    sirius::initialize(1);
    test1();
    test2<double>();
    test2<double_complex>();
    #ifdef __SCALAPACK
    test3<double_complex>();
    #endif
    sirius::finalize();
    return 0;
}
