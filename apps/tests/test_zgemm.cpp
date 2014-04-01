#include <sirius.h>

void test_gemm(int M, int N, int K, mdarray<double_complex, 2>& c)
{
    sirius::Timer t("test_gemm"); 
    
    mdarray<double_complex, 2> a(M, K);
    mdarray<double_complex, 2> b(K, N);

    for (int j = 0; j < K; j++)
    {
        for (int i = 0; i < N; i++) a(i, j) = type_wrapper<double_complex>::random();
    }
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < K; i++) b(i, j) = type_wrapper<double_complex>::random();
    }

    printf("testing serial zgemm with M, N, K = %i, %i, %i\n", M, N, K);
    sirius::Timer t1("gemm_only"); 
    blas<cpu>::gemm(0, 0, M, N, K, a.ptr(), a.ld(), b.ptr(), b.ld(), c.ptr(), c.ld());
    t1.stop();
    printf("execution time (sec) : %12.6f\n", t1.value());
    printf("performance (GFlops) : %12.6f\n", 8e-9 * M * N * K / t1.value());
}

#ifdef _SCALAPACK_
void test_pgemm(int M, int N, int K, int nrow, int ncol)
{
    int context = linalg<scalapack>::create_blacs_context(MPI_COMM_WORLD);
    Cblacs_gridinit(&context, "C", nrow, ncol);

    //== dmatrix<double_complex> a(M, K, context);
    //== dmatrix<double_complex> b(K, N, context);
    //== dmatrix<double_complex> c(M, N, context);
    //== c.zero();

    //== for (int ic = 0; ic < a.num_cols_local(); ic++)
    //== {
    //==     for (int ir = 0; ir < a.num_rows_local(); ir++) a(ir, ic) = type_wrapper<double_complex>::random();
    //== }
    //== for (int ic = 0; ic < b.num_cols_local(); ic++)
    //== {
    //==     for (int ir = 0; ir < b.num_rows_local(); ir++) b(ir, ic) = type_wrapper<double_complex>::random();
    //== }

    //== if (Platform::mpi_rank() == 0)
    //== {
    //==     printf("testing parallel zgemm with M, N, K = %i, %i, %i\n", M, N, K);
    //== }
    //== sirius::Timer t1("gemm_only"); 
    //== blas<cpu>::gemm(0, 0, M, N, K, complex_one, a, b, complex_zero, c);
    //== t1.stop();
    //== if (Platform::mpi_rank() == 0)
    //== {
    //==     printf("execution time (sec) : %12.6f\n", t1.value());
    //==     printf("performance (GFlops) : %12.6f\n", 8e-9 * M * N * K / t1.value() / nrow / ncol);
    //== }

    dmatrix<double_complex> a(K, M, context);
    dmatrix<double_complex> b(K, N, context);
    dmatrix<double_complex> c(M, N, context);
    c.zero();

    for (int ic = 0; ic < a.num_cols_local(); ic++)
    {
        for (int ir = 0; ir < a.num_rows_local(); ir++) a(ir, ic) = type_wrapper<double_complex>::random();
    }
    for (int ic = 0; ic < b.num_cols_local(); ic++)
    {
        for (int ir = 0; ir < b.num_rows_local(); ir++) b(ir, ic) = type_wrapper<double_complex>::random();
    }

    if (Platform::mpi_rank() == 0)
    {
        printf("testing parallel zgemm with M, N, K = %i, %i, %i\n", M, N, K);
    }
    sirius::Timer t1("gemm_only"); 
    blas<cpu>::gemm(2, 0, M, N, K, complex_one, a, b, complex_zero, c);
    t1.stop();
    if (Platform::mpi_rank() == 0)
    {
        printf("execution time (sec) : %12.6f\n", t1.value());
        printf("performance (GFlops) : %12.6f\n", 8e-9 * M * N * K / t1.value() / nrow / ncol);
    }

    linalg<scalapack>::free_blacs_context(context);
}
#endif

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--M=", "{int} M");
    args.register_key("--N=", "{int} N");
    args.register_key("--K=", "{int} K");
    args.register_key("--nrow=", "{int} number of row MPI ranks");
    args.register_key("--ncol=", "{int} number of column MPI ranks");
    args.register_key("--bs=", "{int} cyclic block size");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    int nrow = 1;
    int ncol = 1;

    if (args.exist("nrow")) nrow = args.value<int>("nrow");
    if (args.exist("ncol")) ncol = args.value<int>("ncol");

    int M = args.value<int>("M");
    int N = args.value<int>("N");
    int K = args.value<int>("K");

    Platform::initialize(true);

    if (nrow * ncol == 1)
    {
        mdarray<double_complex, 2> c(M, N);
        for (int j = 0; j < N; j++)
        {
            for (int i = 0; i < M; i++) c(i, j) = complex_zero;
        }
        test_gemm(M, N, K, c);
    }
    else
    {
        #ifdef _SCALAPACK_
        int bs = args.value<int>("bs");
        linalg<scalapack>::set_cyclic_block_size(bs);
        splindex<block_cyclic>::set_cyclic_block_size(bs);
        test_pgemm(M, N, K, nrow, ncol);
        #else
        terminate(__FILE__, __LINE__, "not compiled with ScaLAPACK support");
        #endif
    }

    Platform::finalize();
}
