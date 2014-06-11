#include <sirius.h>

void test_gemm(int M, int N, int K, int transa)
{
    sirius::Timer t("test_gemm"); 
    
    mdarray<double_complex, 2> a, b, c;
    int imax, jmax;
    if (transa == 0)
    {
        imax = N;
        jmax = K;
    }
    else
    {
        imax = K;
        jmax = N;
    }
    a.set_dimensions(imax, jmax);
    b.set_dimensions(K, N);
    c.set_dimensions(M, N);
    #ifdef _GPU_
    a.allocate_page_locked();
    b.allocate_page_locked();
    c.allocate_page_locked();
    #else
    a.allocate();
    b.allocate();
    c.allocate();
    #endif

    for (int j = 0; j < jmax; j++)
    {
        for (int i = 0; i < imax; i++) a(i, j) = type_wrapper<double_complex>::random();
    }

    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < K; i++) b(i, j) = type_wrapper<double_complex>::random();
    }

    c.zero();

    printf("testing serial zgemm with M, N, K = %i, %i, %i, opA = %i\n", M, N, K, transa);
    sirius::Timer t1("gemm_only"); 
    blas<cpu>::gemm(transa, 0, M, N, K, a.ptr(), a.ld(), b.ptr(), b.ld(), c.ptr(), c.ld());
    t1.stop();
    printf("execution time (sec) : %12.6f\n", t1.value());
    printf("performance (GFlops) : %12.6f\n", 8e-9 * M * N * K / t1.value());
}

#ifdef _SCALAPACK_
void test_pgemm(int M, int N, int K, int nrow, int ncol, int transa)
{
    int blacs_handler = linalg<scalapack>::create_blacs_handler(MPI_COMM_WORLD);
    int context = blacs_handler;
    Cblacs_gridinit(&context, "C", nrow, ncol);

    dmatrix<double_complex> a, b, c;
    if (transa == 0)
    {
        a.set_dimensions(M, K, context);
    }
    else
    {
        a.set_dimensions(K, M, context);
    }
    b.set_dimensions(K, N, context);
    c.set_dimensions(M, N, context);
    #ifdef _GPU_
    a.allocate_page_locked();
    b.allocate_page_locked();
    c.allocate_page_locked();
    #else
    a.allocate();
    b.allocate();
    c.allocate();
    #endif

    for (int ic = 0; ic < a.num_cols_local(); ic++)
    {
        for (int ir = 0; ir < a.num_rows_local(); ir++) a(ir, ic) = type_wrapper<double_complex>::random();
    }

    for (int ic = 0; ic < b.num_cols_local(); ic++)
    {
        for (int ir = 0; ir < b.num_rows_local(); ir++) b(ir, ic) = type_wrapper<double_complex>::random();
    }

    c.zero();

    if (Platform::mpi_rank() == 0)
    {
        printf("testing parallel zgemm with M, N, K = %i, %i, %i, opA = %i\n", M, N, K, transa);
        printf("nrow, ncol = %i, %i, bs = %i\n", nrow, ncol, linalg<scalapack>::cyclic_block_size());
    }
    sirius::Timer t1("gemm_only"); 
    blas<cpu>::gemm(transa, 0, M, N, K, complex_one, a, b, complex_zero, c);
    t1.stop();
    if (Platform::mpi_rank() == 0)
    {
        printf("execution time (sec) : %12.6f\n", t1.value());
        printf("performance (GFlops) : %12.6f\n", 8e-9 * M * N * K / t1.value() / nrow / ncol);
    }
    Cblacs_gridexit(context);
    linalg<scalapack>::free_blacs_handler(blacs_handler);
}
#endif

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--M=", "{int} M");
    args.register_key("--N=", "{int} N");
    args.register_key("--K=", "{int} K");
    args.register_key("--opA=", "{0|1|2} 0: op(A) = A, 1: op(A) = A', 2: op(A) = conjg(A')");
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

    int transa = 0;
    if (args.exist("opA")) transa = args.value<int>("opA");

    Platform::initialize(true);

    if (nrow * ncol == 1)
    {
        test_gemm(M, N, K, transa);
    }
    else
    {
        #ifdef _SCALAPACK_
        int bs = args.value<int>("bs");
        linalg<scalapack>::set_cyclic_block_size(bs);
        test_pgemm(M, N, K, nrow, ncol, transa);
        #else
        terminate(__FILE__, __LINE__, "not compiled with ScaLAPACK support");
        #endif
    }

    Platform::finalize();
}
