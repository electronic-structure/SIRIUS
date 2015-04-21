#include <sirius.h>

double test_gemm(int M, int N, int K)
{
    sirius::Timer t("test_gemm"); 

    Communicator comm(MPI_COMM_WORLD);

    splindex<block> spl_M(M, comm.size(), comm.rank());
    
    matrix<double_complex> a(spl_M.local_size(), K);
    matrix<double_complex> b(K, N);
    matrix<double_complex> c(spl_M.local_size(), N);

    for (int j = 0; j < K; j++)
    {
        for (int i = 0; i < (int)spl_M.local_size(); i++) a(i, j) = type_wrapper<double_complex>::random();
        for (int i = 0; i < N; i++) b(j, i) = type_wrapper<double_complex>::random();
    }

    c.zero();

    sirius::Timer t1("gemm_only"); 
    linalg<CPU>::gemm(0, 0, (int)spl_M.local_size(), N, K, a.at<CPU>(), a.ld(), b.at<CPU>(), b.ld(), c.at<CPU>(), c.ld());
    double tval = t1.stop();
    double perf = 8e-9 * int(spl_M.local_size()) * N * K / tval;

    if (Platform::rank() == 0)
    {
        printf("execution time (sec) : %12.6f\n", tval);
        printf("performance (GFlops / rank) : %12.6f\n", perf);
    }
    return perf;
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--M=", "{int} M");
    args.register_key("--N=", "{int} N");
    args.register_key("--K=", "{int} K");
    args.register_key("--repeat=", "{int} repeat test number of times");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    int M = args.value<int>("M");
    int N = args.value<int>("N");
    int K = args.value<int>("K");

    int repeat = args.value<int>("repeat", 1);

    Platform::initialize(true);

    double perf = 0;
    for (int i = 0; i < repeat; i++) perf += test_gemm(M, N, K);
    if (Platform::rank() == 0)
    {
        printf("\n");
        printf("average performance    : %12.6f GFlops / rank\n", perf / repeat);
    }

    Platform::finalize();
}
