#include "test.hpp"

#ifdef __TEST_REAL
typedef double gemm_type;
int const nop_gemm = 2;
#else
typedef double_complex gemm_type;
int const nop_gemm = 8;
#endif


double test_gemm(int M, int N, int K, int transa, linalg_t la__, memory_t mem__)
{
    utils::timer t("test_gemm"); 

    mdarray<gemm_type, 2> a, b, c;
    int imax, jmax;
    if (transa == 0) {
        imax = M;
        jmax = K;
    } else {
        imax = K;
        jmax = M;
    }
    memory_t mem{memory_t::none};
    if (mem__ == memory_t::device) {
        mem = memory_t::host;
    } else {
        mem = mem__;
    }
    a = matrix<gemm_type>(imax, jmax, mem);
    b = matrix<gemm_type>(K, N, mem);
    c = matrix<gemm_type>(M, N, mem);

    for (int j = 0; j < jmax; j++) {
        for (int i = 0; i < imax; i++) {
            a(i, j) = utils::random<gemm_type>();
        }
    }

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < K; i++) {
            b(i, j) = utils::random<gemm_type>();
        }
    }
    c.zero();

    if (mem__ == memory_t::device) {
    }

    printf("testing serial gemm with M, N, K = %i, %i, %i, opA = %i\n", M, N, K, transa);
    printf("a.ld() = %i\n", a.ld());
    printf("b.ld() = %i\n", b.ld());
    printf("c.ld() = %i\n", c.ld());
    utils::timer t1("gemm_only");
    switch (mem__) {
        case memory_t::host:
        case memory_t::host_pinned: {
            experimental::linalg2(la__).gemm(transa, 0, M, N, K, &linalg_const<gemm_type>::one(), a.at<CPU>(), a.ld(), 
                                             b.at<CPU>(), b.ld(), &linalg_const<gemm_type>::zero(), c.at<CPU>(), c.ld());
            break;
        }
        case memory_t::device: {
            a.allocate(memory_t::device);
            b.allocate(memory_t::device);
            c.allocate(memory_t::device);
            a.copy<memory_t::host, memory_t::device>();
            b.copy<memory_t::host, memory_t::device>();
            experimental::linalg2(la__).gemm(transa, 0, M, N, K, &linalg_const<gemm_type>::one(), a.at<GPU>(), a.ld(), 
                                             b.at<GPU>(), b.ld(), &linalg_const<gemm_type>::zero(), c.at<GPU>(), c.ld());
            c.copy<memory_t::device, memory_t::host>();
            break;
        }
        default: {
            throw std::runtime_error("wrong memory type");
        }
    }

    double tval = t1.stop();
    double perf = nop_gemm * 1e-9 * M * N * K / tval;
    printf("execution time (sec) : %12.6f\n", tval);
    printf("performance (GFlops) : %12.6f\n", perf);

    return perf;
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--M=", "{int} M");
    args.register_key("--N=", "{int} N");
    args.register_key("--K=", "{int} K");
    args.register_key("--opA=", "{0|1|2} 0: op(A) = A, 1: op(A) = A', 2: op(A) = conjg(A')");
    args.register_key("--repeat=", "{int} repeat test number of times");
    args.register_key("--linalg_t=", "{string} type of the linear algebra driver");
    args.register_key("--memory_t=", "{string} type of the memory");

    args.parse_args(argn, argv);
    if (argn == 1) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    int M = args.value<int>("M");
    int N = args.value<int>("N");
    int K = args.value<int>("K");

    int transa = args.value<int>("opA", 0);

    int repeat = args.value<int>("repeat", 5);

    std::string linalg_t_str = args.value<std::string>("linalg_t", "blas");
    std::string memory_t_str = args.value<std::string>("memory_t", "host");

    sirius::initialize(true);

    Measurement perf;
    for (int i = 0; i < repeat; i++) {
        perf.push_back(test_gemm(M, N, K, transa, get_linalg_t(linalg_t_str), get_memory_t(memory_t_str)));
    }
    printf("average performance: %12.6f GFlops, sigma: %12.6f\n", perf.average(), perf.sigma());

    sirius::finalize();
}
