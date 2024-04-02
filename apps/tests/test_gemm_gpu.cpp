/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "test.hpp"

#ifdef __TEST_REAL
typedef double gemm_type;
int const nop_gemm = 2;
#else
typedef double_complex gemm_type;
int const nop_gemm = 8;
#endif

double
test_gemm(int M, int N, int K, int transa)
{
    runtime::Timer t("test_gemm");

    matrix<gemm_type> a, b, c;
    int imax, jmax;
    if (transa == 0) {
        imax = M;
        jmax = K;
    } else {
        imax = K;
        jmax = M;
    }
    a = matrix<gemm_type>(imax, jmax);
    b = matrix<gemm_type>(K, N);
    c = matrix<gemm_type>(M, N);

    for (int j = 0; j < jmax; j++) {
        for (int i = 0; i < imax; i++) {
            a(i, j) = 1.0;
        }
    }

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < K; i++) {
            b(i, j) = 1.0;
        }
    }

    a.allocate(memory_t::device);
    a.copy_to_device();
    b.allocate(memory_t::device);
    b.copy_to_device();

    c.allocate(memory_t::device);

    printf("testing serial gemm with M, N, K = %i, %i, %i, opA = %i\n", M, N, K, transa);
    printf("a.ld() = %i\n", a.ld());
    printf("b.ld() = %i\n", b.ld());
    printf("c.ld() = %i\n", c.ld());
    runtime::Timer t1("gemm_only");
    linalg<GPU>::gemm(transa, 0, M, N, K, a.at<GPU>(), a.ld(), b.at<GPU>(), b.ld(), c.at<GPU>(), c.ld());
    c.copy_to_host();
    double tval = t1.stop();
    double perf = nop_gemm * 1e-9 * M * N * K / tval;
    printf("execution time (sec) : %12.6f\n", tval);
    printf("performance (GFlops) : %12.6f\n", perf);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (std::abs(c(j, i) - double(K)) > 1e-12) {
                RTE_THROW("wrong result");
            }
        }
    }

    return perf;
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--M=", "{int} M");
    args.register_key("--N=", "{int} N");
    args.register_key("--K=", "{int} K");
    args.register_key("--opA=", "{0|1|2} 0: op(A) = A, 1: op(A) = A', 2: op(A) = conjg(A')");
    args.register_key("--n=", "{int} skip first n elements in N index");
    args.register_key("--repeat=", "{int} repeat test number of times");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    int M = args.value<int>("M");
    int N = args.value<int>("N");
    int K = args.value<int>("K");

    int transa = args.value<int>("opA", 0);

    int repeat = args.value<int>("repeat", 5);

    sirius::initialize(true);

    Measurement perf;
    for (int i = 0; i < repeat; i++) {
        perf.push_back(test_gemm(M, N, K, transa));
    }
    printf("average performance: %12.6f GFlops / rank,  sigma: %12.6f\n", perf.average(), perf.sigma());

    sirius::finalize();
}
