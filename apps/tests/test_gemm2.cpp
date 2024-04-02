/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "sirius.hpp"
#include "testing.hpp"

using namespace sirius;

#ifdef __TEST_REAL
typedef double gemm_type;
int const nop_gemm = 2;
#else
typedef std::complex<double> gemm_type;
int const nop_gemm = 8;
#endif

double
test_gemm(int M, int N, int K, int transa, la::lib_t la__, memory_t memA__, memory_t memB__, memory_t memC__)
{
    mdarray<gemm_type, 2> a, b, c;
    int imax, jmax;
    if (transa == 0) {
        imax = M;
        jmax = K;
    } else {
        imax = K;
        jmax = M;
    }

    a = matrix<gemm_type>({imax, jmax}, memA__);
    b = matrix<gemm_type>({K, N}, memB__);
    c = matrix<gemm_type>({M, N}, memC__);

    if (!is_host_memory(memA__)) {
        a.allocate(memory_t::host);
    }
    a = [](int64_t i, int64_t j) { return random<gemm_type>(); };
    if (!is_host_memory(memA__)) {
        a.copy_to(memory_t::device);
    }

    if (!is_host_memory(memB__)) {
        b.allocate(memory_t::host);
    }
    b = [](int64_t i, int64_t j) { return random<gemm_type>(); };
    if (!is_host_memory(memB__)) {
        b.copy_to(memory_t::device);
    }

    c.zero(memC__);
    if (!is_host_memory(memC__)) {
        c.allocate(memory_t::host);
    }

    char TA[] = {'N', 'T', 'C'};

    printf("testing serial gemm with M, N, K = %i, %i, %i, opA = %i\n", M, N, K, transa);
    printf("a.ld() = %i\n", a.ld());
    printf("b.ld() = %i\n", b.ld());
    printf("c.ld() = %i\n", c.ld());
    double t = -::sirius::wtime();
    la::wrap(la__).gemm(TA[transa], 'N', M, N, K, &la::constant<gemm_type>::one(), a.at(memA__), a.ld(), b.at(memB__),
                        b.ld(), &la::constant<gemm_type>::zero(), c.at(memC__), c.ld());
    double t2 = t + ::sirius::wtime();
    if (is_device_memory(memC__)) {
        c.copy_to(memory_t::host);
    }

    t += ::sirius::wtime();
    double perf = nop_gemm * 1e-9 * M * N * K / t;
    printf("execution time (sec) : %12.6f\n", t);
    printf("performance (GFlops) : %12.6f\n", perf);
    printf("blas time (sec)      : %12.6f\n", t2);

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
    args.register_key("--repeat=", "{int} repeat test number of times");
    args.register_key("--lib_t=", "{string} type of the linear algebra driver");
    args.register_key("--memA=", "{string} type of memory of matrix A");
    args.register_key("--memB=", "{string} type of memory of matrix B");
    args.register_key("--memC=", "{string} type of memory of matrix C");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    int M = args.value<int>("M", 512);
    int N = args.value<int>("N", 512);
    int K = args.value<int>("K", 512);

    int transa = args.value<int>("opA", 0);

    int repeat = args.value<int>("repeat", 5);

    std::string lib_t_str = args.value<std::string>("lib_t", "blas");
    auto memA             = get_memory_t(args.value<std::string>("memA", "host"));
    auto memB             = get_memory_t(args.value<std::string>("memB", "host"));
    auto memC             = get_memory_t(args.value<std::string>("memC", "host"));

    sirius::initialize(true);

    sirius::Measurement perf;
    for (int i = 0; i < repeat; i++) {
        perf.push_back(test_gemm(M, N, K, transa, la::get_lib_t(lib_t_str), memA, memB, memC));
    }
    printf("average performance: %12.6f GFlops, sigma: %12.6f\n", perf.average(), perf.sigma());

    sirius::finalize();
}
