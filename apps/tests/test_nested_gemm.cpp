/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

using namespace sirius;

void
test_nested_gemm()
{
    int nt = omp_get_max_threads();

    printf("available number of threads: %i\n", nt);

    int m{1000}, n{1000}, k{1000};

    mdarray<double_complex, 2> a(k, m);
    mdarray<double_complex, 2> b(k, n);
    mdarray<double_complex, 2> c(m, n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            a(j, i) = 0.1;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            b(j, i) = 0.1;

    c.zero();

    /* warmup */
    linalg<device_t::CPU>::gemm(2, 0, m, n, k, a, b, c);

    double t = omp_get_wtime();

    linalg<device_t::CPU>::gemm(2, 0, m, n, k, a, b, c);

    double t0   = omp_get_wtime() - t;
    double perf = 8e-9 * m * n * k / t0;

    printf("performance (all threads): %f Gflops\n", perf);

    omp_set_nested(1);

    #pragma omp parallel num_threads(1)
    {
        for (int i = 1; i < nt; i++) {
#ifdef __MKL
            mkl_set_num_threads_local(i);
#endif
            omp_set_num_threads(i);

            double t = omp_get_wtime();
            linalg<device_t::CPU>::gemm(2, 0, m, n, k, a, b, c);
            double perf = 8e-9 * m * n * k / (omp_get_wtime() - t);

            printf("performance (%i threads): %f Gflops\n", i, perf);
        }
    }

    omp_set_nested(0);
#ifdef __MKL
    mkl_set_num_threads_local(0);
#endif
    omp_set_num_threads(nt);
}

int
main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);

    test_nested_gemm();

    sirius::finalize();
}
