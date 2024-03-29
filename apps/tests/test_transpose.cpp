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
test_transpose(int M__, int N__)
{
    matrix<double_complex> a(M__, N__);

    for (int i = 0; i < N__; i++) {
        for (int j = 0; j < M__; j++)
            a(j, i) = type_wrapper<double_complex>::random();
    }

    matrix<double_complex> b(N__, M__);
    b.zero();

    double t = -omp_get_wtime();
    #pragma omp parallel for
    for (int j = 0; j < M__; j++)
        for (int i = 0; i < N__; i++)
            b(i, j) = a(j, i);
    t += omp_get_wtime();

    printf("bandwidth: %f GB/s\n", 2.0 * M__ * N__ * sizeof(double_complex) / (1 << 30) / t);
}

void
test_transpose_v2(int M__, int N__)
{
    matrix<double_complex> a(M__, N__);

    for (int i = 0; i < N__; i++) {
        for (int j = 0; j < M__; j++)
            a(j, i) = type_wrapper<double_complex>::random();
    }

    matrix<double_complex> b(N__, M__);
    b.zero();

    double t = -omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < N__; i++)
        for (int j = 0; j < M__; j++)
            b(i, j) = a(j, i);
    t += omp_get_wtime();

    printf("bandwidth: %f GB/s\n", 2.0 * M__ * N__ * sizeof(double_complex) / (1 << 30) / t);
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--M=", "{int} leading dimension of original matrix");
    args.register_key("--N=", "{int} leading dimension of transposed matrix");

    args.parse_args(argn, argv);
    if (argn == 1) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    int M = args.value<int>("M");
    int N = args.value<int>("N");

    sirius::initialize(1);
    for (int i = 0; i < 10; i++)
        test_transpose(M, N);
    printf("\n");
    for (int i = 0; i < 10; i++)
        test_transpose_v2(M, N);
    sirius::finalize();
}
