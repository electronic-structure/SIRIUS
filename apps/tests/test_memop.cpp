/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>

using namespace sirius;

inline void
memcpy_simple_1(char* dest__, char* src__, size_t n__)
{
    for (size_t i = 0; i < n__; i++) {
        dest__[i] = src__[i];
    }
}

int
main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    int n = 20000000;
    std::vector<double> v1(n, 1.0);
    std::vector<double> v2(n, 2.0);

    double t = -omp_get_wtime();
    std::memcpy(&v1[0], &v2[0], n * sizeof(double));
    t += omp_get_wtime();
    printf("memcpy(stdlib) bandwidth: %f GB/s \n", double(2 * n * sizeof(double)) / t / (1 << 30));

    t = -omp_get_wtime();
    memcpy_simple_1((char*)&v1[0], (char*)&v2[0], n * sizeof(double));
    t += omp_get_wtime();
    printf("memcpy(simple) bandwidth: %f GB/s \n", double(2 * n * sizeof(double)) / t / (1 << 30));

    t = -omp_get_wtime();
    std::copy(v2.begin(), v2.end(), v1.begin());
    t += omp_get_wtime();
    printf("std::copy bandwidth     : %f GB/s \n", double(2 * n * sizeof(double)) / t / (1 << 30));

    t = -omp_get_wtime();
    std::memset(&v1[0], 0, n * sizeof(double));
    t += omp_get_wtime();
    printf("memset(stdlib) bandwidth: %f GB/s \n", double(n * sizeof(double)) / t / (1 << 30));

    t = -omp_get_wtime();
    std::fill(v1.begin(), v1.end(), 0.0);
    t += omp_get_wtime();
    printf("std::fill bandwidth     : %f GB/s \n", double(n * sizeof(double)) / t / (1 << 30));
}
