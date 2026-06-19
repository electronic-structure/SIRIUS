/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>

using namespace sirius;

inline void
memcpy_simple_1(char* dest__, char* src__, size_t n__)
{
    for (size_t i = 0; i < n__; i++) {
        dest__[i] = src__[i];
    }
}

int
test_memop()
{
    size_t n = 2000 * 1000 * 1000;
    std::vector<double> v1(n, 1.0);
    std::vector<double> v2(n, 2.0);

    std::cout << "total size : " << v1.size() * sizeof(double) / 1024 / 1024 / 1024.0 << " GB" << std::endl;

    for (int i = 0; i < 4; i++) {
        std::cout << "pass : " << i << std::endl;
        double t = -omp_get_wtime();
        std::memcpy(&v1[0], &v2[0], n * sizeof(double));
        t += omp_get_wtime();
        std::cout << "memcpy(stdlib) time : " << t << ", bandwidth: " << 2 * n * sizeof(double) / t / (1 << 30)
                  << "GB/s" << std::endl;

        t = -omp_get_wtime();
        memcpy_simple_1((char*)&v1[0], (char*)&v2[0], n * sizeof(double));
        t += omp_get_wtime();
        std::cout << "memcpy(simple) time : " << t << ", bandwidth: " << 2 * n * sizeof(double) / t / (1 << 30)
                  << "GB/s" << std::endl;

        t = -omp_get_wtime();
        std::copy(v2.begin(), v2.end(), v1.begin());
        t += omp_get_wtime();
        std::cout << "std::copy time : " << t << ", bandwidth: " << 2 * n * sizeof(double) / t / (1 << 30) << "GB/s"
                  << std::endl;

        t = -omp_get_wtime();
        std::memset(&v1[0], 0, n * sizeof(double));
        t += omp_get_wtime();
        std::cout << "memset(stdlib) time : " << t << ", bandwidth: " << n * sizeof(double) / t / (1 << 30) << "GB/s"
                  << std::endl;

        t = -omp_get_wtime();
        std::fill(v1.begin(), v1.end(), 0.0);
        t += omp_get_wtime();
        std::cout << "std::fill time : " << t << ", bandwidth: " << n * sizeof(double) / t / (1 << 30) << "GB/s"
                  << std::endl;

        std::cout << std::endl;
    }

    return 0;
}

int
main(int argn, char** argv)
{
    return call_test("test_memop", test_memop);
}
