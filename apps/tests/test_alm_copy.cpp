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

int
test_alm_copy()
{
    size_t m = 200000;
    size_t n = 300;
    size_t k = 20;
    std::vector<std::complex<double>> v1(m * n * k, 1.0);
    std::vector<std::complex<double>> v2(v1.size(), 2.0);

    for (int repeat = 0; repeat < 4; repeat++) {
        double t = omp_get_wtime();
        #pragma omp parallel
        {
            //int tid = omp_get_thread_num();
            #pragma omp for schedule(static, 1)
            for (int i = 0; i < k; i++) {
                auto ptr_in = &v1[m * n * i];
                auto ptr_out = &v2[m * n * i];
                std::copy(ptr_in, ptr_in + m * n, ptr_out);
            }
        }
        t = (omp_get_wtime() - t);
        std::cout << "repeat : " << repeat << ", effective BW : " << 2 * v1.size() * sizeof(std::complex<double>) / t / (1 << 30) << " GB/s" << std::endl;
    }
    return 0;
}

int
main(int argn, char** argv)
{
    return call_test("test_alm_copy", test_alm_copy);
}
