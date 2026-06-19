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

/* use std::vector */
template <typename T>
int
test_alm_copy()
{
    size_t m = 200000;
    size_t n = 300;
    size_t k = 64;
    std::vector<T> v1(m * n * k, 1.0);
    std::vector<T> v2(v1.size(), 2.0);

    for (int repeat = 0; repeat < 4; repeat++) {
        double t = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp for schedule(static, 1)
            for (int i = 0; i < k; i++) {
                auto ptr_in  = &v1[m * n * i];
                auto ptr_out = &v2[m * n * i];
                std::copy(ptr_in, ptr_in + m * n, ptr_out);
            }
        }
        t          = omp_get_wtime() - t;
        auto bytes = 2.0 * v1.size() * sizeof(T);
        std::cout << "repeat : " << repeat << ", time : " << t << ", effective BW : " << bytes / t / (1 << 30)
                  << " GB/s" << std::endl;
    }
    return 0;
}

/* use host-pinned memory */
template <typename T>
int
test_alm_copy_v2()
{
    size_t m = 200000; // number of G+k vectors
    size_t n = 300;    // number of AW coefficients per atom
    size_t k = 64;     // number of atoms in a block

    auto& mp = get_memory_pool(memory_t::host_pinned);

    mdarray<T, 2> alm_all({m, n * k}, mp, mdarray_label("alm_all"));
    mdarray<T, 2> alm_blk({m, n * k}, mp, mdarray_label("alm_blk"));

    #pragma omp parallel for
    for (size_t i = 0; i < alm_all.size(); i++) {
        alm_all[i] = 1;
        alm_blk[i] = 0;
    }

    for (int repeat = 0; repeat < 4; repeat++) {
        double t = omp_get_wtime();

        #pragma omp parallel
        {
            #pragma omp for schedule(static, 1)
            for (int ia = 0; ia < static_cast<int>(k); ia++) {
                auto ptr_in  = alm_all.at(memory_t::host, 0, n * ia);
                auto ptr_out = alm_blk.at(memory_t::host, 0, n * ia);

                std::copy(ptr_in, ptr_in + m * n, ptr_out);
            }
        }

        t = omp_get_wtime() - t;

        auto bytes = 2.0 * alm_all.size() * sizeof(T);
        std::cout << "copy repeat : " << repeat << ", time : " << t << ", effective BW : " << bytes / t / (1 << 30)
                  << " GB/s" << std::endl;
    }

    for (int repeat = 0; repeat < 4; repeat++) {
        double t = omp_get_wtime();

        #pragma omp parallel
        {
            #pragma omp for schedule(static, 1)
            for (int ia = 0; ia < static_cast<int>(k); ia++) {
                auto ptr_in  = alm_all.at(memory_t::host, 0, n * ia);
                auto ptr_out = alm_blk.at(memory_t::host, 0, n * ia);

                for (size_t j = 0; j < m * n; j++) {
                    ptr_out[j] = conj(ptr_in[j]);
                }
            }
        }

        t = omp_get_wtime() - t;

        auto bytes = 2.0 * alm_all.size() * sizeof(std::complex<double>);
        std::cout << "copy+conj repeat : " << repeat << ", effective BW : " << bytes / t / (1 << 30) << " GB/s"
                  << std::endl;
    }

    return 0;
}

int
main(int argn, char** argv)
{
    call_test("test_alm_copy v1 double", test_alm_copy<double>);
    call_test("test_alm_copy v1 double_complex", test_alm_copy<std::complex<double>>);
    call_test("test_alm_copy v2 double", test_alm_copy_v2<double>);
    call_test("test_alm_copy v2 double_complex", test_alm_copy_v2<std::complex<double>>);
    return 0;
}
