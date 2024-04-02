/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <core/cmd_args.hpp>
#include <core/memory.hpp>
#include <complex>
#include <sys/time.h>
#include <random>

using double_complex = std::complex<double>;
using namespace sirius;

void
test2()
{
    memory_pool& mp = get_memory_pool(memory_t::host);
    auto ptr        = mp.allocate<double_complex>(1024);
    mp.free(ptr);
}

void
test2a()
{
    memory_pool& mp = get_memory_pool(memory_t::host);
    auto ptr        = mp.allocate<double_complex>(1024);
    mp.free(ptr);
    ptr = mp.allocate<double_complex>(512);
    mp.free(ptr);
}

void
test3()
{
    memory_pool& mp = get_memory_pool(memory_t::host);
    auto p1         = mp.allocate<double_complex>(1024);
    auto p2         = mp.allocate<double_complex>(2024);
    auto p3         = mp.allocate<double_complex>(3024);
    mp.free(p1);
    mp.free(p2);
    mp.free(p3);
}

void
test5()
{
    memory_pool& mp = get_memory_pool(memory_t::host);

    for (int k = 0; k < 2; k++) {
        std::vector<double*> vp;
        for (size_t i = 1; i < 20; i++) {
            size_t sz   = 1 << i;
            double* ptr = mp.allocate<double>(sz);
            ptr[0]      = 0;
            ptr[sz - 1] = 0;
            vp.push_back(ptr);
        }
        for (auto& e : vp) {
            mp.free(e);
        }
    }
}

/// Wall-clock time in seconds.
inline double
wtime()
{
    timeval t;
    gettimeofday(&t, NULL);
    return double(t.tv_sec) + double(t.tv_usec) / 1e6;
}

double
test_alloc(size_t n)
{
    double t0 = wtime();
    /* time to allocate + fill */
    char* ptr = (char*)std::malloc(n);
    std::fill(ptr, ptr + n, 0);
    double t1 = wtime();
    /* time fo fill */
    std::fill(ptr, ptr + n, 0);
    double t2 = wtime();
    /* harmless: add zero to t0 to prevent full optimization of the code with GCC */
    t0 += ptr[0];
    std::free(ptr);
    double t3 = wtime();

    // return (t1 - t0) - (t2 - t1);
    return (t3 - t0) - 2 * (t2 - t1);
}

double
test_alloc(size_t n, memory_pool& mp)
{
    double t0 = wtime();
    /* time to allocate + fill */
    char* ptr = mp.allocate<char>(n);
    std::fill(ptr, ptr + n, 0);
    double t1 = wtime();
    /* time fo fill */
    std::fill(ptr, ptr + n, 0);
    double t2 = wtime();
    /* harmless: add zero to t0 to prevent full optimization of the code with GCC */
    t0 += ptr[0];
    mp.free(ptr);
    double t3 = wtime();

    // return (t1 - t0) - (t2 - t1);
    return (t3 - t0) - 2 * (t2 - t1);
}

void
test6()
{
    double t0{0};
    for (int k = 0; k < 8; k++) {
        for (int i = 10; i < 30; i++) {
            size_t sz = size_t(1) << i;
            t0 += test_alloc(sz);
        }
    }
    memory_pool& mp = get_memory_pool(memory_t::host);
    double t1{0};
    for (int k = 0; k < 8; k++) {
        for (int i = 10; i < 30; i++) {
            size_t sz = size_t(1) << i;
            t1 += test_alloc(sz, mp);
        }
    }
    std::cout << "std::malloc time: " << t0 << ", memory_pool time: " << t1 << "\n";
}

void
test6a()
{
    double t0{0};
    for (int k = 0; k < 500; k++) {
        for (int i = 2; i < 1024; i++) {
            size_t sz = i;
            t0 += test_alloc(sz);
        }
    }
    memory_pool& mp = get_memory_pool(memory_t::host);
    double t1{0};
    for (int k = 0; k < 500; k++) {
        for (int i = 2; i < 1024; i++) {
            size_t sz = i;
            t1 += test_alloc(sz, mp);
        }
    }
    std::cout << "std::malloc time: " << t0 << ", memory_pool time: " << t1 << "\n";
}

void
test7()
{
    memory_pool& mp = get_memory_pool(memory_t::host);

    int N = 10000;
    std::vector<double*> v(N);
    for (int k = 0; k < 30; k++) {
        for (int i = 0; i < N; i++) {
            auto n = (rand() & 0b1111111111) + 1;
            v[i]   = mp.allocate<double>(n);
        }
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(v.begin(), v.end(), g);
        for (int i = 0; i < N; i++) {
            mp.free(v[i]);
        }
        if (mp.free_size() != mp.total_size()) {
            throw std::runtime_error("wrong free size");
        }
        // if (mp.num_blocks() != 1) {
        //     throw std::runtime_error("wrong number of blocks");
        // }
    }
}

// void test8()
//{
//     memory_pool mp(memory_t::host);
//     mdarray<double_complex, 2> aa(mp, 100, 100);
//     aa.deallocate(memory_t::host);
//     //memory_pool::unique_ptr<double> up;
//     //up = mp.get_unique_ptr<double>(100);
//     //up.reset(nullptr);
//
//     if (mp.free_size() != mp.total_size()) {
//         throw std::runtime_error("wrong free size");
//     }
//     if (mp.num_blocks() != 1) {
//         throw std::runtime_error("wrong number of blocks");
//     }
//     if (mp.num_stored_ptr() != 0) {
//         throw std::runtime_error("wrong number of stored pointers");
//     }
//
// }

double
test_alloc_array(size_t n)
{
    double t0 = wtime();
    /* time to allocate + fill */
    mdarray<char, 1> p({n});
    p.zero();
    double t1 = wtime();
    /* time fo fill */
    p.zero();
    double t2 = wtime();
    /* harmless: add zero to t0 to prevent full optimization of the code with GCC */
    t0 += p[0];
    p.deallocate(memory_t::host);
    double t3 = wtime();

    // return (t1 - t0) - (t2 - t1);
    return (t3 - t0) - 2 * (t2 - t1);
}

double
test_alloc_array(size_t n, memory_pool& mp)
{
    double t0 = wtime();
    /* time to allocate + fill */
    mdarray<char, 1> p({n}, mp);
    p.zero();
    double t1 = wtime();
    /* time fo fill */
    p.zero();
    double t2 = wtime();
    /* harmless: add zero to t0 to prevent full optimization of the code with GCC */
    t0 += p[0];
    p.deallocate(memory_t::host);
    double t3 = wtime();

    // return (t1 - t0) - (t2 - t1);
    return (t3 - t0) - 2 * (t2 - t1);
}

void
test9()
{
    double t0{0};
    for (int k = 0; k < 500; k++) {
        for (int i = 2; i < 1024; i++) {
            size_t sz = i;
            t0 += test_alloc_array(sz);
        }
    }
    memory_pool mp(memory_t::host);
    double t1{0};
    for (int k = 0; k < 500; k++) {
        for (int i = 2; i < 1024; i++) {
            size_t sz = i;
            t1 += test_alloc_array(sz, mp);
        }
    }
    std::cout << "std::malloc time: " << t0 << ", memory_pool time: " << t1 << "\n";
}

int
run_test()
{
    test2();
    test2a();
    test3();
    test5();
    // test6();
    // test6a();
    test7();
    // test8();
    // test9();
    return 0;
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

    printf("%-30s", "testing memory pool: ");
    int result = run_test();
    if (result) {
        printf("\x1b[31m"
               "Failed"
               "\x1b[0m"
               "\n");
    } else {
        printf("\x1b[32m"
               "OK"
               "\x1b[0m"
               "\n");
    }

    return 0;
}
