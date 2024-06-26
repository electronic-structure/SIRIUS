/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <random>
#include <core/cmd_args.hpp>
#include <core/memory.hpp>
#include <testing.hpp>

using double_complex = std::complex<double>;
using namespace sirius;

void
test1()
{
    memory_pool& mp = get_memory_pool(memory_t::host);
    auto ptr        = mp.allocate<double_complex>(1024);
    mp.free(ptr);
}

void
test2()
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
test4()
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

double
test_alloc(size_t n)
{
    auto t0 = time_now();
    /* time to allocate + fill */
    char* ptr = (char*)std::malloc(n);
    std::fill(ptr, ptr + n, 0);
    auto t1 = time_now();
    /* time fo fill */
    std::fill(ptr, ptr + n, 0);
    auto t2 = time_now();
    /* harmless: prevent full optimization of the code with GCC */
    double dt = ptr[0];
    std::free(ptr);
    auto t3 = time_now();

    dt += time_interval(t0, t3) - 2 * time_interval(t1, t2);
    return dt;
}

double
test_alloc(size_t n, memory_pool& mp)
{
    auto t0 = time_now();
    /* time to allocate + fill */
    char* ptr = mp.allocate<char>(n);
    std::fill(ptr, ptr + n, 0);
    auto t1 = time_now();
    /* time fo fill */
    std::fill(ptr, ptr + n, 0);
    auto t2 = time_now();
    /* harmless: prevent full optimization of the code with GCC */
    double dt = ptr[0];
    mp.free(ptr);
    auto t3 = time_now();

    dt += time_interval(t0, t3) - 2 * time_interval(t1, t2);
    return dt;
}

void
test5()
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
test6()
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
    }
}

double
test_alloc_array(size_t n)
{
    auto t0 = time_now();
    /* time to allocate + fill */
    mdarray<char, 1> p({n});
    p.zero();
    auto t1 = time_now();
    /* time fo fill */
    p.zero();
    auto t2 = time_now();
    /* harmless: prevent full optimization of the code with GCC */
    double dt = p[0];
    p.deallocate(memory_t::host);
    auto t3 = time_now();

    dt += time_interval(t0, t3) - 2 * time_interval(t1, t2);
    return dt;
}

double
test_alloc_array(size_t n, memory_pool& mp)
{
    auto t0 = time_now();
    /* time to allocate + fill */
    mdarray<char, 1> p({n}, mp);
    p.zero();
    auto t1 = time_now();
    /* time fo fill */
    p.zero();
    auto t2 = time_now();
    /* harmless: prevent full optimization of the code with GCC */
    double dt = p[0];
    p.deallocate(memory_t::host);
    auto t3 = time_now();

    dt += time_interval(t0, t3) - 2 * time_interval(t1, t2);
    return dt;
}

void
test8()
{
    double t0{0};
    for (int k = 0; k < 500; k++) {
        for (int i = 2; i < 1024; i++) {
            size_t sz = i;
            t0 += test_alloc_array(sz);
        }
    }
    memory_pool& mp = get_memory_pool(memory_t::host);
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
test_mempool()
{
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
    test7();
    test8();
    return 0;
}

int
main(int argn, char** argv)
{
    return call_test(argv[0], test_mempool);
}
