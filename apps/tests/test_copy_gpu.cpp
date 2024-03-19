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
test1()
{
    printf("\n");
    printf("testing bandwidth\n");
    size_t free_mem = cuda_get_free_mem();
    size_t n        = (free_mem - (1 << 26)) / 16;
    printf("array size: %lu\n", n);

    int N = 20;

    mdarray<double_complex, 1> v(nullptr, n);
    v.allocate(0);
    v.randomize();
    v.allocate_on_device();

    Timer t("copy_to_device");
    for (int i = 0; i < N; i++) {
        v.copy_to_device();
    }
    double tval = t.stop();
    printf("time: %12.6f sec., estimated bandwidth: %12.6f GB/sec.\n", tval,
           static_cast<double>(n * N) / tval / (1 << 26));
}

void
cpu_task(mdarray<double_complex, 1>& v)
{
    Timer t("cpu_task");
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < v.size(); i++)
        v(i) = std::exp(v(i));
}

void
test2()
{
    printf("\n");
    printf("testing synchronous copy\n");
    size_t free_mem = cuda_get_free_mem();
    size_t n        = (free_mem - (1 << 26)) / 16;
    printf("array size: %lu\n", n);

    int N = 20;

    mdarray<double_complex, 1> v(nullptr, n);
    v.allocate(0);
    v.randomize();
    v.allocate_on_device();

    mdarray<double_complex, 1> v1(n);
    v1.randomize();

    Timer t("copy_and_execute");
    for (int i = 0; i < N; i++) {
        v.copy_to_device();
        cpu_task(v1);
    }
    double tval = t.stop();
    printf("time: %12.6f sec.\n", tval);
}

void
test3()
{
    printf("\n");
    printf("testing asynchronous copy\n");
    size_t free_mem = cuda_get_free_mem();
    size_t n        = (free_mem - (1 << 26)) / 16;
    printf("array size: %lu\n", n);

    int N = 20;

    mdarray<double_complex, 1> v(nullptr, n);
    v.allocate(1);
    v.randomize();
    v.allocate_on_device();

    mdarray<double_complex, 1> v1(n);
    v1.randomize();

    Timer t("copy_and_execute");
    for (int i = 0; i < N; i++) {
        v.async_copy_to_device(0);
        cpu_task(v1);
    }
    double tval = t.stop();
    printf("time: %12.6f sec.\n", tval);
}

int
main(int argn, char** argv)
{
    Platform::initialize(1);
    test1();
    test2();
    test3();
    Timer::print();
    Platform::finalize();
}
