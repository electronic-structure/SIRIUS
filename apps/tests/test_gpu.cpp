/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

using namespace sirius;

#ifdef SIRIUS_GPU
void
test_gpu(int N)
{
    mdarray<char, 1> buf(N * 1024);

    for (size_t i = 0; i < buf.size(); i++)
        buf(i) = char(i % 255);

    // DUMP("hash(buf): %llX", buf.hash());

    buf.allocate(memory_t::device);
    buf.copy_to_device();
    buf.zero();
    buf.copy_to_host();

    void* ptr  = cuda_malloc_host(N);
    void* ptr1 = std::malloc(N);

    // DUMP("hash(buf): %llX", buf.hash());

    printf("test of GPU pointer: %i\n", cuda_check_device_ptr(buf.at<GPU>()));
    printf("test of CPU pointer: %i\n", cuda_check_device_ptr(ptr));
    printf("test of CPU pointer: %i\n", cuda_check_device_ptr(ptr1));
    printf("test of CPU pointer: %i\n", cuda_check_device_ptr(buf.at<CPU>()));
}
#endif

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--N=", "{int} buffer size (Kb)");

    args.parse_args(argn, argv);
    if (argn == 1) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    int N = args.value<int>("N");

    sirius::initialize(1);
    cuda_device_info();

#ifdef SIRIUS_GPU
    test_gpu(N);
#endif

    sirius::finalize();
}
