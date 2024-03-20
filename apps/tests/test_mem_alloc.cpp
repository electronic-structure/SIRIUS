/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>

using namespace sirius;

void
test(std::vector<int> sizes, memory_t M__)
{
    std::vector<char*> ptrs;
    for (auto sm : sizes) {
        auto s   = sm * (size_t(1) << 20);
        auto t0  = wtime();
        auto ptr = allocate<char>(s, M__);
        ptrs.push_back(ptr);
        if (is_host_memory(M__)) {
            std::fill(ptr, ptr + s, 0);
        } else {
#ifdef SIRIUS_GPU
            acc::zero(ptr, s);
#endif
        }
        auto t1 = wtime();
        if (is_host_memory(M__)) {
            std::fill(ptr, ptr + s, 0);
        } else {
#ifdef SIRIUS_GPU
            acc::zero(ptr, s);
#endif
        }
        auto t2 = wtime();
        // sddk::deallocate(ptr, M__);
        // auto t3 = wtime();

        std::cout << "block size (Mb) : " << sm << ", alloc time : " << (t1 - t0) - (t2 - t1) << "\n";
        print_memory_usage(std::cout, FILE_LINE);
    }
    for (auto p : ptrs) {
        deallocate(p, M__);
    }
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--memory_t=", "{string} type of the memory");
    args.register_key("--sizes=", "{vector} list of chunk sizes in Mb");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    auto sizes = args.value("sizes", std::vector<int>({1024}));
    test(sizes, get_memory_t(args.value<std::string>("memory_t", "host")));
    sirius::finalize();
}
