/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include "testing.hpp"

using namespace sirius;

int
test_mem_alloc(cmd_args const& args)
{
    auto sizes = args.value("sizes", std::vector<int>({1024}));
    auto M     = get_memory_t(args.value<std::string>("memory_t", "host"));

    std::vector<char*> ptrs;
    for (auto sm : sizes) {
        auto s   = sm * (size_t(1) << 20);
        auto t0  = time_now();
        auto ptr = allocate<char>(s, M);
        auto t1  = time_now();
        ptrs.push_back(ptr);
        auto t2 = time_now();
        /* this will be allocation time + fill time */
        if (is_host_memory(M)) {
            std::fill(ptr, ptr + s, 0);
        } else {
#ifdef SIRIUS_GPU
            acc::zero(ptr, s);
#endif
        }
        auto t3 = time_now();
        /* this will be only fill time; memory is already allocated */
        if (is_host_memory(M)) {
            std::fill(ptr, ptr + s, 0);
        } else {
#ifdef SIRIUS_GPU
            acc::zero(ptr, s);
#endif
        }
        auto t4 = time_now();

        double fill_time              = time_interval(t3, t4);
        double allocate_and_fill_time = time_interval(t0, t1) + time_interval(t2, t3);
        double allocate_time          = allocate_and_fill_time - fill_time;

        std::cout << "block size (Mb) : " << sm << ", alloc. time : " << allocate_time
                  << ", alloc. speed : " << sm / allocate_time << " Mb/s"
                  << ", fill time : " << fill_time << ", fill speed : " << sm / fill_time << " Mb/s" << std::endl;
        print_memory_usage(std::cout, FILE_LINE);
    }
    for (auto p : ptrs) {
        deallocate(p, M);
    }
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"memory_t=", "{string} type of the memory"}, {"sizes=", "{vector} list of chunk sizes in Mb"}});

    sirius::initialize(1);
    int result = call_test("test_mem_alloc", test_mem_alloc, args);
    sirius::finalize();
    return result;
}
