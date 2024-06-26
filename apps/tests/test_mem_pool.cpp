/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <algorithm>
#include <random>
#include "testing.hpp"

using namespace sirius;

int
test_memory_pool(cmd_args const& args__)
{
    auto M    = get_memory_t(args__.value<std::string>("memory_t", "host"));
    auto nGb  = args__.value<size_t>("nGb", 2);
    auto gran = args__.value<uint32_t>("gran", 32);

    auto& mpool = get_memory_pool(M);

    std::vector<uint32_t> sizes;
    size_t tot_size{0};

    while (tot_size < nGb * (1 << 30)) {
        auto s = std::max(random_uint32() % (gran * (1 << 20)), uint32_t(1));
        sizes.push_back(s);
        tot_size += s;
    }

    std::cout << "number of memory blocks: " << sizes.size() << "\n";
    std::cout << "total size: " << tot_size << "\n";

    std::vector<mdarray<char, 1>> v;
    for (int k = 0; k < 10; k++) {
        auto t0 = time_now();
        v.clear();
        for (auto s : sizes) {
            v.push_back(mdarray<char, 1>({s}, mpool));
            v.back().zero(M);
        }
        std::shuffle(v.begin(), v.end(), std::mt19937());
        for (auto& e : v) {
            e.deallocate(M);
        }
        double t = time_interval(t0);
        std::cout << "pass : " << k << ", time : " << t << ", effective throughput : " << tot_size / t / (1 << 20)
                  << " Mb/s\n";
    }
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"memory_t=", "{string} type of the memory"},
                   {"nGb=", "{int} total number of Gigabytes to allocate"},
                   {"gran=", "{int} block granularity in Mb"}});

    sirius::initialize(1);
    int result = call_test("test_memory_pool", test_memory_pool, args);
    sirius::finalize();
    return result;
}
