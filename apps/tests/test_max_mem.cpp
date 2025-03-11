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
test_max_mem(cmd_args const& args)
{
    std::cout << "Free avaialble memory : " << (get_available_memory() >> 30) << " Gb" << std::endl;
    auto M      = get_memory_t(args.value<std::string>("memory_t", "host"));
    size_t size = args.value<int>("size", 1);

    //std::cout << "attempting to allocate " << size << " Gb" << std::endl;
    //size *= (1 << 30);
    //mdarray<char, 1> a({size}, M);
    //a.zero(M);

    std::vector<mdarray<char, 1>> v;
    for (int i = 0; i < 100; i++) {
        std::cout << "step : " << i << std::endl;
        std::cout << "attempting to allocate " << size << " Gb" << std::endl;
        v.push_back(mdarray<char, 1>({size * (1 << 30)}, M));
        v.back().zero(M);
        print_memory_usage(std::cout);
        std::cout << std::flush;
    }

    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{"memory_t=", "{string} type of the memory"}, {"size=", "{int} size of array in Gb"}});

    sirius::initialize(1);
    int result = call_test("test_max_mem", test_max_mem, args);
    sirius::finalize();
    return result;
}
