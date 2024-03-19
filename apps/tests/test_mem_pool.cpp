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

using namespace sirius;

// void test(int nGb, int gran, memory_t M__)
// {
//     memory_pool &mpool = get_memory_pool(M__);

//     std::vector<uint32_t> sizes;
//     size_t tot_size{0};

//     while (tot_size < size_t(nGb) * (1 << 30)) {
//         auto s = std::max(utils::rnd() % (size_t(gran) * (1 << 20)), size_t(1));
//         sizes.push_back(s);
//         tot_size += s;
//     }

//     std::cout << "number of memory blocks: " << sizes.size() << "\n";
//     std::cout << "total size: " << tot_size << "\n";

//     std::vector<mdarray<char, 1>> v;
//     for (int k = 0; k < 4; k++) {
//         auto t = -utils::wtime();
//         v.clear();
//         for (auto s: sizes) {
//             v.push_back(mdarray<char, 1>(s, mpool));
//             v.back().zero(M__);
//         }
//         std::shuffle(v.begin(), v.end(), std::mt19937());
//         for (auto& e: v) {
//             e.deallocate(M__);
//         }

// 	mpool.clear();
//         if (mpool.total_size() != tot_size) {
//             std::stringstream s;
//             s << "mpool.total_size() != tot_size" << std::endl
//               << "mpool.total_size() = " << mpool.total_size() << std::endl
//               << "tot_size = " << tot_size;
//             RTE_THROW(s);
//         }
//         if (mpool.free_size() != mpool.total_size()) {
//             throw std::runtime_error("wrong free size");
//         }
//         if (mpool.num_blocks() != 1) {
//             throw std::runtime_error("wrong number of blocks");
//         }
//         t += utils::wtime();
//         std::cout << "pass : " << k << ", time : " << t << "\n";
//     }
// }

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--memory_t=", "{string} type of the memory");
    args.register_key("--nGb=", "{int} total number of Gigabytes to allocate");
    args.register_key("--gran=", "{int} block granularity in Mb");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    // test(args.value<int>("nGb", 2), args.value<int>("gran", 32), get_memory_t(args.value<std::string>("memory_t",
    // "host")));
    sirius::finalize();
}
