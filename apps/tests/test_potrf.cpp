/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#if defined(SIRIUS_CUDA)
#include "core/acc/cusolver.hpp"
#endif

/* template for unit tests */

using namespace sirius;
using namespace sirius::acc;

int
run_test(cmd_args const& args)
{
#if defined(SIRIUS_CUDA)
    std::vector<int> sizes({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 1000});

    using type = std::complex<double>;

    for (auto n : sizes) {
        auto M = random_positive_definite<type>(n);
        M.allocate(get_memory_pool(memory_t::device)).copy_to(memory_t::device);
        std::cout << "n = " << n << std::endl;
        auto info = cusolver::potrf(n, M.at(memory_t::device), M.ld());
        if (info) {
            return info;
        }
    }
#endif
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);

    sirius::initialize(true);
    auto result = call_test("test_potrf", run_test, args);
    sirius::finalize();
    return result;
}
