/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <array>
#include "core/any_ptr.hpp"
#include "core/memory.hpp"
#include "testing.hpp"

using namespace sirius;

int
run_test()
{
    void* ptr = new any_ptr(new mdarray<int, 1>({100}, get_memory_pool(memory_t::host)));
    delete static_cast<any_ptr*>(ptr);
    return 0;
}

int
main(int argn, char** argv)
{
    return call_test(argv[0], run_test);
}
