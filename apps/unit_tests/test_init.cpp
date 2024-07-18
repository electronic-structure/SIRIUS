/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>

using namespace sirius;

int
test_init(cmd_args& args)
{
    sirius::initialize(true);
    sirius::finalize();
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args;
    return call_test(argv[0], test_init, args);
}
