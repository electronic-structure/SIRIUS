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
test_sim_ctx(cmd_args const& args)
{
    Simulation_context ctx;
    ctx.import(args);
    std::cout << ctx.verbosity() << "\n";
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{"control.verbosity=", "{int} verbosity level"}});

    sirius::initialize(true);
    int result = call_test(argv[0], test_sim_ctx, args);
    sirius::finalize();
    return result;
}
