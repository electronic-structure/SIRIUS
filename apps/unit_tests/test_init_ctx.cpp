/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include "testing.hpp"

/* template for unit tests */

using namespace sirius;

int
run_test(cmd_args const& args)
{
    Simulation_context ctx;
    if (ctx.cfg().hubbard().local().size() != 0) {
        return 1;
    }

    nlohmann::json node;
    node["atom_type"] = "Fe";
    node["U"]         = 4.0;
    node["J"]         = 1.0;
    node["l"]         = 2;
    node["n"]         = 3;
    ctx.cfg().hubbard().local().append(node);
    if (ctx.cfg().hubbard().local().size() != 1) {
        return 2;
    }
    if (ctx.cfg().hubbard().local(0).n() != 3 || ctx.cfg().hubbard().local(0).l() != 2 ||
        ctx.cfg().hubbard().local(0).U() != 4.0 || ctx.cfg().hubbard().local(0).J() != 1.0) {
        return 3;
    }

    try {
        ctx.cfg().hubbard().local(0).BE2();
    } catch (nlohmann::json::exception const& e) {
    } catch (...) {
        return 4;
    }
    if (ctx.cfg().hubbard().local(0).contains("coeff")) {
        return 5;
    }

    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);

    sirius::initialize(true);
    auto result = call_test(argv[0], run_test, args);
    sirius::finalize();

    return result;
}
