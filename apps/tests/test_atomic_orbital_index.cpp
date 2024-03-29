/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>

/* template for unit tests */

using namespace sirius;

int
run_test(cmd_args& args)
{
    angular_momentum aqn1(1);
    std::cout << aqn1.l() << " " << aqn1.s() << " " << aqn1.j() << std::endl;

    angular_momentum aqn2(2, -1);
    std::cout << aqn2.l() << " " << aqn2.s() << " " << aqn2.j() << std::endl;

    radial_functions_index ri;

    ri.add(angular_momentum(0, 1));
    ri.add(angular_momentum(1, -1), angular_momentum(1, 1));
    ri.add(angular_momentum(2, -1), angular_momentum(2, 1));
    ri.add(angular_momentum(0));

    std::cout << ri.size() << std::endl;
    std::cout << ri.full_j(0, 0) << std::endl;
    std::cout << ri.full_j(0, 1) << std::endl;

    std::cout << "----" << std::endl;

    for (int l = 0; l < 3; l++) {
        std::cout << ri.max_order(l) << std::endl;
    }
    std::cout << "---" << std::endl;

    for (int l = 0; l < 3; l++) {
        for (int o = 0; o < ri.max_order(l); o++) {
            for (auto j : ri.subshell(l, o)) {
                auto idx = ri.index_of(j, o);
                std::cout << idx << " " << ri.am(idx).l() << " " << ri.am(idx).s() << std::endl;
            }
        }
    }

    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(true);
    int result = run_test(args);
    sirius::finalize();

    return result;
}
