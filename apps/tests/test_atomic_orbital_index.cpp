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
test_atomic_orbital_index(cmd_args const& args)
{
    angular_momentum aqn1(1);
    std::cout << "l : " << aqn1.l() << ", s : " << aqn1.s() << ", j : " << aqn1.j() << std::endl;

    angular_momentum aqn2(2, -1);
    std::cout << "l : " << aqn2.l() << ", s : " << aqn2.s() << ", j : " << aqn2.j() << std::endl;

    radial_functions_index ri;

    ri.add(angular_momentum(0, 1));
    ri.add(angular_momentum(1, -1), angular_momentum(1, 1));
    ri.add(angular_momentum(2, -1), angular_momentum(2, 1));
    ri.add(angular_momentum(0));

    std::cout << "radial index size : " << ri.size() << std::endl;
    std::cout << "is full_j(l=0, order=0) : " << ri.full_j(0, 0) << std::endl;
    std::cout << "is full_j(l=0, order=1) : " << ri.full_j(0, 1) << std::endl;

    std::cout << "----" << std::endl;

    for (int l = 0; l < 3; l++) {
        std::cout << "l : " << l << ", max.order : " << ri.max_order(l) << std::endl;
    }
    std::cout << "---" << std::endl;

    for (int l = 0; l < 3; l++) {
        for (int o = 0; o < ri.max_order(l); o++) {
            for (auto j : ri.subshell(l, o)) {
                auto idx = ri.index_of(j, o);
                std::cout << "l : " << ri.am(idx).l() << ", o : " << o << ", s : " << ri.am(idx).s()
                          << ", idx : " << idx << std::endl;
            }
        }
    }

    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args;
    return call_test("test_atomic_orbital_index", test_atomic_orbital_index, args);
}
