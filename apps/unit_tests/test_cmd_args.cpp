/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <array>
#include "core/cmd_args.hpp"
#include "testing.hpp"

using namespace sirius;

int
run_test(cmd_args const& args)
{
    auto i1 = args.value<int>("i1");
    if (i1 != 100) {
        std::cout << "error parsing integer value" << std::endl;
        return 1;
    }
    auto d1 = args.value<double>("d1");
    if (d1 != 3.14) {
        std::cout << "error parsing double value" << std::endl;
        return 2;
    }
    auto vi1 = args.value<std::vector<int>>("vi1");
    bool ok{true};
    if (vi1.size() == 3) {
        if (vi1[0] != 1 || vi1[1] != 4 || vi1[2] != 3) {
            ok = false;
        }
    } else {
        ok = false;
    }
    if (!ok) {
        std::cout << "error parsing vector of integers" << std::endl;
        return 3;
    }

    std::array<int, 3> a{0, 0, 0};
    auto vi2 = args.value("vi1", a);
    ok       = true;
    if (vi2[0] != 1 || vi2[1] != 4 || vi2[2] != 3) {
        ok = false;
    }
    if (!ok) {
        std::cout << "error parsing array of integers" << std::endl;
        return 3;
    }

    return 0;
}

int
main(int arg, char** argv)
{
    const char* param[] = {"test_cmd_args", "--i1=100", "--d1=3.14", "--vi1=1:4:3"};

    cmd_args args(4, (char**)param,
                  {{"i1=", "(int) integer parameter"},
                   {"d1=", "(double) double parameter"},
                   {"vi1=", "(vector<int>) vector of integers"}});

    return call_test(argv[0], run_test, args);
}
