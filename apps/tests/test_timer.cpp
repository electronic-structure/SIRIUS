/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

using namespace sirius;

void
foo()
{
    PROFILE("foo");
}

double
bar()
{
    utils::timer t1("bar");
    double r{0};
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 1000; j++) {
            r += ((i % 3) - 1) * type_wrapper<double>::random();
        }
    }
    return r;
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

    sirius::initialize(1);
    foo();
    bar();
    bar();
    sirius::finalize();
    utils::timer::print();
}
