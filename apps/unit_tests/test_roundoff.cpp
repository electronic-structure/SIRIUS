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
    int err{0};
    if (std::abs(round(1.525252525, 4) - 1.5253) > 1e-20) {
        err++;
    }
    if (std::abs(round(2.12345678, 4) - 2.1235) > 1e-20) {
        err++;
    }
    if (std::abs(round(2.12344678, 4) - 2.1234) > 1e-20) {
        err++;
    }
    if (std::abs(round(1.999, 0) - 2) > 1e-20) {
        err++;
    }
    if (std::abs(round(1.999, 1) - 2) > 1e-20) {
        err++;
    }
    return err;
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
    printf("running %-30s : ", argv[0]);
    int result = run_test(args);
    if (result) {
        printf("\x1b[31m"
               "Failed"
               "\x1b[0m"
               "\n");
    } else {
        printf("\x1b[32m"
               "OK"
               "\x1b[0m"
               "\n");
    }
    sirius::finalize();

    return result;
}
