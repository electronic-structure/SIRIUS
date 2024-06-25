/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cmath>
#include <testing.hpp>

/* test for NaN and IEEE arithmetics */

using namespace sirius;

int
test_nan()
{
    double val = std::nan("");
    if (val != val) {
        return 0;
    } else {
        return 1;
    }
}

int
main(int argn, char** argv)
{
    return call_test(argv[0], test_nan);
}
