/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

using namespace sirius;

double
rounded(double a, int n)
{
    double a0 = std::floor(a);
    double b  = a - a0;
    b         = std::round(b * std::pow(10, n)) / std::pow(10, n);
    return a0 + b;
}

int
main(int argn, char** argv)
{
    sirius::initialize(1);
    double a = 124.144555443334;
    printf("a: %18.10f\n", a);
    printf("rounded(a, 1): %18.10f\n", rounded(a, 1));
    printf("rounded(a, 4): %18.10f\n", rounded(a, 4));
    printf("rounded(a, 6): %18.10f\n", rounded(a, 6));

    a = -0.0023312221313123;
    printf("a: %20.16f\n", a);
    printf("rounded(a, 1): %20.16f\n", rounded(a, 1));
    printf("rounded(a, 4): %20.16f\n", rounded(a, 4));
    printf("rounded(a, 6): %20.16f\n", rounded(a, 6));
    printf("rounded(a, 12): %20.16f\n", rounded(a, 12));

    sirius::finalize();
}
