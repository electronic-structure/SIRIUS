/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>

using namespace sirius;

int
main(int argn, char** argv)
{
    printf("sizeof(double): %lu\n", sizeof(double));
    printf("sizeof(long double): %lu\n", sizeof(long double));

    long double PI = 3.141592653589793238462643383279502884197L;

    printf("diff (in long double): %40.30Lf\n", std::abs(PI - std::acos(static_cast<long double>(-1))));
    printf("diff (in double): %40.30f\n", std::abs(static_cast<double>(PI) - std::acos(static_cast<double>(-1))));
    printf("diff (double - long double): %40.30Lf\n",
           std::abs(static_cast<long double>(std::acos(static_cast<double>(-1))) -
                    std::acos(static_cast<long double>(-1))));

    return 0;
}
