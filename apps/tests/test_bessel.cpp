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
    printf("%18.12f\n", gsl_sf_bessel_Jnu(0.5, 4.71238898038468967400));
}
