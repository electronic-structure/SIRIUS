/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>

using namespace sirius;

int
test_r3_geom()
{
    r3::vector<double> a(1.1, 2.2, 3.3);
    r3::vector<double> b = a;
    r3::vector<double> c(b);
    r3::vector<double> d;
    d                 = c;
    r3::matrix<int> R = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    auto e            = dot(R, a) + b;
    r3::vector<double> ref(2.2, 4.4, 6.6);
    if ((ref - e).length() > 1e-16) {
        return 1;
    }
    auto x = a + r3::vector<int>(4, 5, 6);
    r3::vector<double> ref1(5.1, 7.2, 9.3);
    if ((ref1 - x).length() > 1e-16) {
        return 2;
    }
    return 0;
}

int
main(int argn, char** argv)
{
    return call_test(argv[0], test_r3_geom);
}
