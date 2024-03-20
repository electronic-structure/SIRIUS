/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <fstream>
#include <iomanip>
#include "radial/radial_solver.hpp"

using namespace sirius;

int
get_nmax(int zn)
{
    int N{0};
    for (int n = 1; n < 20; n++) {
        for (int l = 0; l < n; l++) {
            N += 2 * (2 * l + 1);
            if (zn <= N) {
                return n;
            }
        }
    }
    return 20;
}

int
main(int argn, char** argv)
{
    int N{10000};

    std::vector<std::array<int, 3>> bad;

    #pragma omp parallel for
    for (int zn = 80; zn <= 100; zn++) {

        std::vector<double> p;

        Radial_grid_pow<double> r(N, 1e-6, 40, 3);

        int nmax = get_nmax(zn) + 2;

        std::vector<double> v(N);
        for (int i = 0; i < N; i++) {
            v[i] = -zn / r.x(i);
        }

        for (int n = 1; n <= nmax; n++) {

            double enu = -0.5 * std::pow(double(zn) / n, 2);

            for (int l = 0; l < n; l++) {

                double enu1 = sirius::Bound_state(relativity_t::none, zn, n, l, 0, r, v, -0.15, 0.5, 1.25).enu();
                std::cout << "zn : " << zn << ", n : " << n << ", l : " << l
                          << " |enu-enu0| :  " << std::abs(enu - enu1) << std::endl;
            }
        }
    }

    return 0;
}
