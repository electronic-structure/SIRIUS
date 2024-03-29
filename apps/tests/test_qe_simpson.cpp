/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

using namespace sirius;

extern "C" void
simpson_(int* mesh, double* func, double* rab, double* asum);

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

    json parser;
    std::ifstream("B.pbe-n-kjpaw_psl.0.1.UPF.json") >> parser;
    auto r    = parser["pseudo_potential"]["radial_grid"].get<std::vector<double>>();
    auto rab  = parser["pseudo_potential"]["RAB"].get<std::vector<double>>();
    auto vloc = parser["pseudo_potential"]["local_potential"].get<std::vector<double>>();
    Radial_grid_ext<double> rgrid(static_cast<int>(r.size()), r.data());

    Spline<double> s(rgrid);
    for (int i = 0; i < rgrid.num_points(); i++) {
        double x = rgrid[i];
        // s[i] = std::sin(4 * x) * std::exp(-2 * x * x);
        s[i] = (vloc[i] * x + 3) * x;
    }
    printf("integral(spline): %18.10f\n", s.interpolate().integrate(0));

    int sz = rgrid.num_points();
    double v;
    simpson_(&sz, &s[0], rab.data(), &v);

    printf("integral(QE_simpson): %18.10f\n", v);

    sirius::finalize();
}
