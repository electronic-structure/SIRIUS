/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "sirius.hpp"
#include "testing.hpp"

using namespace sirius;

int
test_radial_solver(cmd_args const& args__)
{
    auto rel = get_relativity_t(args__.value<std::string>("rel", "none"));
    auto zn  = args__.value<int>("zn", 1);
    auto l   = args__.value<int>("l", 0);
    auto dme = args__.value<int>("dme", 0);
    auto enu = args__.value<double>("enu", -0.5);

    Radial_grid_lin_exp<double> rgrid(1500, 1e-7, 3.0);
    std::vector<double> v(rgrid.num_points());
    for (int ir = 0; ir < rgrid.num_points(); ir++) {
        v[ir] = -zn * rgrid.x_inv(ir);
    }

    Radial_solver rsolver(zn, v, rgrid);
    auto result_ref = rsolver.solve(rel, dme, l, enu);

    #pragma omp parallel for
    for (int i = 0; i < 1000; i++) {
        auto result = rsolver.solve(rel, dme, l, enu);
        if ((result.uderiv[1] != result_ref.uderiv[1]) || (result.uderiv[2] != result_ref.uderiv[2])) {
            std::cout << "wrong uderiv" << std::endl;
        }
        for (int j = 0; j < rgrid.num_points(); j++) {
            if (result.p[j] != result_ref.p[j]) {
                std::cout << "wrong p" << std::endl;
            }
        }
    }

    // std::stringstream s;
    // s << "radial_functions_" << args__.value<std::string>("rel", "none") << ".dat";
    // FILE* fout = fopen(s.str().c_str(), "w");

    // for (int ir = 0; ir < rgrid.num_points(); ir++) {
    //     fprintf(fout, "%f ", rgrid[ir]);
    //     fprintf(fout, "%f ", p[ir]);
    //     fprintf(fout, "%f ", rdudr[ir]);
    //     fprintf(fout, "\n");
    // }
    // fclose(fout);

    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"rel=", "(string) type of scalar-relativistic equation"},
                   {"zn=", "(int) nuclear charge"},
                   {"l=", "(int) orbital quantum number"},
                   {"enu=", "(double) energy of the equation"},
                   {"dme=", "(int) energy derivative"}});

    return call_test("test_radial_solver", test_radial_solver, args);
}
