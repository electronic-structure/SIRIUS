/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "radial/spline.hpp"
#include "potential/xc_functional.hpp"
#include "core/math_tools.hpp"

using namespace sirius;

// void get_exc_vxc(int np, double* rhoup, double* rhodn, double* exc, double* vxcup, double* vxcdn)
//{
//     for (int i = 0; i < np; i++) {
//         exc[i] = std::pow(rhoup[i] + rhodn[i], 1 / 3.0) + std::pow(rhoup[i], 1 / 2.0);
//         vxcup[i] =
//     }
// }

int
main(int argn, char** argv)
{
    int np{1000};
    auto rg = Radial_grid_factory<double>(radial_grid_t::linear, np, 0.0, 2.0, 0);

    std::vector<double> rho(np);
    std::vector<double> mag(np);
    std::vector<double> rhoup(np);
    std::vector<double> rhodn(np);

    auto get_rho = [](double x) { return 1 + std::pow(std::cos(x), 2); };

    auto get_mag = [](double x) { return 0.5 + 0.25 * std::pow(std::sin(x), 2); };

    for (int i = 0; i < np; i++) {
        double x = rg[i];
        rho[i]   = get_rho(x);
        mag[i]   = get_mag(x);
        rhoup[i] = 0.5 * (rho[i] + mag[i]);
        rhodn[i] = 0.5 * (rho[i] - mag[i]);
    }

    printf("rhoup[0]: %18.12f   rhodn[0]: %18.12f   mag[0]: %18.12f\n", rhoup[0], rhodn[0], rhoup[0] - rhodn[0]);

    std::vector<double> exc(np);
    std::vector<double> vxcup(np);
    std::vector<double> vxcdn(np);

    XC_functional_base xc("XC_LDA_C_PW", 2);

    xc.get_lda(np, rhoup.data(), rhodn.data(), vxcup.data(), vxcdn.data(), exc.data());
    printf("vxcup[0]: %18.12f   vxcdn[0]: %18.12f   bxc[0]: %18.12f\n", vxcup[0], vxcdn[0], vxcup[0] - vxcdn[0]);
    printf("sign(mag * Bxc): %i\n", sign((rhoup[0] - rhodn[0]) * (vxcup[0] - vxcdn[0])));

    Spline<double> s1(rg);
    Spline<double> s2(rg);
    Spline<double> s3(rg);
    for (int i = 0; i < np; i++) {
        s1(i) = rhoup[i] * vxcup[i];
        s2(i) = rhodn[i] * vxcdn[i];
        s3(i) = (rhoup[i] + rhodn[i]) * exc[i];
    }
    double Exc    = s3.interpolate().integrate(0);
    double evxcup = s1.interpolate().integrate(0);
    double evxcdn = s2.interpolate().integrate(0);
    printf("Exc: %18.12f   evxcup: %18.12f   evxcdn: %18.12f\n", Exc, evxcup, evxcdn);

    double eps{1e-6};

    for (int i = 0; i < np; i++) {
        double x = rg[i];
        rho[i]   = get_rho(x);
        mag[i]   = get_mag(x);
        rhoup[i] = 0.5 * (rho[i] + mag[i]) * (1 + eps);
        rhodn[i] = 0.5 * (rho[i] - mag[i]);
    }
    xc.get_lda(np, rhoup.data(), rhodn.data(), vxcup.data(), vxcdn.data(), exc.data());
    for (int i = 0; i < np; i++) {
        s2(i) = (rhoup[i] + rhodn[i]) * exc[i];
    }
    double Exc_2 = s2.interpolate().integrate(0);
    printf("spin up: %18.12f\n", std::abs((Exc_2 - Exc) / eps - evxcup));

    for (int i = 0; i < np; i++) {
        double x = rg[i];
        rho[i]   = get_rho(x);
        mag[i]   = get_mag(x);
        rhoup[i] = 0.5 * (rho[i] + mag[i]);
        rhodn[i] = 0.5 * (rho[i] - mag[i]) * (1 + eps);
    }
    xc.get_lda(np, rhoup.data(), rhodn.data(), vxcup.data(), vxcdn.data(), exc.data());
    for (int i = 0; i < np; i++) {
        s2(i) = (rhoup[i] + rhodn[i]) * exc[i];
    }
    Exc_2 = s2.interpolate().integrate(0);
    printf("spin dn: %18.12f\n", std::abs((Exc_2 - Exc) / eps - evxcdn));
}
