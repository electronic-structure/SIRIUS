/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>

using namespace sirius;

template <typename T>
double
test()
{
    int lmax{10};
    SHT sht(device_t::CPU, lmax);
    int lmmax = sf::lmmax(lmax);

    auto r = Radial_grid_factory<double>(radial_grid_t::exponential, 1000, 0.01, 2.0, 1.0);

    Spheric_function<function_domain_t::spectral, T> f(lmmax, r);

    for (int ir = 0; ir < r.num_points(); ir++) {
        double x = r[ir];
        for (int l = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++) {
                int lm    = sf::lm(l, m);
                f(lm, ir) = std::exp(-0.1 * (lm + 1) * x) * std::pow(x, l);
            }
        }
    }

    auto lapl_f     = laplacian(f);
    auto grad_f     = gradient(f);
    auto div_grad_f = divergence(grad_f);

    /* chek up to lmax-1 because \grad couples l-1 and l+1 and so \div\grad is incomplete at lmax */
    Spline<double> s(r);
    Spline<double> s1(r);
    for (int ir = 0; ir < r.num_points(); ir++) {
        for (int lm = 0; lm < sf::lmmax(lmax - 1); lm++) {
            s(ir) += std::abs(lapl_f(lm, ir) - div_grad_f(lm, ir));
            s1(ir) += std::abs(lapl_f(lm, ir)) + std::abs(div_grad_f(lm, ir));
        }
    }

    return s.interpolate().integrate(0) / s1.interpolate().integrate(0);
}

int
main(int argn, char** argv)
{
    sirius::initialize(true);

    double diff;

    if ((diff = test<std::complex<double>>()) > 1e-12) {
        printf("error in Ylm expansion: %18.12f\n", diff);
        return 1;
    }

    if ((diff = test<double>()) > 1e-12) {
        printf("error in Rlm expansion: %18.12f\n", diff);
        return 2;
    }

    sirius::finalize();

    return 0;
}
