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

double
test1_angular_radial_double(int lmax__)
{
    SHT sht(device_t::CPU, lmax__);
    int lmmax = sf::lmmax(lmax__);

    auto r = Radial_grid_factory<double>(radial_grid_t::exponential, 1000, 0.01, 2.0, 1.0);

    Spheric_function<function_domain_t::spectral, double> f1(lmmax, r);

    for (int ir = 0; ir < r.num_points(); ir++) {
        for (int lm = 0; lm < lmmax; lm++) {
            f1(lm, ir) = random<double>();
        }
    }
    auto f2 = convert(f1);
    auto f3 = convert(f2);

    double d = 0;
    for (int ir = 0; ir < r.num_points(); ir++) {
        for (int lm = 0; lm < lmmax; lm++) {
            d += std::abs(f1(lm, ir) - f3(lm, ir));
        }
    }
    return d;
}

double
test1_angular_radial_complex(int lmax__)
{
    SHT sht(device_t::CPU, lmax__);
    int lmmax = sf::lmmax(lmax__);

    auto r = Radial_grid_factory<double>(radial_grid_t::exponential, 1000, 0.01, 2.0, 1.0);

    Spheric_function<function_domain_t::spectral, std::complex<double>> f1(lmmax, r);

    for (int ir = 0; ir < r.num_points(); ir++) {
        for (int l = 0; l <= lmax__; l++) {
            f1(sf::lm(l, 0), ir) = random<double>();
            for (int m = 1; m <= l; m++) {
                f1(sf::lm(l, m), ir)  = random<std::complex<double>>();
                f1(sf::lm(l, -m), ir) = std::pow(-1, m) * std::conj(f1(sf::lm(l, m), ir));
            }
        }
    }
    auto f2 = convert(f1);
    auto f3 = convert(f2);

    double d = 0;
    for (int ir = 0; ir < r.num_points(); ir++) {
        for (int lm = 0; lm < lmmax; lm++) {
            d += std::abs(f1(lm, ir) - f3(lm, ir));
        }
    }
    return d;
}

int
test_sht()
{
    if (test1_angular_radial_double(10) > 1e-10) {
        return 1;
    }
    if (test1_angular_radial_complex(10) > 1e-10) {
        return 2;
    }
    return 0;
}

int
main(int argn, char** argv)
{
    return call_test(argv[0], test_sht);
}
