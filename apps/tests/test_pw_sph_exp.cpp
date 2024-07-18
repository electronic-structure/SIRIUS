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

void
test(r3::vector<double> G)
{
    int lmax{7};

    SHT sht(device_t::CPU, lmax);

    std::cout << "lmax = " << lmax << std::endl;
    std::cout << "sht.num_points = " << sht.num_points() << std::endl;

    double r{0.5};

    auto vgs = r3::spherical_coordinates(G);
    std::cout << "G(spherical): " << vgs[0] << " " << vgs[1] << " " << vgs[2] << std::endl;
    std::vector<double> rlm_g(sf::lmmax(lmax));
    sf::spherical_harmonics(lmax, vgs[1], vgs[2], &rlm_g[0]);

    double jl[lmax + 1];
    sf::Spherical_Bessel_functions::sbessel(lmax, r * vgs[0], jl);

    std::vector<double> rlm(sf::lmmax(lmax));

    double diff{0};
    for (int i = 0; i < sht.num_points(); i++) {
        sf::spherical_harmonics(lmax, sht.theta(i), sht.phi(i), &rlm[0]);
        auto coord = sht.coord(i);

        std::complex<double> z(0, 0);

        for (int l = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++) {
                int lm = sf::lm(l, m);
                z += fourpi * std::pow(std::complex<double>(0, 1), l) * jl[l] * rlm[lm] * rlm_g[lm];
            }
        }

        diff += std::abs(z - std::exp(std::complex<double>(0, r * dot(coord, G))));
    }
    std::cout << "diff=" << diff << std::endl;
}

void
test1(int mu, r3::vector<double> G)
{
    int lmax{7};

    SHT sht(device_t::CPU, lmax);
    int lmmax = sf::lmmax(lmax);

    double r{0.5};

    auto vgs = r3::spherical_coordinates(G);

    double theta = vgs[1];
    double phi   = vgs[2];

    std::vector<double> rlm_g(lmmax);
    sf::spherical_harmonics(lmax, theta, phi, &rlm_g[0]);

    mdarray<double, 2> rlm_dg({lmmax, 3});

    sf::dRlm_dr(lmax, G, rlm_dg);

    std::vector<double> rlm(lmmax);

    double jl[lmax + 1];
    std::cout << "computing jl" << std::endl;
    sf::Spherical_Bessel_functions::sbessel(lmax, r * vgs[0], jl);
    double jl_dq[lmax + 1];
    std::cout << "computing djl" << std::endl;
    sf::Spherical_Bessel_functions::sbessel_deriv_q(lmax, vgs[0], r, &jl_dq[0]);

    double diff{0};
    for (int i = 0; i < sht.num_points(); i++) {
        sf::spherical_harmonics(lmax, sht.theta(i), sht.phi(i), &rlm[0]);
        auto coord = sht.coord(i);

        std::complex<double> z(0, 0);

        for (int l = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++) {
                int lm = sf::lm(l, m);
                z += fourpi * std::pow(std::complex<double>(0, 1), l) * rlm[lm] *
                     (rlm_g[lm] * jl_dq[l] * G[mu] / vgs[0] + rlm_dg(lm, mu) * jl[l]);
            }
        }

        diff += std::abs(z -
                         std::complex<double>(0, coord[mu] * r) * std::exp(std::complex<double>(0, r * dot(coord, G))));
    }

    std::cout << "derivatives diff=" << diff << std::endl;
}

int
test_pw_sph_exp()
{
    test({0.4, 0, 0});
    test({0, 0.4, 0});
    test({0, 0, 0.4});
    test({0.1, 0.2, 0.3});

    test1(0, {0.4, 0, 0});
    test1(1, {0.4, 0, 0});
    test1(2, {0.4, 0, 0});

    test1(0, {0.0, 0.4, 0});
    test1(1, {0.0, 0.4, 0});
    test1(2, {0.0, 0.4, 0});

    test1(0, {0.0, 0, 0.4});
    test1(1, {0.0, 0, 0.4});
    test1(2, {0.0, 0, 0.4});

    test1(0, {0.1, 0.2, 0.3});
    test1(1, {0.1, 0.2, 0.3});
    test1(2, {0.1, 0.2, 0.3});

    return 0;
}

int
main(int argn, char** argv)
{
    return call_test("test_pw_sph_exp", test_pw_sph_exp);
}
