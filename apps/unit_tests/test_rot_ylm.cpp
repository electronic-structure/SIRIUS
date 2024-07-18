/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "symmetry/crystal_symmetry.hpp"

/* test rotation of spherical harmonics */

using namespace sirius;

template <typename T>
int
test_rot_ylm_impl(cmd_args const& args)
{
    r3::matrix<double> lattice;
    lattice(0, 0) = 7;
    lattice(1, 1) = 7;
    lattice(2, 2) = 7;

    int num_atoms = 1;
    mdarray<double, 2> positions({3, num_atoms});
    positions.zero();

    mdarray<double, 2> spins({3, num_atoms});
    spins.zero();

    std::vector<int> types(num_atoms, 0);

    bool const spin_orbit{false};
    bool const use_sym{true};
    double const spg_tol{1e-4};

    Crystal_symmetry symmetry(lattice, num_atoms, 1, types, positions, spins, spin_orbit, spg_tol, use_sym);

    /* test P^{-1} R_{lm}(r) = R_{lm}(P r) */

    for (int iter = 0; iter < 10; iter++) {
        for (int isym = 0; isym < symmetry.size(); isym++) {

            int proper_rotation = symmetry[isym].spg_op.proper;
            /* rotation matrix in lattice coordinates */
            auto R = symmetry[isym].spg_op.R;
            /* rotation in Cartesian coord */
            auto Rc = dot(dot(lattice, R), inverse(lattice));
            /* Euler angles of the proper part of rotation */
            auto ang = symmetry[isym].spg_op.euler_angles;

            /* random Cartesian vector */
            r3::vector<double> coord(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
            auto scoord = r3::spherical_coordinates(coord);
            /* rotated coordinates */
            auto coord2  = dot(Rc, coord);
            auto scoord2 = r3::spherical_coordinates(coord2);

            int lmax{10};
            mdarray<T, 1> ylm({sf::lmmax(lmax)});
            /* compute spherical harmonics at original coordinate */
            sf::spherical_harmonics(lmax, scoord[1], scoord[2], &ylm(0));

            mdarray<T, 1> ylm2({sf::lmmax(lmax)});
            /* compute spherical harmonics at rotated coordinates */
            sf::spherical_harmonics(lmax, scoord2[1], scoord2[2], &ylm2(0));

            /* generate rotation matrices; they are block-diagonal in l- index */
            auto ylm_rot_mtrx = sht::rotation_matrix<T>(lmax, ang, proper_rotation);

            mdarray<T, 1> ylm1({sf::lmmax(lmax)});
            ylm1.zero();

            /* rotate original sperical harmonics with P^{-1} */
            for (int l = 0; l <= lmax; l++) {
                for (int i = 0; i < 2 * l + 1; i++) {
                    for (int j = 0; j < 2 * l + 1; j++) {
                        ylm1(l * l + i) += conj(ylm_rot_mtrx[l](i, j)) * ylm(l * l + j);
                    }
                }
            }

            /* compute the difference with the reference */
            double d1{0};
            for (int i = 0; i < sf::lmmax(lmax); i++) {
                d1 += std::abs(ylm1(i) - ylm2(i));
            }
            if (d1 > 1e-10) {
                for (int i = 0; i < sf::lmmax(lmax); i++) {
                    if (std::abs(ylm1(i) - ylm2(i)) > 1e-10) {
                        std::cout << "lm=" << i << " " << ylm1(i) << " " << ylm2(i) << " "
                                  << std::abs(ylm1(i) - ylm2(i)) << std::endl;
                    }
                }
                return 1;
            }
        }
    }
    return 0;
}

int
test_rot_ylm(cmd_args const& args)
{
    int result = test_rot_ylm_impl<double>(args);
    result += test_rot_ylm_impl<std::complex<double>>(args);
    return result;
}

int
main(int argn, char** argv)
{
    cmd_args args;
    return call_test(argv[0], test_rot_ylm, args);
}
