/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include "symmetry/crystal_symmetry.hpp"

/* test rotation of spherical harmonics */

using namespace sirius;

template <typename T>
int
run_test_impl(cmd_args& args)
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
            mdarray<T, 2> ylm_rot_mtrx({sf::lmmax(lmax), sf::lmmax(lmax)});
            sht::rotation_matrix(lmax, ang, proper_rotation, ylm_rot_mtrx);

            mdarray<T, 1> ylm1({sf::lmmax(lmax)});
            ylm1.zero();

            /* rotate original sperical harmonics with P^{-1} */
            for (int i = 0; i < sf::lmmax(lmax); i++) {
                for (int j = 0; j < sf::lmmax(lmax); j++) {
                    ylm1(i) += conj(ylm_rot_mtrx(i, j)) * ylm(j);
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
run_test(cmd_args& args)
{
    int result = run_test_impl<double>(args);
    result += run_test_impl<std::complex<double>>(args);
    return result;
}

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

    sirius::initialize(true);
    printf("running %-30s : ", argv[0]);
    int result = run_test(args);
    if (result) {
        printf("\x1b[31m"
               "Failed"
               "\x1b[0m"
               "\n");
    } else {
        printf("\x1b[32m"
               "OK"
               "\x1b[0m"
               "\n");
    }
    sirius::finalize();

    return result;
}
