/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

using namespace sirius;

void
test_sbess_approx()
{
    int lmax    = 6;
    double R    = 2.0;
    double qmin = 0.1;
    double qmax = 15;
    double eps  = 1e-12;

    Spherical_Bessel_approximant sba(lmax, R, qmin, qmax, eps);

    Radial_grid rgrid(exponential_grid, 2000, 1e-7, R);

    mdarray<Spherical_Bessel_functions, 2> jnu(lmax + 1, sba.nqnu_max());

    printf("qmin, qmax: %f %f\n", qmin, qmax);
    for (int l = 0; l <= lmax; l++) {
        printf("nqnu[l=%i] = %i\n", l, sba.nqnu(l));

        for (int iq = 0; iq < sba.nqnu(l); iq++) {
            jnu(l, iq) = Spherical_Bessel_functions(lmax, rgrid, sba.qnu(iq, l));
        }
    }

    double nu = 3.0;
    Spherical_Bessel_functions sbf(lmax, rgrid, nu);

    FILE* out = fopen("sba.dat", "w");

    for (int l = 0; l < lmax; l++) {
        auto c = sba.approximate(l, nu);

        for (int ir = 0; ir < rgrid.num_points(); ir++) {
            fprintf(out, "%f %f\n", rgrid[ir], sbf(l)[ir]);
        }
        fprintf(out, "\n");
        for (int ir = 0; ir < rgrid.num_points(); ir++) {
            double v = 0;
            for (int iq = 0; iq < sba.nqnu(l); iq++) {
                v += c[iq] * jnu(l, iq)(l)[ir];
            }
            fprintf(out, "%f %f\n", rgrid[ir], v);
        }
        fprintf(out, "\n");
    }
    fclose(out);

    // for (int inu = 0; inu < 100; inu++)
    //{
    //     double nu = qmin + (qmax - qmin) * inu / (100 - 1);
    //     Spherical_Bessel_functions sbf(lmax, rgrid, nu);

    //    printf("nu: %f\n", nu);

    //    for (int l = 0; l < lmax; l++)
    //    {
    //        printf("l: %i\n", l);
    //        auto c = sba.approximate(l, nu);
    //        for (int iq = 0; iq < sba.nqnu(l); iq++)
    //        {
    //            printf("c[iq=%i] = %f\n", iq, c[iq]);
    //        }

    //        for (int ir = 0; ir < rgrid.num_points(); ir++)
    //        {
    //            double v = 0;
    //            for (int iq = 0; iq < sba.nqnu(l); iq++)
    //            {
    //                v += c[iq] * jnu(l, iq)(l)[ir];
    //            }
    //            if (std::abs(v - sbf(l)[ir]) > 1e-4)
    //            {
    //                std::stringstream s;
    //                s << "approximation is wrong" << std::endl
    //                  << "obtained value: " << v << std::endl
    //                  << "target value: " << sbf(l)[ir] << std::endl
    //                  << "ir: " << ir << ", l: " << l;
    //                RTE_THROW(s);
    //            }
    //        }
    //    }
    //}
}

void
test_sbess_approx2()
{
    int lmax    = 6;
    double R    = 100.0;
    double qmin = 0.1;
    double qmax = 25;

    Spherical_Bessel_approximant2 sba(lmax, R, qmin, qmax, 25);

    Radial_grid rgrid(exponential_grid, 2000, 1e-7, R);

    mdarray<Spherical_Bessel_functions, 1> jnu(sba.nqnu());

    printf("nqnu: %i\n", sba.nqnu());

    for (int iq = 0; iq < sba.nqnu(); iq++) {
        jnu(iq) = Spherical_Bessel_functions(lmax, rgrid, sba.qnu(iq));
    }

    double nu = 3.0;
    Spherical_Bessel_functions sbf(lmax, rgrid, nu);

    FILE* out = fopen("sba.dat", "w");

    for (int l = 0; l < lmax; l++) {
        auto c = sba.approximate(l, nu);

        for (int ir = 0; ir < rgrid.num_points(); ir++) {
            fprintf(out, "%f %f\n", rgrid[ir], sbf(l)[ir]);
        }
        fprintf(out, "\n");
        for (int ir = 0; ir < rgrid.num_points(); ir++) {
            double v = 0;
            for (int iq = 0; iq < sba.nqnu(); iq++) {
                v += c[iq] * jnu(iq)(l)[ir];
            }
            fprintf(out, "%f %f\n", rgrid[ir], v);
        }
        fprintf(out, "\n");
    }
    fclose(out);

    // for (int inu = 0; inu < 100; inu++)
    //{
    //     double nu = qmin + (qmax - qmin) * inu / (100 - 1);
    //     Spherical_Bessel_functions sbf(lmax, rgrid, nu);

    //    printf("nu: %f\n", nu);

    //    for (int l = 0; l < lmax; l++)
    //    {
    //        printf("l: %i\n", l);
    //        auto c = sba.approximate(l, nu);
    //        for (int iq = 0; iq < sba.nqnu(l); iq++)
    //        {
    //            printf("c[iq=%i] = %f\n", iq, c[iq]);
    //        }

    //        for (int ir = 0; ir < rgrid.num_points(); ir++)
    //        {
    //            double v = 0;
    //            for (int iq = 0; iq < sba.nqnu(l); iq++)
    //            {
    //                v += c[iq] * jnu(l, iq)(l)[ir];
    //            }
    //            if (std::abs(v - sbf(l)[ir]) > 1e-4)
    //            {
    //                std::stringstream s;
    //                s << "approximation is wrong" << std::endl
    //                  << "obtained value: " << v << std::endl
    //                  << "target value: " << sbf(l)[ir] << std::endl
    //                  << "ir: " << ir << ", l: " << l;
    //                RTE_THROW(s);
    //            }
    //        }
    //    }
    //}
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

    sirius::initialize(1);
    test_sbess_approx2();
    sirius::finalize();
}
