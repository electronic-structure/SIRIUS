/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

using namespace sirius;

#define stdpow(x, y) std::pow(x, y)
#define stdexp(x) std::exp(x)
#define stdsin(x) std::sin(x)
#define stdcos(x) std::cos(x)
#define stdlog(x) std::log(x)

double
check_spline(Spline<double> const& s__, std::function<double(double)> f__, double x0__, double x1__)
{
    Radial_grid_lin<double> rgrid(10000, x0__, x1__);

    double l2norm{0};
    for (int ir = 0; ir < rgrid.num_points(); ir++) {
        double x = rgrid[ir];
        l2norm += std::pow(f__(x), 2);
    }
    l2norm = std::sqrt(l2norm * (x1__ - x0__) / rgrid.num_points());

    FILE* fout = fopen("spline.dat", "w");
    double max_diff{0};
    double rel_diff{0};
    double l2norm_delta{0};
    for (int ir = 0; ir < rgrid.num_points(); ir++) {
        double x = rgrid[ir];
        l2norm_delta += std::pow(s__.at_point(x) - f__(x), 2);
        max_diff = std::max(max_diff, std::abs(s__.at_point(x) - f__(x)));
        rel_diff = std::max(rel_diff, std::abs(s__.at_point(x) - f__(x)) /
                                              (std::abs(s__.at_point(x)) + std::abs(f__(x)) + 1e-12));
        // fprintf(fout, "%18.14f %18.14f %18.14f %18.14f\n", x, f__(x), s__.at_point(x), std::abs(s__.at_point(x) -
        // f__(x)));
        fprintf(fout, "%18.14f %18.14f %18.14f %18.14f\n", x, std::abs(s__.at_point(x) - f__(x)), f__(x),
                s__.at_point(x));
    }
    fclose(fout);
    l2norm_delta = std::sqrt(l2norm_delta * (x1__ - x0__) / rgrid.num_points());
    // return l2norm_delta / l2norm;
    // return rel_diff;
    // max_diff = l2norm_delta;
    return max_diff;
}

void
test_function(std::function<double(double)> f__)
{
    Radial_grid_lin_exp<double> rgrid(3000, 1e-7, 8);

    Spline<double> s(rgrid, f__);
    double diff = check_spline(s, f__, rgrid.first(), rgrid.last());
    printf("difference: %18.10f   ", diff);
    if (diff > 1e-7) {
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
    printf("\n");
}

void
test1()
{
    printf("testing f(x)=1\n");

    test_function([](double x) { return 1; });

    printf("testing f(x)=x\n");

    test_function([](double x) { return x; });

    printf("testing f(x)=x**2\n");

    test_function([](double x) { return stdpow(x, 2); });

    printf("testing f(x)=x**3\n");

    test_function([](double x) { return stdpow(x, 3); });

    printf("testing f(x)=1/(1 + x)\n");

    test_function([](double x) { return stdpow(1 + x, -1); });

    printf("testing f(x)=(1 + x)**(-2)\n");

    test_function([](double x) { return stdpow(1 + x, -2); });

    printf("testing f(x)=Sqrt(x)\n");

    test_function([](double x) { return stdpow(x, 0.5); });

    printf("testing f(x)=exp(-x)\n");

    test_function([](double x) { return stdexp(-x); });

    printf("testing f(x)=Sin(x)\n");

    test_function([](double x) { return stdsin(x); });

    printf("testing f(x)=Sin(2*x)\n");

    test_function([](double x) { return stdsin(2 * x); });

    printf("testing f(x)=exp(x)\n");

    test_function([](double x) { return stdexp(x); });

    printf("testing f(x)=exp(-x**2)\n");

    test_function([](double x) { return stdexp(-stdpow(x, 2)); });

    printf("testing f(x)=exp(x - x**2)\n");

    test_function([](double x) { return stdexp(x - stdpow(x, 2)); });

    printf("testing f(x)=x/exp(x)\n");

    test_function([](double x) { return x / stdexp(x); });

    printf("testing f(x)=x**2/exp(x)\n");

    test_function([](double x) { return stdpow(x, 2) / stdexp(x); });

    printf("testing f(x)=Log(1 + x)\n");

    test_function([](double x) { return stdlog(1 + x); });

    printf("testing f(x)=Sin(x)/exp(2*x)\n");

    test_function([](double x) { return stdsin(x) / stdexp(2 * x); });

    printf("testing f(x)=Sin(x)/x\n");

    test_function([](double x) { return stdsin(x) / x; });
}

int
main(int argn, char** argv)
{
    sirius::initialize(1);

    test1();

    sirius::finalize();

    return 0;
}
