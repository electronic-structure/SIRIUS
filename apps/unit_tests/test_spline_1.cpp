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

#define stdpow(x, y) std::pow(x, y)
#define stdexp(x) std::exp(x)
#define stdsin(x) std::sin(x)
#define stdcos(x) std::cos(x)
#define stdlog(x) std::log(x)

/* check iterpolation vs. original function */
int
check_spline(Spline<double> const& s__, std::function<double(double)> f__, double x0__, double x1__)
{
    int const np{10000};
    Radial_grid_lin<double> rgrid(np, x0__, x1__);
    for (int ir = 0; ir < rgrid.num_points(); ir++) {
        double x    = rgrid[ir];
        double diff = rel_diff(s__.at_point(x), f__(x));
        if (diff > 1e-6) {
            std::cout << "wrong spline interpolation at x = " << x << std::endl
                      << "true value: " << f__(x) << ", spline value: " << s__.at_point(x) << ", diff: " << diff
                      << std::endl;
            return 1;
        }
    }
    return 0;
}

int
test_function(std::function<double(double)> f__, Radial_grid<double> rgrid = Radial_grid_pow<double>(2000, 1e-7, 8, 2))
{
    Spline<double> s(rgrid, f__);
    return check_spline(s, f__, rgrid.first(), rgrid.last());
}

int
test_spline()
{
    int err{0};

    printf("testing f(x)=1\n");
    err += test_function([](double x) { return 1; });

    printf("testing f(x)=x\n");
    err += test_function([](double x) { return x; });

    printf("testing f(x)=x**2\n");
    err += test_function([](double x) { return stdpow(x, 2); });

    printf("testing f(x)=x**3\n");
    err += test_function([](double x) { return stdpow(x, 3); });

    printf("testing f(x)=1/(1 + x)\n");
    err += test_function([](double x) { return stdpow(1 + x, -1); });

    printf("testing f(x)=(1 + x)**(-2)\n");
    err += test_function([](double x) { return stdpow(1 + x, -2); });

    printf("testing f(x)=Sqrt(x)\n");
    err += test_function([](double x) { return stdpow(x, 0.5); });

    printf("testing f(x)=exp(-x)\n");
    err += test_function([](double x) { return stdexp(-x); });

    printf("testing f(x)=Sin(x)\n");
    err += test_function([](double x) { return stdsin(x); }, Radial_grid_lin<double>(1000, 1e-7, 8));

    printf("testing f(x)=Sin(2*x)\n");
    err += test_function([](double x) { return stdsin(2 * x); }, Radial_grid_lin<double>(1000, 1e-7, 8));

    printf("testing f(x)=exp(x)\n");
    err += test_function([](double x) { return stdexp(x); });

    printf("testing f(x)=exp(-x**2)\n");
    err += test_function([](double x) { return stdexp(-stdpow(x, 2)); });

    printf("testing f(x)=exp(x - x**2)\n");
    err += test_function([](double x) { return stdexp(x - stdpow(x, 2)); });

    printf("testing f(x)=x/exp(x)\n");
    err += test_function([](double x) { return x / stdexp(x); });

    printf("testing f(x)=x**2/exp(x)\n");
    err += test_function([](double x) { return stdpow(x, 2) / stdexp(x); });

    printf("testing f(x)=Log(1 + x)\n");
    err += test_function([](double x) { return stdlog(1 + x); });

    printf("testing f(x)=Sin(x)/exp(2*x)\n");
    err += test_function([](double x) { return stdsin(x) / stdexp(2 * x); });

    printf("testing f(x)=Sin(x)/x\n");
    err += test_function([](double x) { return stdsin(x) / x; });

    return err;
}

int
main(int argn, char** argv)
{
    return call_test(argv[0], test_spline);
}
