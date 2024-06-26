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

int
test1(double x0, double x1, int m, double exact_result)
{
    printf("\n");
    printf("test1: integrate sin(x) * x^{%i} and compare with exact result\n", m);
    printf("       lower and upper boundaries: %f %f\n", x0, x1);
    Radial_grid_exp<double> r(5000, x0, x1);
    Spline<double> s(r);

    for (int i = 0; i < 5000; i++) {
        s(i) = std::sin(r[i]);
    }

    double d = s.interpolate().integrate(m);
    if (rel_diff(d, exact_result) > 1e-10) {
        std::cout << "wrong result" << std::endl;
        return 1;
    }
    return 0;
}

template <typename F>
int
test2(F&& f, double x0, double x1, double ref_val)
{
    Radial_grid_pow<double> rgrid(2000, x0, x1, 2);
    Spline<double> s(rgrid, f);
    double val = s.integrate(0);
    auto diff  = rel_diff(val, ref_val);
    if (diff > 1e-9) {
        std::cout << "test2: wrong integral" << std::endl
                  << "       val: " << val << ", ref_val: " << ref_val << ", diff: " << diff << std::endl;
        return 1;
    }
    return 0;
}

// Test product of splines.
int
test3()
{
    Radial_grid_lin_exp<double> rgrid(2000, 1e-7, 4);
    Spline<double> s1(rgrid);
    Spline<double> s2(rgrid);
    Spline<double> s3(rgrid);

    for (int ir = 0; ir < 2000; ir++) {
        s1(ir) = std::sin(rgrid[ir] * 2) / rgrid[ir];
        s2(ir) = std::exp(rgrid[ir]);
        s3(ir) = s1(ir) * s2(ir);
    }
    s1.interpolate();
    s2.interpolate();
    s3.interpolate();

    Spline<double> s12 = s1 * s2;

    Radial_grid_lin<double> rlin(20000, 1e-7, 4);
    double d{0};
    for (int ir = 0; ir < rlin.num_points(); ir++) {
        double x = rlin[ir];
        d += std::pow(s3.at_point(x) - s12.at_point(x), 2);
    }
    d = std::sqrt(d / rlin.num_points());

    printf("RMS diff of spline product: %18.14f\n", d);

    if (d > 1e-6) {
        printf("wrong product of two splines\n");
        return 1;
    }
    return 0;
}

int
test4()
{
    printf("\n");
    printf("test4: value and derivatives of exp(x)\n");

    double x0{1};
    double x1{3};

    int N{2000};
    Radial_grid_lin_exp<double> r(N, x0, x1);
    Spline<double> s(r, [](double x) { return std::exp(x); });

    printf("x = %f, true exp(x) = %f, exp(x) = %f, exp'(x)= %f, exp''(x) = %f\n", x0, s(0), s.deriv(0, 0),
           s.deriv(1, 0), s.deriv(2, 0));
    printf("x = %f, true exp(x) = %f, exp(x) = %f, exp'(x)= %f, exp''(x) = %f\n", x1, s(N - 1), s.deriv(0, N - 1),
           s.deriv(1, N - 1), s.deriv(2, N - 1));

    return 0;
}

int
test5(int m, double x0, double x1, double exact_val)
{
    printf("\n");
    printf("test5\n");

    Radial_grid_exp<double> r(2000, x0, x1);
    Spline<double> s1(r);
    Spline<double> s2(r);
    Spline<double> s3(r);

    for (int i = 0; i < 2000; i++) {
        s1(i) = std::sin(r[i]) / r[i];
        s2(i) = std::exp(-r[i]) * std::pow(r[i], 8.0 / 3.0);
        s3(i) = s1(i) * s2(i);
    }
    s1.interpolate();
    s2.interpolate();
    s3.interpolate();

    double v1 = s3.integrate(m);
    double v2 = inner(s1, s2, m);

    printf("interpolate product of two functions and then integrate with spline   : %16.12f\n", v1);
    printf("interpolate two functions and then integrate the product analytically : %16.12f\n", v2);
    printf("                                                           difference : %16.12f\n", std::abs(v1 - v2));
    printf("                                                         exact result : %16.12f\n", exact_val);

    if (std::abs(v1 - v2) > 1e-10) {
        printf("wrong inner product of splines\n");
        return 1;
    }
    return 0;
}

int
test6()
{
    printf("\n");
    printf("test6: high-order integration\n");

    int N = 2000;
    Radial_grid_exp<double> r(N, 1e-8, 0.9);
    Spline<double> s(r, [](double x) { return std::log(x); });
    double true_value = -0.012395331058672921;
    if (std::abs(s.integrate(7) - true_value) > 1e-10) {
        printf("wrong high-order integration\n");
        return 1;
    }
    return 0;
}

int
test7()
{
    printf("\n");
    printf("test7: 5 points interpolation\n");
    std::vector<double> x = {0, 1, 2, 3, 4};
    Radial_grid_ext<double> r(5, x.data());
    Spline<double> s(r);
    s(0) = 0;
    s(1) = 1;
    s(2) = 0;
    s(3) = 0;
    s(4) = 5;
    s.interpolate();

    double val = s.interpolate().integrate(0);
    if (std::abs(val - 3) > 1e-13) {
        printf("wrong result: %18.12f\n", val);
        return 1;
    }
    return 0;
}

int
test8()
{
    printf("\n");
    printf("test8: 4 points interpolation\n");
    std::vector<double> x = {0, 1, 2, 3};
    Radial_grid_ext<double> r(4, x.data());
    Spline<double> s(r);
    s(0)       = 0;
    s(1)       = 1;
    s(2)       = 0;
    s(3)       = 0;
    double val = s.interpolate().integrate(0);
    if (std::abs(val - 1.125) > 1e-13) {
        return 1;
    }
    return 0;
}

int
test9(std::function<double(double)> f__, std::function<double(double)> d2f__)
{
    printf("\n");
    printf("test9: 2nd derivative\n");

    int N = 2000;
    Radial_grid_exp<double> r(N, 1e-2, 4.0);
    Spline<double> s(r, f__);

    Spline<double> s1(r);
    for (int ir = 0; ir < r.num_points(); ir++) {
        s1(ir) = s.deriv(1, ir);
    }
    s1.interpolate();

    double err1{0}, err2{0};
    for (int ir = 0; ir < r.num_points(); ir++) {
        double x   = r[ir];
        double d2s = d2f__(x);
        err1 += std::abs(d2s - s.deriv(2, ir));
        err2 += std::abs(d2s - s1.deriv(1, ir));
    }
    printf("error of 2nd derivative: %18.10f\n", err1);
    printf("error of two 1st derivatives: %18.10f\n", err2);
    if (err1 > err2) {
        printf("two 1st derivatives are better\n");
    } else {
        printf("2nd derivatives is better\n");
    }
    return 0;
}

template <typename F>
int
test10(F&& f, double exact_val)
{
    int N{2000};
    Radial_grid_exp<double> r(N, 1e-8, 2);
    Spline<double> s1(r, f);
    double v1 = s1.integrate(2);
    Spline<double> s2(r, [&](double x) { return x * x * f(x); });
    double v2 = s2.integrate(0);
    printf("test10: v1 - v2   : %18.16f\n", std::abs(v1 - v2));
    printf("        v1 - exact: %18.16f\n", std::abs(v1 - exact_val));
    printf("        v2 - exact: %18.16f\n", std::abs(v2 - exact_val));

    if (std::abs(v1 - v2) > 1e-9) {
        return 1;
    }
    if (std::abs(v1 - exact_val) > 1e-9) {
        return 2;
    }
    if (std::abs(v2 - exact_val) > 1e-9) {
        return 3;
    }

    return 0;
}

int
test_spline()
{
    int ierr{0};
    ierr += test1(0.1, 7.13, 0, 0.3326313127230704);
    ierr += test1(0.1, 7.13, 1, -3.973877090504168);
    ierr += test1(0.1, 7.13, 2, -23.66503552796384);
    ierr += test1(0.1, 7.13, 3, -101.989998166403);
    ierr += test1(0.1, 7.13, 4, -341.6457111811293);
    ierr += test1(0.1, 7.13, -1, 1.367605245879218);
    ierr += test1(0.1, 7.13, -2, 2.710875755556171);
    ierr += test1(0.1, 7.13, -3, 9.22907091561693);
    ierr += test1(0.1, 7.13, -4, 49.40653515725798);
    ierr += test1(0.1, 7.13, -5, 331.7312413927384);

    ierr += test2([](double x) { return std::exp(-2 * x) * x; }, 1e-7, 2.0, 0.22710545138907753);
    ierr += test2([](double x) { return std::exp(x * (1 - x)); }, 1e-7, 4.5, 1.7302343161598133);
    ierr += test2([](double x) { return std::sqrt(x); }, 1e-7, 4.5, 6.363961021111905);

    ierr += test3();

    ierr += test4();

    ierr += test5(1, 0.0001, 2.0, 0.7029943796175838);
    ierr += test5(2, 0.0001, 2.0, 1.0365460153117974);

    ierr += test6();

    ierr += test7();

    ierr += test8();

    test9([](double x) { return std::pow(x, 2) / std::exp(x); },
          [](double x) { return 2 / std::exp(x) - (4 * x) / std::exp(x) + std::pow(x, 2) / std::exp(x); });
    test9([](double x) { return (100 * (2 * std::pow(x, 3) - 4 * std::pow(x, 5) + std::pow(x, 7))) / std::exp(4 * x); },
          [](double x) {
              return (100 * (12 * x - 80 * std::pow(x, 3) + 42 * std::pow(x, 5))) / std::exp(4 * x) -
                     (800 * (6 * std::pow(x, 2) - 20 * std::pow(x, 4) + 7 * std::pow(x, 6))) / std::exp(4 * x) +
                     (1600 * (2 * std::pow(x, 3) - 4 * std::pow(x, 5) + std::pow(x, 7))) / std::exp(4 * x);
          });
    test9([](double x) { return std::log(0.001 + x); }, [](double x) { return -std::pow(0.001 + x, -2); });
    test9([](double x) { return std::sin(x) / x; },
          [](double x) {
              return (-2 * std::cos(x)) / std::pow(x, 2) + (2 * std::sin(x)) / std::pow(x, 3) - std::sin(x) / x;
          });

    ierr += test10([](double x) { return std::log(0.01 + x); }, 0.9794054710686494);
    ierr += test10([](double x) { return std::exp(-10 * x); }, 0.001999999088970099);

    return ierr;
}

int
main(int argn, char** argv)
{
    return call_test(argv[0], test_spline);
}
