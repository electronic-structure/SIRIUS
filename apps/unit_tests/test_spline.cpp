/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>

using namespace sirius;

void
check_spline(Spline<double> const& s__, std::function<double(double)> f__, double x0, double x1)
{
    Radial_grid_lin<double> rgrid(10000, x0, x1);
    for (int ir = 0; ir < 10000; ir++) {
        double x = rgrid[ir];
        if (std::abs(s__.at_point(x) - f__(x)) > 1e-10) {
            printf("wrong spline interpolation at x = %18.12f\n", x);
            printf("true value: %18.12f, spline value: %18.12f\n", f__(x), s__.at_point(x));
            exit(1);
        }
    }
}

void
test_spline_1a()
{
    auto f = [](double x) { return std::sin(x) / x; };

    double x0 = 1e-7;
    double x1 = 2.0;

    Radial_grid_lin_exp<double> rgrid(2000, 1e-7, 2);
    Spline<double> s(rgrid, f);

    check_spline(s, f, x0, x1);
    double v = s.integrate(0);

    if (std::abs(v - 1.605412876802697) > 1e-10) {
        printf("wrong integral\n");
        exit(1);
    }
}

void
test_spline_1b()
{
    auto f = [](double x) { return std::exp(-2 * x) * x; };

    double x0 = 1e-7;
    double x1 = 2.0;

    Radial_grid_lin_exp<double> rgrid(2000, 1e-7, 2);
    Spline<double> s(rgrid, f);

    check_spline(s, f, x0, x1);
    double v = s.integrate(0);

    if (std::abs(v - 0.22710545138907753) > 1e-10) {
        printf("wrong integral\n");
        exit(1);
    }
}

void
test_spline_3(std::vector<Spline<double>> const& s, std::function<double(double)> f__, double x0, double x1)
{
    for (int k = 0; k < 10; k++)
        check_spline(s[k], f__, x0, x1);
}

// Test vector of splines.
void
test_spline_2()
{
    auto f = [](double x) { return std::sin(x) / x; };

    double x0 = 1e-7;
    double x1 = 2.0;

    Radial_grid_exp<double> rgrid(2000, 1e-7, 2);
    std::vector<Spline<double>> s(10);

    for (int k = 0; k < 10; k++)
        s[k] = Spline<double>(rgrid, f);
    test_spline_3(s, f, x0, x1);
}

// Test product of splines.
void
test_spline_4()
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
    double d = 0;
    for (int ir = 0; ir < rlin.num_points(); ir++) {
        double x = rlin[ir];
        d += std::pow(s3.at_point(x) - s12.at_point(x), 2);
    }
    d = std::sqrt(d / rlin.num_points());

    printf("RMS diff of spline product: %18.14f\n", d);

    if (d > 1e-6) {
        printf("wrong product of two splines\n");
        exit(1);
    }

    // FILE* fout = fopen("splne_prod.dat", "w");
    // Radial_grid rlin(linear_grid, 10000, 1e-7, 4);
    // for (int i = 0; i < rlin.num_points(); i++)
    //{
    //     double x = rlin[i];
    //     fprintf(fout, "%18.10f %18.10f %18.10f\n", x, s3(x), s12(x));
    // }
    // fclose(fout);
}

void
test_spline_5()
{
    int N = 6000;
    int n = 256;

    Radial_grid_exp<double> rgrid(N, 1e-7, 4);
    std::vector<Spline<double>> s1(n);
    std::vector<Spline<double>> s2(n);
    for (int i = 0; i < n; i++) {
        s1[i] = Spline<double>(rgrid);
        s2[i] = Spline<double>(rgrid);
        for (int ir = 0; ir < N; ir++) {
            s1[i](ir) = std::sin(rgrid[ir] * (1 + n * 0.01)) / rgrid[ir];
            s2[i](ir) = std::exp((1 + n * 0.01) * rgrid[ir]);
        }
        s1[i].interpolate();
        s2[i].interpolate();
    }
    mdarray<double, 2> prod({n, n});
    double t = -wtime();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            prod(i, j) = inner(s1[i], s2[j], 2);
        }
    }
    t += wtime();
    printf("inner product time: %12.6f", t);
    printf("performance: %12.6f GFlops", 1e-9 * n * n * N * 85 / t);
}

void
test_spline_6()
{
    mdarray<Spline<double>, 1> array({20});
    Radial_grid_exp<double> rgrid(300, 1e-7, 4);

    for (int i = 0; i < 20; i++) {
        array(i) = Spline<double>(rgrid);
        for (int ir = 0; ir < rgrid.num_points(); ir++)
            array(i)(ir) = std::exp(-rgrid[ir]);
        array(i).interpolate();
    }
}

void
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

    double d   = s.interpolate().integrate(m);
    double err = std::abs(1 - d / exact_result);

    printf("       relative error: %18.12f", err);
    if (err < 1e-10) {
        printf("  OK\n");
    } else {
        printf("  Fail\n");
        exit(1);
    }
}

// void test2(radial_grid_t grid_type, double x0, double x1)
//{
//     printf("\n");
//     printf("test2: value and derivatives of exp(x)\n");
//
//     int N = 5000;
//     Radial_grid r(grid_type, N, x0, x1);
//     Spline<double> s(r, [](double x){return std::exp(x);});
//
//     printf("grid type : %s\n", r.grid_type_name().c_str());
//
//     //== std::string fname = "grid_" + r.grid_type_name() + ".txt";
//     //== FILE* fout = fopen(fname.c_str(), "w");
//     //== for (int i = 0; i < r.num_points(); i++) fprintf(fout,"%i %16.12e\n", i, r[i]);
//     //== fclose(fout);
//
//     printf("x = %f, true exp(x) = %f, exp(x) = %f, exp'(x)= %f, exp''(x) = %f\n", x0, s[0], s.deriv(0, 0), s.deriv(1,
//     0), s.deriv(2, 0)); printf("x = %f, true exp(x) = %f, exp(x) = %f, exp'(x)= %f, exp''(x) = %f\n", x1, s[N - 1],
//     s.deriv(0, N - 1), s.deriv(1, N - 1), s.deriv(2, N - 1));
// }

void
test3(int m, double x0, double x1, double exact_val)
{
    printf("\n");
    printf("test3\n");

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
        exit(1);
    }
}

void
test5()
{
    printf("\n");
    printf("test5: high-order integration\n");

    int N = 2000;
    Radial_grid_exp<double> r(N, 1e-8, 0.9);
    Spline<double> s(r, [](double x) { return std::log(x); });
    double true_value = -0.012395331058672921;
    if (std::abs(s.integrate(7) - true_value) > 1e-10) {
        printf("wrong high-order integration\n");
        exit(1);
    } else {
        printf("OK\n");
    }
}

void
test6a()
{
    printf("\n");
    printf("test6: 5 points interpolation\n");
    std::vector<double> x = {0, 1, 2, 3, 4};
    Radial_grid_ext<double> r(5, x.data());
    Spline<double> s(r);
    s(0) = 0;
    s(1) = 1;
    s(2) = 0;
    s(3) = 0;
    s(4) = 5;
    s.interpolate();

    // Radial_grid_lin<double> rgrid(10000, 0, 4);
    // FILE* fout = fopen("val.dat", "w+");
    // for (int ir = 0; ir < 10000; ir++)
    //{
    //     double x = rgrid[ir];
    //     double val = s.at_point(x);
    //     fprintf(fout, "%18.10f %18.10f\n", x, val);
    // }
    // fclose(fout);

    double val = s.interpolate().integrate(0);
    if (std::abs(val - 3) > 1e-13) {
        printf("wrong result: %18.12f\n", val);
        exit(1);
    } else {
        printf("OK\n");
    }

    //== int N = 4000;
    //== FILE* fout = fopen("spline_test.dat", "w");
    //== for (int i = 0; i < N; i++)
    //== {
    //==     double t = 3.0 * i / (N - 1);
    //==     fprintf(fout, "%18.12f %18.12f\n", t, s(t));
    //== }
    //== fclose(fout);
}

void
test6()
{
    printf("\n");
    printf("test6: 4 points interpolation\n");
    std::vector<double> x = {0, 1, 2, 3};
    Radial_grid_ext<double> r(4, x.data());
    Spline<double> s(r);
    s(0)       = 0;
    s(1)       = 1;
    s(2)       = 0;
    s(3)       = 0;
    double val = s.interpolate().integrate(0);
    if (std::abs(val - 1.125) > 1e-13) {
        printf("wrong result: %18.12f\n", val);
        exit(1);
    } else {
        printf("OK\n");
    }

    //== int N = 4000;
    //== FILE* fout = fopen("spline_test.dat", "w");
    //== for (int i = 0; i < N; i++)
    //== {
    //==     double t = 3.0 * i / (N - 1);
    //==     fprintf(fout, "%18.12f %18.12f\n", t, s(t));
    //== }
    //== fclose(fout);
}

void
test7(std::function<double(double)> f__, std::function<double(double)> d2f__)
{

    int N = 2000;
    Radial_grid_exp<double> r(N, 1e-7, 4.0);
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
}

// template <typename T>
// void test8(int rgrid_t)
//{
//     int N = 400;
//     double r0 = 1e-6;
//     double r1 = 3.0;
//
//     Radial_grid r(static_cast<radial_grid_t>(rgrid_t), N, r0, r1);
//
//     auto int_s0 = [](double x, T a1, T a2) {
//         return (2*a2 + 2*a1*a2*x + std::pow(a1,2)*(-1 + a2*std::pow(x,2)))/(std::pow(a1,3)*std::exp(a1*x));
//     };
//     auto int_s2 = [](double x, T a1, T a2) {
//         return (24*a2 + 24*a1*a2*x + std::pow(a1,4)*std::pow(x,2)*(-1 + a2*std::pow(x,2)) +
//                2*std::pow(a1,2)*(-1 + 6*a2*std::pow(x,2)) + std::pow(a1,3)*(-2*x + 4*a2*std::pow(x,3)))/
//                (std::pow(a1,5)*std::exp(a1*x));
//     };
//     for (int i1 = 1; i1 < 5; i1++) {
//         for (int i2 = 1; i2 < 5; i2++) {
//             T a1 = i1;
//             T a2 = i2;
//             Spline<T> s(r, [a1, a2](double x){return std::exp(-a1 * x) * (1 - a2 * x * x);});
//
//             if (std::is_same<T, long double>::value) {
//                 printf("test8: diff: %18.12Lf\n", std::abs(s.integrate(0) - (int_s0(r1, a1, a2) - int_s0(r0, a1,
//                 a2)))); printf("test8: diff: %18.12Lf\n", std::abs(s.integrate(2) - (int_s2(r1, a1, a2) - int_s2(r0,
//                 a1, a2))));
//             }
//             if (std::is_same<T, double>::value) {
//                 printf("test8: diff: %18.12f\n", std::abs(s.integrate(0) - (int_s0(r1, a1, a2) - int_s0(r0, a1,
//                 a2)))); printf("test8: diff: %18.12f\n", std::abs(s.integrate(2) - (int_s2(r1, a1, a2) - int_s2(r0,
//                 a1, a2))));
//             }
//         }
//     }
// }

// template <typename T>
// void test9(int rgrid_t)
//{
//     int N = 2000;
//     double r0 = 1e-7;
//     double r1 = 3.0;
//
//     Radial_grid r(static_cast<radial_grid_t>(rgrid_t), N, r0, r1);
//     printf("dx(0) = %18.12f\n", r.dx(0));
//
//     auto f = [](double x){return 1.0 / std::pow(static_cast<T>(x), 2);};
//
//     Spline<T> s(r, f);
//
//     if (rgrid_t == 1) {
//         check_spline(s, f, r0, r1);
//     }
//
//     if (std::is_same<T, double>::value) {
//         printf("test9: diff: %18.12f\n", std::abs(s.integrate(2) - (r1 - r0)));
//     }
//     if (std::is_same<T, long double>::value) {
//         printf("test9: diff: %18.12Lf\n", std::abs(s.integrate(2) - (r1 - r0)));
//     }
// }
// template <typename T>
// void check_spline_1(sirius::experimental::Spline<T, T> const& s__, std::function<T(T)> f__, T x0, T x1)
//{
//     sirius::experimental::Radial_grid_lin<T> rgrid(10000, x0, x1);
//     for (int ir = 0; ir < 10000; ir++) {
//         T x = rgrid[ir];
//         if (std::abs(s__(x) - f__(x)) > 1e-10) {
//             printf("wrong spline interpolation at x = %18.12f\n", x);
//             printf("true value: %18.12f, spline value: %18.12f\n", f__(x), s__(x));
//             exit(1);
//         }
//     }
// }
//
// template <typename T>
// void test10()
//{
//     int N = 4000;
//     T r0 = 1e-7;
//     T r1 = 3.0;
//
//     sirius::experimental::Radial_grid_exp<T> rgrid(N, r0, r1);
//     //auto f = [](T x){return 1.0 / std::pow(static_cast<T>(x), 1);};
//     auto f = [](T x){return std::exp(-x);};
//
//     sirius::experimental::Spline<T, T> s(rgrid, f);
//
//     check_spline_1<T>(s, f, r0, r1);
// }

void
test11()
{
    int N                  = 2000;
    const double exact_val = 0.9794054710686494;
    Radial_grid_exp<double> r(N, 1e-8, 2);
    Spline<double> s1(r, [](double x) { return std::log(0.01 + x); });
    double v1 = s1.integrate(2);
    Spline<double> s2(r, [](double x) { return x * x * std::log(0.01 + x); });
    double v2 = s2.integrate(0);
    printf("test11: v1 - v2   : %18.16f\n", std::abs(v1 - v2));
    printf("        v1 - exact: %18.16f\n", std::abs(v1 - exact_val));
    printf("        v2 - exact: %18.16f\n", std::abs(v2 - exact_val));
}

void
test12()
{
    int N                  = 2000;
    const double exact_val = 0.001999999088970099;
    Radial_grid_exp<double> r(N, 1e-8, 2);
    Spline<double> s1(r, [](double x) { return std::exp(-10 * x); });
    double v1 = s1.integrate(2);
    Spline<double> s2(r, [](double x) { return x * x * std::exp(-10 * x); });
    double v2 = s2.integrate(0);
    printf("test11: v1 - v2   : %18.16f\n", std::abs(v1 - v2));
    printf("        v1 - exact: %18.16f\n", std::abs(v1 - exact_val));
    printf("        v2 - exact: %18.16f\n", std::abs(v2 - exact_val));
}

int
main(int argn, char** argv)
{
    sirius::initialize(1);

    test6a();
    test6();

    test1(0.1, 7.13, 0, 0.3326313127230704);
    test1(0.1, 7.13, 1, -3.973877090504168);
    test1(0.1, 7.13, 2, -23.66503552796384);
    test1(0.1, 7.13, 3, -101.989998166403);
    test1(0.1, 7.13, 4, -341.6457111811293);
    test1(0.1, 7.13, -1, 1.367605245879218);
    test1(0.1, 7.13, -2, 2.710875755556171);
    test1(0.1, 7.13, -3, 9.22907091561693);
    test1(0.1, 7.13, -4, 49.40653515725798);
    test1(0.1, 7.13, -5, 331.7312413927384);

    test_spline_1a();
    test_spline_1b();
    test_spline_2();
    test_spline_4();
    // test_spline_5();
    test_spline_6();

    // double x0 = 0.00001;
    // test2(linear_grid, x0, 2.0);
    // test2(exponential_grid, x0, 2.0);
    // test2(scaled_pow_grid, x0, 2.0);
    // test2(pow2_grid, x0, 2.0);
    // test2(pow3_grid, x0, 2.0);

    test3(1, 0.0001, 2.0, 0.7029943796175838);
    test3(2, 0.0001, 2.0, 1.0365460153117974);

    test5();

    test7([](double x) { return std::pow(x, 2) / std::exp(x); },
          [](double x) { return 2 / std::exp(x) - (4 * x) / std::exp(x) + std::pow(x, 2) / std::exp(x); });
    test7([](double x) { return (100 * (2 * std::pow(x, 3) - 4 * std::pow(x, 5) + std::pow(x, 7))) / std::exp(4 * x); },
          [](double x) {
              return (100 * (12 * x - 80 * std::pow(x, 3) + 42 * std::pow(x, 5))) / std::exp(4 * x) -
                     (800 * (6 * std::pow(x, 2) - 20 * std::pow(x, 4) + 7 * std::pow(x, 6))) / std::exp(4 * x) +
                     (1600 * (2 * std::pow(x, 3) - 4 * std::pow(x, 5) + std::pow(x, 7))) / std::exp(4 * x);
          });
    test7([](double x) { return std::log(0.001 + x); }, [](double x) { return -std::pow(0.001 + x, -2); });
    test7([](double x) { return std::sin(x) / x; },
          [](double x) {
              return (-2 * std::cos(x)) / std::pow(x, 2) + (2 * std::sin(x)) / std::pow(x, 3) - std::sin(x) / x;
          });

    test11();
    test12();

    // for (int i = 0; i < 5; i++) {
    //     printf("grid type: %i\n", i);
    //     printf("testing in double\n");
    //     test8<double>(i);
    //     printf("testing in long double\n");
    //     test8<long double>(i);
    // }

    // for (int i = 0; i < 5; i++) {
    //     printf("grid type: %i\n", i);
    //     printf("testing in double\n");
    //     test9<double>(i);
    //     //printf("testing in long double\n");
    //     //test9<long double>(i);
    // }

    sirius::finalize();

    return 0;
}
