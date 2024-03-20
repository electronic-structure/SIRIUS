/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

void
test1(double x0, double x1, int m, double exact_result)
{
    sirius::RadialGrid r(sirius::exponential_grid, 5000, x0, x1);
    sirius::Spline<double> s(5000, r);

    for (int i = 0; i < 5000; i++)
        s[i] = sin(r[i]);

    s.interpolate();

    double d = s.integrate(m);
    printf("power : %i   numerical result : %18.12f   exact_result : %18.12f   difference : %18.12f\n", m, d,
           exact_result, fabs(d - exact_result));
}

void
test2(double x0, double x1)
{
    sirius::RadialGrid r(sirius::exponential_grid, 5000, x0, x1);
    sirius::Spline<double> s(5000, r);

    for (int i = 0; i < 5000; i++)
        s[i] = exp(r[i]);

    s.interpolate();

    std::cout << s[0] << " " << s.deriv(0, 0) << " " << s.deriv(1, 0) << " " << s.deriv(2, 0) << std::endl;
    std::cout << s[4999] << " " << s.deriv(0, 4999) << " " << s.deriv(1, 4999) << " " << s.deriv(2, 4999) << std::endl;
}

void
test3(double x0, double x1)
{
    printf("test3\n");
    sirius::RadialGrid r(sirius::exponential_grid, 2000, x0, x1);
    sirius::Spline<double> s1(2000, r);
    sirius::Spline<double> s2(2000, r);
    sirius::Spline<double> s3(2000, r);

    for (int i = 0; i < 2000; i++) {
        s1[i] = sin(r[i]) / r[i];
        s2[i] = exp(-r[i]) * pow(r[i], 8.0 / 3.0);
        s3[i] = s1[i] * s2[i];
    }
    s1.interpolate();
    s2.interpolate();
    s3.interpolate();

    double v1 = s3.integrate(2);
    double v2 = sirius::Spline<double>::integrate(&s1, &s2);

    printf("integral values: %16.12f %16.12f, diff %16.12f\n", v1, v2, fabs(v1 - v2));

    mdarray<double, 2> s1_coefs(r.size(), 4);
    s1.get_coefs(s1_coefs.get_ptr(), r.size());

    mdarray<double, 2> s2_coefs(r.size(), 4);
    s2.get_coefs(s2_coefs.get_ptr(), r.size());

    mdarray<double, 2> r_dr(r.size(), 2);
    for (int i = 0; i < r.size() - 1; i++) {
        r_dr(i, 0) = r[i];
        r_dr(i, 1) = r.dr(i);
    }
    r_dr(r.size() - 1, 0) = r[r.size() - 1];

    s1_coefs.allocate_on_device();
    s1_coefs.copy_to_device();
    s2_coefs.allocate_on_device();
    s2_coefs.copy_to_device();
    r_dr.allocate_on_device();
    r_dr.copy_to_device();

    spline_inner_product_gpu<double>(r.size(), r_dr.get_ptr_device(), s1_coefs.get_ptr_device(),
                                     s2_coefs.get_ptr_device());

    s1_coefs.deallocate_on_device();
    s2_coefs.deallocate_on_device();
    r_dr.deallocate_on_device();
}

void
test4(double x0, double x1)
{
    sirius::RadialGrid r(sirius::exponential_grid, 5000, x0, x1);
    sirius::Spline<double> s(5000, r);

    for (int i = 0; i < 5000; i++)
        s[i] = exp(-r[i]) * r[i] * r[i];

    s.interpolate();

    std::cout << s[0] << " " << s.deriv(0, 0) << " " << s.deriv(1, 0) << " " << s.deriv(2, 0) << std::endl;
    std::cout << s[4999] << " " << s.deriv(0, 4999) << " " << s.deriv(1, 4999) << " " << s.deriv(2, 4999) << std::endl;
}

int
main(int argn, char** argv)
{
    Platform::initialize(true);

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

    test2(0.1, 2.0);

    test3(0.0001, 2.0);

    test4(0.0001, 1.892184);
}
