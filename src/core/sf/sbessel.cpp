/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gsl/gsl_sf_bessel.h>
#include <cmath>
#include <cassert>

#include "sbessel.hpp"

namespace sirius {

namespace sf {

// compute the spherical bessel functions.
// This implementation is faster than the one provided by GSL, but not necessarily as accurate. For small input, GSL is
// used as a fallback.
static void
custom_bessel(int lmax, double x, double* result)
{
    if (x == 0.0) {
        result[0] = 1.0;
        for (int l = 1; l <= lmax; ++l) {
            result[l] = 0.0;
        }
    } else if (x < 0.1) {
        /* gsl is more accurate for small inputs */
        gsl_sf_bessel_jl_array(lmax, x, result);
    } else {
        const double x_inv = 1.0 / x;
        const double sin_x = std::sin(x);
        result[0]          = sin_x * x_inv;

        if (lmax > 0) {
            result[1] = sin_x * x_inv * x_inv - std::cos(x) * x_inv;
        }

        for (int l = 2; l <= lmax; ++l) {
            result[l] = (2 * (l - 1) + 1) / x * result[l - 1] - result[l - 2];
        }
    }

    /* compare result with gsl in debug mode */
#ifndef NDEBUG
    std::vector<double> ref_result(lmax + 1);
    gsl_sf_bessel_jl_array(lmax, x, ref_result.data());
    for (int l = 0; l <= lmax; ++l) {
        assert(std::abs(result[l] - ref_result[l]) < 1e-6);
    }
#endif
}

Spherical_Bessel_functions::Spherical_Bessel_functions(int lmax__, Radial_grid<double> const& rgrid__, double q__)
    : q_(q__)
    , rgrid_(&rgrid__)
{
    assert(q_ >= 0);

    sbessel_ = std::vector<Spline<double>>(lmax__ + 2);
    for (int l = 0; l <= lmax__ + 1; l++) {
        sbessel_[l] = Spline<double>(rgrid__);
    }

    std::vector<double> jl(lmax__ + 2);
    for (int ir = 0; ir < rgrid__.num_points(); ir++) {
        double t = rgrid__[ir] * q__;
        custom_bessel(lmax__ + 1, t, &jl[0]);
        for (int l = 0; l <= lmax__ + 1; l++) {
            sbessel_[l](ir) = jl[l];
        }
    }

    for (int l = 0; l <= lmax__ + 1; l++) {
        sbessel_[l].interpolate();
    }
}

void
Spherical_Bessel_functions::sbessel(int lmax__, double t__, double* jl__)
{
    gsl_sf_bessel_jl_array(lmax__, t__, jl__);
}

void
Spherical_Bessel_functions::sbessel_deriv_q(int lmax__, double q__, double x__, double* jl_dq__)
{
    std::vector<double> jl(lmax__ + 2);
    /* compute Bessel functions */
    sbessel(lmax__ + 1, x__ * q__, &jl[0]);

    for (int l = 0; l <= lmax__; l++) {
        if (q__ != 0) {
            jl_dq__[l] = (l / q__) * jl[l] - x__ * jl[l + 1];
        } else {
            if (l == 1) {
                jl_dq__[l] = x__ / 3;
            } else {
                jl_dq__[l] = 0;
            }
        }
    }
}

Spline<double> const&
Spherical_Bessel_functions::operator[](int l__) const
{
    return sbessel_[l__];
}

Spline<double>
Spherical_Bessel_functions::deriv_q(int l__)
{
    assert(q_ >= 0);
    Spline<double> s(*rgrid_);
    if (q_ != 0) {
        for (int ir = 0; ir < rgrid_->num_points(); ir++) {
            s(ir) = (l__ / q_) * sbessel_[l__](ir) - (*rgrid_)[ir] * sbessel_[l__ + 1](ir);
        }
    } else {
        if (l__ == 1) {
            for (int ir = 0; ir < rgrid_->num_points(); ir++) {
                s(ir) = (*rgrid_)[ir] / 3;
            }
        }
    }
    s.interpolate();
    return s;
}

} // namespace sf

} // namespace sirius
