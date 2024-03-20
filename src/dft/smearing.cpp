/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cmath>
#include "smearing.hpp"
#include "core/constants.hpp"
#include "core/sf/specfunc.hpp"

namespace sirius {

namespace smearing {

const double pi = 3.1415926535897932385;

const double sqrt2 = std::sqrt(2.0);

double
gaussian::delta(double x__, double width__)
{
    double t = std::pow(x__ / width__, 2);
    return std::exp(-t) / std::sqrt(pi) / width__;
}

double
gaussian::occupancy(double x__, double width__)
{
    return 0.5 * (1 + std::erf(x__ / width__));
}

double
gaussian::entropy(double x__, double width__)
{
    double t = std::pow(x__ / width__, 2);
    return -std::exp(-t) * width__ / 2.0 / std::sqrt(pi);
}

double
fermi_dirac::delta(double x__, double width__)
{
    double t = x__ / 2.0 / width__;
    return 1.0 / std::pow(std::exp(t) + std::exp(-t), 2) / width__;
}

double
fermi_dirac::occupancy(double x__, double width__)
{
    return 1.0 - 1.0 / (1.0 + std::exp(x__ / width__));
}

double
fermi_dirac::entropy(double x__, double width__)
{
    double t = x__ / width__;
    double f = 1.0 / (1.0 + std::exp(t));
    if (std::abs(f - 1.0) * std::abs(f) < 1e-16) {
        return 0;
    }
    return width__ * ((1 - f) * std::log(1 - f) + f * std::log(f));
}

/** Second derivative of occupation function.
 * \f[
 *  -\frac{e^{x/w} \left(e^{x/w}-1\right)}{w^2 \left(e^{x/w}+1\right)^3}
 * \f]
 */
double
fermi_dirac::dxdelta(double x__, double width__)
{
    double exw = std::exp(x__ / width__);
    double w2  = width__ * width__;
    return -exw * (exw - 1) / (std::pow(1 + exw, 3) * w2);
}

double
cold::occupancy(double x__, double width__)
{
    double x  = x__ / width__ - 1.0 / sqrt2;
    double x2 = x * x;
    double f  = std::erf(x) / 2.0 + 0.5;
    if (x2 > 200)
        return f;
    return f + std::exp(-x2) / std::sqrt(2 * pi);
}

double
cold::delta(double x__, double width__)
{
    double x  = x__ / width__ - 1.0 / sqrt2;
    double x2 = x * x;
    if (x2 > 700)
        return 0;
    return std::exp(-x2) * (2 * width__ - sqrt2 * x__) / std::sqrt(pi) / width__ / width__;
}

/** Second derivative of the occupation function \f$f(x,w)\f$.
 *   \f[
 *     \frac{\partial^2 f(x,w)}{\partial x^2} = \frac{e^{-y^2} \left(2 \sqrt{2} y^2-2 y-\sqrt{2}\right)}{\sqrt{\pi }
 * w^2}, \qquad y=\frac{x}{w} - \frac{1}{\sqrt{2}} \f]
 */
double
cold::dxdelta(double x__, double width__)
{
    double sqrt2 = std::sqrt(2.0);
    double z     = x__ / width__ - 1 / sqrt2;
    double z2    = z * z;
    if (z2 > 700)
        return 0;
    double expmz2 = std::exp(-z2);
    return expmz2 * (-sqrt2 - 2 * z + 2 * sqrt2 * z * z) / std::sqrt(pi) / width__ / width__;
}

double
cold::entropy(double x__, double width__)
{
    double x  = x__ / width__ - 1.0 / sqrt2;
    double x2 = x * x;
    if (x2 > 700)
        return 0;
    return -std::exp(-x2) * (width__ - sqrt2 * x__) / 2 / std::sqrt(pi);
}

/**
   Coefficients \f$A_n\f$ required to compute the MP-smearing:
   \f[
   \frac{(-1)^n}{n! 4^n \sqrt{\pi}}
   \f]
 */
double
mp_coefficients(int n)
{
    double sqrtpi = std::sqrt(pi);
    int sign      = n % 2 == 0 ? 1 : -1;
    return sign / tgamma(n + 1) / std::pow(4, n) / sqrtpi;
}

double
methfessel_paxton::occupancy(double x__, double width__, int n__)
{
    double z = -x__ / width__;
    double result{0};
    result = 0.5 * (1 - std::erf(z));
    for (int i = 1; i <= n__; ++i) {
        double A = mp_coefficients(i);
        result += A * sf::hermiteh(2 * i - 1, z) * std::exp(-z * z);
    }
    if (result < 1e-30) {
        return 0;
    }
    return result;
}

double
methfessel_paxton::delta(double x__, double width__, int n__)
{
    double z      = -x__ / width__;
    double result = -std::exp(-z * z) / std::sqrt(pi) / width__ * (-1);
    for (int i = 1; i <= n__; ++i) {
        double A = mp_coefficients(i);
        result += A * sf::hermiteh(2 * i, z) * std::exp(-z * z);
    }
    return result;
}

double
methfessel_paxton::dxdelta(double x__, double width__, int n__)
{
    double z      = -x__ / width__;
    double result = 2 * std::exp(-z * z) * z / std::sqrt(pi) / (width__ * width__);
    for (int i = 1; i <= n__; ++i) {
        double A = mp_coefficients(i);
        result += A * sf::hermiteh(2 * i + 1, z) * std::exp(-z * z);
    }
    return result;
}

double
methfessel_paxton::entropy(double x__, double width__, int n__)
{
    // see Moodules/w1gauss.f90:74 in function w1gauss (QE code)
    double x   = x__ / width__;
    double arg = std::min(200.0, x * x);
    double S   = -0.5 * std::exp(-arg) / std::sqrt(pi);
    if (n__ == 0)
        return S;
    double hd{0};
    double hp = std::exp(-arg);
    int ni    = 0;
    double a  = 1 / std::sqrt(pi);
    for (int i = 1; i <= n__; ++i) {
        hd = 2 * x * hp - 2 * ni * hd;
        ni += 1;
        double hpm1 = hp;
        hp          = 2 * x * hd - 2 * ni * hp;
        ni += 1;
        a = -a / (i + 4.0);
        S = S - a * (0.5 * hp + ni * hpm1);
    }
    return S;
}

} // namespace smearing

} // namespace sirius
