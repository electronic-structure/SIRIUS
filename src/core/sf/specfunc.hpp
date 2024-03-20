/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __SPECFUNC_HPP__
#define __SPECFUNC_HPP__

/** \file specfunc.hpp
 *
 *  \brief Special functions.
 */
#include <vector>
#include <cmath>
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_hermite.h>
#include "core/typedefs.hpp"
#include "core/constants.hpp"
#include "core/rte/rte.hpp"
#include "core/r3/r3.hpp"
#include "core/memory.hpp"

namespace sirius {

/// Special functions.
namespace sf {

/// Maximum number of \f$ \ell, m \f$ combinations for a given \f$ \ell_{max} \f$
inline int
lmmax(int lmax)
{
    return (lmax + 1) * (lmax + 1);
}

/// Get composite lm index by angular index l and azimuthal index m.
inline int
lm(int l, int m)
{
    return (l * l + l + m);
}

/// Get maximum orbital quantum number by the maximum lm index.
inline int
lmax(int lmmax__)
{
    RTE_ASSERT(lmmax__ >= 0);
    int lmax = static_cast<int>(std::sqrt(static_cast<double>(lmmax__)) + 1e-8) - 1;
    if (lmmax(lmax) != lmmax__) {
        std::stringstream s;
        s << "wrong lmmax: " << lmmax__;
        RTE_THROW(s);
    }
    return lmax;
}

/// Get array of orbital quantum numbers for each lm component.
inline std::vector<int>
l_by_lm(int lmax__)
{
    std::vector<int> v(lmmax(lmax__));
    for (int l = 0; l <= lmax__; l++) {
        for (int m = -l; m <= l; m++) {
            v[lm(l, m)] = l;
        }
    }
    return v;
}

inline double
hermiteh(int n, double x)
{
    // phycisists Hermite polynomials,
    // https://www.gnu.org/software/gsl/doc/html/specfunc.html#c.gsl_sf_hermite
    return gsl_sf_hermite(n, x);
}

inline std::vector<double>
hermiteh_array(int n, double x)
{
    std::vector<double> result(n);
    gsl_sf_hermite_array(n, x, result.data());
    return result;
}

inline double
hermiteh_series(int n, double x, const double* a)
{
    return gsl_sf_hermite_series(n, x, a);
}

/// Generate associated Legendre polynomials.
/** Normalised associated Legendre polynomials obey the following recursive relations:
    \f[
    P_{m}^{m}(x) = -\sqrt{1 + \frac{1}{2m}} y P_{m-1}^{m-1}(x)
    \f]
    \f[
    P_{m+1}^{m}(x) = \sqrt{2 m + 3} x P_{m}^{m}(x)
    \f]
    \f[
    P_{\ell}^{m}(x) = a_{\ell}^{m}\big(xP_{\ell-1}^{m}(x) - b_{\ell}^{m}P_{\ell - 2}^{m}(x)\big)
    \f]
    where
    \f{eqnarray*}{
    a_{\ell}^{m} &=& \sqrt{\frac{4 \ell^2 - 1}{\ell^2 - m^2}} \\
    b_{\ell}^{m} &=& \sqrt{\frac{(\ell-1)^2-m^2}{4(\ell-1)^2-1}} \\
    x &=& \cos \theta \\
    y &=& \sin \theta
    \f}
    and
    \f[
    P_{0}^{0} = \sqrt{\frac{1}{4\pi}}
    \f]
 */
template <typename T, typename F>
inline void
legendre_plm(int lmax__, double x__, F&& ilm__, T* plm__)
{
    /* reference paper:
       Associated Legendre Polynomials and Spherical Harmonics Computation for Chemistry Applications
       Taweetham Limpanuparb, Josh Milthorpe
       https://arxiv.org/abs/1410.1748
    */

    /*
    when computing recurrent relations, keep this picture in mind:
    ...
    l=5 m=0,1,2,3,4,5
    l=4 m=0,1,2,3,4
    l=3 m=0,1,2,3
    l=2 m=0,1,2
    l=1 m=0,1
    l=0 m=0
    */

    double y = std::sqrt(1 - x__ * x__);

    plm__[ilm__(0, 0)] = 0.28209479177387814347; // 1.0 / std::sqrt(fourpi)

    /* compute P_{l,l} (diagonal) */
    for (int l = 1; l <= lmax__; l++) {
        plm__[ilm__(l, l)] = -std::sqrt(1 + 0.5 / l) * y * plm__[ilm__(l - 1, l - 1)];
    }
    /* compute P_{l+1,l} (upper diagonal) */
    for (int l = 0; l < lmax__; l++) {
        plm__[ilm__(l + 1, l)] = std::sqrt(2.0 * l + 3) * x__ * plm__[ilm__(l, l)];
    }
    for (int m = 0; m <= lmax__ - 2; m++) {
        for (int l = m + 2; l <= lmax__; l++) {
            double alm = std::sqrt(static_cast<double>((2 * l - 1) * (2 * l + 1)) / (l * l - m * m));
            double blm = std::sqrt(static_cast<double>((l - 1 - m) * (l - 1 + m)) / ((2 * l - 3) * (2 * l - 1)));
            plm__[ilm__(l, m)] = alm * (x__ * plm__[ilm__(l - 1, m)] - blm * plm__[ilm__(l - 2, m)]);
        }
    }
}

/// Generate auxiliary Legendre polynomials which are necessary for spherical harmomic derivatives.
/** Generate the following functions:
    \f{eqnarray*}{
    P_{\ell m}^1(x) &=& y P_{\ell}^{m}(x)' \\
    P_{\ell m}^2(x) &=& \frac{P_{\ell}^{m}(x)}{y}
    \f}
    where \f$ x = \cos \theta \f$, \f$ y = \sin \theta \f$ and \f$ P_{\ell}^{m}(x) \f$ are normalized Legendre
    polynomials.

    Both functions obey the recursive relations similar to \f$ P_{\ell}^{m}(x) \f$ (see sirius::legendre_plm() for
    details). For \f$ P_{\ell m}^1(x)  \f$ this is:
    \f[
    P_{m m}^1(x) = \sqrt{1 + \frac{1}{2m}} \Big( -y P_{m-1,m-1}^{1}(x)' + x P_{m-1}^{m-1}(x) \Big)
    \f]
    \f[
    P_{m+1, m}^1(x) = \sqrt{2 m + 3} \Big( x P_{m,m}^{1}(x) + y P_{m}^{m}(x) \Big)
    \f]
    \f[
    P_{\ell, m}^{1}(x) = a_{\ell}^{m}\Big(x P_{\ell-1,m}^{1}(x) + yP_{\ell-1}^{m}(x) -
     b_{\ell}^{m} P_{\ell - 2,m}^{1}(x) \Big)
    \f]
    where
    \f[
    y = \sqrt{1 - x^2}
    \f]
    and
    \f[
    P_{0,0}^1 = 0
    \f]

    And for \f$ P_{\ell m}^2(x)  \f$ the recursion is:
    \f[
    P_{m,m}^2(x) = -\sqrt{1 + \frac{1}{2m}} P_{m-1}^{m-1}(x)
    \f]
    \f[
    P_{m+1, m}^2(x) = \sqrt{2 m + 3} x P_{m,m}^{2}(x)
    \f]
    \f[
    P_{\ell, m}^2(x) = a_{\ell}^{m}\Big(x P_{\ell-1,m}^{2}(x) - b_{\ell}^{m}P_{\ell - 2,m}^{2}(x) \Big)
    \f]
    where
    \f[
    P_{0,0}^2 = 0
    \f]

    See sirius::legendre_plm() for basic definitions.
 */
template <typename T, typename F>
inline void
legendre_plm_aux(int lmax__, double x__, F&& ilm__, T const* plm__, T* p1lm__, T* p2lm__)
{
    double y = std::sqrt(1 - x__ * x__);

    p1lm__[ilm__(0, 0)] = 0;
    p2lm__[ilm__(0, 0)] = 0;

    for (int l = 1; l <= lmax__; l++) {
        auto a = std::sqrt(1 + 0.5 / l);
        auto b = plm__[ilm__(l - 1, l - 1)];
        /* compute y P_{l,l}' (diagonal) */
        p1lm__[ilm__(l, l)] = a * (-y * p1lm__[ilm__(l - 1, l - 1)] + x__ * b);
        /* compute P_{l,l}' / y (diagonal) */
        p2lm__[ilm__(l, l)] = -a * b;
    }
    for (int l = 0; l < lmax__; l++) {
        auto a = std::sqrt(2.0 * l + 3);
        /* compute y P_{l+1,l}' (upper diagonal) */
        p1lm__[ilm__(l + 1, l)] = a * (x__ * p1lm__[ilm__(l, l)] + y * plm__[ilm__(l, l)]);
        /* compute P_{l+1,l}' / y (upper diagonal) */
        p2lm__[ilm__(l + 1, l)] = a * x__ * p2lm__[ilm__(l, l)];
    }
    for (int m = 0; m <= lmax__ - 2; m++) {
        for (int l = m + 2; l <= lmax__; l++) {
            double alm = std::sqrt(static_cast<double>((2 * l - 1) * (2 * l + 1)) / (l * l - m * m));
            double blm = std::sqrt(static_cast<double>((l - 1 - m) * (l - 1 + m)) / ((2 * l - 3) * (2 * l - 1)));
            p1lm__[ilm__(l, m)] =
                    alm * (x__ * p1lm__[ilm__(l - 1, m)] + y * plm__[ilm__(l - 1, m)] - blm * p1lm__[ilm__(l - 2, m)]);
            p2lm__[ilm__(l, m)] = alm * (x__ * p2lm__[ilm__(l - 1, m)] - blm * p2lm__[ilm__(l - 2, m)]);
        }
    }
}

/// Reference implementation of complex spherical harmonics.
/** Complex spherical harmonics are defined as:
    \f[
    Y_{\ell m}(\theta,\phi) = P_{\ell}^{m}(\cos \theta) e^{im\phi}
    \f]
    where \f$P_{\ell}^m(x) \f$ are associated Legendre polynomials.

    Mathematica code:
    \verbatim
    norm[l_, m_] := 4*Pi*Integrate[LegendreP[l, m, x]*LegendreP[l, m, x], {x, 0, 1}]
    Ylm[l_, m_, t_, p_] := LegendreP[l, m, Cos[t]]*E^(I*m*p)/Sqrt[norm[l, m]]
    Do[Print[ComplexExpand[
     FullSimplify[SphericalHarmonicY[l, m, t, p] - Ylm[l, m, t, p],
      Assumptions -> {0 <= t <= Pi}]]], {l, 0, 5}, {m, -l, l}]
    \endverbatim

    Complex spherical harmonics obey the following symmetry:
    \f[
    Y_{\ell -m}(\theta,\phi) = (-1)^m Y_{\ell m}^{*}(\theta,\phi)
    \f]
    Mathematica code:
    \verbatim
    Do[Print[ComplexExpand[
     FullSimplify[
      SphericalHarmonicY[l, -m, t, p] - (-1)^m*
       Conjugate[SphericalHarmonicY[l, m, t, p]],
        Assumptions -> {0 <= t <= Pi}]]], {l, 0, 4}, {m, 0, l}]
    \endverbatim
 */
inline void
spherical_harmonics_ref(int lmax, double theta, double phi, std::complex<double>* ylm)
{
    double x = std::cos(theta);

    std::vector<double> result_array(gsl_sf_legendre_array_n(lmax));
    gsl_sf_legendre_array(GSL_SF_LEGENDRE_SPHARM, lmax, x, &result_array[0]);

    for (int l = 0; l <= lmax; l++) {
        ylm[sf::lm(l, 0)] = result_array[gsl_sf_legendre_array_index(l, 0)];
    }

    for (int m = 1; m <= lmax; m++) {
        std::complex<double> z = std::exp(std::complex<double>(0.0, m * phi)) * std::pow(-1, m);
        for (int l = m; l <= lmax; l++) {
            ylm[sf::lm(l, m)] = result_array[gsl_sf_legendre_array_index(l, m)] * z;
            if (m % 2) {
                ylm[sf::lm(l, -m)] = -std::conj(ylm[sf::lm(l, m)]);
            } else {
                ylm[sf::lm(l, -m)] = std::conj(ylm[sf::lm(l, m)]);
            }
        }
    }
}

/// Optimized implementation of complex spherical harmonics.
inline void
spherical_harmonics(int lmax, double theta, double phi, std::complex<double>* ylm)
{
    double x = std::cos(theta);

    sf::legendre_plm(lmax, x, sf::lm, ylm);

    double c0 = std::cos(phi);
    double c1 = 1;
    double s0 = -std::sin(phi);
    double s1 = 0;
    double c2 = 2 * c0;

    int phase{-1};

    for (int m = 1; m <= lmax; m++) {
        double c = c2 * c1 - c0;
        c0       = c1;
        c1       = c;
        double s = c2 * s1 - s0;
        s0       = s1;
        s1       = s;
        for (int l = m; l <= lmax; l++) {
            double p           = std::real(ylm[sf::lm(l, m)]);
            double p1          = p * phase;
            ylm[sf::lm(l, m)]  = std::complex<double>(p * c, p * s);
            ylm[sf::lm(l, -m)] = std::complex<double>(p1 * c, -p1 * s);
        }
        phase = -phase;
    }
}

/// Reference implementation of real spherical harmonics Rlm
/** Real spherical harminics are defined as:
    \f[
    R_{\ell m}(\theta,\phi) = \left\{
    \begin{array}{lll}
    \sqrt{2} \Re Y_{\ell m}(\theta,\phi) = \sqrt{2} P_{\ell}^{m}(\cos \theta) \cos m\phi & m > 0 \\
    P_{\ell}^{0}(\cos \theta) & m = 0 \\
    \sqrt{2} \Im Y_{\ell m}(\theta,\phi) = \sqrt{2} (-1)^{|m|} P_{\ell}^{|m|}(\cos \theta) (-\sin |m|\phi) & m < 0
    \end{array}
    \right.
    \f]

    Mathematica code:
    \verbatim
    (* definition of real spherical harmonics, use Plm(l,m) for m\
    \[GreaterEqual]0 only *)

    norm[l_, m_] :=
     4*Pi*Integrate[
       LegendreP[l, Abs[m], x]*LegendreP[l, Abs[m], x], {x, 0, 1}]
    legendre[l_, m_, x_] := LegendreP[l, Abs[m], x]/Sqrt[norm[l, m]]

    (* reference definition *)

    RRlm[l_, m_, th_, ph_] :=
     If[m > 0, Sqrt[2]*ComplexExpand[Re[SphericalHarmonicY[l, m, th, ph]]
        ], If[m < 0,
       Sqrt[2]*ComplexExpand[Im[SphericalHarmonicY[l, m, th, ph]]],
       If[m == 0, ComplexExpand[Re[SphericalHarmonicY[l, 0, th, ph]]]]]]

    (* definition without ComplexExpand *)

    Rlm[l_, m_, th_, ph_] :=
     If[m > 0, legendre[l, m, Cos[th]]*Sqrt[2]*Cos[m*ph],
      If[m < 0, (-1)^m*legendre[l, m, Cos[th]]*Sqrt[2]*(-Sin[Abs[m]*ph]),
       If[m == 0, legendre[l, 0, Cos[th]]]]]

    (* check that both definitions are identical *)
    Do[
     Print[FullSimplify[Rlm[l, m, a, b] - RRlm[l, m, a, b],
       Assumptions -> {0 <= a <= Pi, 0 <= b <= 2*Pi}]], {l, 0, 5}, {m, -l,
       l}]

    \endverbatim
 */
inline void
spherical_harmonics_ref(int lmax, double theta, double phi, double* rlm)
{
    /* reference code */
    int lmmax = (lmax + 1) * (lmax + 1);

    std::vector<std::complex<double>> ylm(lmmax);
    sf::spherical_harmonics_ref(lmax, theta, phi, &ylm[0]);

    double const t = std::sqrt(2.0);

    rlm[0] = y00;

    for (int l = 1; l <= lmax; l++) {
        rlm[sf::lm(l, 0)] = ylm[sf::lm(l, 0)].real();
        for (int m = 1; m <= l; m++) {
            rlm[sf::lm(l, m)]  = t * ylm[sf::lm(l, m)].real();
            rlm[sf::lm(l, -m)] = t * ylm[sf::lm(l, -m)].imag();
        }
    }
}

/// Optimized implementation of real spherical harmonics.
inline void
spherical_harmonics(int lmax, double theta, double phi, double* rlm)
{
    double x = std::cos(theta);

    sf::legendre_plm(lmax, x, sf::lm, rlm);

    double c0 = std::cos(phi);
    double c1 = 1;
    double s0 = -std::sin(phi);
    double s1 = 0;
    double c2 = 2 * c0;

    double const t = std::sqrt(2.0);

    int phase{-1};

    for (int m = 1; m <= lmax; m++) {
        double c = c2 * c1 - c0;
        c0       = c1;
        c1       = c;
        double s = c2 * s1 - s0;
        s0       = s1;
        s1       = s;
        for (int l = m; l <= lmax; l++) {
            double p           = rlm[sf::lm(l, m)];
            rlm[sf::lm(l, m)]  = t * p * c;
            rlm[sf::lm(l, -m)] = -t * p * s * phase;
        }
        phase = -phase;
    }
}

/// Generate \f$ \cos(m x) \f$ for m in [1, n] using recursion.
inline mdarray<double, 1>
cosxn(int n__, double x__)
{
    assert(n__ > 0);
    mdarray<double, 1> data({n__});

    double c0 = std::cos(x__);
    double c1 = 1;
    double c2 = 2 * c0;
    for (int m = 0; m < n__; m++) {
        data[m] = c2 * c1 - c0;
        c0      = c1;
        c1      = data[m];
    }
    return data;
}

/// Generate \f$ \sin(m x) \f$ for m in [1, n] using recursion.
inline mdarray<double, 1>
sinxn(int n__, double x__)
{
    assert(n__ > 0);
    mdarray<double, 1> data({n__});

    double s0 = -std::sin(x__);
    double s1 = 0;
    double c2 = 2 * std::cos(x__);

    for (int m = 0; m < n__; m++) {
        data[m] = c2 * s1 - s0;
        s0      = s1;
        s1      = data[m];
    }
    return data;
}

/// Compute the derivatives of real spherical harmonics over the components of cartesian vector.
/** The following derivative is computed:
    \f[
      \frac{\partial R_{\ell m}(\theta_r, \phi_r)}{\partial r_{\mu}} =
        \frac{\partial R_{\ell m}(\theta_r, \phi_r)}{\partial \theta_r} \frac{\partial \theta_r}{\partial r_{\mu}} +
        \frac{\partial R_{\ell m}(\theta_r, \phi_r)}{\partial \phi_r} \frac{\partial \phi_r}{\partial r_{\mu}}
    \f]
    The derivatives of angles are:
    \f[
       \frac{\partial \theta_r}{\partial r_{x}} = \frac{\cos(\phi_r) \cos(\theta_r)}{r} \\
       \frac{\partial \theta_r}{\partial r_{y}} = \frac{\cos(\theta_r) \sin(\phi_r)}{r} \\
       \frac{\partial \theta_r}{\partial r_{z}} = -\frac{\sin(\theta_r)}{r}
    \f]
    and
    \f[
       \frac{\partial \phi_r}{\partial r_{x}} = -\frac{\sin(\phi_r)}{\sin(\theta_r) r} \\
       \frac{\partial \phi_r}{\partial r_{y}} = \frac{\cos(\phi_r)}{\sin(\theta_r) r} \\
       \frac{\partial \phi_r}{\partial r_{z}} = 0
    \f]
    The derivative of \f$ \phi \f$ has discontinuities at \f$ \theta = 0, \theta=\pi \f$. This, however, is not a
   problem, because multiplication by the the derivative of \f$ R_{\ell m} \f$ removes it. The following functions have
   to be hardcoded: \f[
      \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \theta} \\
      \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \phi} \frac{1}{\sin(\theta)}
    \f]

    Spherical harmonics have a separable form:
    \f[
    R_{\ell m}(\theta, \phi) = P_{\ell}^{m}(\cos \theta) f(\phi)
    \f]
    The derivative over \f$ \theta \f$ is then:
    \f[
    \frac{\partial R_{\ell m}(\theta, \phi)}{\partial \theta} = \frac{\partial P_{\ell}^{m}(x)}{\partial x}
      \frac{\partial x}{\partial \theta} f(\phi) = -\sin \theta \frac{\partial P_{\ell}^{m}(x)}{\partial x} f(\phi)
    \f]
    where \f$ x = \cos \theta \f$

    Mathematica script for spherical harmonic derivatives:
    \verbatim
    Rlm[l_, m_, th_, ph_] :=
     If[m > 0, Sqrt[2]*ComplexExpand[Re[SphericalHarmonicY[l, m, th, ph]]],
       If[m < 0, Sqrt[2]*ComplexExpand[Im[SphericalHarmonicY[l, m, th, ph]]],
         If[m == 0, ComplexExpand[Re[SphericalHarmonicY[l, 0, th, ph]]]]
       ]
     ]
    Do[Print[FullSimplify[D[Rlm[l, m, theta, phi], theta]]], {l, 0, 4}, {m, -l, l}]
    Do[Print[FullSimplify[TrigExpand[D[Rlm[l, m, theta, phi], phi]/Sin[theta]]]], {l, 0, 4}, {m, -l, l}]
    \endverbatim
 */
inline void
dRlm_dr(int lmax__, r3::vector<double>& r__, mdarray<double, 2>& data__, bool divide_by_r__ = true)
{
    /* get spherical coordinates of the Cartesian vector */
    auto vrs = r3::spherical_coordinates(r__);

    if (vrs[0] < 1e-12) {
        data__.zero();
        return;
    }

    int lmmax = (lmax__ + 1) * (lmax__ + 1);

    double theta = vrs[1];
    double phi   = vrs[2];

    double sint = std::sin(theta);
    double sinp = std::sin(phi);
    double cost = std::cos(theta);
    double cosp = std::cos(phi);

    /* nominators of angle derivatives */
    r3::vector<double> dtheta_dr({cost * cosp, cost * sinp, -sint});
    r3::vector<double> dphi_dr({-sinp, cosp, 0});

    std::vector<double> dRlm_dt(lmmax);
    std::vector<double> dRlm_dp_sin_t(lmmax);

    std::vector<double> plm((lmax__ + 1) * (lmax__ + 2) / 2);
    std::vector<double> dplm((lmax__ + 1) * (lmax__ + 2) / 2);
    std::vector<double> plm_y((lmax__ + 1) * (lmax__ + 2) / 2);

    auto ilm = [](int l, int m) { return l * (l + 1) / 2 + m; };

    dRlm_dt[0]       = 0;
    dRlm_dp_sin_t[0] = 0;

    /* compute Legendre polynomials */
    sf::legendre_plm(lmax__, cost, ilm, plm.data());
    /* compute sin(theta) * (dPlm/dx)  and Plm / sin(theta) */
    sf::legendre_plm_aux(lmax__, cost, ilm, plm.data(), dplm.data(), plm_y.data());

    double c0 = cosp;
    double c1 = 1;
    double s0 = -sinp;
    double s1 = 0;
    double c2 = 2 * c0;

    double const t = std::sqrt(2.0);

    for (int l = 0; l <= lmax__; l++) {
        dRlm_dt[sf::lm(l, 0)]       = -dplm[ilm(l, 0)];
        dRlm_dp_sin_t[sf::lm(l, 0)] = 0;
    }

    int phase{-1};
    for (int m = 1; m <= lmax__; m++) {
        double c = c2 * c1 - c0;
        c0       = c1;
        c1       = c;
        double s = c2 * s1 - s0;
        s0       = s1;
        s1       = s;
        for (int l = m; l <= lmax__; l++) {
            double p                     = -dplm[ilm(l, m)];
            dRlm_dt[sf::lm(l, m)]        = t * p * c;
            dRlm_dt[sf::lm(l, -m)]       = -t * p * s * phase;
            p                            = plm_y[ilm(l, m)];
            dRlm_dp_sin_t[sf::lm(l, m)]  = -t * p * s * m;
            dRlm_dp_sin_t[sf::lm(l, -m)] = -t * p * c * m * phase;
        }

        phase = -phase;
    }

    if (!divide_by_r__) {
        vrs[0] = 1;
    }

    for (int mu = 0; mu < 3; mu++) {
        for (int lm = 0; lm < lmmax; lm++) {
            data__(lm, mu) = (dRlm_dt[lm] * dtheta_dr[mu] + dRlm_dp_sin_t[lm] * dphi_dr[mu]) / vrs[0];
        }
    }
}

inline void
dRlm_dr_numerical(int lmax__, r3::vector<double>& r__, mdarray<double, 2>& data__, bool divide_by_r__ = true)
{
    /* get spherical coordinates of the Cartesian vector */
    auto vrs = r3::spherical_coordinates(r__);

    if (vrs[0] < 1e-12) {
        data__.zero();
        return;
    }
    double dr = 1e-5 * vrs[0];

    int lmmax = (lmax__ + 1) * (lmax__ + 1);

    auto pref = divide_by_r__ ? (1.0 / vrs[0]) : 1.0;

    for (int x = 0; x < 3; x++) {

        r3::vector<double> v1 = r__;
        v1[x] += dr;

        r3::vector<double> v2 = r__;
        v2[x] -= dr;

        auto vs1 = r3::spherical_coordinates(v1);
        auto vs2 = r3::spherical_coordinates(v2);
        std::vector<double> rlm1(lmmax);
        std::vector<double> rlm2(lmmax);

        sf::spherical_harmonics(lmax__, vs1[1], vs1[2], &rlm1[0]);
        sf::spherical_harmonics(lmax__, vs2[1], vs2[2], &rlm2[0]);

        for (int lm = 0; lm < lmmax; lm++) {
            data__(lm, x) = pref * (rlm1[lm] - rlm2[lm]) / 2 / dr;
        }
    }
}

} // namespace sf

} // namespace sirius

#endif // __SPECFUNC_HPP__
