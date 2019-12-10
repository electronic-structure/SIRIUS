// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file specfunc.hpp
 *
 *  \brief Special functions.
 */

/// Special functions.
namespace sf {

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
inline void legendre_plm(int lmax__, double x__, F&& ilm__, T* plm__)
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
inline void legendre_plm_aux(int lmax__, double x__, F&& ilm__, T const* plm__, T* p1lm__, T* p2lm__)
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
            p1lm__[ilm__(l, m)] = alm * (x__ * p1lm__[ilm__(l - 1, m)] + y * plm__[ilm__(l - 1, m)] -
                blm * p1lm__[ilm__(l - 2, m)]);
            p2lm__[ilm__(l, m)] = alm * (x__ * p2lm__[ilm__(l - 1, m)] - blm * p2lm__[ilm__(l - 2, m)]);
        }
    }
}

} // namespace sf
