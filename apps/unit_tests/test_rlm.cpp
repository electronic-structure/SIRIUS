/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include <math.h>
#include <complex.h>

using namespace sirius;

/* Generated with the following Mathematica code

Rlm[l_, m_, th_, ph_] :=
 If[m > 0, Sqrt[2]*ComplexExpand[Re[SphericalHarmonicY[l, m, th, ph]]
    ], If[m < 0,
   Sqrt[2]*ComplexExpand[Im[SphericalHarmonicY[l, m, th, ph]]],
   If[m == 0, ComplexExpand[Re[SphericalHarmonicY[l, 0, th, ph]]]]]]

rules = {"Power" -> "std::pow", "Pi" -> "pi", "Sqrt" -> "std::sqrt",
   "Cos" -> "std::cos", "Sin" -> "std::sin",
   "Complex" -> "std::complex<double>"};
Do[
 Print["if (l==", l, " && m == ", m, ") return ",
  StringReplace[
   ToString[
    CForm[
     Rlm[l, m, t, p]
     ]
    ], rules], ";"
  ], {l, 0, 10}, {m, -l, l}]
*/
double
SphericalHarmonicR(int l, int m, double t, double p)
{
    if (l == 0 && m == 0)
        return 1 / (2. * std::sqrt(pi));

    if (l == 1 && m == -1)
        return -(std::sqrt(3 / pi) * std::sin(p) * std::sin(t)) / 2.;

    if (l == 1 && m == 0)
        return (std::sqrt(3 / pi) * std::cos(t)) / 2.;

    if (l == 1 && m == 1)
        return -(std::sqrt(3 / pi) * std::cos(p) * std::sin(t)) / 2.;

    if (l == 2 && m == -2)
        return -(std::sqrt(15 / pi) * std::sin(2 * p) * std::pow(std::sin(t), 2)) / 4.;

    if (l == 2 && m == -1)
        return -(std::sqrt(15 / pi) * std::cos(t) * std::sin(p) * std::sin(t)) / 2.;

    if (l == 2 && m == 0)
        return -std::sqrt(5 / pi) / 4. + (3 * std::sqrt(5 / pi) * std::pow(std::cos(t), 2)) / 4.;

    if (l == 2 && m == 1)
        return -(std::sqrt(15 / pi) * std::cos(p) * std::cos(t) * std::sin(t)) / 2.;

    if (l == 2 && m == 2)
        return (std::sqrt(15 / pi) * std::cos(2 * p) * std::pow(std::sin(t), 2)) / 4.;

    if (l == 3 && m == -3)
        return -(std::sqrt(35 / (2. * pi)) * std::sin(3 * p) * std::pow(std::sin(t), 3)) / 4.;

    if (l == 3 && m == -2)
        return -(std::sqrt(105 / pi) * std::cos(t) * std::sin(2 * p) * std::pow(std::sin(t), 2)) / 4.;

    if (l == 3 && m == -1)
        return std::sqrt(2) * ((std::sqrt(21 / pi) * std::sin(p) * std::sin(t)) / 8. -
                               (5 * std::sqrt(21 / pi) * std::pow(std::cos(t), 2) * std::sin(p) * std::sin(t)) / 8.);

    if (l == 3 && m == 0)
        return (-3 * std::sqrt(7 / pi) * std::cos(t)) / 4. + (5 * std::sqrt(7 / pi) * std::pow(std::cos(t), 3)) / 4.;

    if (l == 3 && m == 1)
        return std::sqrt(2) * ((std::sqrt(21 / pi) * std::cos(p) * std::sin(t)) / 8. -
                               (5 * std::sqrt(21 / pi) * std::cos(p) * std::pow(std::cos(t), 2) * std::sin(t)) / 8.);

    if (l == 3 && m == 2)
        return (std::sqrt(105 / pi) * std::cos(2 * p) * std::cos(t) * std::pow(std::sin(t), 2)) / 4.;

    if (l == 3 && m == 3)
        return -(std::sqrt(35 / (2. * pi)) * std::cos(3 * p) * std::pow(std::sin(t), 3)) / 4.;

    if (l == 4 && m == -4)
        return (-3 * std::sqrt(35 / pi) * std::sin(4 * p) * std::pow(std::sin(t), 4)) / 16.;

    if (l == 4 && m == -3)
        return (-3 * std::sqrt(35 / (2. * pi)) * std::cos(t) * std::sin(3 * p) * std::pow(std::sin(t), 3)) / 4.;

    if (l == 4 && m == -2)
        return std::sqrt(2) * ((3 * std::sqrt(5 / (2. * pi)) * std::sin(2 * p) * std::pow(std::sin(t), 2)) / 8. -
                               (21 * std::sqrt(5 / (2. * pi)) * std::pow(std::cos(t), 2) * std::sin(2 * p) *
                                std::pow(std::sin(t), 2)) /
                                       8.);

    if (l == 4 && m == -1)
        return std::sqrt(2) * ((9 * std::sqrt(5 / pi) * std::cos(t) * std::sin(p) * std::sin(t)) / 8. -
                               (21 * std::sqrt(5 / pi) * std::pow(std::cos(t), 3) * std::sin(p) * std::sin(t)) / 8.);

    if (l == 4 && m == 0)
        return 9 / (16. * std::sqrt(pi)) - (45 * std::pow(std::cos(t), 2)) / (8. * std::sqrt(pi)) +
               (105 * std::pow(std::cos(t), 4)) / (16. * std::sqrt(pi));

    if (l == 4 && m == 1)
        return std::sqrt(2) * ((9 * std::sqrt(5 / pi) * std::cos(p) * std::cos(t) * std::sin(t)) / 8. -
                               (21 * std::sqrt(5 / pi) * std::cos(p) * std::pow(std::cos(t), 3) * std::sin(t)) / 8.);

    if (l == 4 && m == 2)
        return std::sqrt(2) * ((-3 * std::sqrt(5 / (2. * pi)) * std::cos(2 * p) * std::pow(std::sin(t), 2)) / 8. +
                               (21 * std::sqrt(5 / (2. * pi)) * std::cos(2 * p) * std::pow(std::cos(t), 2) *
                                std::pow(std::sin(t), 2)) /
                                       8.);

    if (l == 4 && m == 3)
        return (-3 * std::sqrt(35 / (2. * pi)) * std::cos(3 * p) * std::cos(t) * std::pow(std::sin(t), 3)) / 4.;

    if (l == 4 && m == 4)
        return (3 * std::sqrt(35 / pi) * std::cos(4 * p) * std::pow(std::sin(t), 4)) / 16.;

    if (l == 5 && m == -5)
        return (-3 * std::sqrt(77 / (2. * pi)) * std::sin(5 * p) * std::pow(std::sin(t), 5)) / 16.;

    if (l == 5 && m == -4)
        return (-3 * std::sqrt(385 / pi) * std::cos(t) * std::sin(4 * p) * std::pow(std::sin(t), 4)) / 16.;

    if (l == 5 && m == -3)
        return std::sqrt(2) *
               ((std::sqrt(385 / pi) * std::sin(3 * p) * std::pow(std::sin(t), 3)) / 32. -
                (9 * std::sqrt(385 / pi) * std::pow(std::cos(t), 2) * std::sin(3 * p) * std::pow(std::sin(t), 3)) /
                        32.);

    if (l == 5 && m == -2)
        return std::sqrt(2) *
               ((std::sqrt(1155 / (2. * pi)) * std::cos(t) * std::sin(2 * p) * std::pow(std::sin(t), 2)) / 8. -
                (3 * std::sqrt(1155 / (2. * pi)) * std::pow(std::cos(t), 3) * std::sin(2 * p) *
                 std::pow(std::sin(t), 2)) /
                        8.);

    if (l == 5 && m == -1)
        return std::sqrt(2) *
               (-(std::sqrt(165 / (2. * pi)) * std::sin(p) * std::sin(t)) / 16. +
                (7 * std::sqrt(165 / (2. * pi)) * std::pow(std::cos(t), 2) * std::sin(p) * std::sin(t)) / 8. -
                (21 * std::sqrt(165 / (2. * pi)) * std::pow(std::cos(t), 4) * std::sin(p) * std::sin(t)) / 16.);

    if (l == 5 && m == 0)
        return (15 * std::sqrt(11 / pi) * std::cos(t)) / 16. -
               (35 * std::sqrt(11 / pi) * std::pow(std::cos(t), 3)) / 8. +
               (63 * std::sqrt(11 / pi) * std::pow(std::cos(t), 5)) / 16.;

    if (l == 5 && m == 1)
        return std::sqrt(2) *
               (-(std::sqrt(165 / (2. * pi)) * std::cos(p) * std::sin(t)) / 16. +
                (7 * std::sqrt(165 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 2) * std::sin(t)) / 8. -
                (21 * std::sqrt(165 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 4) * std::sin(t)) / 16.);

    if (l == 5 && m == 2)
        return std::sqrt(2) *
               (-(std::sqrt(1155 / (2. * pi)) * std::cos(2 * p) * std::cos(t) * std::pow(std::sin(t), 2)) / 8. +
                (3 * std::sqrt(1155 / (2. * pi)) * std::cos(2 * p) * std::pow(std::cos(t), 3) *
                 std::pow(std::sin(t), 2)) /
                        8.);

    if (l == 5 && m == 3)
        return std::sqrt(2) *
               ((std::sqrt(385 / pi) * std::cos(3 * p) * std::pow(std::sin(t), 3)) / 32. -
                (9 * std::sqrt(385 / pi) * std::cos(3 * p) * std::pow(std::cos(t), 2) * std::pow(std::sin(t), 3)) /
                        32.);

    if (l == 5 && m == 4)
        return (3 * std::sqrt(385 / pi) * std::cos(4 * p) * std::cos(t) * std::pow(std::sin(t), 4)) / 16.;

    if (l == 5 && m == 5)
        return (-3 * std::sqrt(77 / (2. * pi)) * std::cos(5 * p) * std::pow(std::sin(t), 5)) / 16.;

    if (l == 6 && m == -6)
        return -(std::sqrt(3003 / (2. * pi)) * std::sin(6 * p) * std::pow(std::sin(t), 6)) / 32.;

    if (l == 6 && m == -5)
        return (-3 * std::sqrt(1001 / (2. * pi)) * std::cos(t) * std::sin(5 * p) * std::pow(std::sin(t), 5)) / 16.;

    if (l == 6 && m == -4)
        return std::sqrt(2) * ((3 * std::sqrt(91 / (2. * pi)) * std::sin(4 * p) * std::pow(std::sin(t), 4)) / 32. -
                               (33 * std::sqrt(91 / (2. * pi)) * std::pow(std::cos(t), 2) * std::sin(4 * p) *
                                std::pow(std::sin(t), 4)) /
                                       32.);

    if (l == 6 && m == -3)
        return std::sqrt(2) *
               ((3 * std::sqrt(1365 / pi) * std::cos(t) * std::sin(3 * p) * std::pow(std::sin(t), 3)) / 32. -
                (11 * std::sqrt(1365 / pi) * std::pow(std::cos(t), 3) * std::sin(3 * p) * std::pow(std::sin(t), 3)) /
                        32.);

    if (l == 6 && m == -2)
        return std::sqrt(2) *
               (-(std::sqrt(1365 / pi) * std::sin(2 * p) * std::pow(std::sin(t), 2)) / 64. +
                (9 * std::sqrt(1365 / pi) * std::pow(std::cos(t), 2) * std::sin(2 * p) * std::pow(std::sin(t), 2)) /
                        32. -
                (33 * std::sqrt(1365 / pi) * std::pow(std::cos(t), 4) * std::sin(2 * p) * std::pow(std::sin(t), 2)) /
                        64.);

    if (l == 6 && m == -1)
        return std::sqrt(2) *
               ((-5 * std::sqrt(273 / (2. * pi)) * std::cos(t) * std::sin(p) * std::sin(t)) / 16. +
                (15 * std::sqrt(273 / (2. * pi)) * std::pow(std::cos(t), 3) * std::sin(p) * std::sin(t)) / 8. -
                (33 * std::sqrt(273 / (2. * pi)) * std::pow(std::cos(t), 5) * std::sin(p) * std::sin(t)) / 16.);

    if (l == 6 && m == 0)
        return (-5 * std::sqrt(13 / pi)) / 32. + (105 * std::sqrt(13 / pi) * std::pow(std::cos(t), 2)) / 32. -
               (315 * std::sqrt(13 / pi) * std::pow(std::cos(t), 4)) / 32. +
               (231 * std::sqrt(13 / pi) * std::pow(std::cos(t), 6)) / 32.;

    if (l == 6 && m == 1)
        return std::sqrt(2) *
               ((-5 * std::sqrt(273 / (2. * pi)) * std::cos(p) * std::cos(t) * std::sin(t)) / 16. +
                (15 * std::sqrt(273 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 3) * std::sin(t)) / 8. -
                (33 * std::sqrt(273 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 5) * std::sin(t)) / 16.);

    if (l == 6 && m == 2)
        return std::sqrt(2) *
               ((std::sqrt(1365 / pi) * std::cos(2 * p) * std::pow(std::sin(t), 2)) / 64. -
                (9 * std::sqrt(1365 / pi) * std::cos(2 * p) * std::pow(std::cos(t), 2) * std::pow(std::sin(t), 2)) /
                        32. +
                (33 * std::sqrt(1365 / pi) * std::cos(2 * p) * std::pow(std::cos(t), 4) * std::pow(std::sin(t), 2)) /
                        64.);

    if (l == 6 && m == 3)
        return std::sqrt(2) *
               ((3 * std::sqrt(1365 / pi) * std::cos(3 * p) * std::cos(t) * std::pow(std::sin(t), 3)) / 32. -
                (11 * std::sqrt(1365 / pi) * std::cos(3 * p) * std::pow(std::cos(t), 3) * std::pow(std::sin(t), 3)) /
                        32.);

    if (l == 6 && m == 4)
        return std::sqrt(2) * ((-3 * std::sqrt(91 / (2. * pi)) * std::cos(4 * p) * std::pow(std::sin(t), 4)) / 32. +
                               (33 * std::sqrt(91 / (2. * pi)) * std::cos(4 * p) * std::pow(std::cos(t), 2) *
                                std::pow(std::sin(t), 4)) /
                                       32.);

    if (l == 6 && m == 5)
        return (-3 * std::sqrt(1001 / (2. * pi)) * std::cos(5 * p) * std::cos(t) * std::pow(std::sin(t), 5)) / 16.;

    if (l == 6 && m == 6)
        return (std::sqrt(3003 / (2. * pi)) * std::cos(6 * p) * std::pow(std::sin(t), 6)) / 32.;

    if (l == 7 && m == -7)
        return (-3 * std::sqrt(715 / pi) * std::sin(7 * p) * std::pow(std::sin(t), 7)) / 64.;

    if (l == 7 && m == -6)
        return (-3 * std::sqrt(5005 / (2. * pi)) * std::cos(t) * std::sin(6 * p) * std::pow(std::sin(t), 6)) / 32.;

    if (l == 7 && m == -5)
        return std::sqrt(2) * ((3 * std::sqrt(385 / (2. * pi)) * std::sin(5 * p) * std::pow(std::sin(t), 5)) / 64. -
                               (39 * std::sqrt(385 / (2. * pi)) * std::pow(std::cos(t), 2) * std::sin(5 * p) *
                                std::pow(std::sin(t), 5)) /
                                       64.);

    if (l == 7 && m == -4)
        return std::sqrt(2) *
               ((9 * std::sqrt(385 / (2. * pi)) * std::cos(t) * std::sin(4 * p) * std::pow(std::sin(t), 4)) / 32. -
                (39 * std::sqrt(385 / (2. * pi)) * std::pow(std::cos(t), 3) * std::sin(4 * p) *
                 std::pow(std::sin(t), 4)) /
                        32.);

    if (l == 7 && m == -3)
        return std::sqrt(2) * ((-9 * std::sqrt(35 / (2. * pi)) * std::sin(3 * p) * std::pow(std::sin(t), 3)) / 64. +
                               (99 * std::sqrt(35 / (2. * pi)) * std::pow(std::cos(t), 2) * std::sin(3 * p) *
                                std::pow(std::sin(t), 3)) /
                                       32. -
                               (429 * std::sqrt(35 / (2. * pi)) * std::pow(std::cos(t), 4) * std::sin(3 * p) *
                                std::pow(std::sin(t), 3)) /
                                       64.);

    if (l == 7 && m == -2)
        return std::sqrt(2) *
               ((-45 * std::sqrt(35 / pi) * std::cos(t) * std::sin(2 * p) * std::pow(std::sin(t), 2)) / 64. +
                (165 * std::sqrt(35 / pi) * std::pow(std::cos(t), 3) * std::sin(2 * p) * std::pow(std::sin(t), 2)) /
                        32. -
                (429 * std::sqrt(35 / pi) * std::pow(std::cos(t), 5) * std::sin(2 * p) * std::pow(std::sin(t), 2)) /
                        64.);

    if (l == 7 && m == -1)
        return std::sqrt(2) *
               ((5 * std::sqrt(105 / (2. * pi)) * std::sin(p) * std::sin(t)) / 64. -
                (135 * std::sqrt(105 / (2. * pi)) * std::pow(std::cos(t), 2) * std::sin(p) * std::sin(t)) / 64. +
                (495 * std::sqrt(105 / (2. * pi)) * std::pow(std::cos(t), 4) * std::sin(p) * std::sin(t)) / 64. -
                (429 * std::sqrt(105 / (2. * pi)) * std::pow(std::cos(t), 6) * std::sin(p) * std::sin(t)) / 64.);

    if (l == 7 && m == 0)
        return (-35 * std::sqrt(15 / pi) * std::cos(t)) / 32. +
               (315 * std::sqrt(15 / pi) * std::pow(std::cos(t), 3)) / 32. -
               (693 * std::sqrt(15 / pi) * std::pow(std::cos(t), 5)) / 32. +
               (429 * std::sqrt(15 / pi) * std::pow(std::cos(t), 7)) / 32.;

    if (l == 7 && m == 1)
        return std::sqrt(2) *
               ((5 * std::sqrt(105 / (2. * pi)) * std::cos(p) * std::sin(t)) / 64. -
                (135 * std::sqrt(105 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 2) * std::sin(t)) / 64. +
                (495 * std::sqrt(105 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 4) * std::sin(t)) / 64. -
                (429 * std::sqrt(105 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 6) * std::sin(t)) / 64.);

    if (l == 7 && m == 2)
        return std::sqrt(2) *
               ((45 * std::sqrt(35 / pi) * std::cos(2 * p) * std::cos(t) * std::pow(std::sin(t), 2)) / 64. -
                (165 * std::sqrt(35 / pi) * std::cos(2 * p) * std::pow(std::cos(t), 3) * std::pow(std::sin(t), 2)) /
                        32. +
                (429 * std::sqrt(35 / pi) * std::cos(2 * p) * std::pow(std::cos(t), 5) * std::pow(std::sin(t), 2)) /
                        64.);

    if (l == 7 && m == 3)
        return std::sqrt(2) * ((-9 * std::sqrt(35 / (2. * pi)) * std::cos(3 * p) * std::pow(std::sin(t), 3)) / 64. +
                               (99 * std::sqrt(35 / (2. * pi)) * std::cos(3 * p) * std::pow(std::cos(t), 2) *
                                std::pow(std::sin(t), 3)) /
                                       32. -
                               (429 * std::sqrt(35 / (2. * pi)) * std::cos(3 * p) * std::pow(std::cos(t), 4) *
                                std::pow(std::sin(t), 3)) /
                                       64.);

    if (l == 7 && m == 4)
        return std::sqrt(2) *
               ((-9 * std::sqrt(385 / (2. * pi)) * std::cos(4 * p) * std::cos(t) * std::pow(std::sin(t), 4)) / 32. +
                (39 * std::sqrt(385 / (2. * pi)) * std::cos(4 * p) * std::pow(std::cos(t), 3) *
                 std::pow(std::sin(t), 4)) /
                        32.);

    if (l == 7 && m == 5)
        return std::sqrt(2) * ((3 * std::sqrt(385 / (2. * pi)) * std::cos(5 * p) * std::pow(std::sin(t), 5)) / 64. -
                               (39 * std::sqrt(385 / (2. * pi)) * std::cos(5 * p) * std::pow(std::cos(t), 2) *
                                std::pow(std::sin(t), 5)) /
                                       64.);

    if (l == 7 && m == 6)
        return (3 * std::sqrt(5005 / (2. * pi)) * std::cos(6 * p) * std::cos(t) * std::pow(std::sin(t), 6)) / 32.;

    if (l == 7 && m == 7)
        return (-3 * std::sqrt(715 / pi) * std::cos(7 * p) * std::pow(std::sin(t), 7)) / 64.;

    if (l == 8 && m == -8)
        return (-3 * std::sqrt(12155 / pi) * std::sin(8 * p) * std::pow(std::sin(t), 8)) / 256.;

    if (l == 8 && m == -7)
        return (-3 * std::sqrt(12155 / pi) * std::cos(t) * std::sin(7 * p) * std::pow(std::sin(t), 7)) / 64.;

    if (l == 8 && m == -6)
        return std::sqrt(2) *
               ((std::sqrt(7293 / pi) * std::sin(6 * p) * std::pow(std::sin(t), 6)) / 128. -
                (15 * std::sqrt(7293 / pi) * std::pow(std::cos(t), 2) * std::sin(6 * p) * std::pow(std::sin(t), 6)) /
                        128.);

    if (l == 8 && m == -5)
        return std::sqrt(2) *
               ((3 * std::sqrt(17017 / (2. * pi)) * std::cos(t) * std::sin(5 * p) * std::pow(std::sin(t), 5)) / 64. -
                (15 * std::sqrt(17017 / (2. * pi)) * std::pow(std::cos(t), 3) * std::sin(5 * p) *
                 std::pow(std::sin(t), 5)) /
                        64.);

    if (l == 8 && m == -4)
        return std::sqrt(2) * ((-3 * std::sqrt(1309 / (2. * pi)) * std::sin(4 * p) * std::pow(std::sin(t), 4)) / 128. +
                               (39 * std::sqrt(1309 / (2. * pi)) * std::pow(std::cos(t), 2) * std::sin(4 * p) *
                                std::pow(std::sin(t), 4)) /
                                       64. -
                               (195 * std::sqrt(1309 / (2. * pi)) * std::pow(std::cos(t), 4) * std::sin(4 * p) *
                                std::pow(std::sin(t), 4)) /
                                       128.);

    if (l == 8 && m == -3)
        return std::sqrt(2) *
               ((-3 * std::sqrt(19635 / (2. * pi)) * std::cos(t) * std::sin(3 * p) * std::pow(std::sin(t), 3)) / 64. +
                (13 * std::sqrt(19635 / (2. * pi)) * std::pow(std::cos(t), 3) * std::sin(3 * p) *
                 std::pow(std::sin(t), 3)) /
                        32. -
                (39 * std::sqrt(19635 / (2. * pi)) * std::pow(std::cos(t), 5) * std::sin(3 * p) *
                 std::pow(std::sin(t), 3)) /
                        64.);

    if (l == 8 && m == -2)
        return std::sqrt(2) *
               ((3 * std::sqrt(595 / pi) * std::sin(2 * p) * std::pow(std::sin(t), 2)) / 128. -
                (99 * std::sqrt(595 / pi) * std::pow(std::cos(t), 2) * std::sin(2 * p) * std::pow(std::sin(t), 2)) /
                        128. +
                (429 * std::sqrt(595 / pi) * std::pow(std::cos(t), 4) * std::sin(2 * p) * std::pow(std::sin(t), 2)) /
                        128. -
                (429 * std::sqrt(595 / pi) * std::pow(std::cos(t), 6) * std::sin(2 * p) * std::pow(std::sin(t), 2)) /
                        128.);

    if (l == 8 && m == -1)
        return std::sqrt(2) *
               ((105 * std::sqrt(17 / (2. * pi)) * std::cos(t) * std::sin(p) * std::sin(t)) / 64. -
                (1155 * std::sqrt(17 / (2. * pi)) * std::pow(std::cos(t), 3) * std::sin(p) * std::sin(t)) / 64. +
                (3003 * std::sqrt(17 / (2. * pi)) * std::pow(std::cos(t), 5) * std::sin(p) * std::sin(t)) / 64. -
                (2145 * std::sqrt(17 / (2. * pi)) * std::pow(std::cos(t), 7) * std::sin(p) * std::sin(t)) / 64.);

    if (l == 8 && m == 0)
        return (35 * std::sqrt(17 / pi)) / 256. - (315 * std::sqrt(17 / pi) * std::pow(std::cos(t), 2)) / 64. +
               (3465 * std::sqrt(17 / pi) * std::pow(std::cos(t), 4)) / 128. -
               (3003 * std::sqrt(17 / pi) * std::pow(std::cos(t), 6)) / 64. +
               (6435 * std::sqrt(17 / pi) * std::pow(std::cos(t), 8)) / 256.;

    if (l == 8 && m == 1)
        return std::sqrt(2) *
               ((105 * std::sqrt(17 / (2. * pi)) * std::cos(p) * std::cos(t) * std::sin(t)) / 64. -
                (1155 * std::sqrt(17 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 3) * std::sin(t)) / 64. +
                (3003 * std::sqrt(17 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 5) * std::sin(t)) / 64. -
                (2145 * std::sqrt(17 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 7) * std::sin(t)) / 64.);

    if (l == 8 && m == 2)
        return std::sqrt(2) *
               ((-3 * std::sqrt(595 / pi) * std::cos(2 * p) * std::pow(std::sin(t), 2)) / 128. +
                (99 * std::sqrt(595 / pi) * std::cos(2 * p) * std::pow(std::cos(t), 2) * std::pow(std::sin(t), 2)) /
                        128. -
                (429 * std::sqrt(595 / pi) * std::cos(2 * p) * std::pow(std::cos(t), 4) * std::pow(std::sin(t), 2)) /
                        128. +
                (429 * std::sqrt(595 / pi) * std::cos(2 * p) * std::pow(std::cos(t), 6) * std::pow(std::sin(t), 2)) /
                        128.);

    if (l == 8 && m == 3)
        return std::sqrt(2) *
               ((-3 * std::sqrt(19635 / (2. * pi)) * std::cos(3 * p) * std::cos(t) * std::pow(std::sin(t), 3)) / 64. +
                (13 * std::sqrt(19635 / (2. * pi)) * std::cos(3 * p) * std::pow(std::cos(t), 3) *
                 std::pow(std::sin(t), 3)) /
                        32. -
                (39 * std::sqrt(19635 / (2. * pi)) * std::cos(3 * p) * std::pow(std::cos(t), 5) *
                 std::pow(std::sin(t), 3)) /
                        64.);

    if (l == 8 && m == 4)
        return std::sqrt(2) * ((3 * std::sqrt(1309 / (2. * pi)) * std::cos(4 * p) * std::pow(std::sin(t), 4)) / 128. -
                               (39 * std::sqrt(1309 / (2. * pi)) * std::cos(4 * p) * std::pow(std::cos(t), 2) *
                                std::pow(std::sin(t), 4)) /
                                       64. +
                               (195 * std::sqrt(1309 / (2. * pi)) * std::cos(4 * p) * std::pow(std::cos(t), 4) *
                                std::pow(std::sin(t), 4)) /
                                       128.);

    if (l == 8 && m == 5)
        return std::sqrt(2) *
               ((3 * std::sqrt(17017 / (2. * pi)) * std::cos(5 * p) * std::cos(t) * std::pow(std::sin(t), 5)) / 64. -
                (15 * std::sqrt(17017 / (2. * pi)) * std::cos(5 * p) * std::pow(std::cos(t), 3) *
                 std::pow(std::sin(t), 5)) /
                        64.);

    if (l == 8 && m == 6)
        return std::sqrt(2) *
               (-(std::sqrt(7293 / pi) * std::cos(6 * p) * std::pow(std::sin(t), 6)) / 128. +
                (15 * std::sqrt(7293 / pi) * std::cos(6 * p) * std::pow(std::cos(t), 2) * std::pow(std::sin(t), 6)) /
                        128.);

    if (l == 8 && m == 7)
        return (-3 * std::sqrt(12155 / pi) * std::cos(7 * p) * std::cos(t) * std::pow(std::sin(t), 7)) / 64.;

    if (l == 8 && m == 8)
        return (3 * std::sqrt(12155 / pi) * std::cos(8 * p) * std::pow(std::sin(t), 8)) / 256.;

    if (l == 9 && m == -9)
        return -(std::sqrt(230945 / (2. * pi)) * std::sin(9 * p) * std::pow(std::sin(t), 9)) / 256.;

    if (l == 9 && m == -8)
        return (-3 * std::sqrt(230945 / pi) * std::cos(t) * std::sin(8 * p) * std::pow(std::sin(t), 8)) / 256.;

    if (l == 9 && m == -7)
        return std::sqrt(2) *
               ((3 * std::sqrt(13585 / pi) * std::sin(7 * p) * std::pow(std::sin(t), 7)) / 512. -
                (51 * std::sqrt(13585 / pi) * std::pow(std::cos(t), 2) * std::sin(7 * p) * std::pow(std::sin(t), 7)) /
                        512.);

    if (l == 9 && m == -6)
        return std::sqrt(2) *
               ((3 * std::sqrt(40755 / pi) * std::cos(t) * std::sin(6 * p) * std::pow(std::sin(t), 6)) / 128. -
                (17 * std::sqrt(40755 / pi) * std::pow(std::cos(t), 3) * std::sin(6 * p) * std::pow(std::sin(t), 6)) /
                        128.);

    if (l == 9 && m == -5)
        return std::sqrt(2) *
               ((-3 * std::sqrt(2717 / pi) * std::sin(5 * p) * std::pow(std::sin(t), 5)) / 256. +
                (45 * std::sqrt(2717 / pi) * std::pow(std::cos(t), 2) * std::sin(5 * p) * std::pow(std::sin(t), 5)) /
                        128. -
                (255 * std::sqrt(2717 / pi) * std::pow(std::cos(t), 4) * std::sin(5 * p) * std::pow(std::sin(t), 5)) /
                        256.);

    if (l == 9 && m == -4)
        return std::sqrt(2) *
               ((-3 * std::sqrt(95095 / (2. * pi)) * std::cos(t) * std::sin(4 * p) * std::pow(std::sin(t), 4)) / 128. +
                (15 * std::sqrt(95095 / (2. * pi)) * std::pow(std::cos(t), 3) * std::sin(4 * p) *
                 std::pow(std::sin(t), 4)) /
                        64. -
                (51 * std::sqrt(95095 / (2. * pi)) * std::pow(std::cos(t), 5) * std::sin(4 * p) *
                 std::pow(std::sin(t), 4)) /
                        128.);

    if (l == 9 && m == -3)
        return std::sqrt(2) *
               ((std::sqrt(21945 / pi) * std::sin(3 * p) * std::pow(std::sin(t), 3)) / 256. -
                (39 * std::sqrt(21945 / pi) * std::pow(std::cos(t), 2) * std::sin(3 * p) * std::pow(std::sin(t), 3)) /
                        256. +
                (195 * std::sqrt(21945 / pi) * std::pow(std::cos(t), 4) * std::sin(3 * p) * std::pow(std::sin(t), 3)) /
                        256. -
                (221 * std::sqrt(21945 / pi) * std::pow(std::cos(t), 6) * std::sin(3 * p) * std::pow(std::sin(t), 3)) /
                        256.);

    if (l == 9 && m == -2)
        return std::sqrt(2) *
               ((21 * std::sqrt(1045 / pi) * std::cos(t) * std::sin(2 * p) * std::pow(std::sin(t), 2)) / 128. -
                (273 * std::sqrt(1045 / pi) * std::pow(std::cos(t), 3) * std::sin(2 * p) * std::pow(std::sin(t), 2)) /
                        128. +
                (819 * std::sqrt(1045 / pi) * std::pow(std::cos(t), 5) * std::sin(2 * p) * std::pow(std::sin(t), 2)) /
                        128. -
                (663 * std::sqrt(1045 / pi) * std::pow(std::cos(t), 7) * std::sin(2 * p) * std::pow(std::sin(t), 2)) /
                        128.);

    if (l == 9 && m == -1)
        return std::sqrt(2) *
               ((-21 * std::sqrt(95 / (2. * pi)) * std::sin(p) * std::sin(t)) / 256. +
                (231 * std::sqrt(95 / (2. * pi)) * std::pow(std::cos(t), 2) * std::sin(p) * std::sin(t)) / 64. -
                (3003 * std::sqrt(95 / (2. * pi)) * std::pow(std::cos(t), 4) * std::sin(p) * std::sin(t)) / 128. +
                (3003 * std::sqrt(95 / (2. * pi)) * std::pow(std::cos(t), 6) * std::sin(p) * std::sin(t)) / 64. -
                (7293 * std::sqrt(95 / (2. * pi)) * std::pow(std::cos(t), 8) * std::sin(p) * std::sin(t)) / 256.);

    if (l == 9 && m == 0)
        return (315 * std::sqrt(19 / pi) * std::cos(t)) / 256. -
               (1155 * std::sqrt(19 / pi) * std::pow(std::cos(t), 3)) / 64. +
               (9009 * std::sqrt(19 / pi) * std::pow(std::cos(t), 5)) / 128. -
               (6435 * std::sqrt(19 / pi) * std::pow(std::cos(t), 7)) / 64. +
               (12155 * std::sqrt(19 / pi) * std::pow(std::cos(t), 9)) / 256.;

    if (l == 9 && m == 1)
        return std::sqrt(2) *
               ((-21 * std::sqrt(95 / (2. * pi)) * std::cos(p) * std::sin(t)) / 256. +
                (231 * std::sqrt(95 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 2) * std::sin(t)) / 64. -
                (3003 * std::sqrt(95 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 4) * std::sin(t)) / 128. +
                (3003 * std::sqrt(95 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 6) * std::sin(t)) / 64. -
                (7293 * std::sqrt(95 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 8) * std::sin(t)) / 256.);

    if (l == 9 && m == 2)
        return std::sqrt(2) *
               ((-21 * std::sqrt(1045 / pi) * std::cos(2 * p) * std::cos(t) * std::pow(std::sin(t), 2)) / 128. +
                (273 * std::sqrt(1045 / pi) * std::cos(2 * p) * std::pow(std::cos(t), 3) * std::pow(std::sin(t), 2)) /
                        128. -
                (819 * std::sqrt(1045 / pi) * std::cos(2 * p) * std::pow(std::cos(t), 5) * std::pow(std::sin(t), 2)) /
                        128. +
                (663 * std::sqrt(1045 / pi) * std::cos(2 * p) * std::pow(std::cos(t), 7) * std::pow(std::sin(t), 2)) /
                        128.);

    if (l == 9 && m == 3)
        return std::sqrt(2) *
               ((std::sqrt(21945 / pi) * std::cos(3 * p) * std::pow(std::sin(t), 3)) / 256. -
                (39 * std::sqrt(21945 / pi) * std::cos(3 * p) * std::pow(std::cos(t), 2) * std::pow(std::sin(t), 3)) /
                        256. +
                (195 * std::sqrt(21945 / pi) * std::cos(3 * p) * std::pow(std::cos(t), 4) * std::pow(std::sin(t), 3)) /
                        256. -
                (221 * std::sqrt(21945 / pi) * std::cos(3 * p) * std::pow(std::cos(t), 6) * std::pow(std::sin(t), 3)) /
                        256.);

    if (l == 9 && m == 4)
        return std::sqrt(2) *
               ((3 * std::sqrt(95095 / (2. * pi)) * std::cos(4 * p) * std::cos(t) * std::pow(std::sin(t), 4)) / 128. -
                (15 * std::sqrt(95095 / (2. * pi)) * std::cos(4 * p) * std::pow(std::cos(t), 3) *
                 std::pow(std::sin(t), 4)) /
                        64. +
                (51 * std::sqrt(95095 / (2. * pi)) * std::cos(4 * p) * std::pow(std::cos(t), 5) *
                 std::pow(std::sin(t), 4)) /
                        128.);

    if (l == 9 && m == 5)
        return std::sqrt(2) *
               ((-3 * std::sqrt(2717 / pi) * std::cos(5 * p) * std::pow(std::sin(t), 5)) / 256. +
                (45 * std::sqrt(2717 / pi) * std::cos(5 * p) * std::pow(std::cos(t), 2) * std::pow(std::sin(t), 5)) /
                        128. -
                (255 * std::sqrt(2717 / pi) * std::cos(5 * p) * std::pow(std::cos(t), 4) * std::pow(std::sin(t), 5)) /
                        256.);

    if (l == 9 && m == 6)
        return std::sqrt(2) *
               ((-3 * std::sqrt(40755 / pi) * std::cos(6 * p) * std::cos(t) * std::pow(std::sin(t), 6)) / 128. +
                (17 * std::sqrt(40755 / pi) * std::cos(6 * p) * std::pow(std::cos(t), 3) * std::pow(std::sin(t), 6)) /
                        128.);

    if (l == 9 && m == 7)
        return std::sqrt(2) *
               ((3 * std::sqrt(13585 / pi) * std::cos(7 * p) * std::pow(std::sin(t), 7)) / 512. -
                (51 * std::sqrt(13585 / pi) * std::cos(7 * p) * std::pow(std::cos(t), 2) * std::pow(std::sin(t), 7)) /
                        512.);

    if (l == 9 && m == 8)
        return (3 * std::sqrt(230945 / pi) * std::cos(8 * p) * std::cos(t) * std::pow(std::sin(t), 8)) / 256.;

    if (l == 9 && m == 9)
        return -(std::sqrt(230945 / (2. * pi)) * std::cos(9 * p) * std::pow(std::sin(t), 9)) / 256.;

    if (l == 10 && m == -10)
        return -(std::sqrt(969969 / (2. * pi)) * std::sin(10 * p) * std::pow(std::sin(t), 10)) / 512.;

    if (l == 10 && m == -9)
        return -(std::sqrt(4849845 / (2. * pi)) * std::cos(t) * std::sin(9 * p) * std::pow(std::sin(t), 9)) / 256.;

    if (l == 10 && m == -8)
        return std::sqrt(2) * ((std::sqrt(255255 / (2. * pi)) * std::sin(8 * p) * std::pow(std::sin(t), 8)) / 512. -
                               (19 * std::sqrt(255255 / (2. * pi)) * std::pow(std::cos(t), 2) * std::sin(8 * p) *
                                std::pow(std::sin(t), 8)) /
                                       512.);

    if (l == 10 && m == -7)
        return std::sqrt(2) *
               ((9 * std::sqrt(85085 / pi) * std::cos(t) * std::sin(7 * p) * std::pow(std::sin(t), 7)) / 512. -
                (57 * std::sqrt(85085 / pi) * std::pow(std::cos(t), 3) * std::sin(7 * p) * std::pow(std::sin(t), 7)) /
                        512.);

    if (l == 10 && m == -6)
        return std::sqrt(2) *
               ((-9 * std::sqrt(5005 / pi) * std::sin(6 * p) * std::pow(std::sin(t), 6)) / 1024. +
                (153 * std::sqrt(5005 / pi) * std::pow(std::cos(t), 2) * std::sin(6 * p) * std::pow(std::sin(t), 6)) /
                        512. -
                (969 * std::sqrt(5005 / pi) * std::pow(std::cos(t), 4) * std::sin(6 * p) * std::pow(std::sin(t), 6)) /
                        1024.);

    if (l == 10 && m == -5)
        return std::sqrt(2) *
               ((-45 * std::sqrt(1001 / pi) * std::cos(t) * std::sin(5 * p) * std::pow(std::sin(t), 5)) / 256. +
                (255 * std::sqrt(1001 / pi) * std::pow(std::cos(t), 3) * std::sin(5 * p) * std::pow(std::sin(t), 5)) /
                        128. -
                (969 * std::sqrt(1001 / pi) * std::pow(std::cos(t), 5) * std::sin(5 * p) * std::pow(std::sin(t), 5)) /
                        256.);

    if (l == 10 && m == -4)
        return std::sqrt(2) * ((3 * std::sqrt(5005 / (2. * pi)) * std::sin(4 * p) * std::pow(std::sin(t), 4)) / 256. -
                               (135 * std::sqrt(5005 / (2. * pi)) * std::pow(std::cos(t), 2) * std::sin(4 * p) *
                                std::pow(std::sin(t), 4)) /
                                       256. +
                               (765 * std::sqrt(5005 / (2. * pi)) * std::pow(std::cos(t), 4) * std::sin(4 * p) *
                                std::pow(std::sin(t), 4)) /
                                       256. -
                               (969 * std::sqrt(5005 / (2. * pi)) * std::pow(std::cos(t), 6) * std::sin(4 * p) *
                                std::pow(std::sin(t), 4)) /
                                       256.);

    if (l == 10 && m == -3)
        return std::sqrt(2) *
               ((21 * std::sqrt(5005 / pi) * std::cos(t) * std::sin(3 * p) * std::pow(std::sin(t), 3)) / 256. -
                (315 * std::sqrt(5005 / pi) * std::pow(std::cos(t), 3) * std::sin(3 * p) * std::pow(std::sin(t), 3)) /
                        256. +
                (1071 * std::sqrt(5005 / pi) * std::pow(std::cos(t), 5) * std::sin(3 * p) * std::pow(std::sin(t), 3)) /
                        256. -
                (969 * std::sqrt(5005 / pi) * std::pow(std::cos(t), 7) * std::sin(3 * p) * std::pow(std::sin(t), 3)) /
                        256.);

    if (l == 10 && m == -2)
        return std::sqrt(2) * ((-21 * std::sqrt(385 / (2. * pi)) * std::sin(2 * p) * std::pow(std::sin(t), 2)) / 512. +
                               (273 * std::sqrt(385 / (2. * pi)) * std::pow(std::cos(t), 2) * std::sin(2 * p) *
                                std::pow(std::sin(t), 2)) /
                                       128. -
                               (4095 * std::sqrt(385 / (2. * pi)) * std::pow(std::cos(t), 4) * std::sin(2 * p) *
                                std::pow(std::sin(t), 2)) /
                                       256. +
                               (4641 * std::sqrt(385 / (2. * pi)) * std::pow(std::cos(t), 6) * std::sin(2 * p) *
                                std::pow(std::sin(t), 2)) /
                                       128. -
                               (12597 * std::sqrt(385 / (2. * pi)) * std::pow(std::cos(t), 8) * std::sin(2 * p) *
                                std::pow(std::sin(t), 2)) /
                                       512.);

    if (l == 10 && m == -1)
        return std::sqrt(2) *
               ((-63 * std::sqrt(1155 / (2. * pi)) * std::cos(t) * std::sin(p) * std::sin(t)) / 256. +
                (273 * std::sqrt(1155 / (2. * pi)) * std::pow(std::cos(t), 3) * std::sin(p) * std::sin(t)) / 64. -
                (2457 * std::sqrt(1155 / (2. * pi)) * std::pow(std::cos(t), 5) * std::sin(p) * std::sin(t)) / 128. +
                (1989 * std::sqrt(1155 / (2. * pi)) * std::pow(std::cos(t), 7) * std::sin(p) * std::sin(t)) / 64. -
                (4199 * std::sqrt(1155 / (2. * pi)) * std::pow(std::cos(t), 9) * std::sin(p) * std::sin(t)) / 256.);

    if (l == 10 && m == 0)
        return (-63 * std::sqrt(21 / pi)) / 512. + (3465 * std::sqrt(21 / pi) * std::pow(std::cos(t), 2)) / 512. -
               (15015 * std::sqrt(21 / pi) * std::pow(std::cos(t), 4)) / 256. +
               (45045 * std::sqrt(21 / pi) * std::pow(std::cos(t), 6)) / 256. -
               (109395 * std::sqrt(21 / pi) * std::pow(std::cos(t), 8)) / 512. +
               (46189 * std::sqrt(21 / pi) * std::pow(std::cos(t), 10)) / 512.;

    if (l == 10 && m == 1)
        return std::sqrt(2) *
               ((-63 * std::sqrt(1155 / (2. * pi)) * std::cos(p) * std::cos(t) * std::sin(t)) / 256. +
                (273 * std::sqrt(1155 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 3) * std::sin(t)) / 64. -
                (2457 * std::sqrt(1155 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 5) * std::sin(t)) / 128. +
                (1989 * std::sqrt(1155 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 7) * std::sin(t)) / 64. -
                (4199 * std::sqrt(1155 / (2. * pi)) * std::cos(p) * std::pow(std::cos(t), 9) * std::sin(t)) / 256.);

    if (l == 10 && m == 2)
        return std::sqrt(2) * ((21 * std::sqrt(385 / (2. * pi)) * std::cos(2 * p) * std::pow(std::sin(t), 2)) / 512. -
                               (273 * std::sqrt(385 / (2. * pi)) * std::cos(2 * p) * std::pow(std::cos(t), 2) *
                                std::pow(std::sin(t), 2)) /
                                       128. +
                               (4095 * std::sqrt(385 / (2. * pi)) * std::cos(2 * p) * std::pow(std::cos(t), 4) *
                                std::pow(std::sin(t), 2)) /
                                       256. -
                               (4641 * std::sqrt(385 / (2. * pi)) * std::cos(2 * p) * std::pow(std::cos(t), 6) *
                                std::pow(std::sin(t), 2)) /
                                       128. +
                               (12597 * std::sqrt(385 / (2. * pi)) * std::cos(2 * p) * std::pow(std::cos(t), 8) *
                                std::pow(std::sin(t), 2)) /
                                       512.);

    if (l == 10 && m == 3)
        return std::sqrt(2) *
               ((21 * std::sqrt(5005 / pi) * std::cos(3 * p) * std::cos(t) * std::pow(std::sin(t), 3)) / 256. -
                (315 * std::sqrt(5005 / pi) * std::cos(3 * p) * std::pow(std::cos(t), 3) * std::pow(std::sin(t), 3)) /
                        256. +
                (1071 * std::sqrt(5005 / pi) * std::cos(3 * p) * std::pow(std::cos(t), 5) * std::pow(std::sin(t), 3)) /
                        256. -
                (969 * std::sqrt(5005 / pi) * std::cos(3 * p) * std::pow(std::cos(t), 7) * std::pow(std::sin(t), 3)) /
                        256.);

    if (l == 10 && m == 4)
        return std::sqrt(2) * ((-3 * std::sqrt(5005 / (2. * pi)) * std::cos(4 * p) * std::pow(std::sin(t), 4)) / 256. +
                               (135 * std::sqrt(5005 / (2. * pi)) * std::cos(4 * p) * std::pow(std::cos(t), 2) *
                                std::pow(std::sin(t), 4)) /
                                       256. -
                               (765 * std::sqrt(5005 / (2. * pi)) * std::cos(4 * p) * std::pow(std::cos(t), 4) *
                                std::pow(std::sin(t), 4)) /
                                       256. +
                               (969 * std::sqrt(5005 / (2. * pi)) * std::cos(4 * p) * std::pow(std::cos(t), 6) *
                                std::pow(std::sin(t), 4)) /
                                       256.);

    if (l == 10 && m == 5)
        return std::sqrt(2) *
               ((-45 * std::sqrt(1001 / pi) * std::cos(5 * p) * std::cos(t) * std::pow(std::sin(t), 5)) / 256. +
                (255 * std::sqrt(1001 / pi) * std::cos(5 * p) * std::pow(std::cos(t), 3) * std::pow(std::sin(t), 5)) /
                        128. -
                (969 * std::sqrt(1001 / pi) * std::cos(5 * p) * std::pow(std::cos(t), 5) * std::pow(std::sin(t), 5)) /
                        256.);

    if (l == 10 && m == 6)
        return std::sqrt(2) *
               ((9 * std::sqrt(5005 / pi) * std::cos(6 * p) * std::pow(std::sin(t), 6)) / 1024. -
                (153 * std::sqrt(5005 / pi) * std::cos(6 * p) * std::pow(std::cos(t), 2) * std::pow(std::sin(t), 6)) /
                        512. +
                (969 * std::sqrt(5005 / pi) * std::cos(6 * p) * std::pow(std::cos(t), 4) * std::pow(std::sin(t), 6)) /
                        1024.);

    if (l == 10 && m == 7)
        return std::sqrt(2) *
               ((9 * std::sqrt(85085 / pi) * std::cos(7 * p) * std::cos(t) * std::pow(std::sin(t), 7)) / 512. -
                (57 * std::sqrt(85085 / pi) * std::cos(7 * p) * std::pow(std::cos(t), 3) * std::pow(std::sin(t), 7)) /
                        512.);

    if (l == 10 && m == 8)
        return std::sqrt(2) * (-(std::sqrt(255255 / (2. * pi)) * std::cos(8 * p) * std::pow(std::sin(t), 8)) / 512. +
                               (19 * std::sqrt(255255 / (2. * pi)) * std::cos(8 * p) * std::pow(std::cos(t), 2) *
                                std::pow(std::sin(t), 8)) /
                                       512.);

    if (l == 10 && m == 9)
        return -(std::sqrt(4849845 / (2. * pi)) * std::cos(9 * p) * std::cos(t) * std::pow(std::sin(t), 9)) / 256.;

    if (l == 10 && m == 10)
        return (std::sqrt(969969 / (2. * pi)) * std::cos(10 * p) * std::pow(std::sin(t), 10)) / 512.;

    return 0;
}

int
test_rlm()
{
    int num_points = 500;
    mdarray<double, 2> tp({2, num_points});

    tp(0, 0) = pi;
    tp(1, 0) = 0;

    for (int k = 1; k < num_points - 1; k++) {
        double hk = -1.0 + double(2 * k) / double(num_points - 1);
        tp(0, k)  = std::acos(hk);
        double t  = tp(1, k - 1) + 3.80925122745582 / std::sqrt(double(num_points)) / std::sqrt(1 - hk * hk);
        tp(1, k)  = std::fmod(t, twopi);
    }

    tp(0, num_points - 1) = 0;
    tp(1, num_points - 1) = 0;

    int lmax{10};
    std::vector<double> rlm((lmax + 1) * (lmax + 1));
    std::vector<double> rlm_ref((lmax + 1) * (lmax + 1));

    for (int k = 0; k < num_points; k++) {
        double theta = tp(0, k);
        double phi   = tp(1, k);
        /* generate spherical harmonics */
        sf::spherical_harmonics(lmax, theta, phi, &rlm[0]);
        sf::spherical_harmonics_ref(lmax, theta, phi, &rlm_ref[0]);

        double diff{0};
        for (int l = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++) {
                auto val = SphericalHarmonicR(l, m, theta, phi);
                diff += std::abs(rlm[sf::lm(l, m)] - rlm_ref[sf::lm(l, m)]);
                diff += std::abs(rlm[sf::lm(l, m)] - val);
            }
        }
        if (diff > 1e-10) {
            return 1;
        }
    }
    return 0;
}

int
main(int argn, char** argv)
{
    return call_test(argv[0], test_rlm);
}
