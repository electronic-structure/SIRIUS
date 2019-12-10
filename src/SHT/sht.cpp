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

#include "SHT/sht.hpp"

namespace sirius {

void SHT::wigner_d_matrix(int l, double beta, sddk::mdarray<double, 2>& d_mtrx__)
{
    long double cos_b2 = std::cos((long double)beta / 2.0L);
    long double sin_b2 = std::sin((long double)beta / 2.0L);

    for (int m1 = -l; m1 <= l; m1++) {
        for (int m2 = -l; m2 <= l; m2++) {
            long double d = 0;
            for (int j = 0; j <= std::min(l + m1, l - m2); j++) {
                if ((l - m2 - j) >= 0 && (l + m1 - j) >= 0 && (j + m2 - m1) >= 0) {
                    long double g = (std::sqrt(utils::factorial<long double>(l + m1)) / utils::factorial<long double>(l - m2 - j)) *
                                    (std::sqrt(utils::factorial<long double>(l - m1)) / utils::factorial<long double>(l + m1 - j)) *
                                    (std::sqrt(utils::factorial<long double>(l - m2)) / utils::factorial<long double>(j + m2 - m1)) *
                                    (std::sqrt(utils::factorial<long double>(l + m2)) / utils::factorial<long double>(j));
                    d += g * std::pow(-1, j) * std::pow(cos_b2, 2 * l + m1 - m2 - 2 * j) * std::pow(sin_b2, 2 * j + m2 - m1);
                }
            }
            d_mtrx__(m1 + l, m2 + l) = (double)d;
        }
    }
}

void SHT::rotation_matrix_l(int l, geometry3d::vector3d<double> euler_angles, int proper_rotation,
                            double_complex *rot_mtrx__, int ld) {
    sddk::mdarray<double_complex, 2> rot_mtrx(rot_mtrx__, ld, 2 * l + 1);

    sddk::mdarray<double, 2> d_mtrx(2 * l + 1, 2 * l + 1);
    wigner_d_matrix(l, euler_angles[1], d_mtrx);

    for (int m1 = -l; m1 <= l; m1++) {
        for (int m2 = -l; m2 <= l; m2++) {
            rot_mtrx(m1 + l, m2 + l) = std::exp(double_complex(0, -euler_angles[0] * m1 - euler_angles[2] * m2)) *
                                       d_mtrx(m1 + l, m2 + l) * std::pow(proper_rotation, l);
        }
    }
}

void SHT::rotation_matrix_l(int l, geometry3d::vector3d<double> euler_angles, int proper_rotation,
                            double *rot_mtrx__, int ld) {
    sddk::mdarray<double, 2> rot_mtrx_rlm(rot_mtrx__, ld, 2 * l + 1);
    sddk::mdarray<double_complex, 2> rot_mtrx_ylm(2 * l + 1, 2 * l + 1);

    sddk::mdarray<double, 2> d_mtrx(2 * l + 1, 2 * l + 1);
    wigner_d_matrix(l, euler_angles[1], d_mtrx);

    for (int m1 = -l; m1 <= l; m1++) {
        for (int m2 = -l; m2 <= l; m2++) {
            rot_mtrx_ylm(m1 + l, m2 + l) =
                    std::exp(double_complex(0, -euler_angles[0] * m1 - euler_angles[2] * m2)) *
                    d_mtrx(m1 + l, m2 + l) * std::pow(proper_rotation, l);
        }
    }
    for (int m1 = -l; m1 <= l; m1++) {
        auto i13 = (m1 == 0) ? std::vector<int>({0}) : std::vector<int>({-m1, m1});

        for (int m2 = -l; m2 <= l; m2++) {
            auto i24 = (m2 == 0) ? std::vector<int>({0}) : std::vector<int>({-m2, m2});

            for (int m3 : i13) {
                for (int m4 : i24) {
                    rot_mtrx_rlm(m1 + l, m2 + l) += std::real(rlm_dot_ylm(l, m1, m3) *
                                                              rot_mtrx_ylm(m3 + l, m4 + l) *
                                                              ylm_dot_rlm(l, m4, m2));
                }
            }
        }
    }
}

void SHT::dRlm_dtheta(int lmax, double theta, double phi, sddk::mdarray<double, 1> &data) {
    assert(lmax <= 8);

    data[0] = 0;

    if (lmax == 0)
        return;

    auto cos_theta = SHT::cosxn(lmax, theta);
    auto sin_theta = SHT::sinxn(lmax, theta);
    auto cos_phi = SHT::cosxn(lmax, phi);
    auto sin_phi = SHT::sinxn(lmax, phi);

    data[1] = -(std::sqrt(3 / pi) * cos_theta[0] * sin_phi[0]) / 2.;

    data[2] = -(std::sqrt(3 / pi) * sin_theta[0]) / 2.;

    data[3] = -(std::sqrt(3 / pi) * cos_phi[0] * cos_theta[0]) / 2.;

    if (lmax == 1)
        return;

    data[4] = -(std::sqrt(15 / pi) * cos_phi[0] * cos_theta[0] * sin_phi[0] * sin_theta[0]);

    data[5] = -(std::sqrt(15 / pi) * cos_theta[1] * sin_phi[0]) / 2.;

    data[6] = (-3 * std::sqrt(5 / pi) * cos_theta[0] * sin_theta[0]) / 2.;

    data[7] = -(std::sqrt(15 / pi) * cos_phi[0] * cos_theta[1]) / 2.;

    data[8] = (std::sqrt(15 / pi) * cos_phi[1] * sin_theta[1]) / 4.;

    if (lmax == 2)
        return;

    data[9] = (-3 * std::sqrt(35 / (2. * pi)) * cos_theta[0] * sin_phi[2] * std::pow(sin_theta[0], 2)) / 4.;

    data[10] = (std::sqrt(105 / pi) * sin_phi[1] * (sin_theta[0] - 3 * sin_theta[2])) / 16.;

    data[11] = -(std::sqrt(21 / (2. * pi)) * (cos_theta[0] + 15 * cos_theta[2]) * sin_phi[0]) / 16.;

    data[12] = (-3 * std::sqrt(7 / pi) * (sin_theta[0] + 5 * sin_theta[2])) / 16.;

    data[13] = -(std::sqrt(21 / (2. * pi)) * cos_phi[0] * (cos_theta[0] + 15 * cos_theta[2])) / 16.;

    data[14] = -(std::sqrt(105 / pi) * cos_phi[1] * (sin_theta[0] - 3 * sin_theta[2])) / 16.;

    data[15] = (-3 * std::sqrt(35 / (2. * pi)) * cos_phi[2] * cos_theta[0] * std::pow(sin_theta[0], 2)) / 4.;

    if (lmax == 3)
        return;

    data[16] = (-3 * std::sqrt(35 / pi) * cos_theta[0] * sin_phi[3] * std::pow(sin_theta[0], 3)) / 4.;

    data[17] =
            (-3 * std::sqrt(35 / (2. * pi)) * (1 + 2 * cos_theta[1]) * sin_phi[2] * std::pow(sin_theta[0], 2)) / 4.;

    data[18] = (3 * std::sqrt(5 / pi) * sin_phi[1] * (2 * sin_theta[1] - 7 * sin_theta[3])) / 16.;

    data[19] = (-3 * std::sqrt(5 / (2. * pi)) * (cos_theta[1] + 7 * cos_theta[3]) * sin_phi[0]) / 8.;

    data[20] = (-15 * (2 * sin_theta[1] + 7 * sin_theta[3])) / (32. * std::sqrt(pi));

    data[21] = (-3 * std::sqrt(5 / (2. * pi)) * cos_phi[0] * (cos_theta[1] + 7 * cos_theta[3])) / 8.;

    data[22] = (3 * std::sqrt(5 / pi) * cos_phi[1] * (-2 * sin_theta[1] + 7 * sin_theta[3])) / 16.;

    data[23] =
            (-3 * std::sqrt(35 / (2. * pi)) * cos_phi[2] * (1 + 2 * cos_theta[1]) * std::pow(sin_theta[0], 2)) / 4.;

    data[24] = (3 * std::sqrt(35 / pi) * cos_phi[3] * cos_theta[0] * std::pow(sin_theta[0], 3)) / 4.;

    if (lmax == 4)
        return;

    data[25] = (-15 * std::sqrt(77 / (2. * pi)) * cos_theta[0] * sin_phi[4] * std::pow(sin_theta[0], 4)) / 16.;

    data[26] = (-3 * std::sqrt(385 / pi) * (3 + 5 * cos_theta[1]) * sin_phi[3] * std::pow(sin_theta[0], 3)) / 32.;

    data[27] = (-3 * std::sqrt(385 / (2. * pi)) * cos_theta[0] * (1 + 15 * cos_theta[1]) * sin_phi[2] *
                std::pow(sin_theta[0], 2)) / 32.;

    data[28] =
            (std::sqrt(1155 / pi) * sin_phi[1] * (2 * sin_theta[0] + 3 * (sin_theta[2] - 5 * sin_theta[4]))) / 128.;

    data[29] = -(std::sqrt(165 / pi) * (2 * cos_theta[0] + 21 * (cos_theta[2] + 5 * cos_theta[4])) * sin_phi[0]) /
               256.;

    data[30] = (-15 * std::sqrt(11 / pi) * (2 * sin_theta[0] + 7 * (sin_theta[2] + 3 * sin_theta[4]))) / 256.;

    data[31] = -(std::sqrt(165 / pi) * cos_phi[0] * (2 * cos_theta[0] + 21 * (cos_theta[2] + 5 * cos_theta[4]))) /
               256.;

    data[32] =
            (std::sqrt(1155 / pi) * cos_phi[1] * (-2 * sin_theta[0] - 3 * sin_theta[2] + 15 * sin_theta[4])) / 128.;

    data[33] = (-3 * std::sqrt(385 / (2. * pi)) * cos_phi[2] * (17 * cos_theta[0] + 15 * cos_theta[2]) *
                std::pow(sin_theta[0], 2)) / 64.;

    data[34] = (3 * std::sqrt(385 / pi) * cos_phi[3] * (3 + 5 * cos_theta[1]) * std::pow(sin_theta[0], 3)) / 32.;

    data[35] = (-15 * std::sqrt(77 / (2. * pi)) * cos_phi[4] * cos_theta[0] * std::pow(sin_theta[0], 4)) / 16.;

    if (lmax == 5)
        return;

    data[36] = (-3 * std::sqrt(3003 / (2. * pi)) * cos_theta[0] * sin_phi[5] * std::pow(sin_theta[0], 5)) / 16.;

    data[37] =
            (-3 * std::sqrt(1001 / (2. * pi)) * (2 + 3 * cos_theta[1]) * sin_phi[4] * std::pow(sin_theta[0], 4)) /
            16.;

    data[38] = (-3 * std::sqrt(91 / pi) * cos_theta[0] * (7 + 33 * cos_theta[1]) * sin_phi[3] *
                std::pow(sin_theta[0], 3)) / 32.;

    data[39] = (-3 * std::sqrt(1365 / (2. * pi)) * (7 + 14 * cos_theta[1] + 11 * cos_theta[3]) * sin_phi[2] *
                std::pow(sin_theta[0], 2)) / 64.;

    data[40] = (std::sqrt(1365 / (2. * pi)) * sin_phi[1] *
                (17 * sin_theta[1] + 12 * sin_theta[3] - 99 * sin_theta[5])) / 512.;

    data[41] =
            -(std::sqrt(273 / pi) * (5 * cos_theta[1] + 24 * cos_theta[3] + 99 * cos_theta[5]) * sin_phi[0]) / 256.;

    data[42] = (-21 * std::sqrt(13 / pi) * (5 * sin_theta[1] + 12 * sin_theta[3] + 33 * sin_theta[5])) / 512.;

    data[43] =
            -(std::sqrt(273 / pi) * cos_phi[0] * (5 * cos_theta[1] + 24 * cos_theta[3] + 99 * cos_theta[5])) / 256.;

    data[44] = (std::sqrt(1365 / (2. * pi)) * cos_phi[1] *
                (-17 * sin_theta[1] - 12 * sin_theta[3] + 99 * sin_theta[5])) / 512.;

    data[45] = (-3 * std::sqrt(1365 / (2. * pi)) * cos_phi[2] * (7 + 14 * cos_theta[1] + 11 * cos_theta[3]) *
                std::pow(sin_theta[0], 2)) / 64.;

    data[46] = (3 * std::sqrt(91 / pi) * cos_phi[3] * (47 * cos_theta[0] + 33 * cos_theta[2]) *
                std::pow(sin_theta[0], 3)) / 64.;

    data[47] =
            (-3 * std::sqrt(1001 / (2. * pi)) * cos_phi[4] * (2 + 3 * cos_theta[1]) * std::pow(sin_theta[0], 4)) /
            16.;

    data[48] = (3 * std::sqrt(3003 / (2. * pi)) * cos_phi[5] * cos_theta[0] * std::pow(sin_theta[0], 5)) / 16.;

    if (lmax == 6)
        return;

    data[49] = (-21 * std::sqrt(715 / pi) * cos_theta[0] * sin_phi[6] * std::pow(sin_theta[0], 6)) / 64.;

    data[50] =
            (-3 * std::sqrt(5005 / (2. * pi)) * (5 + 7 * cos_theta[1]) * sin_phi[5] * std::pow(sin_theta[0], 5)) /
            64.;

    data[51] = (-3 * std::sqrt(385 / pi) * cos_theta[0] * (29 + 91 * cos_theta[1]) * sin_phi[4] *
                std::pow(sin_theta[0], 4)) / 128.;

    data[52] = (-3 * std::sqrt(385 / pi) * (81 + 148 * cos_theta[1] + 91 * cos_theta[3]) * sin_phi[3] *
                std::pow(sin_theta[0], 3)) / 256.;

    data[53] = (-3 * std::sqrt(35 / pi) * cos_theta[0] * (523 + 396 * cos_theta[1] + 1001 * cos_theta[3]) *
                sin_phi[2] * std::pow(sin_theta[0], 2)) / 512.;

    data[54] = (3 * std::sqrt(35 / (2. * pi)) * sin_phi[1] *
                (75 * sin_theta[0] + 171 * sin_theta[2] + 55 * sin_theta[4] - 1001 * sin_theta[6])) / 2048.;

    data[55] = -(std::sqrt(105 / pi) *
                 (25 * cos_theta[0] + 243 * cos_theta[2] + 825 * cos_theta[4] + 3003 * cos_theta[6]) * sin_phi[0]) /
               4096.;

    data[56] = (-7 * std::sqrt(15 / pi) *
                (25 * sin_theta[0] + 81 * sin_theta[2] + 165 * sin_theta[4] + 429 * sin_theta[6])) / 2048.;

    data[57] = -(std::sqrt(105 / pi) * cos_phi[0] *
                 (25 * cos_theta[0] + 243 * cos_theta[2] + 825 * cos_theta[4] + 3003 * cos_theta[6])) / 4096.;

    data[58] = (-3 * std::sqrt(35 / (2. * pi)) * cos_phi[1] *
                (75 * sin_theta[0] + 171 * sin_theta[2] + 55 * sin_theta[4] - 1001 * sin_theta[6])) / 2048.;

    data[59] = (-3 * std::sqrt(35 / pi) * cos_phi[2] *
                (1442 * cos_theta[0] + 1397 * cos_theta[2] + 1001 * cos_theta[4]) * std::pow(sin_theta[0], 2)) /
               1024.;

    data[60] = (3 * std::sqrt(385 / pi) * cos_phi[3] * (81 + 148 * cos_theta[1] + 91 * cos_theta[3]) *
                std::pow(sin_theta[0], 3)) / 256.;

    data[61] = (-3 * std::sqrt(385 / pi) * cos_phi[4] * (149 * cos_theta[0] + 91 * cos_theta[2]) *
                std::pow(sin_theta[0], 4)) / 256.;

    data[62] = (3 * std::sqrt(5005 / (2. * pi)) * cos_phi[5] * (5 + 7 * cos_theta[1]) * std::pow(sin_theta[0], 5)) /
               64.;

    data[63] = (-21 * std::sqrt(715 / pi) * cos_phi[6] * cos_theta[0] * std::pow(sin_theta[0], 6)) / 64.;

    if (lmax == 7)
        return;

    data[64] = (-3 * std::sqrt(12155 / pi) * cos_theta[0] * sin_phi[7] * std::pow(sin_theta[0], 7)) / 32.;

    data[65] = (-3 * std::sqrt(12155 / pi) * (3 + 4 * cos_theta[1]) * sin_phi[6] * std::pow(sin_theta[0], 6)) / 64.;

    data[66] = (-3 * std::sqrt(7293 / (2. * pi)) * cos_theta[0] * (2 + 5 * cos_theta[1]) * sin_phi[5] *
                std::pow(sin_theta[0], 5)) / 16.;

    data[67] = (-3 * std::sqrt(17017 / pi) * (11 + 19 * cos_theta[1] + 10 * cos_theta[3]) * sin_phi[4] *
                std::pow(sin_theta[0], 4)) / 128.;

    data[68] =
            (-3 * std::sqrt(1309 / pi) * cos_theta[0] * (43 + 52 * cos_theta[1] + 65 * cos_theta[3]) * sin_phi[3] *
             std::pow(sin_theta[0], 3)) / 128.;

    data[69] = (-3 * std::sqrt(19635 / pi) * (21 + 42 * cos_theta[1] + 39 * cos_theta[3] + 26 * cos_theta[5]) *
                sin_phi[2] * std::pow(sin_theta[0], 2)) / 512.;

    data[70] = (-3 * std::sqrt(595 / (2. * pi)) * (-8 + 121 * cos_theta[1] + 143 * cos_theta[5]) * sin_phi[1] *
                sin_theta[1]) / 512.;

    data[71] = (-3 * std::sqrt(17 / pi) *
                (35 * cos_theta[1] + 154 * cos_theta[3] + 429 * cos_theta[5] + 1430 * cos_theta[7]) * sin_phi[0]) /
               2048.;

    data[72] = (-9 * std::sqrt(17 / pi) *
                (70 * sin_theta[1] + 154 * sin_theta[3] + 286 * sin_theta[5] + 715 * sin_theta[7])) / 4096.;

    data[73] = (-3 * std::sqrt(17 / pi) * cos_phi[0] *
                (35 * cos_theta[1] + 154 * cos_theta[3] + 429 * cos_theta[5] + 1430 * cos_theta[7])) / 2048.;

    data[74] = (3 * std::sqrt(595 / (2. * pi)) * cos_phi[1] *
                (-16 * sin_theta[1] - 22 * sin_theta[3] + 143 * sin_theta[7])) / 1024.;

    data[75] = (-3 * std::sqrt(19635 / pi) * cos_phi[2] *
                (21 + 42 * cos_theta[1] + 39 * cos_theta[3] + 26 * cos_theta[5]) * std::pow(sin_theta[0], 2)) /
               512.;

    data[76] =
            (3 * std::sqrt(1309 / pi) * cos_phi[3] * (138 * cos_theta[0] + 117 * cos_theta[2] + 65 * cos_theta[4]) *
             std::pow(sin_theta[0], 3)) / 256.;

    data[77] = (-3 * std::sqrt(17017 / pi) * cos_phi[4] * (11 + 19 * cos_theta[1] + 10 * cos_theta[3]) *
                std::pow(sin_theta[0], 4)) / 128.;

    data[78] = (3 * std::sqrt(7293 / (2. * pi)) * cos_phi[5] * (9 * cos_theta[0] + 5 * cos_theta[2]) *
                std::pow(sin_theta[0], 5)) / 32.;

    data[79] = (-3 * std::sqrt(12155 / pi) * cos_phi[6] * (3 + 4 * cos_theta[1]) * std::pow(sin_theta[0], 6)) / 64.;

    data[80] = (3 * std::sqrt(12155 / pi) * cos_phi[7] * cos_theta[0] * std::pow(sin_theta[0], 7)) / 32.;
}

void SHT::dRlm_dphi_sin_theta(int lmax, double theta, double phi, sddk::mdarray<double, 1> &data) {
    assert(lmax <= 8);

    data[0] = 0;

    if (lmax == 0)
        return;

    auto cos_theta = SHT::cosxn(lmax, theta);
    auto sin_theta = SHT::sinxn(lmax, theta);
    auto cos_phi = SHT::cosxn(lmax, phi);
    auto sin_phi = SHT::sinxn(lmax, phi);

    data[1] = -(std::sqrt(3 / pi) * cos_phi[0]) / 2.;

    data[2] = 0;

    data[3] = (std::sqrt(3 / pi) * sin_phi[0]) / 2.;

    if (lmax == 1)
        return;

    data[4] = -(std::sqrt(15 / pi) * cos_phi[1] * sin_theta[0]) / 2.;

    data[5] = -(std::sqrt(15 / pi) * cos_phi[0] * cos_theta[0]) / 2.;

    data[6] = 0;

    data[7] = (std::sqrt(15 / pi) * cos_theta[0] * sin_phi[0]) / 2.;

    data[8] = -(std::sqrt(15 / pi) * cos_phi[0] * sin_phi[0] * sin_theta[0]);

    if (lmax == 2)
        return;

    data[9] = (-3 * std::sqrt(35 / (2. * pi)) * cos_phi[2] * std::pow(sin_theta[0], 2)) / 4.;

    data[10] = -(std::sqrt(105 / pi) * cos_phi[1] * sin_theta[1]) / 4.;

    data[11] = -(std::sqrt(21 / (2. * pi)) * cos_phi[0] * (3 + 5 * cos_theta[1])) / 8.;

    data[12] = 0;

    data[13] = (std::sqrt(21 / (2. * pi)) * (3 + 5 * cos_theta[1]) * sin_phi[0]) / 8.;

    data[14] = -(std::sqrt(105 / pi) * cos_phi[0] * cos_theta[0] * sin_phi[0] * sin_theta[0]);

    data[15] = (3 * std::sqrt(35 / (2. * pi)) * sin_phi[2] * std::pow(sin_theta[0], 2)) / 4.;

    if (lmax == 3)
        return;

    data[16] = (-3 * std::sqrt(35 / pi) * cos_phi[3] * std::pow(sin_theta[0], 3)) / 4.;

    data[17] = (-9 * std::sqrt(35 / (2. * pi)) * cos_phi[2] * cos_theta[0] * std::pow(sin_theta[0], 2)) / 4.;

    data[18] = (-3 * std::sqrt(5 / pi) * cos_phi[1] * (3 * sin_theta[0] + 7 * sin_theta[2])) / 16.;

    data[19] = (-3 * std::sqrt(5 / (2. * pi)) * cos_phi[0] * (9 * cos_theta[0] + 7 * cos_theta[2])) / 16.;

    data[20] = 0;

    data[21] = (3 * std::sqrt(5 / (2. * pi)) * cos_theta[0] * (1 + 7 * cos_theta[1]) * sin_phi[0]) / 8.;

    data[22] = (-3 * std::sqrt(5 / pi) * sin_phi[1] * (3 * sin_theta[0] + 7 * sin_theta[2])) / 16.;

    data[23] = (9 * std::sqrt(35 / (2. * pi)) * cos_theta[0] * sin_phi[2] * std::pow(sin_theta[0], 2)) / 4.;

    data[24] = (-3 * std::sqrt(35 / pi) * sin_phi[3] * std::pow(sin_theta[0], 3)) / 4.;

    if (lmax == 4)
        return;

    data[25] = (-15 * std::sqrt(77 / (2. * pi)) * cos_phi[4] * std::pow(sin_theta[0], 4)) / 16.;

    data[26] = (-3 * std::sqrt(385 / pi) * cos_phi[3] * cos_theta[0] * std::pow(sin_theta[0], 3)) / 4.;

    data[27] = (-3 * std::sqrt(385 / (2. * pi)) * cos_phi[2] * (7 + 9 * cos_theta[1]) * std::pow(sin_theta[0], 2)) /
               32.;

    data[28] = -(std::sqrt(1155 / pi) * cos_phi[1] * (2 * sin_theta[1] + 3 * sin_theta[3])) / 32.;

    data[29] = -(std::sqrt(165 / pi) * cos_phi[0] * (15 + 28 * cos_theta[1] + 21 * cos_theta[3])) / 128.;

    data[30] = 0;

    data[31] = (std::sqrt(165 / pi) * (15 + 28 * cos_theta[1] + 21 * cos_theta[3]) * sin_phi[0]) / 128.;

    data[32] = -(std::sqrt(1155 / pi) * sin_phi[1] * (2 * sin_theta[1] + 3 * sin_theta[3])) / 32.;

    data[33] = (3 * std::sqrt(385 / (2. * pi)) * (7 + 9 * cos_theta[1]) * sin_phi[2] * std::pow(sin_theta[0], 2)) /
               32.;

    data[34] = (-3 * std::sqrt(385 / pi) * cos_theta[0] * sin_phi[3] * std::pow(sin_theta[0], 3)) / 4.;

    data[35] = (15 * std::sqrt(77 / (2. * pi)) * sin_phi[4] * std::pow(sin_theta[0], 4)) / 16.;

    if (lmax == 5)
        return;

    data[36] = (-3 * std::sqrt(3003 / (2. * pi)) * cos_phi[5] * std::pow(sin_theta[0], 5)) / 16.;

    data[37] = (-15 * std::sqrt(1001 / (2. * pi)) * cos_phi[4] * cos_theta[0] * std::pow(sin_theta[0], 4)) / 16.;

    data[38] = (-3 * std::sqrt(91 / pi) * cos_phi[3] * (9 + 11 * cos_theta[1]) * std::pow(sin_theta[0], 3)) / 16.;

    data[39] = (-3 * std::sqrt(1365 / (2. * pi)) * cos_phi[2] * (21 * cos_theta[0] + 11 * cos_theta[2]) *
                std::pow(sin_theta[0], 2)) / 64.;

    data[40] = -(std::sqrt(1365 / (2. * pi)) * cos_phi[1] *
                 (10 * sin_theta[0] + 27 * sin_theta[2] + 33 * sin_theta[4])) / 256.;

    data[41] = -(std::sqrt(273 / pi) * cos_phi[0] * (50 * cos_theta[0] + 45 * cos_theta[2] + 33 * cos_theta[4])) /
               256.;

    data[42] = 0;

    data[43] =
            (std::sqrt(273 / pi) * cos_theta[0] * (19 + 12 * cos_theta[1] + 33 * cos_theta[3]) * sin_phi[0]) / 128.;

    data[44] = -(std::sqrt(1365 / (2. * pi)) * sin_phi[1] *
                 (10 * sin_theta[0] + 27 * sin_theta[2] + 33 * sin_theta[4])) / 256.;

    data[45] = (3 * std::sqrt(1365 / (2. * pi)) * cos_theta[0] * (5 + 11 * cos_theta[1]) * sin_phi[2] *
                std::pow(sin_theta[0], 2)) / 32.;

    data[46] = (-3 * std::sqrt(91 / pi) * (9 + 11 * cos_theta[1]) * sin_phi[3] * std::pow(sin_theta[0], 3)) / 16.;

    data[47] = (15 * std::sqrt(1001 / (2. * pi)) * cos_theta[0] * sin_phi[4] * std::pow(sin_theta[0], 4)) / 16.;

    data[48] = (-3 * std::sqrt(3003 / (2. * pi)) * sin_phi[5] * std::pow(sin_theta[0], 5)) / 16.;

    if (lmax == 6)
        return;

    data[49] = (-21 * std::sqrt(715 / pi) * cos_phi[6] * std::pow(sin_theta[0], 6)) / 64.;

    data[50] = (-9 * std::sqrt(5005 / (2. * pi)) * cos_phi[5] * cos_theta[0] * std::pow(sin_theta[0], 5)) / 16.;

    data[51] =
            (-15 * std::sqrt(385 / pi) * cos_phi[4] * (11 + 13 * cos_theta[1]) * std::pow(sin_theta[0], 4)) / 128.;

    data[52] = (-3 * std::sqrt(385 / pi) * cos_phi[3] * (27 * cos_theta[0] + 13 * cos_theta[2]) *
                std::pow(sin_theta[0], 3)) / 32.;

    data[53] = (-9 * std::sqrt(35 / pi) * cos_phi[2] * (189 + 308 * cos_theta[1] + 143 * cos_theta[3]) *
                std::pow(sin_theta[0], 2)) / 512.;

    data[54] = (-3 * std::sqrt(35 / (2. * pi)) * cos_phi[1] *
                (75 * sin_theta[1] + 132 * sin_theta[3] + 143 * sin_theta[5])) / 512.;

    data[55] = -(std::sqrt(105 / pi) * cos_phi[0] *
                 (350 + 675 * cos_theta[1] + 594 * cos_theta[3] + 429 * cos_theta[5])) / 2048.;

    data[56] = 0;

    data[57] = (std::sqrt(105 / pi) * (350 + 675 * cos_theta[1] + 594 * cos_theta[3] + 429 * cos_theta[5]) *
                sin_phi[0]) / 2048.;

    data[58] = (-3 * std::sqrt(35 / (2. * pi)) * sin_phi[1] *
                (75 * sin_theta[1] + 132 * sin_theta[3] + 143 * sin_theta[5])) / 512.;

    data[59] = (9 * std::sqrt(35 / pi) * (189 + 308 * cos_theta[1] + 143 * cos_theta[3]) * sin_phi[2] *
                std::pow(sin_theta[0], 2)) / 512.;

    data[60] = (-3 * std::sqrt(385 / pi) * cos_theta[0] * (7 + 13 * cos_theta[1]) * sin_phi[3] *
                std::pow(sin_theta[0], 3)) / 16.;

    data[61] =
            (15 * std::sqrt(385 / pi) * (11 + 13 * cos_theta[1]) * sin_phi[4] * std::pow(sin_theta[0], 4)) / 128.;

    data[62] = (-9 * std::sqrt(5005 / (2. * pi)) * cos_theta[0] * sin_phi[5] * std::pow(sin_theta[0], 5)) / 16.;

    data[63] = (21 * std::sqrt(715 / pi) * sin_phi[6] * std::pow(sin_theta[0], 6)) / 64.;

    if (lmax == 7)
        return;

    data[64] = (-3 * std::sqrt(12155 / pi) * cos_phi[7] * std::pow(sin_theta[0], 7)) / 32.;

    data[65] = (-21 * std::sqrt(12155 / pi) * cos_phi[6] * cos_theta[0] * std::pow(sin_theta[0], 6)) / 64.;

    data[66] =
            (-3 * std::sqrt(7293 / (2. * pi)) * cos_phi[5] * (13 + 15 * cos_theta[1]) * std::pow(sin_theta[0], 5)) /
            64.;

    data[67] = (-15 * std::sqrt(17017 / pi) * cos_phi[4] * (11 * cos_theta[0] + 5 * cos_theta[2]) *
                std::pow(sin_theta[0], 4)) / 256.;

    data[68] = (-3 * std::sqrt(1309 / pi) * cos_phi[3] * (99 + 156 * cos_theta[1] + 65 * cos_theta[3]) *
                std::pow(sin_theta[0], 3)) / 256.;

    data[69] = (-3 * std::sqrt(19635 / pi) * cos_phi[2] *
                (126 * cos_theta[0] + 91 * cos_theta[2] + 39 * cos_theta[4]) * std::pow(sin_theta[0], 2)) / 1024.;

    data[70] = (-3 * std::sqrt(595 / (2. * pi)) * cos_phi[1] *
                (35 * sin_theta[0] + 11 * (9 * sin_theta[2] + 13 * (sin_theta[4] + sin_theta[6])))) / 2048.;

    data[71] = (-3 * std::sqrt(17 / pi) * cos_phi[0] *
                (1225 * cos_theta[0] + 11 * (105 * cos_theta[2] + 91 * cos_theta[4] + 65 * cos_theta[6]))) / 4096.;

    data[72] = 0;

    data[73] = (3 * std::sqrt(17 / pi) * cos_theta[0] *
                (178 + 869 * cos_theta[1] + 286 * cos_theta[3] + 715 * cos_theta[5]) * sin_phi[0]) / 2048.;

    data[74] = (-3 * std::sqrt(595 / (2. * pi)) * sin_phi[1] *
                (35 * sin_theta[0] + 11 * (9 * sin_theta[2] + 13 * (sin_theta[4] + sin_theta[6])))) / 2048.;

    data[75] =
            (3 * std::sqrt(19635 / pi) * cos_theta[0] * (37 + 52 * cos_theta[1] + 39 * cos_theta[3]) * sin_phi[2] *
             std::pow(sin_theta[0], 2)) / 512.;

    data[76] = (-3 * std::sqrt(1309 / pi) * (99 + 156 * cos_theta[1] + 65 * cos_theta[3]) * sin_phi[3] *
                std::pow(sin_theta[0], 3)) / 256.;

    data[77] = (15 * std::sqrt(17017 / pi) * (11 * cos_theta[0] + 5 * cos_theta[2]) * sin_phi[4] *
                std::pow(sin_theta[0], 4)) / 256.;

    data[78] =
            (-3 * std::sqrt(7293 / (2. * pi)) * (13 + 15 * cos_theta[1]) * sin_phi[5] * std::pow(sin_theta[0], 5)) /
            64.;

    data[79] = (21 * std::sqrt(12155 / pi) * cos_theta[0] * sin_phi[6] * std::pow(sin_theta[0], 6)) / 64.;

    data[80] = (-3 * std::sqrt(12155 / pi) * sin_phi[7] * std::pow(sin_theta[0], 7)) / 32.;
}

void SHT::dRlm_dr(int lmax__, geometry3d::vector3d<double> &r__, sddk::mdarray<double, 2> &data__) {
    /* get spherical coordinates of the Cartesian vector */
    auto vrs = spherical_coordinates(r__);

    if (vrs[0] < 1e-12) {
        data__.zero();
        return;
    }

    int lmmax = (lmax__ + 1) * (lmax__ + 1);

    double theta = vrs[1];
    double phi = vrs[2];

    double sint = std::sin(theta);
    double sinp = std::sin(phi);
    double cost = std::cos(theta);
    double cosp = std::cos(phi);

    /* nominators of angle derivatives */
    geometry3d::vector3d<double> dtheta_dr({cost * cosp, cost * sinp, -sint});
    geometry3d::vector3d<double> dphi_dr({-sinp, cosp, 0});


    std::vector<double> dRlm_dt(lmmax);
    std::vector<double> dRlm_dp_sin_t(lmmax);

    std::vector<double> plm((lmax__ + 1) * (lmax__ + 2) / 2);
    std::vector<double> dplm((lmax__ + 1) * (lmax__ + 2) / 2);
    std::vector<double> plm_y((lmax__ + 1) * (lmax__ + 2) / 2);

    auto ilm = [](int l, int m){return l * (l + 1) / 2 + m;};

    dRlm_dt[0] = 0;
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
       dRlm_dt[utils::lm(l, 0)] = -dplm[ilm(l, 0)];
       dRlm_dp_sin_t[utils::lm(l, 0)] = 0;
    }

    int phase{-1};
    for (int m = 1; m <= lmax__; m++) {
        double c = c2 * c1 - c0;
        c0 = c1;
        c1 = c;
        double s = c2 * s1 - s0;
        s0 = s1;
        s1 = s;
        for (int l = m; l <= lmax__; l++) {
            double p = -dplm[ilm(l, m)];
            dRlm_dt[utils::lm(l, m)] = t * p * c;
            dRlm_dt[utils::lm(l, -m)] = -t * p * s * phase;
            p = plm_y[ilm(l, m)];
            dRlm_dp_sin_t[utils::lm(l, m)] = -t * p * s * m;
            dRlm_dp_sin_t[utils::lm(l, -m)] = -t * p * c * m * phase;
        }

        phase = -phase;
    }

    //dRlm_dtheta(lmax__, theta, phi, dRlm_dt);
    //dRlm_dphi_sin_theta(lmax__, theta, phi, dRlm_dp_sin_t);

    for (int mu = 0; mu < 3; mu++) {
        for (int lm = 0; lm < lmmax; lm++) {
            data__(lm, mu) = (dRlm_dt[lm] * dtheta_dr[mu] + dRlm_dp_sin_t[lm] * dphi_dr[mu]) / vrs[0];
        }
    }
}

double SHT::ClebschGordan(const int l, const double j, const double mj, const int spin) {
    // l : orbital angular momentum
    // m:  projection of the total angular momentum $m \pm /frac12$
    // spin: Component of the spinor, 0 up, 1 down

    double CG = 0.0; // Clebsch Gordan coeeficient cf PRB 71, 115106 page 3 first column

    if ((spin != 0) && (spin != 1)) {
        std::printf("Error : unkown spin direction\n");
    }

    const double denom = sqrt(1.0 / (2.0 * l + 1.0));

    if (std::abs(j - l - 0.5) < 1e-8) { // check for j = l + 1/2
        int m = static_cast<int>(mj - 0.5); // if mj is integer (2 * m), then int m = (mj-1) >> 1;
        if (spin == 0) {
            CG = sqrt(l + m + 1.0);
        }
        if (spin == 1) {
            CG = sqrt((l - m));
        }
    } else {
        if (std::abs(j - l + 0.5) < 1e-8) { // check for j = l - 1/2
            int m = static_cast<int>(mj + 0.5);
            if (m < (1 - l)) {
                CG = 0.0;
            } else {
                if (spin == 0) {
                    CG = sqrt(l - m + 1);
                }
                if (spin == 1) {
                    CG = -sqrt(l + m);
                }
            }
        } else {
            std::printf("Clebsch gordan coefficients do not exist for this combination of j=%.5lf and l=%d\n", j, l);
            exit(0);
        }
    }
    return (CG * denom);
}


// this function computes the U^sigma_{ljm mj} coefficient that
// rotates the complex spherical harmonics to the real one for the
// spin orbit case

// mj is normally half integer from -j to j but to avoid computation
// error it is considered as integer so mj = 2 mj

double_complex
SHT::calculate_U_sigma_m(const int l, const double j, const int mj, const int mp, const int sigma) {

    if ((sigma != 0) && (sigma != 1)) {
        std::printf("SphericalIndex function : unkown spin direction\n");
        return 0;
    }

    if (std::abs(j - l - 0.5) < 1e-8) {
        // j = l + 1/2
        // m = mj - 1/2

        int m1 = (mj - 1) >> 1;
        if (sigma == 0) { // up spin
            if (m1 < -l) { // convention U^s_{mj,m'} = 0
                return 0.0;
            } else {// U^s_{mj,mp} =
                return SHT::rlm_dot_ylm(l, m1, mp);
            }
        } else { // down spin
            if ((m1 + 1) > l) {
                return 0.0;
            } else {
                return SHT::rlm_dot_ylm(l, m1 + 1, mp);
            }
        }
    } else {
        if (std::abs(j - l + 0.5) < 1e-8) {
            int m1 = (mj + 1) >> 1;
            if (sigma == 0) {
                return SHT::rlm_dot_ylm(l, m1 - 1, mp);
            } else {
                return SHT::rlm_dot_ylm(l, m1, mp);
            }
        } else {
            std::printf("Spherical Index function : l and j are not compatible\n");
            exit(0);
        }
    }
}

template<>
void SHT::backward_transform<double>(int ld, double const *flm, int nr, int lmmax, double *ftp) const {
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    sddk::linalg2(sddk::linalg_t::blas).gemm('T', 'N', num_points_, nr, lmmax, &sddk::linalg_const<double>::one(),
        &rlm_backward_(0, 0), lmmax_, flm, ld, &sddk::linalg_const<double>::zero(), ftp, num_points_);
}

template<>
void SHT::backward_transform<double_complex>(int ld, double_complex const *flm, int nr, int lmmax,
                                             double_complex *ftp) const {
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    sddk::linalg2(sddk::linalg_t::blas).gemm('T', 'N', num_points_, nr, lmmax,
        &sddk::linalg_const<double_complex>::one(), &ylm_backward_(0, 0), lmmax_, flm, ld,
        &sddk::linalg_const<double_complex>::zero(), ftp, num_points_);
}

template<>
void SHT::forward_transform<double>(double const *ftp, int nr, int lmmax, int ld, double *flm) const {
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    sddk::linalg2(sddk::linalg_t::blas).gemm('T', 'N', lmmax, nr, num_points_, &sddk::linalg_const<double>::one(),
        &rlm_forward_(0, 0), num_points_, ftp, num_points_, &sddk::linalg_const<double>::zero(), flm, ld);
}

template<>
void SHT::forward_transform<double_complex>(double_complex const *ftp, int nr, int lmmax, int ld,
                                            double_complex *flm) const {
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    sddk::linalg2(sddk::linalg_t::blas).gemm('T', 'N', lmmax, nr, num_points_, &sddk::linalg_const<double_complex>::one(),
        &ylm_forward_(0, 0), num_points_, ftp, num_points_, &sddk::linalg_const<double_complex>::zero(), flm, ld);
}

}
