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
void SHT::backward_transform<double>(int ld, double const *flm, int nr, int lmmax, double *ftp) const
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    sddk::linalg(sddk::linalg_t::blas).gemm('T', 'N', num_points_, nr, lmmax, &sddk::linalg_const<double>::one(),
        &rlm_backward_(0, 0), lmmax_, flm, ld, &sddk::linalg_const<double>::zero(), ftp, num_points_);
}

template<>
void SHT::backward_transform<double_complex>(int ld, double_complex const *flm, int nr, int lmmax,
                                             double_complex *ftp) const
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    sddk::linalg(sddk::linalg_t::blas).gemm('T', 'N', num_points_, nr, lmmax,
        &sddk::linalg_const<double_complex>::one(), &ylm_backward_(0, 0), lmmax_, flm, ld,
        &sddk::linalg_const<double_complex>::zero(), ftp, num_points_);
}

template<>
void SHT::forward_transform<double>(double const *ftp, int nr, int lmmax, int ld, double *flm) const
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    sddk::linalg(sddk::linalg_t::blas).gemm('T', 'N', lmmax, nr, num_points_, &sddk::linalg_const<double>::one(),
        &rlm_forward_(0, 0), num_points_, ftp, num_points_, &sddk::linalg_const<double>::zero(), flm, ld);
}

template<>
void SHT::forward_transform<double_complex>(double_complex const *ftp, int nr, int lmmax, int ld,
                                            double_complex *flm) const
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    sddk::linalg(sddk::linalg_t::blas).gemm('T', 'N', lmmax, nr, num_points_, &sddk::linalg_const<double_complex>::one(),
        &ylm_forward_(0, 0), num_points_, ftp, num_points_, &sddk::linalg_const<double_complex>::zero(), flm, ld);
}

void SHT::check() const
{
    double dr = 0;
    double dy = 0;

    for (int lm = 0; lm < lmmax_; lm++) {
        for (int lm1 = 0; lm1 < lmmax_; lm1++) {
            double         t = 0;
            double_complex zt(0, 0);
            for (int itp = 0; itp < num_points_; itp++) {
                zt += ylm_forward_(itp, lm) * ylm_backward_(lm1, itp);
                t += rlm_forward_(itp, lm) * rlm_backward_(lm1, itp);
            }

            if (lm == lm1) {
                zt -= 1.0;
                t -= 1.0;
            }
            dr += std::abs(t);
            dy += std::abs(zt);
        }
    }
    dr = dr / lmmax_ / lmmax_;
    dy = dy / lmmax_ / lmmax_;

    if (dr > 1e-15 || dy > 1e-15) {
        std::stringstream s;
        s << "spherical mesh error is too big" << std::endl
          << "  real spherical integration error " << dr << std::endl
          << "  complex spherical integration error " << dy;
        WARNING(s.str())
    }

    std::vector<double> flm(lmmax_);
    std::vector<double> ftp(num_points_);
    for (int lm = 0; lm < lmmax_; lm++) {
        std::memset(&flm[0], 0, lmmax_ * sizeof(double));
        flm[lm] = 1.0;
        backward_transform(lmmax_, &flm[0], 1, lmmax_, &ftp[0]);
        forward_transform(&ftp[0], 1, lmmax_, lmmax_, &flm[0]);
        flm[lm] -= 1.0;

        double t = 0.0;
        for (int lm1 = 0; lm1 < lmmax_; lm1++) {
            t += std::abs(flm[lm1]);
        }

        t /= lmmax_;

        if (t > 1e-15) {
            std::stringstream s;
            s << "test of backward / forward real SHT failed" << std::endl
              << "  total error " << t;
            WARNING(s.str());
        }
    }
}

}
