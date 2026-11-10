/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "core/sht/sht.hpp"
#include "core/math_tools.hpp"
#include "lebedev_grids.hpp"
#include "core/la/linalg.hpp"

namespace sirius {

namespace sht { // TODO: move most of this to special functions

/// Generate Wigner-D matrix.
mdarray<double, 2>
wigner_d_matrix(int l, double beta)
{
    mdarray<double, 2> d_mtrx({2 * l + 1, 2 * l + 1});

    long double cos_b2 = std::cos((long double)beta / 2.0L);
    long double sin_b2 = std::sin((long double)beta / 2.0L);

    for (int m1 = -l; m1 <= l; m1++) {
        for (int m2 = -l; m2 <= l; m2++) {
            long double d = 0;
            for (int j = 0; j <= std::min(l + m1, l - m2); j++) {
                if ((l - m2 - j) >= 0 && (l + m1 - j) >= 0 && (j + m2 - m1) >= 0) {
                    long double g = (std::sqrt(factorial<long double>(l + m1)) / factorial<long double>(l - m2 - j)) *
                                    (std::sqrt(factorial<long double>(l - m1)) / factorial<long double>(l + m1 - j)) *
                                    (std::sqrt(factorial<long double>(l - m2)) / factorial<long double>(j + m2 - m1)) *
                                    (std::sqrt(factorial<long double>(l + m2)) / factorial<long double>(j));
                    d += g * std::pow(-1, j) * std::pow(cos_b2, 2 * l + m1 - m2 - 2 * j) *
                         std::pow(sin_b2, 2 * j + m2 - m1);
                }
            }
            d_mtrx(m1 + l, m2 + l) = (double)d;
        }
    }

    return d_mtrx;
}

/// Rotation of a set of complex spherical harmonics with l orbital quantum number.
template <>
mdarray<std::complex<double>, 2>
rotation_matrix_l<std::complex<double>>(int l, r3::vector<double> euler_angles, int proper_rotation)
{
    mdarray<std::complex<double>, 2> rot_mtrx({2 * l + 1, 2 * l + 1});

    auto d_mtrx = wigner_d_matrix(l, euler_angles[1]);

    for (int m1 = -l; m1 <= l; m1++) {
        for (int m2 = -l; m2 <= l; m2++) {
            rot_mtrx(m1 + l, m2 + l) = std::exp(std::complex<double>(0, -euler_angles[0] * m1 - euler_angles[2] * m2)) *
                                       d_mtrx(m1 + l, m2 + l) * std::pow(proper_rotation, l);
        }
    }

    return rot_mtrx;
}

/// Rotation of a set of real spherical harmonics with l orbital quantum number.
template <>
mdarray<double, 2>
rotation_matrix_l<double>(int l, r3::vector<double> euler_angles, int proper_rotation)
{
    auto rot_mtrx_ylm = rotation_matrix_l<std::complex<double>>(l, euler_angles, proper_rotation);

    mdarray<double, 2> rot_mtrx({2 * l + 1, 2 * l + 1});
    rot_mtrx.zero();

    for (int m1 = -l; m1 <= l; m1++) {
        auto i13 = (m1 == 0) ? std::vector<int>({0}) : std::vector<int>({-m1, m1});

        for (int m2 = -l; m2 <= l; m2++) {
            auto i24 = (m2 == 0) ? std::vector<int>({0}) : std::vector<int>({-m2, m2});

            for (int m3 : i13) {
                for (int m4 : i24) {
                    rot_mtrx(m1 + l, m2 + l) += std::real(SHT::rlm_dot_ylm(l, m1, m3) * rot_mtrx_ylm(m3 + l, m4 + l) *
                                                          SHT::ylm_dot_rlm(l, m4, m2));
                }
            }
        }
    }
    return rot_mtrx;
}

/// Return vector of rotation matrices for each of the l-shells.
template <typename T>
std::vector<mdarray<T, 2>>
rotation_matrix(int lmax, r3::vector<double> euler_angles, int proper_rotation)
{
    std::vector<mdarray<T, 2>> result(lmax + 1);

    for (int l = 0; l <= lmax; l++) {
        result[l] = rotation_matrix_l<T>(l, euler_angles, proper_rotation);
    }
    return result;
}

/// Instantiation for real spherical harmonics.
template std::vector<mdarray<double, 2>>
rotation_matrix<double>(int lmax, r3::vector<double> euler_angles, int proper_rotation);

/// Instantiation for complex spherical harmonics.
template std::vector<mdarray<std::complex<double>, 2>>
rotation_matrix<std::complex<double>>(int lmax, r3::vector<double> euler_angles, int proper_rotation);

double
ClebschGordan(const int l, const double j, const double mj, const int spin)
{
    // l : orbital angular momentum
    // m:  projection of the total angular momentum $m \pm /frac12$
    // spin: Component of the spinor, 0 up, 1 down

    double CG = 0.0; // Clebsch Gordan coeeficient cf PRB 71, 115106 page 3 first column

    if ((spin != 0) && (spin != 1)) {
        RTE_THROW("Error : unknown spin direction");
    }

    const double denom = std::sqrt(1.0 / (2.0 * l + 1.0));

    if (std::abs(j - l - 0.5) < 1e-8) {     // check for j = l + 1/2
        int m = static_cast<int>(mj - 0.5); // if mj is integer (2 * m), then int m = (mj-1) >> 1;
        if (spin == 0) {
            CG = std::sqrt(l + m + 1.0);
        }
        if (spin == 1) {
            CG = std::sqrt((l - m));
        }
    } else {
        if (std::abs(j - l + 0.5) < 1e-8) { // check for j = l - 1/2
            int m = static_cast<int>(mj + 0.5);
            if (m < (1 - l)) {
                CG = 0.0;
            } else {
                if (spin == 0) {
                    CG = std::sqrt(l - m + 1);
                }
                if (spin == 1) {
                    CG = -std::sqrt(l + m);
                }
            }
        } else {
            std::stringstream s;
            s << "Clebsch-Gordan coefficients do not exist for this combination of j=" << j << " and l=" << l;
            RTE_THROW(s);
        }
    }
    return (CG * denom);
}

// this function computes the U^sigma_{ljm mj} coefficient that
// rotates the complex spherical harmonics to the real one for the
// spin orbit case

// mj is normally half integer from -j to j but to avoid computation
// error it is considered as integer so mj = 2 mj

std::complex<double>
calculate_U_sigma_m(const int l, const double j, const int mj, const int mp, const int sigma)
{
    if ((sigma != 0) && (sigma != 1)) {
        RTE_THROW("SphericalIndex function : unknown spin direction");
    }

    if (std::abs(j - l - 0.5) < 1e-8) {
        // j = l + 1/2
        // m = mj - 1/2

        int m1 = (mj - 1) >> 1;
        if (sigma == 0) {  // up spin
            if (m1 < -l) { // convention U^s_{mj,m'} = 0
                return 0.0;
            } else { // U^s_{mj,mp} =
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
            RTE_THROW("Spherical Index function : l and j are not compatible");
        }
    }
    return 0; // make compiler happy
}

} // namespace sht

SHT::SHT(device_t pu__, int lmax__, int mesh_type__)
    : pu_(pu__)
    , lmax_(lmax__)
    , mesh_type_(mesh_type__)
{
    lmmax_ = (lmax_ + 1) * (lmax_ + 1);

    switch (mesh_type_) {
        case 0: {
            num_points_ = Lebedev_Laikov_npoint(2 * lmax_);
            break;
        }
        case 1: {
            num_points_ = lmmax_;
            break;
        }
        default: {
            std::stringstream s;
            s << "[SHT] wrong spherical coverage parameter : " << mesh_type_;
            throw std::runtime_error(s.str());
        }
    }

    std::vector<double> x(num_points_);
    std::vector<double> y(num_points_);
    std::vector<double> z(num_points_);

    coord_ = mdarray<double, 2>({3, num_points_});

    tp_ = mdarray<double, 2>({2, num_points_});

    w_.resize(num_points_);

    switch (mesh_type_) {
        case 0: {
            Lebedev_Laikov_sphere(num_points_, &x[0], &y[0], &z[0], &w_[0]);
            break;
        }
        case 1: {
            uniform_coverage();
            break;
        }
    }

    ylm_backward_ = mdarray<std::complex<double>, 2>({lmmax_, num_points_});

    ylm_forward_ = mdarray<std::complex<double>, 2>({num_points_, lmmax_});

    rlm_backward_ = mdarray<double, 2>({lmmax_, num_points_});

    rlm_forward_ = mdarray<double, 2>({num_points_, lmmax_});

    for (int itp = 0; itp < num_points_; itp++) {
        switch (mesh_type_) {
            case 0: {
                coord_(0, itp) = x[itp];
                coord_(1, itp) = y[itp];
                coord_(2, itp) = z[itp];

                auto vs = spherical_coordinates(r3::vector<double>(x[itp], y[itp], z[itp]));

                tp_(0, itp) = vs[1];
                tp_(1, itp) = vs[2];
                sf::spherical_harmonics(lmax_, vs[1], vs[2], &ylm_backward_(0, itp));
                sf::spherical_harmonics(lmax_, vs[1], vs[2], &rlm_backward_(0, itp));
                for (int lm = 0; lm < lmmax_; lm++) {
                    ylm_forward_(itp, lm) = std::conj(ylm_backward_(lm, itp)) * w_[itp] * fourpi;
                    rlm_forward_(itp, lm) = rlm_backward_(lm, itp) * w_[itp] * fourpi;
                }
                break;
            }
            case 1: {
                double t = tp_(0, itp);
                double p = tp_(1, itp);

                coord_(0, itp) = std::sin(t) * std::cos(p);
                coord_(1, itp) = std::sin(t) * std::sin(p);
                coord_(2, itp) = std::cos(t);

                sf::spherical_harmonics(lmax_, t, p, &ylm_backward_(0, itp));
                sf::spherical_harmonics(lmax_, t, p, &rlm_backward_(0, itp));

                for (int lm = 0; lm < lmmax_; lm++) {
                    ylm_forward_(lm, itp) = ylm_backward_(lm, itp);
                    rlm_forward_(lm, itp) = rlm_backward_(lm, itp);
                }
                break;
            }
        }
    }

    if (mesh_type_ == 1) {
        la::wrap(la::lib_t::lapack).geinv(lmmax_, ylm_forward_);
        la::wrap(la::lib_t::lapack).geinv(lmmax_, rlm_forward_);
    }

    switch (pu_) {
        case device_t::GPU: {
            ylm_forward_.allocate(memory_t::device).copy_to(memory_t::device);
            rlm_forward_.allocate(memory_t::device).copy_to(memory_t::device);
            ylm_backward_.allocate(memory_t::device).copy_to(memory_t::device);
            rlm_backward_.allocate(memory_t::device).copy_to(memory_t::device);
            break;
        }
        case device_t::CPU: {
            break;
        }
    }
}

template <>
void
SHT::backward_transform<double>(int ld, double const* flm, int nr, int lmmax, double* ftp) const
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    la::wrap(la::lib_t::blas)
            .gemm('T', 'N', num_points_, nr, lmmax, &la::constant<double>::one(), &rlm_backward_(0, 0), lmmax_, flm, ld,
                  &la::constant<double>::zero(), ftp, num_points_);
}

template <>
void
SHT::backward_transform<std::complex<double>>(int ld, std::complex<double> const* flm, int nr, int lmmax,
                                              std::complex<double>* ftp) const
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    la::wrap(la::lib_t::blas)
            .gemm('T', 'N', num_points_, nr, lmmax, &la::constant<std::complex<double>>::one(), &ylm_backward_(0, 0),
                  lmmax_, flm, ld, &la::constant<std::complex<double>>::zero(), ftp, num_points_);
}

template <>
void
SHT::forward_transform<double>(double const* ftp, int nr, int lmmax, int ld, double* flm) const
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    la::wrap(la::lib_t::blas)
            .gemm('T', 'N', lmmax, nr, num_points_, &la::constant<double>::one(), &rlm_forward_(0, 0), num_points_, ftp,
                  num_points_, &la::constant<double>::zero(), flm, ld);
}

template <>
void
SHT::forward_transform<std::complex<double>>(std::complex<double> const* ftp, int nr, int lmmax, int ld,
                                             std::complex<double>* flm) const
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    la::wrap(la::lib_t::blas)
            .gemm('T', 'N', lmmax, nr, num_points_, &la::constant<std::complex<double>>::one(), &ylm_forward_(0, 0),
                  num_points_, ftp, num_points_, &la::constant<std::complex<double>>::zero(), flm, ld);
}

void
SHT::check() const
{
    double dr = 0;
    double dy = 0;

    for (int lm = 0; lm < lmmax_; lm++) {
        for (int lm1 = 0; lm1 < lmmax_; lm1++) {
            double t = 0;
            std::complex<double> zt(0, 0);
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
        RTE_WARNING(s.str())
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
            s << "test of backward / forward real SHT failed" << std::endl << "  total error " << t;
            RTE_WARNING(s.str());
        }
    }
}

} // namespace sirius
