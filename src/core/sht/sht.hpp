/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file sht.hpp
 *
 *  \brief Contains declaration and particular implementation of sirius::SHT class.
 */

#ifndef __SHT_HPP__
#define __SHT_HPP__

#include <math.h>
#include <stddef.h>
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_legendre.h>
#include <string.h>
#include <vector>
#include <algorithm>

#include "core/constants.hpp"
#include "core/typedefs.hpp"
#include "core/r3/r3.hpp"
#include "core/sf/specfunc.hpp"
#include "core/la/linalg.hpp"
#include "lebedev_grids.hpp"

namespace sirius {

namespace sht {

mdarray<double, 2>
wigner_d_matrix(int l, double beta);

template <typename T>
mdarray<T, 2>
rotation_matrix_l(int l, r3::vector<double> euler_angles, int proper_rotation);

template <typename T>
std::vector<mdarray<T, 2>>
rotation_matrix(int lmax, r3::vector<double> euler_angles, int proper_rotation);

double
ClebschGordan(const int l, const double j, const double mj, const int spin);

// this function computes the U^sigma_{ljm mj} coefficient that
// rotates the complex spherical harmonics to the real one for the
// spin orbit case

// mj is normally half integer from -j to j but to avoid computation
// error it is considered as integer so mj = 2 mj
std::complex<double>
calculate_U_sigma_m(const int l, const double j, const int mj, const int mp, const int sigma);

} // namespace sht

/// Spherical harmonics transformations and related oprtations.
/** This class is responsible for the generation of complex and real spherical harmonics, generation of transformation
 *  matrices, transformation between spectral and real-space representations, generation of Gaunt and Clebsch-Gordan
 *  coefficients and calculation of spherical harmonic derivatives */
class SHT // TODO: better name
{
  private:
    /// Type of processing unit.
    device_t pu_;

    /// Maximum \f$ \ell \f$ of spherical harmonics.
    int lmax_;

    /// Maximum number of \f$ \ell, m \f$ components.
    int lmmax_;

    /// Number of real-space \f$ (\theta, \phi) \f$ points on the sphere.
    int num_points_;

    /// Cartesian coordinates of points (normalized to 1).
    mdarray<double, 2> coord_;

    /// \f$ (\theta, \phi) \f$ angles of points.
    mdarray<double, 2> tp_;

    /// Point weights.
    std::vector<double> w_;

    /// Backward transformation from Ylm to spherical coordinates.
    mdarray<std::complex<double>, 2> ylm_backward_;

    /// Forward transformation from spherical coordinates to Ylm.
    mdarray<std::complex<double>, 2> ylm_forward_;

    /// Backward transformation from Rlm to spherical coordinates.
    mdarray<double, 2> rlm_backward_;

    /// Forward transformation from spherical coordinates to Rlm.
    mdarray<double, 2> rlm_forward_;

    /// Type of spherical grid (0: Lebedev-Laikov, 1: uniform).
    int mesh_type_{0};

  public:
    /// Default constructor.
    SHT(device_t pu__, int lmax__, int mesh_type__ = 0)
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

    /// Check the transformations.
    void
    check() const;

    /// Perform a backward transformation from spherical harmonics to spherical coordinates.
    /** \f[
     *      f(\theta, \phi, r) = \sum_{\ell m} f_{\ell m}(r) Y_{\ell m}(\theta, \phi)
     *  \f]
     *
     *  \param [in] ld Size of leading dimension of flm.
     *  \param [in] flm Raw pointer to \f$ f_{\ell m}(r) \f$.
     *  \param [in] nr Number of radial points.
     *  \param [in] lmmax Maximum number of lm- harmonics to take into sum.
     *  \param [out] ftp Raw pointer to \f$ f(\theta, \phi, r) \f$.
     */
    template <typename T>
    void
    backward_transform(int ld, T const* flm, int nr, int lmmax, T* ftp) const;

    /// Perform a forward transformation from spherical coordinates to spherical harmonics.
    /** \f[
     *      f_{\ell m}(r) = \iint  f(\theta, \phi, r) Y_{\ell m}^{*}(\theta, \phi) \sin \theta d\phi d\theta =
     *        \sum_{i} f(\theta_i, \phi_i, r) Y_{\ell m}^{*}(\theta_i, \phi_i) w_i
     *  \f]
     *
     *  \param [in] ftp Raw pointer to \f$ f(\theta, \phi, r) \f$.
     *  \param [in] nr Number of radial points.
     *  \param [in] lmmax Maximum number of lm- coefficients to generate.
     *  \param [in] ld Size of leading dimension of flm.
     *  \param [out] flm Raw pointer to \f$ f_{\ell m}(r) \f$.
     */
    template <typename T>
    void
    forward_transform(T const* ftp, int nr, int lmmax, int ld, T* flm) const;

    /// Convert form Rlm to Ylm representation.
    static void
    convert(int lmax__, double const* f_rlm__, std::complex<double>* f_ylm__)
    {
        int lm = 0;
        for (int l = 0; l <= lmax__; l++) {
            for (int m = -l; m <= l; m++) {
                if (m == 0) {
                    f_ylm__[lm] = f_rlm__[lm];
                } else {
                    int lm1     = sf::lm(l, -m);
                    f_ylm__[lm] = ylm_dot_rlm(l, m, m) * f_rlm__[lm] + ylm_dot_rlm(l, m, -m) * f_rlm__[lm1];
                }
                lm++;
            }
        }
    }

    /// Convert from Ylm to Rlm representation.
    static void
    convert(int lmax__, std::complex<double> const* f_ylm__, double* f_rlm__)
    {
        int lm = 0;
        for (int l = 0; l <= lmax__; l++) {
            for (int m = -l; m <= l; m++) {
                if (m == 0) {
                    f_rlm__[lm] = std::real(f_ylm__[lm]);
                } else {
                    int lm1     = sf::lm(l, -m);
                    f_rlm__[lm] = std::real(rlm_dot_ylm(l, m, m) * f_ylm__[lm] + rlm_dot_ylm(l, m, -m) * f_ylm__[lm1]);
                }
                lm++;
            }
        }
    }

    // void rlm_forward_iterative_transform(double *ftp__, int lmmax, int ncol, double* flm)
    //{
    //     Timer t("sirius::SHT::rlm_forward_iterative_transform");
    //
    //     RTE_ASSERT(lmmax <= lmmax_);

    //    mdarray<double, 2> ftp(ftp__, num_points_, ncol);
    //    mdarray<double, 2> ftp1(num_points_, ncol);
    //
    //    blas<cpu>::gemm(1, 0, lmmax, ncol, num_points_, 1.0, &rlm_forward_(0, 0), num_points_, &ftp(0, 0),
    //    num_points_, 0.0,
    //                    flm, lmmax);
    //
    //    for (int i = 0; i < 2; i++)
    //    {
    //        rlm_backward_transform(flm, lmmax, ncol, &ftp1(0, 0));
    //        double tdiff = 0.0;
    //        for (int ir = 0; ir < ncol; ir++)
    //        {
    //            for (int itp = 0; itp < num_points_; itp++)
    //            {
    //                ftp1(itp, ir) = ftp(itp, ir) - ftp1(itp, ir);
    //                //tdiff += fabs(ftp1(itp, ir));
    //            }
    //        }
    //
    //        for (int itp = 0; itp < num_points_; itp++)
    //        {
    //            tdiff += fabs(ftp1(itp, ncol - 1));
    //        }
    //        std::cout << "iter : " << i << " avg. MT diff = " << tdiff / num_points_ << std::endl;
    //        blas<cpu>::gemm(1, 0, lmmax, ncol, num_points_, 1.0, &rlm_forward_(0, 0), num_points_, &ftp1(0, 0),
    //        num_points_, 1.0,
    //                        flm, lmmax);
    //    }
    //}

    /// Compute element of the transformation matrix from complex to real spherical harmonics.
    /** Real spherical harmonic can be written as a linear combination of complex harmonics:
        \f[
            R_{\ell m}(\theta, \phi) = \sum_{m'} a^{\ell}_{m' m}Y_{\ell m'}(\theta, \phi)
        \f]
        where
        \f[
            a^{\ell}_{m' m} = \langle Y_{\ell m'} | R_{\ell m} \rangle
        \f]
        which gives the name to this function.

        Transformation from real to complex spherical harmonics is conjugate transpose:

        \f[
            Y_{\ell m}(\theta, \phi) = \sum_{m'} a^{\ell*}_{m m'}R_{\ell m'}(\theta, \phi)
        \f]

        Mathematica code:
        \verbatim
        b[m1_, m2_] :=
         If[m1 == 0, 1,
         If[m1 < 0 && m2 < 0, -I/Sqrt[2],
         If[m1 > 0 && m2 < 0, (-1)^m1*I/Sqrt[2],
         If[m1 < 0 && m2 > 0, (-1)^m2/Sqrt[2],
         If[m1 > 0 && m2 > 0, 1/Sqrt[2]]]]]]

        a[m1_, m2_] := If[Abs[m1] == Abs[m2], b[m1, m2], 0]

        Rlm[l_, m_, t_, p_] := Sum[a[m1, m]*SphericalHarmonicY[l, m1, t, p], {m1, -l, l}]
        \endverbatim
     */
    static inline std::complex<double>
    ylm_dot_rlm(int l, int m1, int m2)
    {
        double const isqrt2 = 1.0 / std::sqrt(2);

        RTE_ASSERT(l >= 0 && std::abs(m1) <= l && std::abs(m2) <= l);

        if (!((m1 == m2) || (m1 == -m2))) {
            return std::complex<double>(0, 0);
        }

        if (m1 == 0) {
            return std::complex<double>(1, 0);
        }

        if (m1 < 0) {
            if (m2 < 0) {
                return -std::complex<double>(0, isqrt2);
            } else {
                return std::pow(-1.0, m2) * std::complex<double>(isqrt2, 0);
            }
        } else {
            if (m2 < 0) {
                return std::pow(-1.0, m1) * std::complex<double>(0, isqrt2);
            } else {
                return std::complex<double>(isqrt2, 0);
            }
        }
    }

    static inline std::complex<double>
    rlm_dot_ylm(int l, int m1, int m2)
    {
        return std::conj(ylm_dot_rlm(l, m2, m1));
    }

    /// Gaunt coefficent of three complex spherical harmonics.
    /**
     *  \f[
     *    \langle Y_{\ell_1 m_1} | Y_{\ell_2 m_2} | Y_{\ell_3 m_3} \rangle
     *  \f]
     */
    static double
    gaunt_yyy(int l1, int l2, int l3, int m1, int m2, int m3)
    {
        RTE_ASSERT(l1 >= 0);
        RTE_ASSERT(l2 >= 0);
        RTE_ASSERT(l3 >= 0);
        RTE_ASSERT(m1 >= -l1 && m1 <= l1);
        RTE_ASSERT(m2 >= -l2 && m2 <= l2);
        RTE_ASSERT(m3 >= -l3 && m3 <= l3);

        return std::pow(-1.0, std::abs(m1)) *
               std::sqrt(double(2 * l1 + 1) * double(2 * l2 + 1) * double(2 * l3 + 1) / fourpi) *
               gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, 0, 0, 0) *
               gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, -2 * m1, 2 * m2, 2 * m3);
    }

    /// Gaunt coefficent of three real spherical harmonics.
    /**
     *  \f[
     *    \langle R_{\ell_1 m_1} | R_{\ell_2 m_2} | R_{\ell_3 m_3} \rangle
     *  \f]
     */
    static double
    gaunt_rrr(int l1, int l2, int l3, int m1, int m2, int m3)
    {
        RTE_ASSERT(l1 >= 0);
        RTE_ASSERT(l2 >= 0);
        RTE_ASSERT(l3 >= 0);
        RTE_ASSERT(m1 >= -l1 && m1 <= l1);
        RTE_ASSERT(m2 >= -l2 && m2 <= l2);
        RTE_ASSERT(m3 >= -l3 && m3 <= l3);

        double d = 0;
        for (int k1 = -l1; k1 <= l1; k1++) {
            for (int k2 = -l2; k2 <= l2; k2++) {
                for (int k3 = -l3; k3 <= l3; k3++) {
                    d += std::real(std::conj(SHT::ylm_dot_rlm(l1, k1, m1)) * SHT::ylm_dot_rlm(l2, k2, m2) *
                                   SHT::ylm_dot_rlm(l3, k3, m3)) *
                         SHT::gaunt_yyy(l1, l2, l3, k1, k2, k3);
                }
            }
        }
        return d;
    }

    /// Gaunt coefficent of two real spherical harmonics with a complex one.
    /**
     *  \f[
     *    \langle R_{\ell_1 m_1} | Y_{\ell_2 m_2} | R_{\ell_3 m_3} \rangle
     *  \f]
     */
    static double
    gaunt_rlm_ylm_rlm(int l1, int l2, int l3, int m1, int m2, int m3)
    {
        RTE_ASSERT(l1 >= 0);
        RTE_ASSERT(l2 >= 0);
        RTE_ASSERT(l3 >= 0);
        RTE_ASSERT(m1 >= -l1 && m1 <= l1);
        RTE_ASSERT(m2 >= -l2 && m2 <= l2);
        RTE_ASSERT(m3 >= -l3 && m3 <= l3);

        double d = 0;
        for (int k1 = -l1; k1 <= l1; k1++) {
            for (int k3 = -l3; k3 <= l3; k3++) {
                d += std::real(std::conj(SHT::ylm_dot_rlm(l1, k1, m1)) * SHT::ylm_dot_rlm(l3, k3, m3)) *
                     SHT::gaunt_yyy(l1, l2, l3, k1, m2, k3);
            }
        }
        return d;
    }

    /// Gaunt coefficent of two complex and one real spherical harmonics.
    /**
     *  \f[
     *    \langle Y_{\ell_1 m_1} | R_{\ell_2 m_2} | Y_{\ell_3 m_3} \rangle
     *  \f]
     */
    static std::complex<double>
    gaunt_hybrid(int l1, int l2, int l3, int m1, int m2, int m3)
    {
        RTE_ASSERT(l1 >= 0);
        RTE_ASSERT(l2 >= 0);
        RTE_ASSERT(l3 >= 0);
        RTE_ASSERT(m1 >= -l1 && m1 <= l1);
        RTE_ASSERT(m2 >= -l2 && m2 <= l2);
        RTE_ASSERT(m3 >= -l3 && m3 <= l3);

        if (m2 == 0) {
            return std::complex<double>(gaunt_yyy(l1, l2, l3, m1, m2, m3), 0.0);
        } else {
            return (ylm_dot_rlm(l2, m2, m2) * gaunt_yyy(l1, l2, l3, m1, m2, m3) +
                    ylm_dot_rlm(l2, -m2, m2) * gaunt_yyy(l1, l2, l3, m1, -m2, m3));
        }
    }

    void
    uniform_coverage()
    {
        tp_(0, 0) = pi;
        tp_(1, 0) = 0;

        for (int k = 1; k < num_points_ - 1; k++) {
            double hk = -1.0 + double(2 * k) / double(num_points_ - 1);
            tp_(0, k) = std::acos(hk);
            double t  = tp_(1, k - 1) + 3.80925122745582 / std::sqrt(double(num_points_)) / std::sqrt(1 - hk * hk);
            tp_(1, k) = std::fmod(t, twopi);
        }

        tp_(0, num_points_ - 1) = 0;
        tp_(1, num_points_ - 1) = 0;
    }

    /// Return Clebsch-Gordan coefficient.
    /** Clebsch-Gordan coefficients arise when two angular momenta are combined into a
     *  total angular momentum.
     */
    static inline double
    clebsch_gordan(int l1, int l2, int l3, int m1, int m2, int m3)
    {
        RTE_ASSERT(l1 >= 0);
        RTE_ASSERT(l2 >= 0);
        RTE_ASSERT(l3 >= 0);
        RTE_ASSERT(m1 >= -l1 && m1 <= l1);
        RTE_ASSERT(m2 >= -l2 && m2 <= l2);
        RTE_ASSERT(m3 >= -l3 && m3 <= l3);

        return std::pow(-1, l1 - l2 + m3) * std::sqrt(double(2 * l3 + 1)) *
               gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, 2 * m1, 2 * m2, -2 * m3);
    }

    inline auto
    ylm_backward(int lm, int itp) const
    {
        return ylm_backward_(lm, itp);
    }

    inline auto
    rlm_backward(int lm, int itp) const
    {
        return rlm_backward_(lm, itp);
    }

    inline auto
    coord(int x, int itp) const
    {
        return coord_(x, itp);
    }

    inline auto
    coord(int idx__) const
    {
        return r3::vector<double>(coord_(0, idx__), coord_(1, idx__), coord(2, idx__));
    }

    inline auto
    theta(int idx__) const
    {
        return tp_(0, idx__);
    }

    inline auto
    phi(int idx__) const
    {
        return tp_(1, idx__);
    }

    inline auto
    weight(int idx__) const
    {
        return w_[idx__];
    }

    inline auto
    num_points() const
    {
        return num_points_;
    }

    inline auto
    lmax() const
    {
        return lmax_;
    }

    inline auto
    lmmax() const
    {
        return lmmax_;
    }
};

} // namespace sirius

#endif // __SHT_HPP__
