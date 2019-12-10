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
#include <geometry3d.hpp>
#include <constants.hpp>

#include "typedefs.hpp"
#include "linalg.hpp"
#include "lebedev_grids.hpp"
#include "specfunc/specfunc.hpp"

namespace sirius {

namespace sht {

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
inline void spherical_harmonics_ref(int lmax, double theta, double phi, double_complex* ylm)
{
    double x = std::cos(theta);

    std::vector<double> result_array(gsl_sf_legendre_array_n(lmax));
    gsl_sf_legendre_array(GSL_SF_LEGENDRE_SPHARM, lmax, x, &result_array[0]);

    for (int l = 0; l <= lmax; l++) {
        ylm[utils::lm(l, 0)] = result_array[gsl_sf_legendre_array_index(l, 0)];
    }

    for (int m = 1; m <= lmax; m++) {
        double_complex z = std::exp(double_complex(0.0, m * phi)) * std::pow(-1, m);
        for (int l = m; l <= lmax; l++) {
            ylm[utils::lm(l, m)] = result_array[gsl_sf_legendre_array_index(l, m)] * z;
            if (m % 2) {
                ylm[utils::lm(l, -m)] = -std::conj(ylm[utils::lm(l, m)]);
            } else {
                ylm[utils::lm(l, -m)] = std::conj(ylm[utils::lm(l, m)]);
            }
        }
    }
}

/// Optimized implementation of complex spherical harmonics.
inline void spherical_harmonics(int lmax, double theta, double phi, double_complex* ylm)
{
    double x = std::cos(theta);

    sf::legendre_plm(lmax, x, utils::lm, ylm);

    double c0 = std::cos(phi);
    double c1 = 1;
    double s0 = -std::sin(phi);
    double s1 = 0;
    double c2 = 2 * c0;

    int phase{-1};

    for (int m = 1; m <= lmax; m++) {
        double c = c2 * c1 - c0;
        c0 = c1;
        c1 = c;
        double s = c2 * s1 - s0;
        s0 = s1;
        s1 = s;
        for (int l = m; l <= lmax; l++) {
            double p = std::real(ylm[utils::lm(l, m)]);
            double p1 = p * phase;
            ylm[utils::lm(l, m)] = double_complex(p * c, p * s);
            ylm[utils::lm(l, -m)] = double_complex(p1 * c, -p1 * s);
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
inline void spherical_harmonics_ref(int lmax, double theta, double phi, double* rlm)
{
    /* reference code */
    int lmmax = (lmax + 1) * (lmax + 1);

    std::vector<double_complex> ylm(lmmax);
    sht::spherical_harmonics_ref(lmax, theta, phi, &ylm[0]);

    double const t = std::sqrt(2.0);

    rlm[0] = y00;

    for (int l = 1; l <= lmax; l++) {
        rlm[utils::lm(l, 0)] = ylm[utils::lm(l, 0)].real();
        for (int m = 1; m <= l; m++) {
            rlm[utils::lm(l, m)]  = t * ylm[utils::lm(l, m)].real();
            rlm[utils::lm(l, -m)] = t * ylm[utils::lm(l, -m)].imag();
        }
    }
}

/// Optimized implementation of real spherical harmonics.
inline void spherical_harmonics(int lmax, double theta, double phi, double* rlm)
{
    double x = std::cos(theta);

    sf::legendre_plm(lmax, x, utils::lm, rlm);

    double c0 = std::cos(phi);
    double c1 = 1;
    double s0 = -std::sin(phi);
    double s1 = 0;
    double c2 = 2 * c0;

    double const t = std::sqrt(2.0);

    int phase{-1};

    for (int m = 1; m <= lmax; m++) {
        double c = c2 * c1 - c0;
        c0 = c1;
        c1 = c;
        double s = c2 * s1 - s0;
        s0 = s1;
        s1 = s;
        for (int l = m; l <= lmax; l++) {
            double p = rlm[utils::lm(l, m)];
            rlm[utils::lm(l, m)] = t * p * c;
            rlm[utils::lm(l, -m)] = -t * p * s * phase;
        }
        phase = -phase;
    }
}

/// Generate \f$ \cos(m x) \f$ for m in [1, n] using recursion.
inline sddk::mdarray<double, 1> cosxn(int n__, double x__)
{
    assert(n__ > 0);
    sddk::mdarray<double, 1> data(n__);

    double c0 = std::cos(x__);
    double c1 = 1;
    double c2 = 2 * c0;
    for (int m = 0; m < n__; m++) {
        data[m] = c2 * c1 - c0;
        c0 = c1;
        c1 = data[m];
    }
    return data;
}

/// Generate \f$ \sin(m x) \f$ for m in [1, n] using recursion.
inline sddk::mdarray<double, 1> sinxn(int n__, double x__)
{
    assert(n__ > 0);
    sddk::mdarray<double, 1> data(n__);

    double s0 = -std::sin(x__);
    double s1 = 0;
    double c2 = 2 * std::cos(x__);

    for (int m = 0; m < n__; m++) {
        data[m] = c2 * s1 - s0;
        s0 = s1;
        s1 = data[m];
    }
    return data;
}

} // namespace sht

/// Spherical harmonics transformations and related oprtations.
/** This class is responsible for the generation of complex and real spherical harmonics, generation of transformation
 *  matrices, transformation between spectral and real-space representations, generation of Gaunt and Clebsch-Gordan
 *  coefficients and calculation of spherical harmonic derivatives */
class SHT // TODO: better name
{
  private:
    /// Maximum \f$ \ell \f$ of spherical harmonics.
    int lmax_;

    /// Maximum number of \f$ \ell, m \f$ components.
    int lmmax_;

    /// Number of real-space \f$ (\theta, \phi) \f$ points on the sphere.
    int num_points_;

    /// Cartesian coordinates of points (normalized to 1).
    sddk::mdarray<double, 2> coord_;

    /// \f$ (\theta, \phi) \f$ angles of points.
    sddk::mdarray<double, 2> tp_;

    /// Point weights.
    std::vector<double> w_;

    /// Backward transformation from Ylm to spherical coordinates.
    sddk::mdarray<double_complex, 2> ylm_backward_;

    /// Forward transformation from spherical coordinates to Ylm.
    sddk::mdarray<double_complex, 2> ylm_forward_;

    /// Backward transformation from Rlm to spherical coordinates.
    sddk::mdarray<double, 2> rlm_backward_;

    /// Forward transformation from spherical coordinates to Rlm.
    sddk::mdarray<double, 2> rlm_forward_;

    /// Type of spherical grid (0: Lebedev-Laikov, 1: uniform).
    int mesh_type_{0};

  public:
    /// Default constructor.
    SHT(int lmax__)
        : lmax_(lmax__)
    {
        lmmax_ = (lmax_ + 1) * (lmax_ + 1);

        if (mesh_type_ == 0) {
            num_points_ = Lebedev_Laikov_npoint(2 * lmax_);
        }
        if (mesh_type_ == 1) {
            num_points_ = lmmax_;
        }

        std::vector<double> x(num_points_);
        std::vector<double> y(num_points_);
        std::vector<double> z(num_points_);

        coord_ = sddk::mdarray<double, 2>(3, num_points_);

        tp_ = sddk::mdarray<double, 2>(2, num_points_);

        w_.resize(num_points_);

        if (mesh_type_ == 0) {
            Lebedev_Laikov_sphere(num_points_, &x[0], &y[0], &z[0], &w_[0]);
        }
        if (mesh_type_ == 1) {
            uniform_coverage();
        }

        ylm_backward_ = sddk::mdarray<double_complex, 2>(lmmax_, num_points_);

        ylm_forward_ = sddk::mdarray<double_complex, 2>(num_points_, lmmax_);

        rlm_backward_ = sddk::mdarray<double, 2>(lmmax_, num_points_);

        rlm_forward_ = sddk::mdarray<double, 2>(num_points_, lmmax_);

        for (int itp = 0; itp < num_points_; itp++) {
            if (mesh_type_ == 0) {
                coord_(0, itp) = x[itp];
                coord_(1, itp) = y[itp];
                coord_(2, itp) = z[itp];

                auto vs     = spherical_coordinates(geometry3d::vector3d<double>(x[itp], y[itp], z[itp]));
                tp_(0, itp) = vs[1];
                tp_(1, itp) = vs[2];
                sht::spherical_harmonics(lmax_, vs[1], vs[2], &ylm_backward_(0, itp));
                sht::spherical_harmonics(lmax_, vs[1], vs[2], &rlm_backward_(0, itp));
                for (int lm = 0; lm < lmmax_; lm++) {
                    ylm_forward_(itp, lm) = std::conj(ylm_backward_(lm, itp)) * w_[itp] * fourpi;
                    rlm_forward_(itp, lm) = rlm_backward_(lm, itp) * w_[itp] * fourpi;
                }
            }
            if (mesh_type_ == 1) {
                double t = tp_(0, itp);
                double p = tp_(1, itp);

                coord_(0, itp) = sin(t) * cos(p);
                coord_(1, itp) = sin(t) * sin(p);
                coord_(2, itp) = cos(t);

                sht::spherical_harmonics(lmax_, t, p, &ylm_backward_(0, itp));
                sht::spherical_harmonics(lmax_, t, p, &rlm_backward_(0, itp));

                for (int lm = 0; lm < lmmax_; lm++) {
                    ylm_forward_(lm, itp) = ylm_backward_(lm, itp);
                    rlm_forward_(lm, itp) = rlm_backward_(lm, itp);
                }
            }
        }

        if (mesh_type_ == 1) {
            sddk::linalg2(sddk::linalg_t::lapack).geinv(lmmax_, ylm_forward_);
            sddk::linalg2(sddk::linalg_t::lapack).geinv(lmmax_, rlm_forward_);
        }

#if (__VERIFICATION > 0)
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
#endif
    }

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
    void backward_transform(int ld, T const* flm, int nr, int lmmax, T* ftp) const;

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
    void forward_transform(T const* ftp, int nr, int lmmax, int ld, T* flm) const;

    /// Convert form Rlm to Ylm representation.
    static void convert(int lmax__, double const* f_rlm__, double_complex* f_ylm__)
    {
        int lm = 0;
        for (int l = 0; l <= lmax__; l++) {
            for (int m = -l; m <= l; m++) {
                if (m == 0) {
                    f_ylm__[lm] = f_rlm__[lm];
                } else {
                    int lm1     = utils::lm(l, -m);
                    f_ylm__[lm] = ylm_dot_rlm(l, m, m) * f_rlm__[lm] + ylm_dot_rlm(l, m, -m) * f_rlm__[lm1];
                }
                lm++;
            }
        }
    }

    /// Convert from Ylm to Rlm representation.
    static void convert(int lmax__, double_complex const* f_ylm__, double* f_rlm__)
    {
        int lm = 0;
        for (int l = 0; l <= lmax__; l++) {
            for (int m = -l; m <= l; m++) {
                if (m == 0) {
                    f_rlm__[lm] = std::real(f_ylm__[lm]);
                } else {
                    int lm1     = utils::lm(l, -m);
                    f_rlm__[lm] = std::real(rlm_dot_ylm(l, m, m) * f_ylm__[lm] + rlm_dot_ylm(l, m, -m) * f_ylm__[lm1]);
                }
                lm++;
            }
        }
    }

    //void rlm_forward_iterative_transform(double *ftp__, int lmmax, int ncol, double* flm)
    //{
    //    Timer t("sirius::SHT::rlm_forward_iterative_transform");
    //
    //    assert(lmmax <= lmmax_);

    //    mdarray<double, 2> ftp(ftp__, num_points_, ncol);
    //    mdarray<double, 2> ftp1(num_points_, ncol);
    //
    //    blas<cpu>::gemm(1, 0, lmmax, ncol, num_points_, 1.0, &rlm_forward_(0, 0), num_points_, &ftp(0, 0), num_points_, 0.0,
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
    //        blas<cpu>::gemm(1, 0, lmmax, ncol, num_points_, 1.0, &rlm_forward_(0, 0), num_points_, &ftp1(0, 0), num_points_, 1.0,
    //                        flm, lmmax);
    //    }
    //}

    /// Transform Cartesian coordinates [x,y,z] to spherical coordinates [r,theta,phi]
    static geometry3d::vector3d<double> spherical_coordinates(geometry3d::vector3d<double> vc)
    {
        geometry3d::vector3d<double> vs;

        const double eps{1e-12};

        vs[0] = vc.length();

        if (vs[0] <= eps) {
            vs[1] = 0.0;
            vs[2] = 0.0;
        } else {
            vs[1] = std::acos(vc[2] / vs[0]); // theta = cos^{-1}(z/r)

            if (std::abs(vc[0]) > eps || std::abs(vc[1]) > eps) {
                vs[2] = std::atan2(vc[1], vc[0]); // phi = tan^{-1}(y/x)
                if (vs[2] < 0.0) {
                    vs[2] += twopi;
                }
            } else {
                vs[2] = 0.0;
            }
        }

        return vs;
    }

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
    static inline double_complex ylm_dot_rlm(int l, int m1, int m2)
    {
        double const isqrt2 = 1.0 / std::sqrt(2);

        assert(l >= 0 && std::abs(m1) <= l && std::abs(m2) <= l);

        if (!((m1 == m2) || (m1 == -m2))) {
            return double_complex(0, 0);
        }

        if (m1 == 0) {
            return double_complex(1, 0);
        }

        if (m1 < 0) {
            if (m2 < 0) {
                return -double_complex(0, isqrt2);
            } else {
                return std::pow(-1.0, m2) * double_complex(isqrt2, 0);
            }
        } else {
            if (m2 < 0) {
                return std::pow(-1.0, m1) * double_complex(0, isqrt2);
            } else {
                return double_complex(isqrt2, 0);
            }
        }
    }

    static inline double_complex rlm_dot_ylm(int l, int m1, int m2)
    {
        return std::conj(ylm_dot_rlm(l, m2, m1));
    }

    /// Gaunt coefficent of three complex spherical harmonics.
    /**
     *  \f[
     *    \langle Y_{\ell_1 m_1} | Y_{\ell_2 m_2} | Y_{\ell_3 m_3} \rangle
     *  \f]
     */
    static double gaunt_ylm(int l1, int l2, int l3, int m1, int m2, int m3)
    {
        assert(l1 >= 0);
        assert(l2 >= 0);
        assert(l3 >= 0);
        assert(m1 >= -l1 && m1 <= l1);
        assert(m2 >= -l2 && m2 <= l2);
        assert(m3 >= -l3 && m3 <= l3);

        return std::pow(-1.0, std::abs(m1)) * std::sqrt(double(2 * l1 + 1) * double(2 * l2 + 1) * double(2 * l3 + 1) / fourpi) *
               gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, 0, 0, 0) *
               gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, -2 * m1, 2 * m2, 2 * m3);
    }

    /// Gaunt coefficent of three real spherical harmonics.
    /**
     *  \f[
     *    \langle R_{\ell_1 m_1} | R_{\ell_2 m_2} | R_{\ell_3 m_3} \rangle
     *  \f]
     */
    static double gaunt_rlm(int l1, int l2, int l3, int m1, int m2, int m3)
    {
        assert(l1 >= 0);
        assert(l2 >= 0);
        assert(l3 >= 0);
        assert(m1 >= -l1 && m1 <= l1);
        assert(m2 >= -l2 && m2 <= l2);
        assert(m3 >= -l3 && m3 <= l3);

        double d = 0;
        for (int k1 = -l1; k1 <= l1; k1++) {
            for (int k2 = -l2; k2 <= l2; k2++) {
                for (int k3 = -l3; k3 <= l3; k3++) {
                    d += std::real(std::conj(SHT::ylm_dot_rlm(l1, k1, m1)) *
                                   SHT::ylm_dot_rlm(l2, k2, m2) *
                                   SHT::ylm_dot_rlm(l3, k3, m3)) *
                         SHT::gaunt_ylm(l1, l2, l3, k1, k2, k3);
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
    static double gaunt_rlm_ylm_rlm(int l1, int l2, int l3, int m1, int m2, int m3)
    {
        assert(l1 >= 0);
        assert(l2 >= 0);
        assert(l3 >= 0);
        assert(m1 >= -l1 && m1 <= l1);
        assert(m2 >= -l2 && m2 <= l2);
        assert(m3 >= -l3 && m3 <= l3);

        double d = 0;
        for (int k1 = -l1; k1 <= l1; k1++) {
            for (int k3 = -l3; k3 <= l3; k3++) {
                d += std::real(std::conj(SHT::ylm_dot_rlm(l1, k1, m1)) *
                               SHT::ylm_dot_rlm(l3, k3, m3)) *
                     SHT::gaunt_ylm(l1, l2, l3, k1, m2, k3);
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
    static double_complex gaunt_hybrid(int l1, int l2, int l3, int m1, int m2, int m3)
    {
        assert(l1 >= 0);
        assert(l2 >= 0);
        assert(l3 >= 0);
        assert(m1 >= -l1 && m1 <= l1);
        assert(m2 >= -l2 && m2 <= l2);
        assert(m3 >= -l3 && m3 <= l3);

        if (m2 == 0) {
            return double_complex(gaunt_ylm(l1, l2, l3, m1, m2, m3), 0.0);
        } else {
            return (ylm_dot_rlm(l2, m2, m2) * gaunt_ylm(l1, l2, l3, m1, m2, m3) +
                    ylm_dot_rlm(l2, -m2, m2) * gaunt_ylm(l1, l2, l3, m1, -m2, m3));
        }
    }

    void uniform_coverage()
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
    static inline double clebsch_gordan(int l1, int l2, int l3, int m1, int m2, int m3)
    {
        assert(l1 >= 0);
        assert(l2 >= 0);
        assert(l3 >= 0);
        assert(m1 >= -l1 && m1 <= l1);
        assert(m2 >= -l2 && m2 <= l2);
        assert(m3 >= -l3 && m3 <= l3);

        return std::pow(-1, l1 - l2 + m3) * std::sqrt(double(2 * l3 + 1)) *
               gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, 2 * m1, 2 * m2, -2 * m3);
    }

    inline double_complex ylm_backward(int lm, int itp) const
    {
        return ylm_backward_(lm, itp);
    }

    inline double rlm_backward(int lm, int itp) const
    {
        return rlm_backward_(lm, itp);
    }

    inline double coord(int x, int itp) const
    {
        return coord_(x, itp);
    }

    inline geometry3d::vector3d<double> coord(int idx__) const
    {
        return geometry3d::vector3d<double>(coord_(0, idx__), coord_(1, idx__), coord(2, idx__));
    }

    inline double theta(int idx__) const
    {
        return tp_(0, idx__);
    }

    inline double phi(int idx__) const
    {
        return tp_(1, idx__);
    }

    inline double weight(int idx__) const
    {
        return w_[idx__];
    }

    inline int num_points() const
    {
        return num_points_;
    }

    inline int lmax() const
    {
        return lmax_;
    }

    inline int lmmax() const
    {
        return lmmax_;
    }

    static void wigner_d_matrix(int l, double beta, sddk::mdarray<double, 2>& d_mtrx__);

    static void rotation_matrix_l(int l, geometry3d::vector3d<double> euler_angles, int proper_rotation,
                                  double_complex* rot_mtrx__, int ld);

    static void rotation_matrix_l(int l, geometry3d::vector3d<double> euler_angles, int proper_rotation,
                                  double* rot_mtrx__, int ld);

    template <typename T>
    static void rotation_matrix(int              lmax,
                                geometry3d::vector3d<double> euler_angles,
                                int              proper_rotation,
                                sddk::mdarray<T, 2>&   rotm)
    {
        rotm.zero();

        for (int l = 0; l <= lmax; l++) {
            rotation_matrix_l(l, euler_angles, proper_rotation, &rotm(l * l, l * l), rotm.ld());
        }
    }

    /// Compute derivative of real-spherical harmonic with respect to theta angle.
    static void dRlm_dtheta(int lmax, double theta, double phi, sddk::mdarray<double, 1>& data);

    ///  Compute derivative of real-spherical harmonic with respect to phi angle and divide by sin(theta).
    static void dRlm_dphi_sin_theta(int lmax, double theta, double phi, sddk::mdarray<double, 1>& data);

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
        The derivative of \f$ \phi \f$ has discontinuities at \f$ \theta = 0, \theta=\pi \f$. This, however, is not a problem, because
        multiplication by the the derivative of \f$ R_{\ell m} \f$ removes it. The following functions have to be hardcoded:
        \f[
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
    static void dRlm_dr(int lmax__, geometry3d::vector3d<double>& r__, sddk::mdarray<double, 2>& data__);

    /// Generate \f$ \cos(m x) \f$ for m in [1, n] using recursion.
    static sddk::mdarray<double, 1> cosxn(int n__, double x__)
    {
        assert(n__ > 0);
        sddk::mdarray<double, 1> data(n__);
        data[0] = std::cos(x__);
        if (n__ > 1) {
            data[1] = std::cos(2 * x__);
            for (int i = 2; i < n__; i++) {
                data[i] = 2 * data[0] * data[i - 1] - data[i - 2];
            }
        }
        return data;
    }

    /// Generate \f$ \sin(m x) \f$ for m in [1, n] using recursion.
    static sddk::mdarray<double, 1> sinxn(int n__, double x__)
    {
        assert(n__ > 0);
        sddk::mdarray<double, 1> data(n__);
        auto               cosx = std::cos(x__);
        data[0]                 = std::sin(x__);
        if (n__ > 1) {
            data[1] = std::sin(2 * x__);
            for (int i = 2; i < n__; i++) {
                data[i] = 2 * cosx * data[i - 1] - data[i - 2];
            }
        }
        return data;
    }

    static double ClebschGordan(const int l, const double j, const double mj, const int spin);

    // this function computes the U^sigma_{ljm mj} coefficient that
    // rotates the complex spherical harmonics to the real one for the
    // spin orbit case

    // mj is normally half integer from -j to j but to avoid computation
    // error it is considered as integer so mj = 2 mj
    static double_complex
    calculate_U_sigma_m(const int l, const double j, const int mj, const int mp, const int sigma);
};
} // namespace sirius

#endif // __SHT_HPP__
