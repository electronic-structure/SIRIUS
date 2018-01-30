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

/** \file sbessel.h
 *   
 *  \brief Contains implementation of sirius::Spherical_Bessel_functions and sirius::sbessel_approx classes.
 */

#ifndef __SBESSEL_PW_H__
#define __SBESSEL_PW_H__

#include <gsl/gsl_sf_bessel.h>
#include "eigenproblem.h"
#include "Unit_cell/unit_cell.h"

namespace sirius
{

/// Spherical Bessel functions \f$ j_{\ell}(q x) \f$ up to lmax.
class Spherical_Bessel_functions
{
    private:
        int lmax_{-1};
        double q_{0};
        Radial_grid<double> const* rgrid_{nullptr};

        std::vector<Spline<double>> sbessel_;

    public:

        Spherical_Bessel_functions()
        {
        }

        Spherical_Bessel_functions(int lmax__, Radial_grid<double> const& rgrid__, double q__)
            : lmax_(lmax__)
            , q_(q__)
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
                gsl_sf_bessel_jl_array(lmax__ + 1, t, &jl[0]);
                for (int l = 0; l <= lmax__ + 1; l++) {
                    sbessel_[l][ir] = jl[l];
                }
            }
            
            for (int l = 0; l <= lmax__ + 1; l++) {
                sbessel_[l].interpolate();
            }
        }

        static void sbessel(int lmax__, double t__, double* jl__)
        {
            gsl_sf_bessel_jl_array(lmax__, t__, jl__);
        }

        static void sbessel_deriv_q(int lmax__, double q__, double x__, double* jl_dq__)
        {
            std::vector<double> jl(lmax__ + 2);
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

        Spline<double> const& operator[](int l__) const
        {
            assert(l__ <= lmax_);
            return sbessel_[l__];
        }
        
        /// Derivative of Bessel function with respect to q.
        /** \f[
         *    \frac{\partial j_{\ell}(q x)}{\partial q} = \frac{\ell}{q} j_{\ell}(q x) - x j_{\ell+1}(q x)
         *  \f]
         */
        Spline<double> deriv_q(int l__)
        {
            assert(l__ <= lmax_);
            assert(q_ >= 0);
            Spline<double> s(*rgrid_);
            if (q_ != 0) {
                for (int ir = 0; ir < rgrid_->num_points(); ir++) {
                    s[ir] = (l__ / q_) * sbessel_[l__][ir] - (*rgrid_)[ir] * sbessel_[l__ + 1][ir];
                }
            } else {
                if (l__ == 1) {
                    for (int ir = 0; ir < rgrid_->num_points(); ir++) {
                        s[ir] = (*rgrid_)[ir] / 3;
                    }
                }
            }
            s.interpolate();
            return std::move(s);
        }
};

class sbessel_approx
{
    private:

        Unit_cell const& unit_cell_;

        int lmax_;

        mdarray<std::vector<double>, 2> qnu_;
        mdarray<double, 4> coeffs_; 

        int nqnu_max_;

        static double sbessel_l2norm(double nu, int l, double R)
        {
            if (std::abs(nu) < 1e-10) TERMINATE_NOT_IMPLEMENTED;

            if (l == 0)
            {
                return (nu * R * 2 - std::sin(nu * R * 2)) / 4 / std::pow(nu, 3);
            }
            else
            {
                double jl[l + 2];
                gsl_sf_bessel_jl_array(l + 1, R * nu, &jl[0]);
                return std::pow(R, 3) * (jl[l] * jl[l] - jl[l + 1] * jl[l - 1]) / 2;
            }
        }

    public:

        sbessel_approx(Unit_cell const& unit_cell__,
                       int lmax__,
                       double const qmin__,
                       double const qmax__,
                       double const eps__)
            : unit_cell_(unit_cell__),
              lmax_(lmax__)
        {
            PROFILE("sirius::sbessel_approx");

            qnu_ = mdarray<std::vector<double>, 2>(lmax_ + 1, unit_cell_.num_atom_types());

            #pragma omp parallel for
            for (int l = 0; l <= lmax_; l++)
            {
                for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
                {
                    qnu_(l, iat) = build_approx_freq(qmin__, qmax__, l, unit_cell_.atom_type(iat).mt_radius(), eps__);
                }
            }

            nqnu_max_ = 0;
            for (int l = 0; l <= lmax_; l++)
            {
                for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
                {
                    nqnu_max_ = std::max(nqnu_max_, static_cast<int>(qnu_(l, iat).size()));
                }
            }
        }

        void approximate(std::vector<double> const& q__)
        {
            PROFILE("sirius::sbessel_approx::approximate");

            coeffs_ = mdarray<double, 4>(nqnu_max_, q__.size(), lmax_ + 1, unit_cell_.num_atom_types());
            
            #pragma omp parallel for
            for (int l = 0; l <= lmax_; l++)
            {
                for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
                {
                    int n = nqnu(l, iat);

                    mdarray<double, 2> A(n, n);
                    for (int iq = 0; iq < n; iq++)
                    { 
                        for (int jq = 0; jq <= iq; jq++)
                        {
                            A(jq, iq) = A(iq, jq) = overlap(qnu_(l, iat)[jq], qnu_(l, iat)[iq], l,
                                                            unit_cell_.atom_type(iat).mt_radius());
                        }

                        for (int j = 0; j < (int)q__.size(); j++)
                        {
                            if (std::abs(q__[j]) < 1e-12)
                            {
                                coeffs_(iq, j, l, iat) = 0;
                            }
                            else
                            {
                                coeffs_(iq, j, l, iat) = overlap(qnu_(l, iat)[iq], q__[j], l, 
                                                                 unit_cell_.atom_type(iat).mt_radius());
                            }
                        }
                    }
                    linalg<CPU>::gesv(n, (int)q__.size(), A.at<CPU>(), A.ld(), &coeffs_(0, 0, l, iat), coeffs_.ld());
                }
            }
        }

        inline double qnu(int const iq, int const l, int const iat)
        {
            return qnu_(l, iat)[iq];
        }

        inline int nqnu(int const l, int const iat)
        {
            return static_cast<int>(qnu_(l, iat).size());
        }

        inline int nqnu_max()
        {
            return nqnu_max_;
        }

        inline double coeff(int const iq, int const j, int const l, int const iat)
        {
            return coeffs_(iq, j, l, iat);
        }
        
        // \int_0^{R} j(nu1 * r) * j(nu2 * r) * r^2 dr
        // this integral can be computed analytically
        static double overlap(double nu1__, double nu2__, int l__, double R__)
        {
            if (std::abs(nu1__) < 1e-10 || std::abs(nu2__) < 1e-10) TERMINATE_NOT_IMPLEMENTED;

            if (std::abs(nu1__ - nu2__) < 1e-12)
            {
                if (l__ == 0)
                {
                    return (nu2__ * R__ * 2 - std::sin(nu2__ * R__ * 2)) / 4 / std::pow(nu2__, 3);
                }
                else
                {
                    double jl[l__ + 2];
                    gsl_sf_bessel_jl_array(l__ + 1, R__ * nu2__, &jl[0]);
                    return std::pow(R__, 3) * (jl[l__] * jl[l__] - jl[l__ + 1] * jl[l__ - 1]) / 2;
                }
            }
            else
            {
                if (l__ == 0)
                {
                    return (nu2__ * std::cos(nu2__ * R__) * std::sin(nu1__ * R__) - nu1__ * std::cos(nu1__ * R__) * std::sin(nu2__ * R__)) /
                           (std::pow(nu1__, 3) * nu2__ - nu1__ * std::pow(nu2__, 3));
                }
                else
                {
                    double j1[l__ + 2];
                    gsl_sf_bessel_jl_array(l__ + 1, R__ * nu1__, &j1[0]);

                    double j2[l__ + 2];
                    gsl_sf_bessel_jl_array(l__ + 1, R__ * nu2__, &j2[0]);

                    return std::pow(R__, 2) * (nu2__ * j2[l__ - 1] * j1[l__] - nu1__ * j1[l__ - 1] * j2[l__]) / (std::pow(nu1__, 2) - std::pow(nu2__, 2));
                }
            }
        }

        std::vector<double> build_approx_freq(double const qmin__,
                                              double const qmax__,
                                              int const l__,
                                              double const R__,
                                              double const eps__)
        {
            std::vector<double> qnu;

            double min_val;
            int n = 2;

            do
            {
                n++;
                qnu.resize(n);
                for (int i = 0; i < n; i++) qnu[i] = qmin__ + (qmax__ - qmin__) * i / (n - 1);
                
                dmatrix<double_complex> ovlp(n, n);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        double o = overlap(qnu[j], qnu[i], l__, R__);
                        ovlp(j, i) = o / sbessel_l2norm(qnu[i], l__, R__) / sbessel_l2norm(qnu[j], l__, R__);
                    }
                }
                
                std::vector<double> eval(n);
                dmatrix<double_complex> z(n, n);

                Eigensolver_lapack<double_complex> solver;
                solver.solve(n, ovlp, &eval[0], z);
                min_val = eval[0];

            } while (min_val > eps__);

            return qnu;
        }
};

class Spherical_Bessel_approximant
{
    private:

        int lmax_;

        double R_;
        
        /// List of Bessel function scaling factors for each angular momentum.
        std::vector< std::vector<double> > qnu_;


        //mdarray<double, 4> coeffs_; 

        int nqnu_max_;

        static double sbessel_l2norm(double nu, int l, double R)
        {
            if (std::abs(nu) < 1e-10)
            {
                if (l == 0) return std::pow(R, 3) / 3.0;
                return 0;
            }

            if (l == 0)
            {
                return (nu * R * 2 - std::sin(nu * R * 2)) / 4 / std::pow(nu, 3);
            }
            else
            {
                double jl[l + 2];
                gsl_sf_bessel_jl_array(l + 1, R * nu, &jl[0]);
                return std::pow(R, 3) * (jl[l] * jl[l] - jl[l + 1] * jl[l - 1]) / 2;
            }
        }

    public:

        Spherical_Bessel_approximant(int lmax__,
                                     double R__,
                                     double const qmin__,
                                     double const qmax__,
                                     double const eps__)
            : lmax_(lmax__),
              R_(R__)
        {
            PROFILE("sirius::Spherical_Bessel_approximant");

            qnu_ = std::vector< std::vector<double> >(lmax_ + 1);

            #pragma omp parallel for
            for (int l = 0; l <= lmax_; l++)
                qnu_[l] = build_approx_freq(qmin__, qmax__, l, R_, eps__);

            nqnu_max_ = 0;
            for (int l = 0; l <= lmax_; l++)
                nqnu_max_ = std::max(nqnu_max_, nqnu(l));
        }

        std::vector<double> approximate(int l__, double nu__)
        {
            int n = nqnu(l__);
            std::vector<double> x(n);
            matrix<double> A(n, n);

            for (int iq = 0; iq < n; iq++)
            { 
                for (int jq = 0; jq <= iq; jq++)
                {
                    A(jq, iq) = A(iq, jq) = overlap(qnu(jq, l__), qnu(iq, l__), l__, R_);
                }
                x[iq] = overlap(qnu(iq, l__), nu__, l__, R_);
            }
            linalg<CPU>::gesv(n, 1, A.at<CPU>(), A.ld(), &x[0], n);
            return x;
        }

        void approximate(std::vector<double> const& q__)
        {
            //runtime::Timer t("sirius::sbessel_approx::approximate");

            //coeffs_ = mdarray<double, 4>(nqnu_max_, q__.size(), lmax_ + 1, unit_cell_.num_atom_types());
            //
            //#pragma omp parallel for
            //for (int l = 0; l <= lmax_; l++)
            //{
            //    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
            //    {
            //        int n = nqnu(l, iat);

            //        mdarray<double, 2> A(n, n);
            //        for (int iq = 0; iq < n; iq++)
            //        { 
            //            for (int jq = 0; jq <= iq; jq++)
            //            {
            //                A(jq, iq) = A(iq, jq) = overlap(qnu_(l, iat)[jq], qnu_(l, iat)[iq], l,
            //                                                unit_cell_.atom_type(iat).mt_radius());
            //            }

            //            for (int j = 0; j < (int)q__.size(); j++)
            //            {
            //                if (std::abs(q__[j]) < 1e-12)
            //                {
            //                    coeffs_(iq, j, l, iat) = 0;
            //                }
            //                else
            //                {
            //                    coeffs_(iq, j, l, iat) = overlap(qnu_(l, iat)[iq], q__[j], l, 
            //                                                     unit_cell_.atom_type(iat).mt_radius());
            //                }
            //            }
            //        }
            //        linalg<CPU>::gesv(n, (int)q__.size(), A.at<CPU>(), A.ld(), &coeffs_(0, 0, l, iat), coeffs_.ld());
            //    }
            //}
        }

        inline double qnu(int const iq, int const l) const
        {
            return qnu_[l][iq];
        }

        inline int nqnu(int const l) const
        {
            return static_cast<int>(qnu_[l].size());
        }

        inline int nqnu_max() const
        {
            return nqnu_max_;
        }

        //inline double coeff(int const iq, int const j, int const l, int const iat)
        //{
        //    return coeffs_(iq, j, l, iat);
        //}
        //

        // \int_0^{R} j(nu1 * r) * j(nu2 * r) * r^2 dr
        // this integral can be computed analytically
        static double overlap(double nu1__, double nu2__, int l__, double R__)
        {
            if (std::abs(nu1__) < 1e-10 && std::abs(nu2__) < 1e-10 && l__ == 0) return std::pow(R__, 3) / 3.0;

            if ((std::abs(nu1__) < 1e-10 || std::abs(nu2__) < 1e-10) && l__ > 0) return 0;

            if ((std::abs(nu1__) < 1e-10 || std::abs(nu2__) < 1e-10) && l__ == 0)
            {
                double nu = std::max(nu1__, nu2__);
                double nuR = nu * R__;
                return (std::sin(nuR) - nuR * std::cos(nuR)) / std::pow(nu, 3);
            }

            if (std::abs(nu1__ - nu2__) < 1e-12)
            {
                if (l__ == 0)
                {
                    return (nu2__ * R__ * 2 - std::sin(nu2__ * R__ * 2)) / 4 / std::pow(nu2__, 3);
                }
                else
                {
                    double jl[l__ + 2];
                    gsl_sf_bessel_jl_array(l__ + 1, R__ * nu2__, &jl[0]);
                    return std::pow(R__, 3) * (jl[l__] * jl[l__] - jl[l__ + 1] * jl[l__ - 1]) / 2;
                }
            }
            else
            {
                if (l__ == 0)
                {
                    return (nu2__ * std::cos(nu2__ * R__) * std::sin(nu1__ * R__) - nu1__ * std::cos(nu1__ * R__) * std::sin(nu2__ * R__)) /
                           (std::pow(nu1__, 3) * nu2__ - nu1__ * std::pow(nu2__, 3));
                }
                else
                {
                    double j1[l__ + 2];
                    gsl_sf_bessel_jl_array(l__ + 1, R__ * nu1__, &j1[0]);

                    double j2[l__ + 2];
                    gsl_sf_bessel_jl_array(l__ + 1, R__ * nu2__, &j2[0]);

                    return std::pow(R__, 2) * (nu2__ * j2[l__ - 1] * j1[l__] - nu1__ * j1[l__ - 1] * j2[l__]) / (std::pow(nu1__, 2) - std::pow(nu2__, 2));
                }
            }
            TERMINATE("this is wrong");
            return -1;
        }

        std::vector<double> build_approx_freq(double const qmin__,
                                              double const qmax__,
                                              int const l__,
                                              double const R__,
                                              double const eps__)
        {
            std::vector<double> qnu;

            double min_val;
            int n = 2;

            do
            {
                n++;
                qnu.resize(n);
                for (int i = 0; i < n; i++) qnu[i] = qmin__ + (qmax__ - qmin__) * i / (n - 1);
                
                dmatrix<double> ovlp(n, n);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        double o = overlap(qnu[j], qnu[i], l__, R__);
                        ovlp(j, i) = o / sbessel_l2norm(qnu[i], l__, R__) / sbessel_l2norm(qnu[j], l__, R__);
                    }
                }
                
                std::vector<double> eval(n);
                dmatrix<double> z(n, n);

                Eigensolver_lapack<double> solver;
                solver.solve(n, ovlp, &eval[0], z);
                min_val = eval[0];

            } while (min_val > eps__);

            return qnu;
        }
};

class Spherical_Bessel_approximant2
{
    private:

        int lmax_;

        double R_;
        
        /// List of Bessel function scaling factors for each angular momentum.
        std::vector<double> qnu_;

        static double sbessel_l2norm(double nu, int l, double R)
        {
            if (std::abs(nu) < 1e-10)
            {
                if (l == 0) return std::pow(R, 3) / 3.0;
                return 0;
            }

            if (l == 0)
            {
                return (nu * R * 2 - std::sin(nu * R * 2)) / 4 / std::pow(nu, 3);
            }
            else
            {
                double jl[l + 2];
                gsl_sf_bessel_jl_array(l + 1, R * nu, &jl[0]);
                return std::pow(R, 3) * (jl[l] * jl[l] - jl[l + 1] * jl[l - 1]) / 2;
            }
        }

    public:

        Spherical_Bessel_approximant2(int lmax__,
                                     double R__,
                                     double const qmin__,
                                     double const qmax__,
                                     int nq__)
            : lmax_(lmax__),
              R_(R__)
        {
            PROFILE("sirius::Spherical_Bessel_approximant");

            int nq = nfreq(qmin__, qmax__, 0, R__, 1e-12);
            qnu_.resize(nq);
            for (int i = 0; i < nq; i++) qnu_[i] = qmin__ + (qmax__ - qmin__) * i / (nq - 1);

        }

        std::vector<double> approximate(int l__, double nu__)
        {
            int n = nqnu();
            std::vector<double> x(n);
            matrix<double> A(n, n);

            for (int iq = 0; iq < n; iq++)
            { 
                for (int jq = 0; jq <= iq; jq++)
                {
                    A(jq, iq) = A(iq, jq) = overlap(qnu(jq), qnu(iq), l__, R_);
                }
                x[iq] = overlap(qnu(iq), nu__, l__, R_);
            }
            linalg<CPU>::gesv(n, 1, A.at<CPU>(), A.ld(), &x[0], n);
            return x;
        }

        inline double qnu(int const iq) const
        {
            return qnu_[iq];
        }

        inline int nqnu() const
        {
            return static_cast<int>(qnu_.size());
        }

        // \int_0^{R} j(nu1 * r) * j(nu2 * r) * r^2 dr
        // this integral can be computed analytically
        static double overlap(double nu1__, double nu2__, int l__, double R__)
        {
            if (std::abs(nu1__) < 1e-10 && std::abs(nu2__) < 1e-10 && l__ == 0) return std::pow(R__, 3) / 3.0;

            if ((std::abs(nu1__) < 1e-10 || std::abs(nu2__) < 1e-10) && l__ > 0) return 0;

            if ((std::abs(nu1__) < 1e-10 || std::abs(nu2__) < 1e-10) && l__ == 0)
            {
                double nu = std::max(nu1__, nu2__);
                double nuR = nu * R__;
                return (std::sin(nuR) - nuR * std::cos(nuR)) / std::pow(nu, 3);
            }

            if (std::abs(nu1__ - nu2__) < 1e-12)
            {
                if (l__ == 0)
                {
                    return (nu2__ * R__ * 2 - std::sin(nu2__ * R__ * 2)) / 4 / std::pow(nu2__, 3);
                }
                else
                {
                    std::vector<double> jl(l__ + 2);
                    gsl_sf_bessel_jl_array(l__ + 1, R__ * nu2__, &jl[0]);
                    return std::pow(R__, 3) * (jl[l__] * jl[l__] - jl[l__ + 1] * jl[l__ - 1]) / 2;
                }
            }
            else
            {
                if (l__ == 0)
                {
                    return (nu2__ * std::cos(nu2__ * R__) * std::sin(nu1__ * R__) - nu1__ * std::cos(nu1__ * R__) * std::sin(nu2__ * R__)) /
                           (std::pow(nu1__, 3) * nu2__ - nu1__ * std::pow(nu2__, 3));
                }
                else
                {
                    std::vector<double> j1(l__ + 2);
                    gsl_sf_bessel_jl_array(l__ + 1, R__ * nu1__, &j1[0]);

                    std::vector<double> j2(l__ + 2);
                    gsl_sf_bessel_jl_array(l__ + 1, R__ * nu2__, &j2[0]);

                    return std::pow(R__, 2) * (nu2__ * j2[l__ - 1] * j1[l__] - nu1__ * j1[l__ - 1] * j2[l__]) / (std::pow(nu1__, 2) - std::pow(nu2__, 2));
                }
            }
            TERMINATE("this is wrong");
            return -1;
        }

        int nfreq(double const qmin__,
                  double const qmax__,
                  int const l__,
                  double const R__,
                  double const eps__)
        {
            std::vector<double> qnu;

            double min_val;
            int n = 2;

            do
            {
                n++;
                qnu.resize(n);
                for (int i = 0; i < n; i++) qnu[i] = qmin__ + (qmax__ - qmin__) * i / (n - 1);
                
                dmatrix<double> ovlp(n, n);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        double o = overlap(qnu[j], qnu[i], l__, R__);
                        ovlp(j, i) = o / sbessel_l2norm(qnu[i], l__, R__) / sbessel_l2norm(qnu[j], l__, R__);
                    }
                }
                
                std::vector<double> eval(n);
                dmatrix<double> z(n, n);

                Eigensolver_lapack<double> solver;
                solver.solve(n, ovlp, &eval[0], z);
                min_val = eval[0];

                if (n > 100) return 100;

            } while (min_val > eps__);

            return n;
        }
};


};

#endif
