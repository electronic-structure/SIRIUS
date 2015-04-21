// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file sbessel_pw.h
 *   
 *  \brief Contains implementation of sirius::sbessel_pw class.
 */

#ifndef __SBESSEL_PW_H__
#define __SBESSEL_PW_H__

#include <gsl/gsl_sf_bessel.h>
#include "evp_solver.h"

namespace sirius
{

/// Spherical bessel functions of a plane-wave expansion inside muffin-tins.
template <typename T> 
class sbessel_pw
{
    private:

        Unit_cell* unit_cell_;

        int lmax_;

        mdarray<Spline<T>*, 2> sjl_; 

    public:

        sbessel_pw(Unit_cell* unit_cell__, int lmax__) : unit_cell_(unit_cell__), lmax_(lmax__)
        {
            sjl_ = mdarray<Spline<T>*, 2>(lmax_ + 1, unit_cell_->num_atom_types());

            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
            {
                for (int l = 0; l <= lmax_; l++)
                {
                    sjl_(l, iat) = new Spline<T>(unit_cell_->atom_type(iat)->radial_grid());
                }
            }
        }
        
        ~sbessel_pw()
        {
            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
            {
                for (int l = 0; l <= lmax_; l++) delete sjl_(l, iat);
            }
            sjl_.deallocate();
        }

        void load(double q)
        {
            std::vector<double> jl(lmax_ + 1);
            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
            {
                for (int ir = 0; ir < unit_cell_->atom_type(iat)->num_mt_points(); ir++)
                {
                    double x = unit_cell_->atom_type(iat)->radial_grid(ir) * q;
                    gsl_sf_bessel_jl_array(lmax_, x, &jl[0]);
                    for (int l = 0; l <= lmax_; l++) (*sjl_(l, iat))[ir] = jl[l];
                }
            }
        }

        void interpolate(double q)
        {
            load(q);
            
            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
            {
                for (int l = 0; l <= lmax_; l++) sjl_(l, iat)->interpolate();
            }
        }

        inline T operator()(int ir, int l, int iat)
        {
            return (*sjl_(l, iat))[ir];
        }

        inline Spline<T>* operator()(int l, int iat)
        {
            return sjl_(l, iat);
        }
};

class Spherical_Bessel_functions
{
    private:

        std::vector< Spline<double>* > sbessel_;

    public:

        Spherical_Bessel_functions(int lmax__, Radial_grid const& rgrid__, double nu__)
        {
            sbessel_ = std::vector< Spline<double>* >(lmax__ + 1);
            for (int l = 0; l <= lmax__; l++) sbessel_[l] = new Spline<double>(rgrid__);

            std::vector<double> jl(lmax__ + 1);
            for (int ir = 0; ir < rgrid__.num_points(); ir++)
            {
                double v = rgrid__[ir] * nu__;
                gsl_sf_bessel_jl_array(lmax__, v, &jl[0]);
                for (int l = 0; l <= lmax__; l++) (*sbessel_[l])[ir] = jl[l];
            }
            
            for (int l = 0; l < lmax__; l++) sbessel_[l]->interpolate();
        }

        ~Spherical_Bessel_functions()
        {
            for (auto s: sbessel_) delete s;
        }

        Spline<double> const& operator()(int i__) const
        {
            return *sbessel_[i__];
        }
};

class sbessel_approx
{
    private:

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

        Unit_cell* unit_cell_;

        int lmax_;

        mdarray<std::vector<double>, 2> qnu_;
        mdarray<double, 4> coeffs_; 

        int nqnu_max_;


    public:

        sbessel_approx(Unit_cell* const unit_cell__,
                       int lmax__,
                       double const qmin__,
                       double const qmax__,
                       double const eps__)
            : unit_cell_(unit_cell__),
              lmax_(lmax__)
        {
            Timer t("sirius::sbessel_approx");

            qnu_ = mdarray<std::vector<double>, 2>(lmax_ + 1, unit_cell_->num_atom_types());

            #pragma omp parallel for
            for (int l = 0; l <= lmax_; l++)
            {
                for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
                {
                    qnu_(l, iat) = build_approx_freq(qmin__, qmax__, l, unit_cell_->atom_type(iat)->mt_radius(), eps__);
                }
            }

            nqnu_max_ = 0;
            for (int l = 0; l <= lmax_; l++)
            {
                for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
                {
                    nqnu_max_ = std::max(nqnu_max_, static_cast<int>(qnu_(l, iat).size()));
                }
            }
        }

        void approximate(std::vector<double> const& q__)
        {
            Timer t("sirius::sbessel_approx::approximate");

            coeffs_ = mdarray<double, 4>(nqnu_max_, q__.size(), lmax_ + 1, unit_cell_->num_atom_types());
            
            #pragma omp parallel for
            for (int l = 0; l <= lmax_; l++)
            {
                for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
                {
                    int n = nqnu(l, iat);

                    mdarray<double, 2> A(n, n);
                    for (int iq = 0; iq < n; iq++)
                    { 
                        for (int jq = 0; jq <= iq; jq++)
                        {
                            A(jq, iq) = A(iq, jq) = overlap(qnu_(l, iat)[jq], qnu_(l, iat)[iq], l,
                                                            unit_cell_->atom_type(iat)->mt_radius());
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
                                                                 unit_cell_->atom_type(iat)->mt_radius());
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
                
                mdarray<double_complex, 2> ovlp(n, n);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        double o = overlap(qnu[j], qnu[i], l__, R__);
                        ovlp(j, i) = o / sbessel_l2norm(qnu[i], l__, R__) / sbessel_l2norm(qnu[j], l__, R__);
                    }
                }
                
                std::vector<double> eval(n);
                mdarray<double_complex, 2> z(n, n);

                standard_evp_lapack solver;
                solver.solve(n, ovlp.at<CPU>(), n, &eval[0], z.at<CPU>(), n);
                min_val = eval[0];

            } while (min_val > eps__);

            return qnu;
        }
};


};

#endif
