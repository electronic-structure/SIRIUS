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
            sjl_.set_dimensions(lmax_ + 1, unit_cell_->num_atom_types());
            sjl_.allocate();

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
            std::vector<double> jl(lmax_+ 1);
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

class sbessel_approx
{
    private:

        static double sbessel_l2norm(double nu, int l, double R)
        {
            double d1;
            if (l == 0)
            {
                d1 = -gsl_sf_bessel_Ynu(0.5, nu * R);
            }
            else
            {
                d1 = gsl_sf_bessel_Jnu(l - 0.5, nu * R);
            }
            return R * std::sqrt(pi * (std::pow(gsl_sf_bessel_Jnu(0.5 + l, nu * R), 2) - d1 * gsl_sf_bessel_Jnu(l + 1.5, nu * R)) / 4 / nu);
        }

    public:
        
        // \int_0^{R} j(nu1*r) * j(nu2 * r) * r^2 dr
        // this integral can be computed analytically
        static double overlap(double nu1, double nu2, int l, double R)
        {
            if (std::abs(nu1 - nu2) < 1e-12)
            {
                double d1;
                if (l == 0)
                {
                    d1 = -gsl_sf_bessel_Ynu(0.5, nu1 * R);
                }
                else
                {
                    d1 = gsl_sf_bessel_Jnu(l - 0.5, nu1 * R);
                }
                return pi * R * R * (std::pow(gsl_sf_bessel_Jnu(0.5 + l, nu1 * R), 2) - d1 * gsl_sf_bessel_Jnu(l + 1.5, nu1 * R)) / 4 / nu1;
            }
            else
            {
                double d1, d2;
                if (l == 0)
                {
                    d1 = -gsl_sf_bessel_Ynu(0.5, nu1 * R);
                    d2 = -gsl_sf_bessel_Ynu(0.5, nu2 * R);
                }
                else
                {
                    d1 = gsl_sf_bessel_Jnu(l - 0.5, nu1 * R);
                    d2 = gsl_sf_bessel_Jnu(l - 0.5, nu2 * R);
                }
                
                double d = nu2 * d2 * gsl_sf_bessel_Jnu(l + 0.5, nu1 * R) - nu1 * d1 * gsl_sf_bessel_Jnu(l + 0.5, nu2 * R);
                return (pi * R * d / ( 2 * (std::pow(nu1, 2) - std::pow(nu2, 2)) * std::sqrt(nu1 * nu2)));
            }
        }

        static void build_approx_freq(double const qmin__,
                                      double const qmax__,
                                      int const l__,
                                      double const R__,
                                      double const eps__,
                                      std::vector<double>& nu__)
        {
            double min_val = 1e10;
            int n = 2;

            do
            {
                n++;
                nu__.resize(n);
                for (int i = 0; i < n; i++) nu__[i] = qmin__ + (qmax__ - qmin__) * i / (n - 1);
                
                mdarray<double_complex, 2> ovlp(n, n);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        ovlp(j, i) = overlap(nu__[j], nu__[i], l__, R__) / sbessel_l2norm(nu__[i], l__, R__) / sbessel_l2norm(nu__[j], l__, R__);
                    }

                }
                
                std::vector<double> eval(n);
                mdarray<double_complex, 2> z(n, n);

                standard_evp_lapack solver;
                solver.solve(n, ovlp.ptr(), n, &eval[0], z.ptr(), n);
                min_val = eval[0];

            } while (min_val > eps__);
        }
            


};


};

#endif
