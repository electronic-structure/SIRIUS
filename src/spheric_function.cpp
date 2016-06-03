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

/** \file spheric_function.cpp
 *   
 *  \brief Contains gradient() and operator*().
 */

#include "sht.h"
#include "spheric_function.h"
#include "utils.h"

namespace sirius 
{

Spheric_function_gradient<spectral, double_complex> gradient(Spheric_function<spectral, double_complex>& f)
{
    Spheric_function_gradient<spectral, double_complex> g(f.angular_domain_size(), f.radial_grid());
    for (int i = 0; i < 3; i++)
    {
        g[i] = Spheric_function<spectral, double_complex>(f.angular_domain_size(), f.radial_grid());
        g[i].zero();
    }
            
    int lmax = Utils::lmax_by_lmmax(f.angular_domain_size());

    //Spline<double_complex> s(f.radial_grid());

    for (int l = 0; l <= lmax; l++)
    {
        double d1 = sqrt(double(l + 1) / double(2 * l + 3));
        double d2 = sqrt(double(l) / double(2 * l - 1));

        for (int m = -l; m <= l; m++)
        {
            int lm = Utils::lm_by_l_m(l, m);
            auto s = f.component(lm);

            for (int mu = -1; mu <= 1; mu++)
            {
                int j = (mu + 2) % 3; // map -1,0,1 to 1,2,0

                if ((l + 1) <= lmax && abs(m + mu) <= l + 1)
                {
                    int lm1 = Utils::lm_by_l_m(l + 1, m + mu); 
                    double d = d1 * SHT::clebsch_gordan(l, 1, l + 1, m, mu, m + mu);
                    for (int ir = 0; ir < f.radial_grid().num_points(); ir++)
                        g[j](lm1, ir) += (s.deriv(1, ir) - f(lm, ir) * f.radial_grid().x_inv(ir) * double(l)) * d;  
                }
                if ((l - 1) >= 0 && abs(m + mu) <= l - 1)
                {
                    int lm1 = Utils::lm_by_l_m(l - 1, m + mu); 
                    double d = d2 * SHT::clebsch_gordan(l, 1, l - 1, m, mu, m + mu); 
                    for (int ir = 0; ir < f.radial_grid().num_points(); ir++)
                        g[j](lm1, ir) -= (s.deriv(1, ir) + f(lm, ir) * f.radial_grid().x_inv(ir) * double(l + 1)) * d;
                }
            }
        }
    }

    double_complex d1(1.0 / sqrt(2.0), 0);
    double_complex d2(0, 1.0 / sqrt(2.0));

    for (int ir = 0; ir < f.radial_grid().num_points(); ir++)
    {
        for (int lm = 0; lm < f.angular_domain_size(); lm++)
        {
            double_complex g_p = g[0](lm, ir);
            double_complex g_m = g[1](lm, ir);
            g[0](lm, ir) = d1 * (g_m - g_p);
            g[1](lm, ir) = d2 * (g_m + g_p);
        }
    }

    return g;
}

Spheric_function_gradient<spectral, double> gradient(Spheric_function<spectral, double>& f)
{
    int lmax = Utils::lmax_by_lmmax(f.angular_domain_size());
    SHT sht(lmax);
    auto zf = convert(f);
    auto zg = gradient(zf);
    Spheric_function_gradient<spectral, double> g(f.angular_domain_size(), f.radial_grid());
    for (int x: {0, 1, 2}) {
        g[x] = convert(zg[x]);
    }
    return g;
}

Spheric_function<spatial, double> operator*(Spheric_function_gradient<spatial, double>& f, 
                                            Spheric_function_gradient<spatial, double>& g)
{
    for (int x: {0, 1, 2}) {
        if (f[x].radial_grid().hash() != g[x].radial_grid().hash()) {
            TERMINATE("wrong radial grids");
        }
        
        if (f[x].angular_domain_size() != g[x].angular_domain_size()) {
            TERMINATE("wrong number of angular points");
        }
    }

    Spheric_function<spatial, double> result(f.angular_domain_size(), f.radial_grid());
    result.zero();

    for (int x: {0, 1, 2}) {
        for (int ir = 0; ir < f[x].radial_grid().num_points(); ir++) {
            for (int tp = 0; tp < f[x].angular_domain_size(); tp++) {
                result(tp, ir) += f[x](tp, ir) * g[x](tp, ir);
            }
        }
    }

    return result;
}

}
