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

/** \file radial_solver.cpp
 *   
 *  \brief Contains remaining implementation of sirius::Radial_solver class.
 */

#include "radial_solver.h"

namespace sirius {

int Radial_solver::integrate(int l, 
                             double enu, 
                             Spline<double>& ve,
                             Spline<double>& mp, 
                             std::vector<double>& p, 
                             std::vector<double>& dpdr, 
                             std::vector<double>& q, 
                             std::vector<double>& dqdr)
{
    /* number of mesh points */
    int nr = radial_grid_.num_points();
    
    double alpha2 = 0.5 * pow((1 / speed_of_light), 2);
    if (!relativistic_) alpha2 = 0.0;

    double enu0 = 0.0;
    if (relativistic_) enu0 = enu;

    double ll2 = 0.5 * l * (l + 1);

    double x2 = radial_grid_[0];
    double x2inv = radial_grid_.x_inv(0);
    double v2 = ve[0] + zn_ / x2;
    double m2 = 1 - (v2 - enu0) * alpha2;

    p.resize(nr);
    dpdr.resize(nr);
    q.resize(nr);
    dqdr.resize(nr);

    // TODO: check r->0 asymptotic
    p[0] = pow(radial_grid_[0], l + 1) * exp(zn_ * radial_grid_[0] / (l + 1));
    q[0] = (0.5 / m2) * p[0] * (l / radial_grid_[0] + zn_ / (l + 1));

    double p2 = p[0];
    double q2 = q[0];
    double mp2 = mp[0];
    double vl2 = ll2 / m2 / pow(x2, 2);

    double v2enuvl2 = (v2 - enu + vl2);

    double pk[4];
    double qk[4];
    
    for (int i = 0; i < nr - 1; i++)
    {
        double x0 = x2;
        x2 = radial_grid_[i + 1];
        double x0inv = x2inv;
        x2inv = radial_grid_.x_inv(i + 1);
        double h = radial_grid_.dx(i);
        double h1 = h / 2;

        double x1 = x0 + h1;
        double x1inv = 1.0 / x1;
        double p0 = p2;
        double q0 = q2;
        double m0 = m2;
        v2 = ve[i + 1] + zn_ * x2inv;

        double mp0 = mp2;
        mp2 = mp[i + 1];
        double mp1 = mp(i, h1);
        double v1 = ve(i, h1) + zn_ * x1inv;
        double m1 = 1 - (v1 - enu0) * alpha2;
        m2 = 1 - (v2 - enu0) * alpha2;
        vl2 = ll2 / m2 / pow(x2, 2);
        
        double v0enuvl0 = v2enuvl2;
        v2enuvl2 = (v2 - enu + vl2);
        
        double vl1 = ll2 / m1 / pow(x1, 2);

        double v1enuvl1 = (v1 - enu + vl1);
        
        // k0 = F(Y(x), x)
        pk[0] = 2 * m0 * q0 + p0 * x0inv;
        qk[0] = v0enuvl0 * p0 - q0 * x0inv - mp0;

        // k1 = F(Y(x) + k0 * h/2, x + h/2)
        pk[1] = 2 * m1 * (q0 + qk[0] * h1) + (p0 + pk[0] * h1) * x1inv;
        qk[1] = v1enuvl1 * (p0 + pk[0] * h1) - (q0 + qk[0] * h1) * x1inv - mp1;

        // k2 = F(Y(x) + k1 * h/2, x + h/2)
        pk[2] = 2 * m1 * (q0 + qk[1] * h1) + (p0 + pk[1] * h1) * x1inv; 
        qk[2] = v1enuvl1 * (p0 + pk[1] * h1) - (q0 + qk[1] * h1) * x1inv - mp1;

        // k3 = F(Y(x) + k2 * h, x + h)
        pk[3] = 2 * m2 * (q0 + qk[2] * h) + (p0 + pk[2] * h) * x2inv; 
        qk[3] = v2enuvl2 * (p0 + pk[2] * h) - (q0 + qk[2] * h) * x2inv - mp2;
        
        // Y(x + h) = Y(x) + h * (k0 + 2 * k1 + 2 * k2 + k3) / 6
        p2 = p0 + (pk[0] + 2 * (pk[1] + pk[2]) + pk[3]) * h / 6.0;
        q2 = q0 + (qk[0] + 2 * (qk[1] + qk[2]) + qk[3]) * h / 6.0;
       
        p2 = std::max(std::min(1e10, p2), -1e10);
        q2 = std::max(std::min(1e10, q2), -1e10);
       
        p[i + 1] = p2;
        q[i + 1] = q2;
    }
    
    /* get number of nodes */
    int nn = 0;
    for (int i = 0; i < nr - 1; i++) if (p[i] * p[i + 1] < 0.0) nn++;

    for (int i = 0; i < nr; i++)
    {
        double V = ve[i] + zn_ * radial_grid_.x_inv(i); 
        double M = 1.0 - (V - enu0) * alpha2;

        /* P' = 2MQ + \frac{P}{r} */
        dpdr[i] = 2 * M * q[i] + p[i] * radial_grid_.x_inv(i);

        /* Q' = (V - E + \frac{\ell(\ell + 1)}{2 M r^2}) P - \frac{Q}{r} */
        dqdr[i] = (V - enu + double(l * (l + 1)) / (2 * M * pow(radial_grid_[i], 2))) * p[i] - 
                  q[i] * radial_grid_.x_inv(i) - mp[i];
    }

    return nn;
}

int Radial_solver::solve(int l, 
                         double enu, 
                         int m, 
                         std::vector<double>& v, 
                         std::vector<double>& p0, 
                         std::vector<double>& p1, 
                         std::vector<double>& q0, 
                         std::vector<double>& q1)
{
    assert(radial_grid_.num_points() == (int)v.size());

    /* subtract the nucleus part and keep the smooth part of the potential */
    Spline<double> vs(radial_grid_);
    for (int i = 0; i < radial_grid_.num_points(); i++) vs[i] = v[i] - zn_ / radial_grid_[i];
    vs.interpolate();
    
    Spline<double> mp(radial_grid_, 0);
    mp.interpolate();

    int nn = 0;
    
    /* loop until the m-th order is reached */
    for (int j = 0; j <= m; j++)
    {
        if (j)
        {
            for (int i = 0; i < radial_grid_.num_points(); i++) mp[i] = j * p0[i];
            mp.interpolate();
        }
        
        nn = integrate(l, enu, vs, mp, p0, p1, q0, q1);
    }

    return nn;
}

int Radial_solver::solve(int l, 
                         double enu, 
                         int m, 
                         std::vector<double>& v, 
                         std::vector<double>& p, 
                         std::vector<double>& hp, 
                         double& dpdr_R)
{
    std::vector<double> q;
    std::vector<double> dpdr;
    std::vector<double> dqdr;

    int nn = solve(l, enu, m, v, p, dpdr, q, dqdr);
    
    hp.resize(radial_grid_.num_points());
    double alph2 = 0.0;
    if (relativistic_) alph2 = pow((1.0 / speed_of_light), 2);
    for (int i = 0; i < radial_grid_.num_points(); i++)
    {
        double t1 = 2.0 - v[i] * alph2;
        hp[i] = (double(l * (l + 1)) / t1 / pow(radial_grid_[i], 2.0) + v[i]) * p[i] - q[i] / radial_grid_[i] - dqdr[i];
    }
    dpdr_R = dpdr[radial_grid_.num_points() - 1];
    return nn;
}

double Radial_solver::find_enu(int n, int l, std::vector<double>& v, double enu0)
{
    // TODO: explain this

    std::vector<double> p;
    std::vector<double> hp;
    
    double enu = enu0;
    double de = 0.001;
    bool found = false;
    double dpdr;
    int nndp = 0;
    for (int i = 0; i < 1000; i++)
    {
        int nnd = solve(l, enu, 0, v, p, hp, dpdr) - (n - l - 1);
        if (nnd > 0)
        {
            enu -= de;
        }
        else
        {
            enu += de;
        }
        if (i > 0)
        {
            if (nnd != nndp) 
            {
                de *= 0.5;
            }
            else
            {
                de *= 1.25;
            }
        }
        nndp = nnd;
        if (fabs(de) < 1e-10)
        {
            found = true;
            break;
        }
    }
    if (!found)
    {   
        error_local(__FILE__, __LINE__, "top of the band is not found");
    }
    double etop = enu;
    de = -0.001;
    found = false;
    double p1p = 0;
    for (int i = 0; i < 1000; i++)
    {
        solve(l, enu, 0, v, p, hp, dpdr);

        if (i > 0)
        {
            if (dpdr * p1p < 0.0)
            {
                if (fabs(de) < 1e-10)
                {
                    found = true;
                    break;
                }
                de *= -0.5;
            }
            else
            {
                de *= 1.25;
            }
        }
        p1p = dpdr;
        enu += de;
    }
    if (!found)
    {   
        error_local(__FILE__, __LINE__, "bottom of the band is not found");
    }
    return (enu + etop) / 2.0;
}
                        
        
double Radial_solver::bound_state(int n, int l, double enu, std::vector<double>& v, std::vector<double>& p)
{
    int np = radial_grid_.num_points();

    Spline<double> vs(radial_grid_);
    for (int i = 0; i < np; i++) vs[i] = v[i] - zn_ / radial_grid_[i];
    vs.interpolate();

    Spline<double> mp(radial_grid_, 0);
    mp.interpolate();
    
    std::vector<double> q(np);
    std::vector<double> dpdr(np);
    std::vector<double> dqdr(np);
    
    int s = 1;
    int sp;
    double denu = enu_tolerance_;

    for (int iter = 0; iter < 1000; iter++)
    {
        int nn = integrate(l, enu, vs, mp, p, dpdr, q, dqdr);
        
        sp = s;
        s = (nn > (n - l - 1)) ? -1 : 1;
        denu = s * fabs(denu);
        denu = (s != sp) ? denu * 0.5 : denu * 1.25;
        enu += denu;
        
        if (fabs(denu) < enu_tolerance_ && iter > 4) break;
    }
    
    if (fabs(denu) >= enu_tolerance_) 
    {
        std::stringstream s;
        s << "enu is not converged for n = " << n << " and l = " << l; 
        error_local(__FILE__, __LINE__, s);
    }

    /* search for the turning point */
    int idxtp = np - 1;
    for (int i = 0; i < np; i++)
    {
        if (v[i] > enu)
        {
            idxtp = i;
            break;
        }
    }

    /* zero the tail of the wave-function */
    double t1 = 1e100;
    for (int i = idxtp; i < np; i++)
    {
        if ((fabs(p[i]) < t1) && (p[i - 1] * p[i] > 0))
        {
            t1 = fabs(p[i]);
        }
        else
        {
            t1 = 0.0;
            p[i] = 0.0;
        }
    }

    Spline<double> rho(radial_grid_);
    for (int i = 0; i < np; i++) rho[i] = p[i] * p[i];

    /* p is not divided by r, so we integrate with r^0 prefactor */
    double norm = rho.interpolate().integrate(0);
    
    for (int i = 0; i < np; i++) p[i] /= sqrt(norm);

    /* count number of nodes of the function */
    int nn = 0;
    for (int i = 0; i < np - 1; i++) if (p[i] * p[i + 1] < 0.0) nn++;

    if (nn != (n - l - 1))
    {
        std::stringstream s;
        s << "n = " << n << std::endl 
          << "l = " << l << std::endl
          << "enu = " << enu << std::endl
          << "wrong number of nodes : " << nn << " instead of " << (n - l - 1);
        error_local(__FILE__, __LINE__, s);
    }

    return enu;
}

}

