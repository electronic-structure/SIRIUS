#ifndef __RADIAL_SOLVER_H__
#define __RADIAL_SOLVER_H__

#include "sirius.h"

/*! \brief solves a scalar relativistic equation
    
    \f{eqnarray*}{
       P' &=& 2 M Q + \frac{P}{r} \\
       Q' &=& (V - E + \frac{\ell(\ell + 1)}{2 M r^2}) P - \frac{Q}{r}
    \f}
*/

class radial_solver
{
    private:

        bool relativistic;
        
        double zn;
        
        sirius::radial_grid& r;
        
        double enu_tolerance;
        
        int integrate(int nr, int l, double enu, sirius::spline& ve, sirius::spline& mp, 
                      std::vector<double>& p, std::vector<double>& q);

    public:

        radial_solver(bool relativistic, double zn, sirius::radial_grid& r) : relativistic(relativistic), zn(zn), r(r)
        {
            enu_tolerance = 1e-10;
        }
        
        void solve_in_mt(int l, double enu, int m, std::vector<double>& v, std::vector<double>& p, 
                         std::vector<double>& hp);

        void bound_state(int n, int l, double& enu, std::vector<double>& v, std::vector<double>& p);

};

int radial_solver::integrate(int nr, int l, double enu, sirius::spline& ve, sirius::spline& mp, 
                             std::vector<double>& p, std::vector<double>& q)
{
    double alpha2 = 0.5 * pow((1 / speed_of_light), 2);
    if (!relativistic) alpha2 = 0.0;

    double enu0 = 0.0;
    if (relativistic) enu0 = enu;

    double ll2 = 0.5 * l * (l + 1);

    double x2 = r[0];
    double v2 = ve[0] + zn / x2;
    double m2 = 1 - (v2 - enu0) * alpha2;

    p.resize(nr);
    q.resize(nr);

    // TODO: check r->0 asymptotic
    p[0] = pow(r[0], l + 1) * exp(zn * r[0] / (l + 1));
    q[0] = 0.0; //(0.5 / m2) * p[0] * (l / r[0] + zn / (l + 1));

    double p2 = p[0];
    double q2 = q[0];
    double mp2 = mp[0];
    double vl2 = ll2 / m2 / pow(x2, 2);

    double pk[4];
    double qk[4];

    for (int i = 0; i < nr - 1; i++)
    {
        double x0 = x2;
        x2 = r[i + 1];
        double h = r.dr(i);
        double h1 = h / 2;

        double x1 = x0 + h1;
        double p0 = p2;
        double q0 = q2;
        double m0 = m2;
        double vl0 = vl2;
        double v0 = v2;
        v2 = ve[i + 1] + zn / x2;

        double mp0 = mp2;
        mp2 = mp[i + 1];
        double mp1 = mp(i, h1);
        double v1 = ve(i, h1) + zn / x1;
        double m1 = 1 - (v1 - enu0) * alpha2;
        m2 = 1 - (v2 - enu0) * alpha2;
        
        // k0 = F(Y(x), x)
        pk[0] = 2 * m0 * q0 + p0 / x0;
        qk[0] = (v0 - enu + vl0) * p0 - q0 / x0 - mp0;

        double vl1 = ll2 / m1 / pow(x1, 2);
        // k1 = F(Y(x) + k0 * h/2, x + h/2)
        pk[1] = 2 * m1 * (q0 + qk[0] * h1) + (p0 + pk[0] * h1) / x1;
        qk[1] = (v1 - enu + vl1) * (p0 + pk[0] * h1) - (q0 + qk[0] * h1) / x1 - mp1;

        // k2 = F(Y(x) + k1 * h/2, x + h/2)
        pk[2] = 2 * m1 * (q0 + qk[1] * h1) + (p0 + pk[1] * h1) / x1; 
        qk[2] = (v1 - enu + vl1) * (p0 + pk[1] * h1) - (q0 + qk[1] * h1) / x1 - mp1;

        vl2 = ll2 / m2 / pow(x2, 2);
        // k3 = F(Y(x) + k2 * h, x + h)
        pk[3] = 2 * m2 * (q0 + qk[2] * h) + (p0 + pk[2] * h) / x2; 
        qk[3] = (v2 - enu + vl2) * (p0 + pk[2] * h) - (q0 + qk[2] * h) / x2 - mp2;
        
        // Y(x + h) = Y(x) + h * (k0 + 2 * k1 + 2 * k2 + k3) / 6
        p2 = p0 + (pk[0] + 2 * pk[1] + 2 * pk[2] + pk[3]) * h / 6.0;
        q2 = q0 + (qk[0] + 2 * qk[1] + 2 * qk[2] + qk[3]) * h / 6.0;
        
        p[i + 1] = p2;
        q[i + 1] = q2;
    }

    int nn = 0;
    for (int i = 0; i < nr - 1; i++)
    {
        if (p[i] * p[i + 1] < 0.0) nn++;
    }

    return nn;
}

void radial_solver::bound_state(int n, int l, double& enu, std::vector<double>& v, std::vector<double>& p)
{
    std::vector<double> ve(r.size());
    for (int i = 0; i < r.size(); i++)
        ve[i] = v[i] - zn / r[i];
    
    sirius::spline ve_spline(r.size(), r, ve);
    sirius::spline mp_spline(r.size(), r);
    
    std::vector<double> q;
    
    int s = 1;
    int sp;
    double denu = 0.01;

    for (int iter = 0; iter < 1000; iter++)
    {
        int nn = integrate(r.size(), l, enu, ve_spline, mp_spline, p, q);
        
        sp = s;
        
        if (nn > (n - l - 1)) 
            s = -1;
        else
            s = 1;
        
        denu = s * fabs(denu);
        
        if (s != sp) 
            denu *= 0.5;
        else
            denu *= 1.25;
        
        if (fabs(denu) < enu_tolerance) break;
        
        enu += denu;
    }
    
    if (fabs(denu) >= enu_tolerance) 
        stop(std::cout << "enu is not converged");

    // search for the turning point
    int idxtp = r.size() - 1;
    for (int i = 0; i < r.size(); i++)
    {
        if (v[i] > enu)
        {
            idxtp = i;
            break;
        }
    }

    // zero the tail of the wave-function
    double t1 = 1e100;
    for (int i = idxtp; i < r.size(); i++)
    {
        if ((fabs(p[i]) < t1) && (p[i - 1] * p[i] > 0))
            t1 = fabs(p[i]);
        else
        {
            t1 = 0.0;
            p[i] = 0.0;
        }
    }

    std::vector<double> rho(r.size());
    for (int i = 0; i < r.size(); i++)
        rho[i] = p[i] * p[i];

    sirius::spline rho_spline(r.size(), r, rho);

    double norm = rho_spline.integrate();
    
    for (int i = 0; i < r.size(); i++)
        p[i] /= sqrt(norm);

    // count number of nodes
    int nn = 0;
    for (int i = 0; i < r.size() - 1; i++)
        if (p[i] * p[i + 1] < 0.0) nn++;

    if (nn != (n - l - 1))
    {
        std::ofstream fout("bound_state.dat");
        for (int i =0; i < r.size(); i++)
        {
            fout << r[i] << " " << p[i] << std::endl;
        }
        fout.close();
        stop(std::cout << "wrong number of nodes : " << nn << " instead of " << (n - l - 1));
    }
}

void radial_solver::solve_in_mt(int l, double enu, int m, std::vector<double>& v, std::vector<double>& p, std::vector<double>& hp)
{
    std::vector<double> ve(r.mt_nr());
    for (int i = 0; i < r.mt_nr(); i++)
        ve[i] = v[i] - zn / r[i];
    
    sirius::spline ve_spline(r.mt_nr(), r, ve);

    std::vector<double> q;
    std::vector<double> mp(r.mt_nr(), 0.0);

    sirius::spline mp_spline(r.mt_nr(), r);
    
    for (int j = 0; j <= m; j++)
    {
        if (j)
        {
            for (int i = 0; i < r.mt_nr(); i++)
                mp[i] = j * p[i];
            
            mp_spline.interpolate(mp);
        }
        
        integrate(r.mt_nr(), l, enu, ve_spline, mp_spline, p, q);
    }
}


#endif // __RADIAL_SOLVER_H__
