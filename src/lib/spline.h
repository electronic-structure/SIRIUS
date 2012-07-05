#ifndef __SPLINE_H__
#define __SPLINE_H__

#include "mdarray.h"
#include "radial_grid.h"
#include "sirius.h"
#include "linalg.h"

namespace sirius {

/// \brief cubic spline with a not-a-knot boundary conditions
class spline
{
    public:
        
        spline(int n, double *y, sirius::radial_grid *r) : n(n), r(r)
        {
            std::vector<double> d(n);
            std::vector<double> dl(n - 1);
            std::vector<double> du(n - 1);
            std::vector<double> m(n);
            std::vector<double> dy(n - 1);
            
            for (int i = 0; i < n - 1; i++) 
            {
                dy[i] = (y[i + 1] - y[i]) / r->h(i);
            }
            
            // setup "B" vector of AX=B equation
            for (int i = 0; i < n - 2; i++)
            {
                m[i + 1] = 6 * (dy[i + 1] - dy[i]);
            }
            m[0] = -m[1];
            m[n - 1] = -m[n - 2];
            
            // main diagonal
            for (int i = 0; i < n - 2; i++)
            {
                d[i + 1] = 2 * (r->h(i) + r->h(i + 1));
            }
            d[0] = (r->h(1) / r->h(0)) * r->h(1) - r->h(0);
            d[n - 1] = (r->h(n - 3) / r->h(n - 2)) * r->h(n - 3) - r->h(n - 2);
            
            // subdiagonals
            for (int i = 0; i < n - 1; i++)
            {
                du[i] = r->h(i);
                dl[i] = r->h(i);
            }
            du[0] = -(r->h(1) * (1 + r->h(1) / r->h(0)) + d[1]);
            dl[n - 2] = -(r->h(n - 3) * (1 + r->h(n - 3) / r->h(n - 2)) + d[n - 2]); 

            int info = dgtsv(n, 1, &dl[0], &d[0], &du[0], &m[0], n);
            if (info)
            {
                std::cout << std::endl << "Error: dgtsv returned " << info << std::endl;
                stop();
            }

            a.set_dimensions(4, n - 1);
            a.allocate();
            a.zero();

            for (int i = 0; i < n - 1; i++)
            {
                a(0, i) = y[i];
                a(2, i) = m[i] / 2.0;
                double t = (m[i + 1] - m[i]) / 6.0;
                a(1, i) = dy[i] - (a(2, i) + t) * r->h(i);
                a(3, i) = t / r->h(i);
            }
        }

        double integrate(int m = 0)
        {
            std::vector<double> g(n);
            
            if (m == 0)
            {
                for (int i = 0; i < n - 1; i++)
                {
                    double dx = r->h(i);
                    g[i + 1] = g[i] + (((a(3, i) * dx / 4 + a(2,i) / 3) * dx + a(1, i) / 2) * dx + a(0, i)) * dx;
                }
            }
            else
            {
                for (int i = 0; i < n - 1; i++)
                {
                    double x0 = (*r)[i];
                    double x1 = (*r)[i + 1];
                    double a0 = a(0, i);
                    double a1 = a(1, i);
                    double a2 = a(2, i);
                    double a3 = a(3, i);

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(m)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],Assumptions->{m>=0,Element[{x0,x1},Reals],x1>x0>0}]
                    g[i + 1] = g[i] + (pow(x0, 1 + m) * (-(a0 * (2 + m) * (3 + m) * (4 + m)) + 
                        x0 * (a1 * (3 + m) * (4 + m) - 2 * a2 * (4 + m) * x0 + 6 * a3 * pow(x0, 2)))) / ((1 + m) * (2 + m) * (3 + m) * (4 + m)) + 
                        pow(x1, 1 + m) * ((a0 - x0 * (a1 + x0 * (-a2 + a3 * x0))) / (1 + m) + ((a1 + x0 * (-2 * a2 + 3 * a3 * x0)) * x1) / (2 + m) + 
                        ((a2 - 3 * a3 * x0) * pow(x1, 2)) / (3 + m) + (a3 * pow(x1, 3))/(4 + m));
                }
            
            }
            
            return g[n - 1];
        }
        
        
        double operator()(double x)
        {
            int i;
            for (i = 0; i < n; i++) 
            {
                if (x >= (*r)[i] && x < (*r)[i + 1]) break;
            }
            double t = (x - (*r)[i]);
            return a(0, i) + t * (a(1,i) + t * (a(2, i) + t * a(3, i)));
        }
        
    private:
    
        /// number of interpolating points
        int n;
    
        /// polynomial coefficients
        mdarray<double,2> a;
        
        /// radial grid
        sirius::radial_grid *r;


};

};

#endif // __SPLINE_H__
