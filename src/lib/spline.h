#ifndef __SPLINE_H__
#define __SPLINE_H__

namespace sirius {

/// \brief cubic spline with a not-a-knot boundary conditions
class Spline
{
    private:
        
        /// number of interpolating points
        int num_points;
    
        /// radial grid
        sirius::RadialGrid& radial_grid;
        
        std::vector<double> a;
        std::vector<double> b;
        std::vector<double> c;
        std::vector<double> d;

        void init()
        {
            a = std::vector<double>(num_points, 0.0);
            b = std::vector<double>(num_points - 1, 0.0);
            c = std::vector<double>(num_points - 1, 0.0);
            d = std::vector<double>(num_points - 1, 0.0);
        }

    public:
    
        Spline(int num_points, 
               sirius::RadialGrid& radial_grid) : num_points(num_points), 
                                                  radial_grid(radial_grid)
        {
            init();
        }
        
        Spline(int num_points, 
               sirius::RadialGrid& radial_grid, 
               std::vector<double>& y) : num_points(num_points), 
                                         radial_grid(radial_grid)
        {
            init();
            interpolate(y);
        }
        
        void interpolate(std::vector<double>& y)
        {
            a = y;
            interpolate();
        }
        
        void interpolate()
        {
            std::vector<double> diag_main(num_points);
            std::vector<double> diag_lower(num_points - 1);
            std::vector<double> diag_upper(num_points - 1);
            std::vector<double> m(num_points);
            std::vector<double> dy(num_points - 1);
            
            // derivative of y
            for (int i = 0; i < num_points - 1; i++) 
                dy[i] = (a[i + 1] - a[i]) / radial_grid.dr(i);
            
            // setup "B" vector of AX=B equation
            for (int i = 0; i < num_points - 2; i++) 
                m[i + 1] = 6 * (dy[i + 1] - dy[i]);
            m[0] = -m[1];
            m[num_points - 1] = -m[num_points - 2];
            
            // main diagonal of "A" matrix
            for (int i = 0; i < num_points - 2; i++)
                diag_main[i + 1] = 2 * (radial_grid.dr(i) + radial_grid.dr(i + 1));
            double h0 = radial_grid.dr(0);
            double h1 = radial_grid.dr(1);
            double h2 = radial_grid.dr(num_points - 2);
            double h3 = radial_grid.dr(num_points - 3);
            diag_main[0] = (h1 / h0) * h1 - h0;
            diag_main[num_points - 1] = (h3 / h2) * h3 - h2;
            
            // subdiagonals of "A" matrix
            for (int i = 0; i < num_points - 1; i++)
            {
                diag_upper[i] = radial_grid.dr(i);
                diag_lower[i] = radial_grid.dr(i);
            }
            diag_upper[0] = -(h1 * (1 + h1 / h0) + diag_main[1]);
            diag_lower[num_points - 2] = -(h3 * (1 + h3 / h2) + diag_main[num_points - 2]); 

            // solve tridiagonal system
            int info = dgtsv(num_points, 1, &diag_lower[0], &diag_main[0], &diag_upper[0], &m[0], num_points);
            if (info)
            {
                std::stringstream s;
                s << "dgtsv returned " << info;
                error(__FILE__, __LINE__, s);
            }
            
            b.resize(num_points - 1);
            c.resize(num_points - 1);
            d.resize(num_points - 1);

            for (int i = 0; i < num_points - 1; i++)
            {
                c[i] = m[i] / 2.0;
                double t = (m[i + 1] - m[i]) / 6.0;
                b[i] = dy[i] - (c[i] + t) * radial_grid.dr(i);
                d[i] = t / radial_grid.dr(i);
            }
        }
        
        double integrate(int m = 0)
        {
            std::vector<double> g(num_points);
    
            return integrate(g, m);
        }
        
        double integrate(int n, int m)
        {
            std::vector<double> g(num_points);
    
            integrate(g, m);

            return g[n];
        }

        double integrate(std::vector<double>& g, int m = 0)
        {
            g = std::vector<double>(num_points, 0.0);
            
            if (m == 0)
            {
                for (int i = 0; i < num_points - 1; i++)
                {
                    double dx = radial_grid.dr(i);
                    g[i + 1] = g[i] + (((d[i] * dx / 4 + c[i] / 3) * dx + b[i] / 2) * dx + a[i]) * dx;
                }
            }
            
            if (m > 0 || m < -4)
            {
                for (int i = 0; i < num_points - 1; i++)
                {
                    double x0 = radial_grid[i];
                    double x1 = radial_grid[i + 1];
                    double a0 = a[i];
                    double a1 = b[i];
                    double a2 = c[i];
                    double a3 = d[i];

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(m)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    g[i + 1] = g[i] + (pow(x0, 1 + m) * (-(a0 * (2 + m) * (3 + m) * (4 + m)) + 
                        x0 * (a1 * (3 + m) * (4 + m) - 2 * a2 * (4 + m) * x0 + 6 * a3 * pow(x0, 2)))) / ((1 + m) * (2 + m) * (3 + m) * (4 + m)) + 
                        pow(x1, 1 + m) * ((a0 - x0 * (a1 + x0 * (-a2 + a3 * x0))) / (1 + m) + ((a1 + x0 * (-2 * a2 + 3 * a3 * x0)) * x1) / (2 + m) + 
                        ((a2 - 3 * a3 * x0) * pow(x1, 2)) / (3 + m) + (a3 * pow(x1, 3)) / (4 + m));
                }
            }

            if (m == -1)
            {
                for (int i = 0; i < num_points - 1; i++)
                {
                    double x0 = radial_grid[i];
                    double x1 = radial_grid[i + 1];
                    double a0 = a[i];
                    double a1 = b[i];
                    double a2 = c[i];
                    double a3 = d[i];

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(-1)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    g[i + 1] = g[i] + (-((x0 - x1) * (6 * a1 - 9 * a2 * x0 + 11 * a3 * pow(x0, 2) + 3 * a2 * x1 - 7 * a3 * x0 * x1 + 2 * a3 * pow(x1, 2))) / 6.0 + 
                        (-a0 + x0 * (a1 - a2 * x0 + a3 * pow(x0, 2))) * log(x0 / x1));
                }
            }
            
            if (m == -2)
            {
                for (int i = 0; i < num_points - 1; i++)
                {
                    double x0 = radial_grid[i];
                    double x1 = radial_grid[i + 1];
                    double a0 = a[i];
                    double a1 = b[i];
                    double a2 = c[i];
                    double a3 = d[i];

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(-2)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    g[i + 1] = g[i] + (((x0 - x1) * (-2 * a0 + x0 * (2 * a1 - 2 * a2 * (x0 + x1) + a3 * (2 * pow(x0, 2) + 5 * x0 * x1 - pow(x1, 2)))) + 
                        2 * x0 * (a1 + x0 * (-2 * a2 + 3 * a3 * x0)) * x1 * log(x1 / x0)) / (2.0 * x0 * x1));
                }
            }

            if (m == -3)
            {
                for (int i = 0; i < num_points - 1; i++)
                {
                    double x0 = radial_grid[i];
                    double x1 = radial_grid[i + 1];
                    double a0 = a[i];
                    double a1 = b[i];
                    double a2 = c[i];
                    double a3 = d[i];

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(-3)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    g[i + 1] = g[i] + (-((x0 - x1) * (a0 * (x0 + x1) + x0 * (a1 * (-x0 + x1) + x0 * (a2 * x0 - a3 * pow(x0, 2) - 3 * a2 * x1 + 5 * a3 * x0 * x1 + 
                        2 * a3 * pow(x1, 2)))) + 2 * pow(x0, 2) * (a2 - 3 * a3 * x0) * pow(x1, 2) * log(x0 / x1)) / (2.0 * pow(x0, 2) * pow(x1, 2)));
                }
            }

            if (m == -4)
            {
                for (int i = 0; i < num_points - 1; i++)
                {
                    double x0 = radial_grid[i];
                    double x1 = radial_grid[i + 1];
                    double a0 = a[i];
                    double a1 = b[i];
                    double a2 = c[i];
                    double a3 = d[i];

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(-4)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    g[i + 1] = g[i] + ((2 * a0 * (-pow(x0, 3) + pow(x1, 3)) + x0 * (x0 - x1) * (a1 * (x0 - x1) * (2 * x0 + x1) + x0 * (-2 * a2 * pow(x0 - x1, 2) + 
                        a3 * x0 * (2 * pow(x0, 2) - 7 * x0 * x1 + 11 * pow(x1, 2)))) + 6 * a3 * pow(x0, 3) * pow(x1, 3) * log(x1 / x0)) / (6.0 * pow(x0, 3) * pow(x1, 3)));
                }
            }

            return g[num_points - 1];
        }

        std::vector<double>& data_points()
        {
            return a;
        }
        
        inline int size()
        {
            return num_points;
        }
                
        double operator()(double x)
        {
            int i;
            for (i = 0; i < num_points; i++) 
            {
                if (x >= radial_grid[i] && x < radial_grid[i + 1]) break;
            }
            double t = (x - radial_grid[i]);
            return a[i] + t * (b[i] + t * (c[i] + t * d[i]));
        }

        double operator()(const int i, double dx)
        {
            return a[i] + dx * (b[i] + dx * (c[i] + dx * d[i]));
        }
        
        double& operator[](const int i)
        {
            return a[i];
        }
};

};

#endif // __SPLINE_H__
