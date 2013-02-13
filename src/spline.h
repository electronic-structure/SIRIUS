#ifndef __SPLINE_H__
#define __SPLINE_H__

namespace sirius {

/// \brief cubic spline with a not-a-knot boundary conditions
template <typename T> class Spline
{
    private:
        
        /// number of interpolating points
        int num_points;
    
        /// radial grid
        sirius::RadialGrid& radial_grid;

        std::vector<T> a;
        std::vector<T> b;
        std::vector<T> c;
        std::vector<T> d;

    public:
    
        Spline(int num_points, 
               sirius::RadialGrid& radial_grid) : num_points(num_points), 
                                                  radial_grid(radial_grid)
        {
            a = std::vector<T>(num_points);
            b = std::vector<T>(num_points - 1);
            c = std::vector<T>(num_points - 1);
            d = std::vector<T>(num_points - 1);

            memset(&a[0], 0, num_points * sizeof(T));
            memset(&b[0], 0, (num_points - 1) * sizeof(T));
            memset(&c[0], 0, (num_points - 1) * sizeof(T));
            memset(&d[0], 0, (num_points - 1) * sizeof(T));
        }
        
        Spline(int num_points, sirius::RadialGrid& radial_grid, std::vector<T>& y) : num_points(num_points), 
                                                                                     radial_grid(radial_grid)
        {
            interpolate(y);
        }
        
        void interpolate(std::vector<T>& y)
        {
            a = y;
            interpolate();
        }
        
        void interpolate()
        {
            std::vector<T> diag_main(num_points);
            std::vector<T> diag_lower(num_points - 1);
            std::vector<T> diag_upper(num_points - 1);
            std::vector<T> m(num_points);
            std::vector<T> dy(num_points - 1);
            
            // derivative of y
            for (int i = 0; i < num_points - 1; i++) 
                dy[i] = (a[i + 1] - a[i]) / radial_grid.dr(i);
            
            // setup "B" vector of AX=B equation
            for (int i = 0; i < num_points - 2; i++) 
                m[i + 1] = (dy[i + 1] - dy[i]) * 6.0;
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
            int info = linalg<lapack>::gtsv(num_points, 1, &diag_lower[0], &diag_main[0], &diag_upper[0], &m[0], 
                                            num_points);

            if (info)
            {
                std::stringstream s;
                s << "gtsv returned " << info;
                error(__FILE__, __LINE__, s);
            }
            
            b.resize(num_points - 1);
            c.resize(num_points - 1);
            d.resize(num_points - 1);

            for (int i = 0; i < num_points - 1; i++)
            {
                c[i] = m[i] / 2.0;
                T t = (m[i + 1] - m[i]) / 6.0;
                b[i] = dy[i] - (c[i] + t) * radial_grid.dr(i);
                d[i] = t / radial_grid.dr(i);
            }
        }
        
        T integrate(int m = 0)
        {
            std::vector<T> g(num_points);
    
            return integrate(g, m);
        }
        
        T integrate(int n, int m)
        {
            std::vector<T> g(num_points);
    
            integrate(g, m);

            return g[n];
        }

        T integrate(std::vector<T>& g, int m = 0)
        {
            g = std::vector<T>(num_points);

            g[0] = 0.0;
            
            if (m == 0)
            {
                double t = 1.0 / 3.0;
                for (int i = 0; i < num_points - 1; i++)
                {
                    double dx = radial_grid.dr(i);
                    g[i + 1] = g[i] + (((d[i] * dx * 0.25 + c[i] * t) * dx + b[i] * 0.5) * dx + a[i]) * dx;
                }
            }
            
            if (m == 2)
            {
                for (int i = 0; i < num_points - 1; i++)
                {
                    double x0 = radial_grid[i];
                    double x1 = radial_grid[i + 1];
                    double dx = radial_grid.dr(i);
                    T a0 = a[i];
                    T a1 = b[i];
                    T a2 = c[i];
                    T a3 = d[i];

                    T x0_2 = x0 * x0;
                    T x0_3 = x0_2 * x0;
                    T x1_2 = x1 * x1;
                    T x1_3 = x1_2 * x1;

                    g[i + 1] = g[i] + (20.0 * a0 * (x1_3 - x0_3) + 5.0 * a1 * (x0 * x0_3 + x1_3 * (3.0 * dx - x0)) - dx * dx * dx * 
                        (-2.0 * a2 * (x0_2 + 3.0 * x0 * x1 + 6.0 * x1_2) - a3 * dx * (x0_2 + 4.0 * x0 * x1 + 10.0 * x1_2))) / 60.0;
                }
            }
            
            if ((m > 0 && m != 2) || m < -4)
            {
                for (int i = 0; i < num_points - 1; i++)
                {
                    double x0 = radial_grid[i];
                    double x1 = radial_grid[i + 1];
                    T a0 = a[i];
                    T a1 = b[i];
                    T a2 = c[i];
                    T a3 = d[i];

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(m)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    g[i + 1] = g[i] + (pow(x0, 1 + m) * (-(a0 * double((2 + m) * (3 + m) * (4 + m))) + 
                        x0 * (a1 * double((3 + m) * (4 + m)) - 2.0 * a2 * double(4 + m) * x0 + 6.0 * a3 * pow(x0, 2)))) / double((1 + m) * (2 + m) * (3 + m) * (4 + m)) + 
                        pow(x1, 1 + m) * ((a0 - x0 * (a1 + x0 * (-a2 + a3 * x0))) / double(1 + m) + ((a1 + x0 * (-2.0 * a2 + 3.0 * a3 * x0)) * x1) / double(2 + m) + 
                        ((a2 - 3.0 * a3 * x0) * pow(x1, 2)) / double(3 + m) + (a3 * pow(x1, 3)) / double(4 + m));
                }
            }

            if (m == -1)
            {
                for (int i = 0; i < num_points - 1; i++)
                {
                    double x0 = radial_grid[i];
                    double x1 = radial_grid[i + 1];
                    T a0 = a[i];
                    T a1 = b[i];
                    T a2 = c[i];
                    T a3 = d[i];

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(-1)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    g[i + 1] = g[i] + (-((x0 - x1) * (6.0 * a1 - 9.0 * a2 * x0 + 11.0 * a3 * pow(x0, 2) + 3.0 * a2 * x1 - 7.0 * a3 * x0 * x1 + 2.0 * a3 * pow(x1, 2))) / 6.0 + 
                        (-a0 + x0 * (a1 - a2 * x0 + a3 * pow(x0, 2))) * log(x0 / x1));
                }
            }
            
            if (m == -2)
            {
                for (int i = 0; i < num_points - 1; i++)
                {
                    double x0 = radial_grid[i];
                    double x1 = radial_grid[i + 1];
                    T a0 = a[i];
                    T a1 = b[i];
                    T a2 = c[i];
                    T a3 = d[i];

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(-2)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    g[i + 1] = g[i] + (((x0 - x1) * (-2.0 * a0 + x0 * (2.0 * a1 - 2.0 * a2 * (x0 + x1) + a3 * (2.0 * pow(x0, 2) + 5.0 * x0 * x1 - pow(x1, 2)))) + 
                        2.0 * x0 * (a1 + x0 * (-2.0 * a2 + 3.0 * a3 * x0)) * x1 * log(x1 / x0)) / (2.0 * x0 * x1));
                }
            }

            if (m == -3)
            {
                for (int i = 0; i < num_points - 1; i++)
                {
                    double x0 = radial_grid[i];
                    double x1 = radial_grid[i + 1];
                    T a0 = a[i];
                    T a1 = b[i];
                    T a2 = c[i];
                    T a3 = d[i];

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(-3)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    g[i + 1] = g[i] + (-((x0 - x1) * (a0 * (x0 + x1) + x0 * (a1 * (-x0 + x1) + x0 * (a2 * x0 - a3 * pow(x0, 2) - 3.0 * a2 * x1 + 5.0 * a3 * x0 * x1 + 
                        2.0 * a3 * pow(x1, 2)))) + 2.0 * pow(x0, 2) * (a2 - 3.0 * a3 * x0) * pow(x1, 2) * log(x0 / x1)) / (2.0 * pow(x0, 2) * pow(x1, 2)));
                }
            }

            if (m == -4)
            {
                for (int i = 0; i < num_points - 1; i++)
                {
                    double x0 = radial_grid[i];
                    double x1 = radial_grid[i + 1];
                    T a0 = a[i];
                    T a1 = b[i];
                    T a2 = c[i];
                    T a3 = d[i];

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(-4)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    g[i + 1] = g[i] + ((2.0 * a0 * (-pow(x0, 3) + pow(x1, 3)) + x0 * (x0 - x1) * (a1 * (x0 - x1) * (2.0 * x0 + x1) + x0 * (-2.0 * a2 * pow(x0 - x1, 2) + 
                        a3 * x0 * (2.0 * pow(x0, 2) - 7.0 * x0 * x1 + 11.0 * pow(x1, 2)))) + 6.0 * a3 * pow(x0, 3) * pow(x1, 3) * log(x1 / x0)) / (6.0 * pow(x0, 3) * pow(x1, 3)));
                }
            }

            return g[num_points - 1];
        }

        std::vector<T>& data_points()
        {
            return a;
        }
        
        inline int size()
        {
            return num_points;
        }
                
        /*T operator()(double x)
        {
            int i;
            for (i = 0; i < num_points; i++) 
            {
                if (x >= radial_grid[i] && x < radial_grid[i + 1]) break;
            }
            double t = (x - radial_grid[i]);
            return a[i] + t * (b[i] + t * (c[i] + t * d[i]));
        }*/

        T operator()(const int i, double dx)
        {
            return a[i] + dx * (b[i] + dx * (c[i] + dx * d[i]));
        }
        
        T& operator[](const int i)
        {
            return a[i];
        }

        inline T deriv(const int dm, const int i, const double dx)
        {
            switch (dm)
            {
                case 0:
                    return a[i] + dx * (b[i] + dx * (c[i] + dx * d[i]));
                    break;
                case 1:
                    return b[i] + dx * (2 * c[i] + 3 * dx * d[i]);
                    break;
                case 2:
                    return 2 * c[i] + 6 * dx * d[i];
                    break;
                case 3:
                    return 6 * d[i];
                    break;
                default:
                    error(__FILE__, __LINE__, "wrong order of derivative");
                    return 0.0;
                    break;
            }
        }

        inline T deriv(const int dm, const int i)
        {
            if (i == num_points - 1) return deriv(dm, i - 1, radial_grid.dr(i - 1));
            else return deriv(dm, i, 0.0);
            //if (i == 0) return deriv(dm, 0, 0.0);
            //else return deriv(dm, i - 1, radial_grid.dr(i));
        }
};

};

#endif // __SPLINE_H__
