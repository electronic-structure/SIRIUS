#ifndef __SPLINE_H__
#define __SPLINE_H__

namespace sirius {

/// \brief cubic spline with a not-a-knot boundary conditions
class spline
{
    private:
        
        /// number of interpolating points
        int number_of_points;
    
        /// radial grid
        sirius::radial_grid& r;
        
        std::vector<double> a;
        std::vector<double> b;
        std::vector<double> c;
        std::vector<double> d;

        void init();
        
    public:
    
        spline(int number_of_points, sirius::radial_grid& r) : number_of_points(number_of_points), r(r)
        {
            init();
        }
        
        spline(int number_of_points, sirius::radial_grid& r, std::vector<double>& y) : number_of_points(number_of_points), r(r)
        {
            init();
            interpolate(y);
        }
        
        void interpolate(std::vector<double>& y)
        {
            memcpy(&a[0], &y[0], number_of_points * sizeof(double));
            interpolate();
        }
        
        void interpolate();
        
        double integrate(int m = 0);
        
        double integrate(std::vector<double>& g, int m = 0);
        
        std::vector<double>& data_points()
        {
            return a;
        }
        
        inline int size()
        {
            return number_of_points;
        }
                
        double operator()(double x)
        {
            int i;
            for (i = 0; i < number_of_points; i++) 
            {
                if (x >= r[i] && x < r[i + 1]) break;
            }
            double t = (x - r[i]);
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

void spline::init()
{
    a.resize(number_of_points);
    memset(&a[0], 0, number_of_points * sizeof(double));
    
    b.resize(number_of_points - 1);
    memset(&b[0], 0, (number_of_points - 1) * sizeof(double));
    
    c.resize(number_of_points - 1);
    memset(&c[0], 0, (number_of_points - 1) * sizeof(double));
    
    d.resize(number_of_points - 1);
    memset(&d[0], 0, (number_of_points - 1) * sizeof(double));
}

void spline::interpolate()
{
    std::vector<double> diag_main(number_of_points);
    std::vector<double> diag_lower(number_of_points - 1);
    std::vector<double> diag_upper(number_of_points - 1);
    std::vector<double> m(number_of_points);
    std::vector<double> dy(number_of_points - 1);
    
    // copy original function to "a" coefficients
    //a = y;

    // derivative of y
    for (int i = 0; i < number_of_points - 1; i++) 
        dy[i] = (a[i + 1] - a[i]) / r.dr(i);
    
    // setup "B" vector of AX=B equation
    for (int i = 0; i < number_of_points - 2; i++) 
        m[i + 1] = 6 * (dy[i + 1] - dy[i]);
    m[0] = -m[1];
    m[number_of_points - 1] = -m[number_of_points - 2];
    
    // main diagonal of "A" matrix
    for (int i = 0; i < number_of_points - 2; i++)
        diag_main[i + 1] = 2 * (r.dr(i) + r.dr(i + 1));
    diag_main[0] = (r.dr(1) / r.dr(0)) * r.dr(1) - r.dr(0);
    diag_main[number_of_points - 1] = (r.dr(number_of_points - 3) / r.dr(number_of_points - 2)) * r.dr(number_of_points - 3) - r.dr(number_of_points - 2);
    
    // subdiagonals of "A" matrix
    for (int i = 0; i < number_of_points - 1; i++)
    {
        diag_upper[i] = r.dr(i);
        diag_lower[i] = r.dr(i);
    }
    diag_upper[0] = -(r.dr(1) * (1 + r.dr(1) / r.dr(0)) + diag_main[1]);
    diag_lower[number_of_points - 2] = -(r.dr(number_of_points - 3) * (1 + r.dr(number_of_points - 3) / r.dr(number_of_points - 2)) + diag_main[number_of_points - 2]); 

    // solve tridiagonal system
    int info = dgtsv(number_of_points, 1, &diag_lower[0], &diag_main[0], &diag_upper[0], &m[0], number_of_points);
    if (info)
        stop(std::cout << "dgtsv returned " << info);
    
    b.resize(number_of_points - 1);
    c.resize(number_of_points - 1);
    d.resize(number_of_points - 1);

    for (int i = 0; i < number_of_points - 1; i++)
    {
        c[i] = m[i] / 2.0;
        double t = (m[i + 1] - m[i]) / 6.0;
        b[i] = dy[i] - (c[i] + t) * r.dr(i);
        d[i] = t / r.dr(i);
    }
}

double spline::integrate(std::vector<double>& g, int m)
{
    g.resize(number_of_points);
    memset(&g[0], 0, number_of_points * sizeof(double));
    
    if (m == 0)
    {
        for (int i = 0; i < number_of_points - 1; i++)
        {
            double dx = r.dr(i);
            g[i + 1] = g[i] + (((d[i] * dx / 4 + c[i] / 3) * dx + b[i] / 2) * dx + a[i]) * dx;
        }
    }
    else
    {
        for (int i = 0; i < number_of_points - 1; i++)
        {
            double x0 = r[i];
            double x1 = r[i + 1];
            double a0 = a[i];
            double a1 = b[i];
            double a2 = c[i];
            double a3 = d[i];

            // obtained with the following Mathematica code:
            //   FullSimplify[Integrate[x^(m)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],Assumptions->{m>=0,Element[{x0,x1},Reals],x1>x0>0}]
            g[i + 1] = g[i] + (pow(x0, 1 + m) * (-(a0 * (2 + m) * (3 + m) * (4 + m)) + 
                x0 * (a1 * (3 + m) * (4 + m) - 2 * a2 * (4 + m) * x0 + 6 * a3 * pow(x0, 2)))) / ((1 + m) * (2 + m) * (3 + m) * (4 + m)) + 
                pow(x1, 1 + m) * ((a0 - x0 * (a1 + x0 * (-a2 + a3 * x0))) / (1 + m) + ((a1 + x0 * (-2 * a2 + 3 * a3 * x0)) * x1) / (2 + m) + 
                ((a2 - 3 * a3 * x0) * pow(x1, 2)) / (3 + m) + (a3 * pow(x1, 3))/(4 + m));
        }
    }
    
    return g[number_of_points - 1];
}

double spline::integrate(int m)
{
    std::vector<double> g(number_of_points);
    
    return integrate(g, m);
}

};

#endif // __SPLINE_H__
