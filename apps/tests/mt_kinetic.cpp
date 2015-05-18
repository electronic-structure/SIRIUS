#include <sirius.h>

using namespace sirius;

void test1()
{
    double x0 = 1e-6;
    double x1 = 1.5;
    const int N = 5000;
    Radial_grid r(exponential_grid, N, x0, x1);
    Spline<double> f(r);
    Spline<double> g(r);

    for (int ir = 0; ir < N; ir++)
    {
        f[ir] = exp(-r[ir] * r[ir]);
        g[ir] = sin(r[ir]) / r[ir];
    }
    f.interpolate();
    g.interpolate();

    Spline<double> f1(r);
    Spline<double> g1(r);
    for (int ir = 0; ir < N; ir++)
    {
        f1[ir] = f.deriv(1, ir) * r[ir] * r[ir];
        g1[ir] = g.deriv(1, ir) * r[ir] * r[ir];
    }
    f1.interpolate();
    g1.interpolate();

    Spline<double> f2(r);
    Spline<double> g2(r);
    for (int ir = 0; ir < N; ir++)
    {
        f2[ir] = f1.deriv(1, ir) / r[ir] / r[ir];
        g2[ir] = g1.deriv(1, ir) / r[ir] / r[ir];
    }
    f2.interpolate();
    g2.interpolate();

    std::cout << " <g|nabla^2|f> = " << inner(f2, g, 2) - g[N - 1] * f.deriv(1, N - 1) * x1 * x1 << std::endl;
    std::cout << " <f|nabla^2|g> = " << inner(g2, f, 2) - f[N - 1] * g.deriv(1, N - 1) * x1 * x1 << std::endl;
}

void test2()
{

    double x0 = 1e-6;
    double x1 = 1.892184;
    const int N = 5000;
    Radial_grid r(exponential_grid, N, x0, x1);
    Spline<double> u0(r);
    Spline<double> u1(r);
    Spline<double> d2_u0(r);
    Spline<double> d2_u1(r);
    Spline<double> s(r);

    for (int ir = 0; ir < N; ir++)
    {
        u0[ir] = pow(r[ir], 0.5) * pow(r[ir], 2);
        u1[ir] = pow(r[ir], 0.5) * pow(r[ir], 3);
    }
    u0.interpolate();
    u1.interpolate();

    for (int ir = 0; ir < N; ir++) 
    {
        d2_u0[ir] = u0.deriv(2, ir);
        d2_u1[ir] = u1.deriv(2, ir);
    }

    for (int ir = 0; ir < N; ir++) s[ir] = u0[ir] * u0[ir];

    double norm = s.interpolate().integrate(0);
    norm = 1.0 / sqrt(norm);
    std::cout << "norm0=" << norm << std::endl;
    for (int ir = 0; ir < N; ir++)
    {
        u0[ir] *= norm;
        d2_u0[ir] *= norm;
    }

    for (int ir = 0; ir < N; ir++) s[ir] = u1[ir] * u1[ir];

    norm = s.interpolate().integrate(0);
    norm = 1.0 / sqrt(norm);
    std::cout << "norm1=" << norm << std::endl;
    for (int ir = 0; ir < N; ir++) 
    {
        u1[ir] *= norm;
        d2_u1[ir] *= norm;
    }

    //for (int ir = 0; ir < N; ir++) s[ir] = u0[ir] * u1[ir];
    //double t1 = s.interpolate().integrate(0);

    //for (int ir = 0; ir < N; ir++) 
    //{
    //    u1[ir] -= u0[ir] * t1;
    //    d2_u1[ir] -= d2_u0[ir] * t1;
    //}
    //std::cout << "t1="<<t1<<std::endl;


    for (int ir = 0; ir < N; ir++) s[ir] = (u0[ir] / r[ir]) * (d2_u1[ir] / r[ir]);
    std::cout << "<0|d2|1> = " << -0.5 * s.interpolate().integrate(2) << std::endl;
    
    for (int ir = 0; ir < N; ir++) s[ir] = (u1[ir] / r[ir]) * (d2_u0[ir] / r[ir]);
    std::cout << "<1|d2|0> = " << -0.5 * s.interpolate().integrate(2) << std::endl;
    
    for (int ir = 0; ir < N; ir++) s[ir] = (u0[ir] / r[ir]) * (d2_u0[ir] / r[ir]);
    std::cout << "<0|d2|0> = " << -0.5 * s.interpolate().integrate(2) << std::endl;

    for (int ir = 0; ir < N; ir++) s[ir] = (u1[ir] / r[ir]) * (d2_u1[ir] / r[ir]);
    std::cout << "<1|d2|1> = " << -0.5 * s.interpolate().integrate(2) << std::endl;
}


int main(int argn, char** argv)
{
    test1();
    test2();

}
