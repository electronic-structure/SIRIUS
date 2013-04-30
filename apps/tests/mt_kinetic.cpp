#include <sirius.h>

void test1()
{

    double x0 = 1e-6;
    double x1 = 1.5;
    const int N = 5000;
    sirius::RadialGrid r(sirius::exponential_grid, N, x0, x1);
    sirius::Spline<double> f(N, r);
    sirius::Spline<double> g(N, r);

    for (int ir = 0; ir < N; ir++)
    {
        f[ir] = exp( -r[ir] * r[ir]);
        g[ir] = sin(r[ir]) / r[ir];
    }
    f.interpolate();
    g.interpolate();

    sirius::Spline<double> f1(N, r);
    sirius::Spline<double> g1(N, r);
    for (int ir = 0; ir < N; ir++)
    {
        f1[ir] = f.deriv(1, ir) * r[ir] * r[ir];
        g1[ir] = g.deriv(1, ir) * r[ir] * r[ir];
    }
    f1.interpolate();
    g1.interpolate();

    sirius::Spline<double> f2(N, r);
    sirius::Spline<double> g2(N, r);
    for (int ir = 0; ir < N; ir++)
    {
        f2[ir] = f1.deriv(1, ir) / r[ir] / r[ir];
        g2[ir] = g1.deriv(1, ir) / r[ir] / r[ir];
    }
    f2.interpolate();
    g2.interpolate();

    std::cout << " <g|nabla^2|f> = " << sirius::Spline<double>::integrate(&f2, &g) - g[N - 1] * f.deriv(1, N - 1) * x1 * x1 << std::endl;
    std::cout << " <f|nabla^2|g> = " << sirius::Spline<double>::integrate(&g2, &f) - f[N - 1] * g.deriv(1, N - 1) * x1 * x1 << std::endl;
}

void test2()
{

    double x0 = 1e-6;
    double x1 = 1.892184;
    const int N = 5000;
    sirius::RadialGrid r(sirius::exponential_grid, N, x0, x1);
    sirius::Spline<double> u0(N, r);
    sirius::Spline<double> u1(N, r);
    sirius::Spline<double> d2_u0(N, r);
    sirius::Spline<double> d2_u1(N, r);
    sirius::Spline<double> s(N, r);

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
    test2();

}
