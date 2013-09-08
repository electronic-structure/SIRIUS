#include <sirius.h>

using namespace sirius;

void test1(double x0, double x1, int m, double exact_result)
{
    printf("\n");
    printf("test1: integrate sin(x) * x^{%i} and compare with exact result\n", m);
    printf("       lower and upper boundaries: %f %f\n", x0, x1);
    Radial_grid r(exponential_grid, 5000, x0, x1);
    Spline<double> s(5000, r);
    
    for (int i = 0; i < 5000; i++) s[i] = sin(r[i]);
    
    double d = s.interpolate().integrate(m);
    double abs_err = fabs(d - exact_result);
    
    printf("       absolute error: %18.12f", abs_err);
    if (abs_err < 1e-10) 
    {
        printf("  OK\n");
    }
    else
    {
        printf("  Fail\n");
    }
}

void test2(radial_grid_type grid_type, double x0, double x1)
{
    printf("\n");
    printf("test2: value and derivatives of exp(x)\n");

    int N = 5000;
    Radial_grid r(grid_type, N, x0, x1, 3 * x1);
    Spline<double> s(N, r);
    
    for (int i = 0; i < N; i++) s[i] = exp(r[i]);
    
    s.interpolate();

    printf("grid type : %s\n", r.grid_type_name().c_str());

    std::string fname = "grid_" + r.grid_type_name() + ".txt";
    FILE* fout = fopen(fname.c_str(), "w");
    for (int i = 0; i < r.size(); i++) fprintf(fout,"%i %16.12e\n", i, r[i]);
    fclose(fout);
    
    printf("x = %f, exp(x) = %f, exp(x) = %f, exp'(x)= %f, exp''(x) = %f\n", x0, s[0], s.deriv(0, 0), s.deriv(1, 0), s.deriv(2, 0));
    printf("x = %f, exp(x) = %f, exp(x) = %f, exp'(x)= %f, exp''(x) = %f\n", x1, s[N - 1], s.deriv(0, N - 1), s.deriv(1, N - 1), s.deriv(2, N - 1));
}

void test3(double x0, double x1, double exact_val)
{
    printf("\n");
    printf("test3\n");
    
    Radial_grid r(exponential_grid, 2000, x0, x1);
    Spline<double> s1(2000, r);
    Spline<double> s2(2000, r);
    Spline<double> s3(2000, r);

    for (int i = 0; i < 2000; i++)
    {
        s1[i] = sin(r[i]) / r[i];
        s2[i] = exp(-r[i]) * pow(r[i], 8.0 / 3.0);
        s3[i] = s1[i] * s2[i];
    }
    s1.interpolate();
    s2.interpolate();
    s3.interpolate();

    double v1 = s3.integrate(2);
    double v2 = Spline<double>::integrate(&s1, &s2);

    printf("interpolate product of two functions and then integrate with spline   : %16.12f\n", v1);
    printf("interpolate two functions and then integrate the product analytically : %16.12f\n", v2);
    printf("                                                           difference : %16.12f\n", fabs(v1 - v2));
    printf("                                                         exact result : %16.12f\n", exact_val);
}

//void test4(double x0, double x1)
//{
//    Radial_grid r(exponential_grid, 5000, x0, x1);
//    Spline<double> s(5000, r);
//    
//    for (int i = 0; i < 5000; i++) s[i] = exp(-r[i]) * r[i] * r[i];
//    
//    s.interpolate();
//    
//    std::cout << s[0] << " " << s.deriv(0, 0) << " " << s.deriv(1, 0) << " " << s.deriv(2, 0) << std::endl;
//    std::cout << s[4999] << " " << s.deriv(0, 4999) << " " << s.deriv(1, 4999) << " " << s.deriv(2, 4999) << std::endl;
//}

int main(int argn, char **argv)
{
    Platform::initialize(1);

    test1(0.1, 7.13, 0, 0.3326313127230704);
    test1(0.1, 7.13, 1, -3.973877090504168);
    test1(0.1, 7.13, 2, -23.66503552796384);
    test1(0.1, 7.13, 3, -101.989998166403);
    test1(0.1, 7.13, 4, -341.6457111811293);
    test1(0.1, 7.13, -1, 1.367605245879218);
    test1(0.1, 7.13, -2, 2.710875755556171);
    test1(0.1, 7.13, -3, 9.22907091561693);
    test1(0.1, 7.13, -4, 49.40653515725798);
    test1(0.1, 7.13, -5, 331.7312413927384);
    
    double x0 = 0.00001;
    test2(linear_grid, x0, 2.0);
    test2(exponential_grid, x0, 2.0);
    test2(linear_exponential_grid, x0, 2.0);
    test2(pow_grid, x0, 2.0);
    //test2(hyperbolic_grid, x0, 2.0);

    test3(0.0001, 2.0, 1.0365460153117974);

    //test4(0.0001, 1.892184);
}
