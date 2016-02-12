#include <sirius.h>

using namespace sirius;

void test_spline_1()
{
    Radial_grid rgrid(exponential_grid, 2000, 1e-7, 2);
    Spline<double> s(rgrid);

    for (int i = 0; i < rgrid.num_points(); i++)
    {
        s[i] = std::sin(rgrid[i]) / rgrid[i];
    }
    s.interpolate();
    printf("value at x=1: %18.12f\n", s(1.0));
    printf("integral: %18.12f\n", s.integrate(0));
}

void test_spline_3(std::vector< Spline<double> > const& s)
{
    for (int k = 0; k < 10; k++)
    {
        printf("value at x=1: %18.12f\n", s[k](1.0));
        printf("integral: %18.12f\n", s[k].integrate(0));
    }

}

void test_spline_2()
{
    Radial_grid rgrid(exponential_grid, 2000, 1e-7, 2);
    std::vector< Spline<double> > s(10);

    for (int k = 0; k < 10; k++)
    {
        s[k] = Spline<double>(rgrid);

        for (int i = 0; i < rgrid.num_points(); i++)
        {
            s[k][i] = std::sin(rgrid[i]) / rgrid[i];
        }
        s[k].interpolate();
        printf("value at x=1: %18.12f\n", s[k](1.0));
        printf("integral: %18.12f\n", s[k].integrate(0));
    }
    test_spline_3(s);
}

void test_spline_4()
{
    Radial_grid rgrid(exponential_grid, 2000, 1e-7, 4);
    Spline<double> s1(rgrid);
    Spline<double> s2(rgrid);
    Spline<double> s3(rgrid);

    for (int ir = 0; ir < 2000; ir++)
    {
        s1[ir] = std::sin(rgrid[ir] * 2) / rgrid[ir];
        s2[ir] = std::exp(rgrid[ir]);
        s3[ir] = s1[ir] * s2[ir];
    }
    s1.interpolate();
    s2.interpolate();
    s3.interpolate();

    Spline<double> s12 = s1 * s2;

    //auto& coefs1 = s1.coefs();
    //auto& coefs2 = s2.coefs();
    //auto& coefs3 = s3.coefs();
    //mdarray<double, 2> coefs12(2000, 4);
    //for (int ir = 0; ir < 2000; ir++)
    //{
    //    coefs12(ir, 0) = coefs1(ir, 0) * coefs2(ir, 0);
    //    coefs12(ir, 1) = coefs1(ir, 1) * coefs2(ir, 0) + coefs1(ir, 0) * coefs2(ir, 1);
    //    coefs12(ir, 2) = coefs1(ir, 2) * coefs2(ir, 0) + coefs1(ir, 1) * coefs2(ir, 1) + coefs1(ir, 0) * coefs2(ir, 2);
    //    coefs12(ir, 3) = coefs1(ir, 3) * coefs2(ir, 0) + coefs1(ir, 2) * coefs2(ir, 1) + coefs1(ir, 1) * coefs2(ir, 2) + coefs1(ir, 0) * coefs2(ir, 3);
    //}

    //Spline<double> s12(rgrid);
    //s12.set_coefs(coefs12);

    FILE* fout = fopen("splne_prod.dat", "w");
    Radial_grid rlin(linear_grid, 10000, 1e-7, 4);
    double d = 0;
    for (int i = 0; i < rlin.num_points(); i++)
    {
        double x = rlin[i];
        d += std::abs(s3(x) - s12(x));
        fprintf(fout, "%18.10f %18.10f %18.10f\n", x, s3(x), s12(x));
    }
    fclose(fout);
    printf("diff = %18.10f\n", d);

    //d = 0;
    //for (int ir = 0; ir < 2000; ir++)
    //{
    //    double d1 = 0;
    //    for (int i = 0; i < 4; i++) d1 += std::abs(coefs3(ir, i) - coefs12(ir, i));
    //    d += d1;
    //    std::cout << ir << " " << d1 << std::endl;
    //}
    //std::cout << "diff = " << d << std::endl;

}

void test_spline_5()
{
    int N = 6000;
    int n = 256;

    Radial_grid rgrid(exponential_grid, N, 1e-7, 4);
    std::vector< Spline<double> > s1(n);
    std::vector< Spline<double> > s2(n);
    for (int i = 0; i < n; i++)
    {
        s1[i] = Spline<double>(rgrid);
        s2[i] = Spline<double>(rgrid);
        for (int ir = 0; ir < N; ir++)
        {
            s1[i][ir] = std::sin(rgrid[ir] * (1 + n * 0.01)) / rgrid[ir];
            s2[i][ir] = std::exp((1 + n * 0.01) * rgrid[ir]);
        }
        s1[i].interpolate();
        s2[i].interpolate();
    }
    mdarray<double, 2> prod(n, n);
    runtime::Timer t("spline|inner");
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            prod(i, j) = inner(s1[i], s2[j], 2);
        }
    }
    double tval = t.stop();
    DUMP("inner product time: %12.6f", tval);
    DUMP("performance: %12.6f GFlops", 1e-9 * n * n * N * 85 / tval);
}

void test_spline_6()
{
    mdarray<Spline<double>, 1> array(20);
    Radial_grid rgrid(exponential_grid, 300, 1e-7, 4);

    for (int i = 0; i < 20; i++)
    {
        array(i) = Spline<double>(rgrid);
        for (int ir = 0; ir < rgrid.num_points(); ir++) array(i)[ir] = std::exp(-rgrid[ir]);
        array(i).interpolate();
    }
}

int main(int argn, char** argv)
{
    sirius::initialize(1);

    test_spline_1();
    test_spline_2();
    test_spline_4();
    test_spline_5();
    test_spline_6();

    sirius::finalize();
}
