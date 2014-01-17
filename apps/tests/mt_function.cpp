#include <sirius.h>

using namespace sirius;

template<typename T, typename U>
void test1_radial_angular()
{
    int lmmax = 64;

    Radial_grid r(exponential_grid, 1000, 0.01, 2.0);

    Spheric_function<T> f1(r, lmmax);
    Spheric_function<U> f2(r, lmmax);
    Spheric_function<T> f3(r, lmmax);

    srand((int)time(NULL));

    for (int lm = 0; lm < lmmax; lm++)
    {
        for (int ir = 0; ir < r.num_mt_points(); ir++) 
        {
            f1(ir, lm) = primitive_type_wrapper<T>::sift(complex16(double(rand()) / RAND_MAX, 
                                                                   double(rand()) / RAND_MAX));
        }
    }
    f1.sh_convert(f2);
    f2.sh_convert(f3);

    double d = 0;
    for (int lm = 0; lm < lmmax; lm++)
    {
        for (int ir = 0; ir < r.num_mt_points(); ir++) d += primitive_type_wrapper<T>::abs(f1(ir, lm) - f3(ir, lm));
    }
    
    if (d < 1e-10)
    {
        std::cout << "OK" << std::endl;
    }
    else
    {
        std::cout << "failed, diff : " << d << std::endl;
    }
}

template<typename T, typename U>
void test1_angular_radial()
{
    int lmmax = 64;

    Radial_grid r(exponential_grid, 1000, 0.01, 2.0);

    Spheric_function<T> f1(lmmax, r);
    Spheric_function<U> f2(lmmax, r);
    Spheric_function<T> f3(lmmax, r);

    srand((int)time(NULL));

    for (int ir = 0; ir < r.num_mt_points(); ir++) 
    {
        for (int lm = 0; lm < lmmax; lm++)
        {
            f1(lm, ir) = primitive_type_wrapper<T>::sift(complex16(double(rand()) / RAND_MAX, 
                                                                   double(rand()) / RAND_MAX));
        }
    }
    f1.sh_convert(f2);
    f2.sh_convert(f3);

    double d = 0;
    for (int ir = 0; ir < r.num_mt_points(); ir++) 
    {
        for (int lm = 0; lm < lmmax; lm++) d += primitive_type_wrapper<T>::abs(f1(lm, ir) - f3(lm, ir));
    }

    std::cout << "diff : " << d << std::endl;
}

template <typename T>
void test2(int lmax, int nr)
{ 
    int lmmax = Utils::lmmax(lmax);
    Radial_grid r(exponential_grid, nr, 0.01, 2.0);

    SHT sht(lmax);
    Spheric_function<T> f1(lmmax, r);
    Spheric_function<T> f2(sht, r);
    Spheric_function<T> f3(lmmax, r);
    

    for (int ir = 0; ir < nr; ir++)
    {
        for (int lm = 0; lm < lmmax; lm++)
        {
            f1(lm, ir) = primitive_type_wrapper<T>::sift(complex16(double(rand()) / RAND_MAX, 
                                                                   double(rand()) / RAND_MAX));
        }
    }
    f1.sh_transform(f2);
    f2.sh_transform(f3);

    double d = 0;
    for (int ir = 0; ir < nr; ir++)
    {
        for (int lm = 0; lm < lmmax; lm++)
        {
            d += primitive_type_wrapper<T>::abs(f1(lm, ir) - f3(lm, ir));
        }
    }

    std::cout << "diff : " << d << std::endl;
}

void test3(int lmax, int nr)
{ 
    int lmmax = Utils::lmmax(lmax);
    Radial_grid r(exponential_grid, nr, 0.01, 2.0);
    SHT sht(lmax);

    Spheric_function<double> f1(lmmax, r);
    Spheric_function<double> f2(sht, r);
    Spheric_function<complex16> f3(sht, r);
    Spheric_function<complex16> f4(lmmax, r);
    Spheric_function<double> f5(lmmax, r);

    for (int ir = 0; ir < nr; ir++)
    {
        for (int lm = 0; lm < lmmax; lm++) f1(lm, ir) = double(rand()) / RAND_MAX;
    }
    f1.sh_transform(f2);
    for (int ir = 0; ir < nr; ir++)
    {
        for (int tp = 0; tp < sht.num_points(); tp++) f3(tp, ir) = f2(tp, ir);
    }

    f3.sh_transform(f4);
    f4.sh_convert(f5);

    double d = 0;
    for (int ir = 0; ir < nr; ir++)
    {
        for (int lm = 0; lm < lmmax; lm++)
        {
            d += primitive_type_wrapper<double>::abs(f1(lm, ir) - f5(lm, ir));
        }
    }

    std::cout << "diff : " << d << std::endl;
}
///== 
///== void test4()
///== {
///==     Radial_grid r(linear_exponential_grid, 1000, 0.01, 2.0, 2.0);
///== 
///==     MT_function<double> f(Argument(arg_radial, 1000), Argument(arg_lm, 64));
///==     f.zero();
///==     for (int ir = 0; ir < 1000; ir++)
///==     {
///==         f(ir, 3) = r[ir] * (-2 * sqrt(pi / 3.0));
///==     }
///==         
///==     MT_function<complex16> zf(&f, true);
///==     
///==     double v[] = {2.0, 0.0, 0.0};
///==     double rtp[3];
///==     SHT::spherical_coordinates(v, rtp);
///== 
///==     std::vector<complex16> ylm(64);
///==     std::vector<double> rlm(64);
///==     
///==     SHT::spherical_harmonics(7, rtp[1], rtp[2], &ylm[0]);
///==     SHT::spherical_harmonics(7, rtp[1], rtp[2], &rlm[0]);
///== 
///==     complex16 z1(0, 0);
///==     for (int lm = 0; lm < 64; lm++)
///==     {
///==         z1 += ylm[lm] * zf(999, lm);
///==     }
///==     std::cout << "Value at the MT : " << z1 << std::endl;
///== 
///==     MT_function<complex16>* g[3];
///==     MT_function<double>* gr[3];
///==     for (int i = 0; i < 3; i++) 
///==     {
///==         g[i] = new MT_function<complex16>(&zf, false);
///==         gr[i] = new MT_function<double>(&f, false);
///==     }
///== 
///==     gradient(r, &zf, g[0], g[1], g[2]);
///==     gradient(r, &f, gr[0], gr[1], gr[2]);
///==     
///==     std::cout << "Gradient value at MT : " << std::endl;
///==     for (int j = 0; j < 3; j++)
///==     {   
///==         z1 = complex16(0, 0);
///==         double d1 = 0.0;
///==         for (int lm = 0; lm < 64; lm++)
///==         {
///==             z1 += ylm[lm] * (*g[j])(999, lm);
///==             d1 += rlm[lm] * (*gr[j])(999, lm);
///==         }
///==         std::cout << z1 << " " << d1 << std::endl;
///==     }
///==     for (int i = 0; i < 3; i++) 
///==     {
///==         delete g[i];
///==         delete gr[i];
///==     }
///== }
///== 

/* The following Mathematica code was used to check the gadient:
    
    <<VectorAnalysis`
    SetCoordinates[Spherical]
    f[r_,t_,p_]:=Exp[-r*r]*(SphericalHarmonicY[0,0,t,p]+SphericalHarmonicY[1,-1,t,p]+SphericalHarmonicY[2,-2,t,p]);
    g=Grad[f[Rr,Ttheta,Pphi]];
    G=FullSimplify[
    g[[1]]*{Cos[Pphi] Sin[Ttheta],Sin[Pphi] Sin[Ttheta],Cos[Ttheta]}+
    g[[2]]*{Cos[Pphi] Cos[Ttheta], Cos[Ttheta] Sin[Pphi],-Sin[Ttheta]}+
    g[[3]]*{- Sin[Pphi],Cos[Pphi],0}
    ];
    Integrate[G*Conjugate[f[Rr,Ttheta,Pphi]]*Rr*Rr*Sin[Ttheta],{Rr,0.01,2},{Ttheta,0,Pi},{Pphi,-Pi,Pi}]

    Output: {0.00106237, 0. - 0.650989 I, 0}
*/
void test5()
{
    Radial_grid r(linear_exponential_grid, 1000, 0.01, 2.0, 2.0);

    int lmmax = 64;
    Spheric_function<complex16> f(r, lmmax);
    f.zero();
    
    for (int ir = 0; ir < 1000; ir++)
    {
        f(ir, 0) = exp(-pow(r[ir], 2));
        f(ir, 1) = exp(-pow(r[ir], 2));
        f(ir, 4) = exp(-pow(r[ir], 2));
    }

    Spheric_function_gradient<complex16> grad(f);

    vector3d<complex16> v = inner(f, grad);

    std::cout << "grad : ";
    for (int i = 0; i < 3; i++) std::cout << v[i] << " ";
    std::cout << std::endl;
}

void test6()
{
    int nr = 2000;

    Radial_grid r(linear_exponential_grid, nr, 0.01, 2.0, 2.0);

    Spheric_function<double> f(r, 64);

    for (int l1 = 0; l1 <= 5; l1++)
    {
        for (int m1 = -l1; m1 <= l1; m1++)
        {
            f.zero();
            for (int ir = 0; ir < nr; ir++) f(ir, Utils::lm_by_l_m(l1, m1)) = exp(-r[ir]) * cos(l1 * r[ir]) * sin(m1 + r[ir]);
            Spheric_function_gradient<double> grad(f);

            for (int l2 = 0; l2 <= 5; l2++)
            {
                for (int m2 = -l2; m2 <= l2; m2++)
                {
                    f.zero();
                    for (int ir = 0; ir < nr; ir++) f(ir, Utils::lm_by_l_m(l2, m2)) = exp(-r[ir]) * cos(l2 * r[ir]) * sin(m2 + r[ir]);
                    
                    vector3d<double> v = inner(f, grad);

                    std::cout << "<lm2=" << Utils::lm_by_l_m(l2, m2) << "|grad|lm1=" << Utils::lm_by_l_m(l1, m1) << "> : ";
                    for (int i = 0; i < 3; i++) std::cout << v[i] << " ";
                    std::cout << std::endl;
                }
            }
        }
    }

}

///== void test7()
///== {
///==     int nr = 2000;
///== 
///==     Radial_grid r(linear_exponential_grid, nr, 0.01, 2.0, 2.0);
///== 
///==     MT_function<double> f(Argument(arg_lm, 64), Argument(arg_radial, nr));
///==     MT_function<double>* g[3];
///==     for (int i = 0; i < 3; i++) g[i] = new MT_function<double>(Argument(arg_lm, 64), Argument(arg_radial, nr));
///== 
///==     for (int l1 = 0; l1 <= 5; l1++)
///==     {
///==         for (int m1 = -l1; m1 <= l1; m1++)
///==         {
///==             f.zero();
///==             for (int ir = 0; ir < nr; ir++) f(Utils::lm_by_l_m(l1, m1), ir) = exp(-r[ir]) * cos(l1 * r[ir]) * sin(m1 + r[ir]);
///==             gradient(r, &f, g[0], g[1], g[2]);
///== 
///==             for (int l2 = 0; l2 <= 5; l2++)
///==             {
///==                 for (int m2 = -l2; m2 <= l2; m2++)
///==                 {
///==                     f.zero();
///==                     for (int ir = 0; ir < nr; ir++) f(Utils::lm_by_l_m(l2, m2), ir) = exp(-r[ir]) * cos(l2 * r[ir]) * sin(m2 + r[ir]);
///== 
///==                     std::cout << "<lm2=" << Utils::lm_by_l_m(l2, m2) << "|grad|lm1=" << Utils::lm_by_l_m(l1, m1) << "> : ";
///==                     for (int i = 0; i < 3; i++) std::cout << inner(r, &f, g[i]) << " ";
///==                     std::cout << std::endl;
///==                 }
///==             }
///==         }
///==     }
///== 
///==     for (int i = 0; i < 3; i++) delete g[i]; 
///== }
///== 
///== void test8()
///== {
///== 
///==     int nr = 2000;
///== 
///==     Radial_grid r(linear_exponential_grid, nr, 0.01, 1.0, 1.0);
///== 
///==     MT_function<double> f(Argument(arg_lm, 64), Argument(arg_radial, nr));
///==     MT_function<double>* g[3];
///==     for (int i = 0; i < 3; i++) g[i] = new MT_function<double>(Argument(arg_lm, 64), Argument(arg_radial, nr));
///== 
///==     
///==     int l = 2;
///==     int m = 2;
///==     f.zero();
///==     for (int ir = 0; ir < nr; ir++) f(Utils::lm_by_l_m(l, m), ir) = exp(-r[ir]) * cos(l * r[ir]) * sin(m + r[ir]);
///==     gradient(r, &f, g[0], g[1], g[2]);
///== 
///==     std::vector<double> rlm(64);
///==     
///==     SHT::spherical_harmonics(7, 1.1, 2.5, &rlm[0]);
///== 
///== 
///==     std::cout << "Gradient :  ";
///==     for (int j = 0; j < 3; j++)
///==     {   
///==         double d1 = 0.0;
///==         for (int lm = 0; lm < 64; lm++) d1 += rlm[lm] * (*g[j])(lm, nr - 1);
///==         std::cout << d1 << " ";
///==     }
///==     std::cout << std::endl;
///== 
///==     for (int i = 0; i < 3; i++) delete g[i]; 
///== }
///== 
///== extern "C"void gradrfmt_(int* lmax, int* nr, double* r, int* ld1, int* ld2, double* rfmt, double* grfmt);
///== 
///== void test9()
///== {
///==     
///==     int nr = 4000;
///== 
///==     int lmax = 14;
///==     int lmmax = Utils::lmmax(lmax);
///== 
///==     Radial_grid r(linear_exponential_grid, nr, 0.01, 1.0, 1.0);
///== 
///==     MT_function<double> f(Argument(arg_lm, lmmax), Argument(arg_radial, nr));
///==     MT_function<double>* g[3];
///==     for (int i = 0; i < 3; i++) g[i] = new MT_function<double>(Argument(arg_lm, lmmax), Argument(arg_radial, nr));
///== 
///==     
///==     f.zero();
///==     for (int l = 0; l <= lmax; l++)
///==     {
///==         for (int m = -l; m <= l; m++)
///==         {
///==             for (int ir = 0; ir < nr; ir++) f(Utils::lm_by_l_m(l, m), ir) = exp(-r[ir]) * cos(l * r[ir]) * sin(m + r[ir]);
///==         }
///==     }
///==     gradient(r, &f, g[0], g[1], g[2]);
///== 
///==     std::vector<double> rgrid(nr);
///==     r.get_radial_points(&rgrid[0]);
///==     mdarray<double, 3> ftn_grad(lmmax, nr, 3);
///== 
///==     gradrfmt_(&lmax, &nr, &rgrid[0], &lmmax, &nr, &f(0, 0), &ftn_grad(0, 0, 0));
///== 
///==     for (int x = 0; x < 3; x++)
///==     {
///==         double d = 0.0;
///==         for (int ir = 0; ir < nr; ir++)
///==         {
///==             for (int lm = 0; lm < lmmax; lm++) d += fabs((*g[x])(lm, ir) - ftn_grad(lm, ir, x));
///==         }
///==         printf("coord : %i, diff : %12.6f, average diff : %18.12f\n", x, d, d / lmmax / nr);
///==     }
///== 
///== 
///== 
///== 
///== 
///==     for (int i = 0; i < 3; i++) delete g[i]; 
///== }
///== 
///== 
///== 
///== 
///== 
///== //void test6()
///== //{
///== //    SHT sht(7);
///== //
///== //    mt_function<double> f1(Argument(arg_rlm, 64), Argument(arg_radial, 1000));
///== //
///== //    for (int ir = 0; ir < 1000; ir++)
///== //    {
///== //        for (int lm = 0; lm < 64; lm++)
///== //        {
///== //            f1(lm, ir) = double(rand()) / RAND_MAX;
///== //        }
///== //    }
///== //    
///== //    mt_function<double> f2(Argument(arg_tp, sht.num_points()), Argument(arg_radial, 1000));
///== //    f1.sh_transform(&sht, &f2);
///== //
///== //    mt_function<complex16> f3(Argument(arg_tp, sht.num_points()), Argument(arg_radial, 1000));
///== //    for (int ir = 0; ir < 1000; ir++)
///== //    {
///== //        for (int itp = 0; itp < sht.num_points(); itp++)
///== //        {
///== //            f3(itp, ir) = f2(itp, ir);
///== //        }
///== //    }
///== //    mt_function<complex16> f4(Argument(arg_ylm, 64), Argument(arg_radial, 1000));
///== //    f3.sh_transform(&sht, &f4);
///== //
///== //    mt_function<double> f5(Argument(arg_rlm, 64), Argument(arg_radial, 1000));
///== //    f4.sh_convert(&f5);
///== //
///== //    double d = 0;
///== //    for (int ir = 0; ir < 1000; ir++)
///== //    {
///== //        for (int lm = 0; lm < 64; lm++)
///== //        {
///== //            d += fabs(f1(lm, ir) - f5(lm, ir));
///== //        }
///== //    }
///== //
///== //    std::cout << "diff : " << d << std::endl;
///== //
///== //
///== //}

int main(int argn, char** argv)
{
    Platform::initialize(true);

    std::cout << "Rlm -> Ylm -> Rlm transformation, radial index first: ";
    test1_radial_angular<double, complex16>();
    
    std::cout << "Rlm -> Ylm -> Rlm transformation, angular index first" << std::endl;
    test1_angular_radial<double, complex16>();
    
    std::cout << "Rlm -> (t,p) -> Rlm transformation, angular index first" << std::endl;
    test2<double>(10, 1000);

    std::cout << "Ylm -> (t,p) -> Ylm transformation, angular index first" << std::endl;
    test2<complex16>(10, 1000);

    std::cout << "Rlm -> (t,p) -> Ylm -> Rlm transformation" << std::endl;
    test3(10, 1000);

    //== test4();

    std::cout << "Gradient of a function" << std::endl;
    test5();
    
    std::cout << "Matrix elements of a gradient, radial index first" << std::endl;
    test6();
    //== //test7();

    //== //test8();
    //== 
    //== test9();

    //== //test6();
    
}
