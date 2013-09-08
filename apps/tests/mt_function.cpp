#include <sirius.h>

using namespace sirius;

template<typename T, typename U, argument_t a0, int N0, argument_t a1, int N1>
void test1()
{

    MT_function<T> f1(Argument(a0, N0), Argument(a1, N1));
    MT_function<U> f2(Argument(a0, N0), Argument(a1, N1));
    MT_function<T> f3(Argument(a0, N0), Argument(a1, N1));

    srand((int)time(NULL));

    for (int i1 = 0; i1 < N1; i1++)
    {
        for (int i0 = 0; i0 < N0; i0++) 
        {
            f1(i0, i1) = primitive_type_wrapper<T>::sift(complex16(double(rand()) / RAND_MAX, 
                                                                   double(rand()) / RAND_MAX));
        }
    }
    f1.sh_convert(&f2);
    f2.sh_convert(&f3);

    double d = 0;
    for (int i1 = 0; i1 < N1; i1++)
    {
        for (int i0 = 0; i0 < N0; i0++) d += primitive_type_wrapper<T>::abs(f1(i0, i1) - f3(i0, i1));
    }

    std::cout << "diff : " << d << std::endl;
}

template <typename T, int lmax, int Nr>
void test2()
{ 
    int lmmax = Utils::lmmax_by_lmax(lmax);
    SHT sht(lmax);
    MT_function<T> f1(Argument(arg_lm, lmmax), Argument(arg_radial, Nr));
    MT_function<T> f2(Argument(arg_tp, sht.num_points()), Argument(arg_radial, Nr));
    MT_function<T> f3(Argument(arg_lm, lmmax), Argument(arg_radial, Nr));
    

    for (int ir = 0; ir < Nr; ir++)
    {
        for (int lm = 0; lm < lmmax; lm++)
        {
            f1(lm, ir) = primitive_type_wrapper<T>::sift(complex16(double(rand()) / RAND_MAX, 
                                                                   double(rand()) / RAND_MAX));
        }
    }
    f1.sh_transform(&sht, &f2);
    f2.sh_transform(&sht, &f3);

    double d = 0;
    for (int ir = 0; ir < Nr; ir++)
    {
        for (int lm = 0; lm < lmmax; lm++)
        {
            d += primitive_type_wrapper<T>::abs(f1(lm, ir) - f3(lm, ir));
        }
    }

    std::cout << "diff : " << d << std::endl;
}

template <int lmax, int Nr>
void test3()
{ 
    int lmmax = Utils::lmmax_by_lmax(lmax);
    SHT sht(lmax);
    MT_function<double> f1(Argument(arg_lm, lmmax), Argument(arg_radial, Nr));
    MT_function<double> f2(Argument(arg_tp, sht.num_points()), Argument(arg_radial, Nr));
    MT_function<complex16> f3(Argument(arg_tp, sht.num_points()), Argument(arg_radial, Nr));
    MT_function<complex16> f4(Argument(arg_lm, lmmax), Argument(arg_radial, Nr));
    MT_function<double> f5(Argument(arg_lm, lmmax), Argument(arg_radial, Nr));

    for (int ir = 0; ir < Nr; ir++)
    {
        for (int lm = 0; lm < lmmax; lm++) f1(lm, ir) = double(rand()) / RAND_MAX;
    }
    f1.sh_transform(&sht, &f2);
    for (int ir = 0; ir < Nr; ir++)
    {
        for (int tp = 0; tp < sht.num_points(); tp++) f3(tp, ir) = f2(tp, ir);
    }


    f3.sh_transform(&sht, &f4);
    f4.sh_convert(&f5);

    double d = 0;
    for (int ir = 0; ir < Nr; ir++)
    {
        for (int lm = 0; lm < lmmax; lm++)
        {
            d += primitive_type_wrapper<double>::abs(f1(lm, ir) - f5(lm, ir));
        }
    }

    std::cout << "diff : " << d << std::endl;
}

void test4()
{
    Radial_grid r(linear_exponential_grid, 1000, 0.01, 2.0, 2.0);

    MT_function<double> f(Argument(arg_radial, 1000), Argument(arg_lm, 64));
    f.zero();
    for (int ir = 0; ir < 1000; ir++)
    {
        f(ir, 3) = r[ir] * (-2 * sqrt(pi / 3.0));
    }
        
    MT_function<complex16> zf(&f, true);
    
    double v[] = {2.0, 0.0, 0.0};
    double rtp[3];
    SHT::spherical_coordinates(v, rtp);

    std::vector<complex16> ylm(64);
    std::vector<double> rlm(64);
    
    SHT::spherical_harmonics(7, rtp[1], rtp[2], &ylm[0]);
    SHT::spherical_harmonics(7, rtp[1], rtp[2], &rlm[0]);

    complex16 z1(0, 0);
    for (int lm = 0; lm < 64; lm++)
    {
        z1 += ylm[lm] * zf(999, lm);
    }
    std::cout << "Value at the MT : " << z1 << std::endl;

    MT_function<complex16>* g[3];
    MT_function<double>* gr[3];
    for (int i = 0; i < 3; i++) 
    {
        g[i] = new MT_function<complex16>(&zf, false);
        gr[i] = new MT_function<double>(&f, false);
    }

    gradient(r, &zf, g[0], g[1], g[2]);
    gradient(r, &f, gr[0], gr[1], gr[2]);
    
    std::cout << "Gradient value at MT : " << std::endl;
    for (int j = 0; j < 3; j++)
    {   
        z1 = complex16(0, 0);
        double d1 = 0.0;
        for (int lm = 0; lm < 64; lm++)
        {
            z1 += ylm[lm] * (*g[j])(999, lm);
            d1 += rlm[lm] * (*gr[j])(999, lm);
        }
        std::cout << z1 << " " << d1 << std::endl;
    }
    for (int i = 0; i < 3; i++) 
    {
        delete g[i];
        delete gr[i];
    }
}

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

    MT_function<complex16> zf(Argument(arg_radial, 1000), Argument(arg_lm, 64));
    zf.zero();
    
    for (int ir = 0; ir < 1000; ir++)
    {
        zf(ir, 0) = exp(-pow(r[ir], 2));
        zf(ir, 1) = exp(-pow(r[ir], 2));
        zf(ir, 4) = exp(-pow(r[ir], 2));
    }

    MT_function<complex16>* g[3];
    for (int i = 0; i < 3; i++) 
        g[i] = new MT_function<complex16>(Argument(arg_radial, 1000), Argument(arg_lm, 64));

    gradient(r, &zf, g[0], g[1], g[2]);

    complex16 zsum(0, 0);
    for (int i = 0; i < 3; i++)
    {
        std::cout << "<f|gf> : " << inner(r, &zf, g[i]) << std::endl; 
    }
    
    for (int i = 0; i < 3; i++) delete g[i]; 
}

//void test6()
//{
//    SHT sht(7);
//
//    mt_function<double> f1(Argument(arg_rlm, 64), Argument(arg_radial, 1000));
//
//    for (int ir = 0; ir < 1000; ir++)
//    {
//        for (int lm = 0; lm < 64; lm++)
//        {
//            f1(lm, ir) = double(rand()) / RAND_MAX;
//        }
//    }
//    
//    mt_function<double> f2(Argument(arg_tp, sht.num_points()), Argument(arg_radial, 1000));
//    f1.sh_transform(&sht, &f2);
//
//    mt_function<complex16> f3(Argument(arg_tp, sht.num_points()), Argument(arg_radial, 1000));
//    for (int ir = 0; ir < 1000; ir++)
//    {
//        for (int itp = 0; itp < sht.num_points(); itp++)
//        {
//            f3(itp, ir) = f2(itp, ir);
//        }
//    }
//    mt_function<complex16> f4(Argument(arg_ylm, 64), Argument(arg_radial, 1000));
//    f3.sh_transform(&sht, &f4);
//
//    mt_function<double> f5(Argument(arg_rlm, 64), Argument(arg_radial, 1000));
//    f4.sh_convert(&f5);
//
//    double d = 0;
//    for (int ir = 0; ir < 1000; ir++)
//    {
//        for (int lm = 0; lm < 64; lm++)
//        {
//            d += fabs(f1(lm, ir) - f5(lm, ir));
//        }
//    }
//
//    std::cout << "diff : " << d << std::endl;
//
//
//}

int main(int argn, char** argv)
{
    Platform::initialize(true);

    std::cout << "Rlm -> Ylm -> Rlm transformation, radial index first" << std::endl;
    test1<double, complex16, arg_radial, 1000, arg_lm, 64>();
    std::cout << "Rlm -> Ylm -> Rlm transformation, radial index second" << std::endl;
    test1<double, complex16, arg_lm, 64, arg_radial, 1000>();
    
    std::cout << "Rlm -> (t,p) -> Rlm transformation" << std::endl;
    test2<double, 10, 1000>();
    std::cout << "Ylm -> (t,p) -> Ylm transformation" << std::endl;
    test2<complex16, 10, 1000>();

    std::cout << "Rlm -> (t,p) -> Ylm -> Rlm transformation" << std::endl;
    test3<10, 1000>();

    test4();

    test5();

    //test6();
    
}
