#include <sirius.h>

using namespace sirius;

void test1()
{
    RadialGrid r(pow3_grid, 1000, 0.01, 2.0, 2.0);

    Function<double, 2> f(Argument(arg_radial, 1000), Argument(arg_lm, 64));
    f.data_.zero();
    for (int ir = 0; ir < 1000; ir++)
    {
        f.data_(ir, 3) = r[ir] * (-2 * sqrt(pi / 3.0));
    }
        
    Function<complex16, 2> zf(Argument(arg_radial, 1000), Argument(arg_lm, 64));
    zf.data_.zero();
    zf.convert_to_ylm(f);

    double v[] = {2.0, 0.0, 0.0};
    double rtp[3];
    SHT::spherical_coordinates(v, rtp);

    std::vector<complex16> ylm(64);
    
    SHT::spherical_harmonics(7, rtp[1], rtp[2], &ylm[0]);

    complex16 z1(0, 0);
    for (int lm = 0; lm < 64; lm++)
    {
        z1 += ylm[lm] * zf.data_(999, lm);
    }
    std::cout << "Value at the MT : " << z1 << std::endl;

    Function<complex16, 2>* g[3];
    for (int i = 0; i < 3; i++) 
        g[i] = new Function<complex16, 2>(Argument(arg_radial, 1000), Argument(arg_lm, 64));

    gradient(r, zf, g);
    
    std::cout << "Gradient value at MT : " << std::endl;
    for (int j = 0; j < 3; j++)
    {   
        z1 = complex16(0, 0);
        for (int lm = 0; lm < 64; lm++)
        {
            z1 += ylm[lm] * g[j]->data_(999, lm);
        }
        std::cout << z1 << std::endl;
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
void test2()
{
    RadialGrid r(pow3_grid, 1000, 0.01, 2.0, 2.0);

    Function<complex16, 2> zf(Argument(arg_radial, 1000), Argument(arg_lm, 64));
    zf.data_.zero();
    
    for (int ir = 0; ir < 1000; ir++)
    {
        zf.data_(ir, 0) = exp(-pow(r[ir], 2));
        zf.data_(ir, 1) = exp(-pow(r[ir], 2));
        zf.data_(ir, 4) = exp(-pow(r[ir], 2));
    }

    Function<complex16, 2>* g[3];
    for (int i = 0; i < 3; i++) 
        g[i] = new Function<complex16, 2>(Argument(arg_radial, 1000), Argument(arg_lm, 64));

    gradient(r, zf, g);

    complex16 zsum(0, 0);
    for (int i = 0; i < 3; i++)
    {
        std::cout << "<f|gf> : " << inner(r, &zf, g[i]) << std::endl; 
    }
}

int main(int argn, char** argv)
{
    Platform::initialize(true);

    test1();
    test2();
    
}
