/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "sirius.hpp"
#include "testing.hpp"

using namespace sirius;

template <typename T>
int
test1()
{
    SHT sht(device_t::CPU, 7);
    int lmmax = 64;

    auto r = Radial_grid_factory<double>(radial_grid_t::exponential, 1000, 0.01, 2.0, 1.0);

    Spheric_function<function_domain_t::spectral, T> f1(lmmax, r);

    for (int ir = 0; ir < r.num_points(); ir++) {
        for (int lm = 0; lm < lmmax; lm++) {
            f1(lm, ir) = random<T>();
        }
    }
    auto f2 = convert(f1);
    auto f3 = convert(f2);

    double d{0};
    for (int ir = 0; ir < r.num_points(); ir++) {
        for (int lm = 0; lm < lmmax; lm++) {
            d += std::abs(f1(lm, ir) - f3(lm, ir));
        }
    }
    if (d < 1e-10) {
        return 0;
    } else {
        return 1;
    }
}

template <typename T>
int
test2(int lmax, int nr)
{
    int lmmax = sf::lmmax(lmax);
    auto r    = Radial_grid_factory<double>(radial_grid_t::exponential, nr, 0.01, 2.0, 1.0);

    SHT sht(device_t::CPU, lmax);
    Spheric_function<function_domain_t::spectral, T> f1(lmmax, r);

    for (int ir = 0; ir < nr; ir++) {
        for (int lm = 0; lm < lmmax; lm++) {
            f1(lm, ir) = random<T>();
        }
    }
    auto f2 = transform(sht, f1);
    auto f3 = transform(sht, f2);

    double d{0};
    for (int ir = 0; ir < nr; ir++) {
        for (int lm = 0; lm < lmmax; lm++) {
            d += std::abs(f1(lm, ir) - f3(lm, ir));
        }
    }
    if (d < 1e-9) {
        return 0;
    } else {
        return 1;
    }
}

int
test3(int lmax, int nr)
{
    int lmmax = sf::lmmax(lmax);
    auto r    = Radial_grid_factory<double>(radial_grid_t::exponential, nr, 0.01, 2.0, 1.0);
    SHT sht(device_t::CPU, lmax);

    Spheric_function<function_domain_t::spectral, double> f1(lmmax, r);
    Spheric_function<function_domain_t::spatial, std::complex<double>> f3(sht.num_points(), r);

    /* real Rlm coefficients */
    for (int ir = 0; ir < nr; ir++) {
        for (int lm = 0; lm < lmmax; lm++) {
            f1(lm, ir) = random<double>();
        }
    }
    /* reansform to (theta, phi) */
    auto f2 = transform(sht, f1);
    /* copy to complex (theta, phi) */
    for (int ir = 0; ir < nr; ir++) {
        for (int tp = 0; tp < sht.num_points(); tp++) {
            f3(tp, ir) = f2(tp, ir);
        }
    }
    /* transform to Ylm */
    auto f4 = transform(sht, f3);
    /* convert to Rlm */
    auto f5 = convert(f4);

    /* compare with initial function */
    double d{0};
    for (int ir = 0; ir < nr; ir++) {
        for (int lm = 0; lm < lmmax; lm++) {
            d += std::abs(f1(lm, ir) - f5(lm, ir));
        }
    }
    if (d < 1e-10) {
        return 0;
    } else {
        return 1;
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
int
test4()
{
    r3::vector<std::complex<double>> ref_v(0.00106237, std::complex<double>(0, -0.650989), 0);

    auto r = Radial_grid_factory<double>(radial_grid_t::exponential, 1000, 0.01, 2.0, 1.0);

    int lmmax = 64;
    Spheric_function<function_domain_t::spectral, std::complex<double>> f(lmmax, r);
    f.zero();

    for (int ir = 0; ir < 1000; ir++) {
        f(0, ir) = std::exp(-std::pow(r[ir], 2));
        f(1, ir) = std::exp(-std::pow(r[ir], 2));
        f(4, ir) = std::exp(-std::pow(r[ir], 2));
    }

    auto grad_f = gradient(f);

    r3::vector<std::complex<double>> v;
    for (int x = 0; x < 3; x++) {
        v[x] = inner(f, grad_f[x]);
    }

    double d{0};
    for (int x : {0, 1, 2}) {
        d += std::abs(v[x] - ref_v[x]);
    }
    if (d < 1e-7) {
        return 0;
    } else {
        return 1;
    }
}

/* matrix elements were generated with the following Mathematica code

  R1[l_, m_, th_, ph_] :=
  If[m > 0,
   Sqrt[2]*ComplexExpand[Re[SphericalHarmonicY[l, m, th, ph]]
     ], If[m < 0,
    Sqrt[2]*ComplexExpand[Im[SphericalHarmonicY[l, m, th, ph]]],
    If[m == 0, ComplexExpand[Re[SphericalHarmonicY[l, 0, th, ph]]]]]];

  f[l_, m_, r_, t_, p_] := Exp[-r]*Cos[l*r]*Sin[m + r]*R1[l, m, t, p];

  lmax = 5
  lmmax = (lmax + 1)^2
  result = Table[0, {lm1, 1, lmmax}, {lm2, 1, lmmax}];

  Timing[For[l1 = 0, l1 <= lmax, l1++,
  For[m1 = -l1, m1 <= l1, m1++,
   g = Grad[f[l1, m1, Rr, Ttheta, Pphi], {Rr, Ttheta, Pphi},
     "Spherical"];

   G = g[[1]]*{Cos[Pphi] Sin[Ttheta], Sin[Pphi] Sin[Ttheta],
       Cos[Ttheta]} +
     g[[2]]*{Cos[Pphi] Cos[Ttheta],
       Cos[Ttheta] Sin[Pphi], -Sin[Ttheta]} +
     g[[3]]*{-Sin[Pphi], Cos[Pphi], 0};

   For[l2 = 0, l2 <= lmax, l2++,
    For[m2 = -l2, m2 <= l2, m2++,
     a = Integrate[
       G*f[l2, m2, Rr, Ttheta, Pphi]*Rr*Rr*Sin[Ttheta], {Pphi, -Pi,
        Pi}];
     b = Integrate[a, {Ttheta, 0, Pi}];
     result[[l1*l1 + m1 + l1 + 1]][[l2*l2 + m2 + l2 + 1]] =
      Integrate[b, {Rr, 0.01, 2}]
     ]
    ]
   ]
  ]
 ]
*/
int
test5()
{
    int const lmax{3};
    int const lmmax = (lmax + 1) * (lmax + 1);

    double ref_val[lmmax][lmmax][3] = {{{0, 0, 0},
                                        {0, 0.12265731552086959, 0},
                                        {0, 0, -0.12265731552086959},
                                        {0.12265731552086959, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0}},
                                       {{0, -0.20725306194522689, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {-0.1801736478370834, 0, 0},
                                        {0, 0, -0.1801736478370834},
                                        {0, -0.10402330407961692, 0},
                                        {0, 0, 0},
                                        {0, -0.1801736478370834, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0}},
                                       {{0, 0, 0.20725306194522689},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0.1801736478370834, 0},
                                        {0, 0, -0.20804660815923384},
                                        {0.1801736478370834, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0}},
                                       {{-0.20725306194522689, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, -0.1801736478370834, 0},
                                        {0, 0, 0},
                                        {-0.10402330407961692, 0, 0},
                                        {0, 0, -0.1801736478370834},
                                        {0.1801736478370834, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0}},
                                       {{0, 0, 0},
                                        {0.44228573487515693, 0, 0},
                                        {0, 0, 0},
                                        {0, 0.44228573487515693, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {-0.4660083350514234, 0, 0},
                                        {0, 0, -0.38049421225330937},
                                        {0.1203228347232006, 0, 0},
                                        {0, 0, 0},
                                        {0, 0.1203228347232006, 0},
                                        {0, 0, 0},
                                        {0, 0.4660083350514234, 0}},
                                       {{0, 0, 0},
                                        {0, 0, 0.44228573487515693},
                                        {0, -0.44228573487515693, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {-0.38049421225330937, 0, 0},
                                        {0, 0, -0.4812913388928024},
                                        {0, -0.2947295494770755, 0},
                                        {0, 0, 0},
                                        {0, -0.38049421225330937, 0},
                                        {0, 0, 0}},
                                       {{0, 0, 0},
                                        {0, 0.2553537880889033, 0},
                                        {0, 0, 0.5107075761778065},
                                        {0.2553537880889033, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0.4168105261025923, 0},
                                        {0, 0, -0.5104865541861799},
                                        {0.4168105261025923, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0}},
                                       {{0, 0, 0},
                                        {0, 0, 0},
                                        {-0.44228573487515693, 0, 0},
                                        {0, 0, 0.44228573487515693},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, -0.38049421225330937, 0},
                                        {0, 0, 0},
                                        {-0.2947295494770755, 0, 0},
                                        {0, 0, -0.4812913388928024},
                                        {0.38049421225330937, 0, 0},
                                        {0, 0, 0}},
                                       {{0, 0, 0},
                                        {0, 0.44228573487515693, 0},
                                        {0, 0, 0},
                                        {-0.44228573487515693, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0.4660083350514234, 0},
                                        {0, 0, 0},
                                        {0, 0.1203228347232006, 0},
                                        {0, 0, 0},
                                        {-0.1203228347232006, 0, 0},
                                        {0, 0, -0.38049421225330937},
                                        {0.4660083350514234, 0, 0}},
                                       {{0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {1.5512554789468367, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, -1.5512554789468367, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0}},
                                       {{0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 1.2665947947054943},
                                        {1.2665947947054943, 0, 0},
                                        {0, 0, 0},
                                        {0, 1.2665947947054943, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0}},
                                       {{0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {-0.4005324423782739, 0, 0},
                                        {0, 0, 1.6021297695130956},
                                        {0, -1.3874850805576484, 0},
                                        {0, 0, 0},
                                        {0, -0.4005324423782739, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0}},
                                       {{0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0.9811001092574764, 0},
                                        {0, 0, 1.6993152365453255},
                                        {0.9811001092574764, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0}},
                                       {{0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, -0.4005324423782739, 0},
                                        {0, 0, 0},
                                        {-1.3874850805576484, 0, 0},
                                        {0, 0, 1.6021297695130956},
                                        {0.4005324423782739, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0}},
                                       {{0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 1.2665947947054943, 0},
                                        {0, 0, 0},
                                        {-1.2665947947054943, 0, 0},
                                        {0, 0, 1.2665947947054943},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0}},
                                       {{0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, -1.5512554789468367, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {-1.5512554789468367, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0},
                                        {0, 0, 0}}};

    double val[lmmax][lmmax][3];

    int nr = 2000;

    auto r = Radial_grid_factory<double>(radial_grid_t::exponential, nr, 0.01, 2.0, 1.0);

    Spheric_function<function_domain_t::spectral, double> f(64, r);

    for (int l1 = 0; l1 <= lmax; l1++) {
        for (int m1 = -l1; m1 <= l1; m1++) {
            f.zero();
            for (int ir = 0; ir < nr; ir++) {
                f(sf::lm(l1, m1), ir) = std::exp(-r[ir]) * std::pow(r[ir], l1);
            }
            auto grad_f = gradient(f);

            for (int l2 = 0; l2 <= lmax; l2++) {
                for (int m2 = -l2; m2 <= l2; m2++) {
                    f.zero();
                    for (int ir = 0; ir < nr; ir++) {
                        f(sf::lm(l2, m2), ir) = std::exp(-r[ir]) * std::pow(r[ir], l2);
                    }

                    for (int x = 0; x < 3; x++) {
                        val[sf::lm(l1, m1)][sf::lm(l2, m2)][x] = inner(f, grad_f[x]);
                    }
                }
            }
        }
    }
    double d{0};
    for (int lm1 = 0; lm1 < lmmax; lm1++) {
        for (int lm2 = 0; lm2 < lmmax; lm2++) {
            for (int x : {0, 1, 2}) {
                d += std::abs(val[lm1][lm2][x] - ref_val[lm1][lm2][x]);
            }
        }
    }
    if (d < 1e-9) {
        return 0;
    } else {
        return 1;
    }
}

int
test6()
{
    auto rgrid = Radial_grid_factory<double>(radial_grid_t::exponential, 2000, 1e-7, 2.0, 1.0);
    Spheric_function<function_domain_t::spectral, double> rho_up_lm(64, rgrid);
    Spheric_function<function_domain_t::spectral, double> rho_dn_lm(64, rgrid);

    for (int ir = 0; ir < rgrid.num_points(); ir++) {
        for (int lm = 0; lm < 64; lm++) {
            rho_up_lm(lm, ir) = random<double>();
            rho_dn_lm(lm, ir) = random<double>();
        }
    }

    SHT sht(device_t::CPU, 8);

    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_up_tp(sht.num_points(), rgrid);
    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_dn_tp(sht.num_points(), rgrid);

    /* compute gradient in Rlm spherical harmonics */
    auto grad_rho_up_lm = gradient(rho_up_lm);
    auto grad_rho_dn_lm = gradient(rho_dn_lm);

    /* backward transform gradient from Rlm to (theta, phi) */
    for (int x = 0; x < 3; x++) {
        grad_rho_up_tp[x] = transform(sht, grad_rho_up_lm[x]);
        grad_rho_dn_tp[x] = transform(sht, grad_rho_dn_lm[x]);
    }

    Spheric_function<function_domain_t::spatial, double> grad_rho_up_grad_rho_up_tp;
    Spheric_function<function_domain_t::spatial, double> grad_rho_dn_grad_rho_dn_tp;
    Spheric_function<function_domain_t::spatial, double> grad_rho_up_grad_rho_dn_tp;

    /* compute density gradient products */
    grad_rho_up_grad_rho_up_tp = grad_rho_up_tp * grad_rho_up_tp;
    grad_rho_up_grad_rho_dn_tp = grad_rho_up_tp * grad_rho_dn_tp;
    grad_rho_dn_grad_rho_dn_tp = grad_rho_dn_tp * grad_rho_dn_tp;

    return 0;
}

int
main(int argn, char** argv)
{
    int err{0};

    err += call_test("Rlm -> Ylm -> Rlm conversion", []() { return test1<double>(); });
    err += call_test("Rlm -> (t,p) -> Rlm transformation", []() { return test2<double>(10, 1000); });
    err += call_test("Ylm -> (t,p) -> Ylm transformation", []() { return test2<std::complex<double>>(10, 1000); });
    err += call_test("Rlm -> (t,p) -> Ylm -> Rlm transformation", []() { return test3(10, 1000); });
    err += call_test("Gradient", test4);
    err += call_test("Many gradients", test5);
    err += call_test("MT rho", test6);

    return std::min(err, 1);
}
