// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file sht.cpp
 *   
 *  \brief Contains remaining implementation and full template specializations of sirius::SHT class.
 */

#include "sht.h"

namespace sirius
{

template <>
void SHT::backward_transform<double>(int ld, double* flm, int nr, int lmmax, double* ftp)
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    linalg<CPU>::gemm(1, 0, num_points_, nr, lmmax, &rlm_backward_(0, 0), lmmax_, flm, ld, ftp, num_points_);
}

template <>
void SHT::backward_transform<double_complex>(int ld, double_complex* flm, int nr, int lmmax, double_complex* ftp)
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    linalg<CPU>::gemm(1, 0, num_points_, nr, lmmax, &ylm_backward_(0, 0), lmmax_, flm, ld, ftp, num_points_);
}

template <>
void SHT::forward_transform<double>(double* ftp, int nr, int lmmax, int ld, double* flm)
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    linalg<CPU>::gemm(1, 0, lmmax, nr, num_points_, &rlm_forward_(0, 0), num_points_, ftp, num_points_, flm, ld);
}

template <>
void SHT::forward_transform<double_complex>(double_complex* ftp, int nr, int lmmax, int ld, double_complex* flm)
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    linalg<CPU>::gemm(1, 0, lmmax, nr, num_points_, &ylm_forward_(0, 0), num_points_, ftp, num_points_, flm, ld);
}

/** Specialization for real Gaunt coefficients between three complex spherical harmonics
 *  \f[ 
 *      \langle Y_{\ell_1 m_1} | Y_{\ell_2 m_2} | Y_{\ell_3 m_3} \rangle
 *  \f]
 */
template<> 
double SHT::gaunt<double>(int l1, int l2, int l3, int m1, int m2, int m3)
{
    assert(l1 >= 0);
    assert(l2 >= 0);
    assert(l3 >= 0);
    assert(m1 >= -l1 && m1 <= l1);
    assert(m2 >= -l2 && m2 <= l2);
    assert(m3 >= -l3 && m3 <= l3);
    
    return pow(-1.0, abs(m1)) * sqrt(double(2 * l1 + 1) * double(2 * l2 + 1) * double(2 * l3 + 1) / fourpi) * 
           gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, 0, 0, 0) *
           gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, -2 * m1, 2 * m2, 2 * m3);
}

/** Specialization for complex Gaunt coefficients between two complex and one real spherical harmonics
 *  \f[ 
 *      \langle Y_{\ell_1 m_1} | R_{\ell_2 m_2} | Y_{\ell_3 m_3} \rangle
 *  \f]
 */
template<> double_complex SHT::gaunt<double_complex>(int l1, int l2, int l3, int m1, int m2, int m3)
{
    assert(l1 >= 0);
    assert(l2 >= 0);
    assert(l3 >= 0);
    assert(m1 >= -l1 && m1 <= l1);
    assert(m2 >= -l2 && m2 <= l2);
    assert(m3 >= -l3 && m3 <= l3);

    if (m2 == 0) 
    {
        return double_complex(gaunt<double>(l1, l2, l3, m1, m2, m3), 0.0);
    }
    else 
    {
        return (ylm_dot_rlm(l2, m2, m2) * gaunt<double>(l1, l2, l3, m1, m2, m3) +  
                ylm_dot_rlm(l2, -m2, m2) * gaunt<double>(l1, l2, l3, m1, -m2, m3));
    }
}


SHT::SHT(int lmax__) : lmax_(lmax__), mesh_type_(0)
{
    lmmax_ = (lmax_ + 1) * (lmax_ + 1);
    
    if (mesh_type_ == 0) num_points_ = Lebedev_Laikov_npoint(2 * lmax_);
    if (mesh_type_ == 1) num_points_ = lmmax_;
    
    std::vector<double> x(num_points_);
    std::vector<double> y(num_points_);
    std::vector<double> z(num_points_);

    coord_ = mdarray<double, 2>(3, num_points_);

    tp_ = mdarray<double, 2>(2, num_points_);

    w_.resize(num_points_);

    if (mesh_type_ == 0) Lebedev_Laikov_sphere(num_points_, &x[0], &y[0], &z[0], &w_[0]);
    if (mesh_type_ == 1) uniform_coverage();

    ylm_backward_ = mdarray<double_complex, 2>(lmmax_, num_points_);

    ylm_forward_ = mdarray<double_complex, 2>(num_points_, lmmax_);

    rlm_backward_ = mdarray<double, 2>(lmmax_, num_points_);

    rlm_forward_ = mdarray<double, 2>(num_points_, lmmax_);

    for (int itp = 0; itp < num_points_; itp++)
    {
        if (mesh_type_ == 0)
        {
            coord_(0, itp) = x[itp];
            coord_(1, itp) = y[itp];
            coord_(2, itp) = z[itp];
            
            auto vs = spherical_coordinates(vector3d<double>(x[itp], y[itp], z[itp]));
            spherical_harmonics(lmax_, vs[1], vs[2], &ylm_backward_(0, itp));
            spherical_harmonics(lmax_, vs[1], vs[2], &rlm_backward_(0, itp));
            for (int lm = 0; lm < lmmax_; lm++)
            {
                ylm_forward_(itp, lm) = conj(ylm_backward_(lm, itp)) * w_[itp] * fourpi;
                rlm_forward_(itp, lm) = rlm_backward_(lm, itp) * w_[itp] * fourpi;
            }
        }
        if (mesh_type_ == 1)
        {
            double t = tp_(0, itp);
            double p = tp_(1, itp);

            coord_(0, itp) = sin(t) * cos(p);
            coord_(1, itp) = sin(t) * sin(p);
            coord_(2, itp) = cos(t);

            spherical_harmonics(lmax_, t, p, &ylm_backward_(0, itp));
            spherical_harmonics(lmax_, t, p, &rlm_backward_(0, itp));

            for (int lm = 0; lm < lmmax_; lm++)
            {
                ylm_forward_(lm, itp) = ylm_backward_(lm, itp);
                rlm_forward_(lm, itp) = rlm_backward_(lm, itp);
            }
        }
    }

    if (mesh_type_ == 1)
    {
        lin_alg<lapack>::invert_ge(&ylm_forward_(0, 0), lmmax_);
        lin_alg<lapack>::invert_ge(&rlm_forward_(0, 0), lmmax_);
    }
    
    if (debug_level > 0)
    {
        double dr = 0;
        double dy = 0;

        for (int lm = 0; lm < lmmax_; lm++)
        {
            for (int lm1 = 0; lm1 < lmmax_; lm1++)
            {
                double t = 0;
                double_complex zt(0, 0);
                for (int itp = 0; itp < num_points_; itp++)
                {
                    zt += ylm_forward_(itp, lm) * ylm_backward_(lm1, itp);
                    t += rlm_forward_(itp, lm) * rlm_backward_(lm1, itp);
                }
                
                if (lm == lm1) 
                {
                    zt -= 1.0;
                    t -= 1.0;
                }
                dr += fabs(t);
                dy += abs(zt);
            }
        }
        dr = dr / lmmax_ / lmmax_;
        dy = dy / lmmax_ / lmmax_;

        if (dr > 1e-15 || dy > 1e-15)
        {
            std::stringstream s;
            s << "spherical mesh error is too big" << std::endl
              << "  real spherical integration error " << dr << std::endl
              << "  complex spherical integration error " << dy;
            warning_local(__FILE__, __LINE__, s);
        }

        std::vector<double> flm(lmmax_);
        std::vector<double> ftp(num_points_);
        for (int lm = 0; lm < lmmax_; lm++)
        {
            memset(&flm[0], 0, lmmax_ * sizeof(double));
            flm[lm] = 1.0;
            backward_transform(lmmax_, &flm[0], 1, lmmax_, &ftp[0]);
            forward_transform(&ftp[0], 1, lmmax_, lmmax_, &flm[0]);
            flm[lm] -= 1.0;

            double t = 0.0;
            for (int lm1 = 0; lm1 < lmmax_; lm1++) t += fabs(flm[lm1]);

            t /= lmmax_;

            if (t > 1e-15) 
            {
                std::stringstream s;
                s << "test of backward / forward real SHT failed" << std::endl
                  << "  total error " << t;
                warning_local(__FILE__, __LINE__, s);
            }
        }
    }
}

vector3d<double> SHT::spherical_coordinates(vector3d<double> vc)
{
    vector3d<double> vs;

    double eps = 1e-12;

    vs[0] = vc.length();

    if (vs[0] <= eps)
    {
        vs[1] = 0.0;
        vs[2] = 0.0;
    } 
    else
    {
        vs[1] = acos(vc[2] / vs[0]); // theta = cos^{-1}(z/r)

        if (fabs(vc[0]) > eps || fabs(vc[1]) > eps)
        {
            vs[2] = atan2(vc[1], vc[0]); // phi = tan^{-1}(y/x)
            if (vs[2] < 0.0) vs[2] += twopi;
        }
        else
        {
            vs[2] = 0.0;
        }
    }

    return vs;
}

void SHT::spherical_harmonics(int lmax, double theta, double phi, double_complex* ylm)
{
    double x = cos(theta);
    std::vector<double> result_array(lmax + 1);

    for (int m = 0; m <= lmax; m++)
    {
        double_complex z = exp(double_complex(0.0, m * phi)); 
        
        gsl_sf_legendre_sphPlm_array(lmax, m, x, &result_array[0]);

        for (int l = m; l <= lmax; l++)
        {
            ylm[Utils::lm_by_l_m(l, m)] = result_array[l - m] * z;
            if (m % 2) 
            {
                ylm[Utils::lm_by_l_m(l, -m)] = -conj(ylm[Utils::lm_by_l_m(l, m)]);
            }
            else
            {
                ylm[Utils::lm_by_l_m(l, -m)] = conj(ylm[Utils::lm_by_l_m(l, m)]);        
            }
        }
    }
}

void SHT::spherical_harmonics(int lmax, double theta, double phi, double* rlm)
{
    int lmmax = (lmax + 1) * (lmax + 1);
    std::vector<double_complex> ylm(lmmax);
    spherical_harmonics(lmax, theta, phi, &ylm[0]);
    
    double t = sqrt(2.0);
    
    rlm[0] = y00;

    for (int l = 1; l <= lmax; l++)
    {
        for (int m = -l; m < 0; m++) 
            rlm[Utils::lm_by_l_m(l, m)] = t * imag(ylm[Utils::lm_by_l_m(l, m)]);
        
        rlm[Utils::lm_by_l_m(l, 0)] = real(ylm[Utils::lm_by_l_m(l, 0)]);
         
        for (int m = 1; m <= l; m++) 
            rlm[Utils::lm_by_l_m(l, m)] = t * real(ylm[Utils::lm_by_l_m(l, m)]);
    }
}
                
double_complex SHT::ylm_dot_rlm(int l, int m1, int m2)
{
    const double isqrt2 = 0.70710678118654752440;

    assert(l >= 0 && abs(m1) <= l && abs(m2) <= l);

    if (!((m1 == m2) || (m1 == -m2))) return double_complex(0, 0);

    if (m1 == 0) return double_complex(1, 0);

    if (m1 < 0)
    {
        if (m2 < 0) return -double_complex(0, isqrt2);
        else return pow(-1.0, m2) * double_complex(isqrt2, 0);
    }
    else
    {
        if (m2 < 0) return pow(-1.0, m1) * double_complex(0, isqrt2);
        else return double_complex(isqrt2, 0);
    }
}

void SHT::uniform_coverage()
{
    tp_(0, 0) = pi;
    tp_(1, 0) = 0;

    for (int k = 1; k < num_points_ - 1; k++)
    {
        double hk = -1.0 + double(2 * k) / double(num_points_ - 1);
        tp_(0, k) = acos(hk);
        double t = tp_(1, k - 1) + 3.80925122745582 / sqrt(double(num_points_)) / sqrt(1 - hk * hk);
        tp_(1, k) = fmod(t, twopi);
    }
    
    tp_(0, num_points_ - 1) = 0;
    tp_(1, num_points_ - 1) = 0;
}

Spheric_function<spectral, double> SHT::convert(Spheric_function<spectral, double_complex>& f)
{
    Spheric_function<spectral, double> g;

    /* radial index is first */
    if (f.radial_domain_idx() == 0)
    {
        g = Spheric_function<spectral, double>(f.radial_grid(), f.angular_domain_size());
    }
    else
    {
        g = Spheric_function<spectral, double>(f.angular_domain_size(), f.radial_grid());
    }

    convert(f, g);

    return g;
}

Spheric_function<spectral, double_complex> SHT::convert(Spheric_function<spectral, double>& f)
{
    Spheric_function<spectral, double_complex> g;
    
    int lmax = Utils::lmax_by_lmmax(f.angular_domain_size());

    /* cache transformation arrays */
    std::vector<double_complex> tpp(f.angular_domain_size());
    std::vector<double_complex> tpm(f.angular_domain_size());
    for (int l = 0; l <= lmax; l++)
    {
        for (int m = -l; m <= l; m++) 
        {
            int lm = Utils::lm_by_l_m(l, m);
            tpp[lm] = ylm_dot_rlm(l, m, m);
            tpm[lm] = ylm_dot_rlm(l, m, -m);
        }
    }

    /* radial index is first */
    if (f.radial_domain_idx() == 0)
    {
        g = Spheric_function<spectral, double_complex>(f.radial_grid(), f.angular_domain_size());

        int lm = 0;
        for (int l = 0; l <= lmax; l++)
        {
            for (int m = -l; m <= l; m++)
            {
                if (m == 0)
                {
                    for (int ir = 0; ir < f.radial_grid().num_points(); ir++) g(ir, lm) = f(ir, lm);
                }
                else 
                {
                    int lm1 = Utils::lm_by_l_m(l, -m);
                    for (int ir = 0; ir < f.radial_grid().num_points(); ir++)
                        g(ir, lm) = tpp[lm] * f(ir, lm) + tpm[lm] * f(ir, lm1);
                }
                lm++;
            }
        }
    }
    else
    {
        g = Spheric_function<spectral, double_complex>(f.angular_domain_size(), f.radial_grid());
        for (int ir = 0; ir < f.radial_grid().num_points(); ir++)
        {
            int lm = 0;
            for (int l = 0; l <= lmax; l++)
            {
                for (int m = -l; m <= l; m++)
                {
                    if (m == 0)
                    {
                        g(lm, ir) = f(lm, ir);
                    }
                    else 
                    {
                        int lm1 = Utils::lm_by_l_m(l, -m);
                        g(lm, ir) = tpp[lm] * f(lm, ir) + tpm[lm] * f(lm1, ir);
                    }
                    lm++;
                }
            }
        }
    }

    return g;
}

void SHT::convert(Spheric_function<spectral, double_complex>& f, Spheric_function<spectral, double>& g)
{
    if (f.radial_domain_idx() != g.radial_domain_idx())
        error_local(__FILE__, __LINE__, "wrong radial domain index");
    
    if (f.radial_grid().hash() != g.radial_grid().hash())
        error_local(__FILE__, __LINE__, "radial grids don't match");

    int lmmax = std::min(f.angular_domain_size(), g.angular_domain_size());
    int lmax = Utils::lmax_by_lmmax(lmmax);

    /* cache transformation arrays */
    std::vector<double_complex> tpp(lmmax);
    std::vector<double_complex> tpm(lmmax);
    for (int l = 0; l <= lmax; l++)
    {
        for (int m = -l; m <= l; m++) 
        {
            int lm = Utils::lm_by_l_m(l, m);
            tpp[lm] = rlm_dot_ylm(l, m, m);
            tpm[lm] = rlm_dot_ylm(l, m, -m);
        }
    }

    /* radial index is first */
    if (f.radial_domain_idx() == 0)
    {
        int lm = 0;
        for (int l = 0; l <= lmax; l++)
        {
            for (int m = -l; m <= l; m++)
            {
                if (m == 0)
                {
                    for (int ir = 0; ir < f.radial_grid().num_points(); ir++) 
                        g(ir, lm) = real(f(ir, lm));
                }
                else 
                {
                    int lm1 = Utils::lm_by_l_m(l, -m);
                    for (int ir = 0; ir < f.radial_grid().num_points(); ir++)
                        g(ir, lm) = real(tpp[lm] * f(ir, lm) + tpm[lm] * f(ir, lm1));
                }
                lm++;
            }
        }
    }
    else
    {
        for (int ir = 0; ir < f.radial_grid().num_points(); ir++)
        {
            int lm = 0;
            for (int l = 0; l <= lmax; l++)
            {
                for (int m = -l; m <= l; m++)
                {
                    if (m == 0)
                    {
                        g(lm, ir) = real(f(lm, ir));
                    }
                    else 
                    {
                        int lm1 = Utils::lm_by_l_m(l, -m);
                        g(lm, ir) = real(tpp[lm] * f(lm, ir) + tpm[lm] * f(lm1, ir));
                    }
                    lm++;
                }
            }
        }
    }
}

}
