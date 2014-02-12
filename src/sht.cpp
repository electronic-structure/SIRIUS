#include "sht.h"

namespace sirius
{

SHT::SHT(int lmax__) : lmax_(lmax__), mesh_type_(0)
{
    lmmax_ = (lmax_ + 1) * (lmax_ + 1);
    
    if (mesh_type_ == 0) num_points_ = Lebedev_Laikov_npoint(2 * lmax_);
    if (mesh_type_ == 1) num_points_ = lmmax_;
    
    std::vector<double> x(num_points_);
    std::vector<double> y(num_points_);
    std::vector<double> z(num_points_);

    coord_.set_dimensions(3, num_points_);
    coord_.allocate();

    tp_.set_dimensions(2, num_points_);
    tp_.allocate();

    w_.resize(num_points_);

    if (mesh_type_ == 0) Lebedev_Laikov_sphere(num_points_, &x[0], &y[0], &z[0], &w_[0]);
    if (mesh_type_ == 1) uniform_coverage();

    ylm_backward_.set_dimensions(lmmax_, num_points_);
    ylm_backward_.allocate();

    ylm_forward_.set_dimensions(num_points_, lmmax_);
    ylm_forward_.allocate();

    rlm_backward_.set_dimensions(lmmax_, num_points_);
    rlm_backward_.allocate();

    rlm_forward_.set_dimensions(num_points_, lmmax_);
    rlm_forward_.allocate();

    for (int itp = 0; itp < num_points_; itp++)
    {
        if (mesh_type_ == 0)
        {
            coord_(0, itp) = x[itp];
            coord_(1, itp) = y[itp];
            coord_(2, itp) = z[itp];
            
            double vs[3];

            spherical_coordinates(vector3d<double>(x[itp], y[itp], z[itp]), vs);
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
        linalg<lapack>::invert_ge(&ylm_forward_(0, 0), lmmax_);
        linalg<lapack>::invert_ge(&rlm_forward_(0, 0), lmmax_);
    }

    double dr = 0;
    double dy = 0;

    for (int lm = 0; lm < lmmax_; lm++)
    {
        for (int lm1 = 0; lm1 < lmmax_; lm1++)
        {
            double t = 0;
            complex16 zt(0, 0);
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
        backward_transform(&flm[0], lmmax_, 1, &ftp[0]);
        forward_transform(&ftp[0], lmmax_, 1, &flm[0]);
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

void SHT::spherical_coordinates(vector3d<double> vc, double* vs)
{
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
}

void SHT::spherical_harmonics(int lmax, double theta, double phi, complex16* ylm)
{
    double x = cos(theta);
    std::vector<double> result_array(lmax + 1);

    for (int m = 0; m <= lmax; m++)
    {
        complex16 z = exp(complex16(0.0, m * phi)); 
        
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
    std::vector<complex16> ylm(lmmax);
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
                
complex16 SHT::ylm_dot_rlm(int l, int m1, int m2)
{
    const double isqrt2 = 0.70710678118654752440;

    assert(l >= 0 && abs(m1) <= l && abs(m2) <= l);

    if (!((m1 == m2) || (m1 == -m2))) return complex16(0, 0);

    if (m1 == 0) return complex16(1, 0);

    if (m1 < 0)
    {
        if (m2 < 0) return -complex16(0, isqrt2);
        else return pow(-1.0, m2) * complex16(isqrt2, 0);
    }
    else
    {
        if (m2 < 0) return pow(-1.0, m1) * complex16(0, isqrt2);
        else return complex16(isqrt2, 0);
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

}
