// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file spheric_function.h
 *   
 *  \brief Contains declaration and implementation of sirius::Spheric_function and 
 *         sirius::Spheric_function_gradient classes.
 */

#ifndef __SPHERIC_FUNCTION_H__
#define __SPHERIC_FUNCTION_H__

#include <array>
#include <typeinfo>
#include "radial_grid.h"
#include "spline.h"
#include "sht.h"

namespace sirius {

/// Function in spherical harmonics or spherical coordinates representation.
template <function_domain_t domain_t, typename T = double_complex>
class Spheric_function: public mdarray<T, 2>
{
    private:

        /// Radial grid.
        Radial_grid<double> const* radial_grid_{nullptr};
        
        int angular_domain_size_;

        Spheric_function(Spheric_function<domain_t, T> const& src__) = delete;

        Spheric_function<domain_t, T>& operator=(Spheric_function<domain_t, T> const& src__) = delete;

    public:

        Spheric_function()
        {
        }

        Spheric_function(int angular_domain_size__, Radial_grid<double> const& radial_grid__) 
            : mdarray<T, 2>(angular_domain_size__, radial_grid__.num_points())
            , radial_grid_(&radial_grid__)
            , angular_domain_size_(angular_domain_size__)
        {
        }

        Spheric_function(T* ptr__, int angular_domain_size__, Radial_grid<double> const& radial_grid__) 
            : mdarray<T, 2>(ptr__, angular_domain_size__, radial_grid__.num_points())
            , radial_grid_(&radial_grid__)
            , angular_domain_size_(angular_domain_size__)
        {
        }

        Spheric_function(Spheric_function<domain_t, T>&& src__)
            : mdarray<T, 2>(std::move(src__))
        {
            radial_grid_         = src__.radial_grid_;
            angular_domain_size_ = src__.angular_domain_size_;
        }

        Spheric_function<domain_t, T>& operator=(Spheric_function<domain_t, T>&& src__)
        {
            if (this != &src__) {
                mdarray<T, 2>::operator=(std::move(src__));
                radial_grid_         = src__.radial_grid_;
                angular_domain_size_ = src__.angular_domain_size_;
            }
            return *this;
        }

        inline Spheric_function<domain_t, T>& operator+=(Spheric_function<domain_t, T> const& rhs__)
        {
            for (size_t i1 = 0; i1 < this->size(1); i1++) {
                for (size_t i0 = 0; i0 < this->size(0); i0++) {
                    (*this)(i0, i1) += rhs__(i0, i1);
                }
            }
            
            return *this;
        }

        inline Spheric_function<domain_t, T>& operator+=(Spheric_function<domain_t, T>&& rhs__)
        {
            for (size_t i1 = 0; i1 < this->size(1); i1++) {
                for (size_t i0 = 0; i0 < this->size(0); i0++) {
                    (*this)(i0, i1) += rhs__(i0, i1);
                }
            }

            return *this;
        }

        inline int angular_domain_size() const
        {
            return angular_domain_size_;
        }

        inline Radial_grid<double> const& radial_grid() const
        {
            return *radial_grid_;
        }

        Spline<T> component(int lm__) const
        {
            if (domain_t != spectral) {
                TERMINATE("function is not is spectral domain");
            }

            Spline<T> s(radial_grid());
            for (int ir = 0; ir < radial_grid_->num_points(); ir++) {
                s(ir) = (*this)(lm__, ir);
            }
            return std::move(s.interpolate());
        }

        T value(double theta__, double phi__, int jr__, double dr__) const
        {
            assert(domain_t == spectral);

            int lmax = Utils::lmax_by_lmmax(angular_domain_size_);
            std::vector<T> ylm(angular_domain_size_);
            SHT::spherical_harmonics(lmax, theta__, phi__, &ylm[0]);
            T p = 0.0;
            for (int lm = 0; lm < angular_domain_size_; lm++) {
                double deriv = ((*this)(lm, jr__ + 1) - (*this)(lm, jr__)) / radial_grid_->dx(jr__);
                p += ylm[lm] * ((*this)(lm, jr__) + deriv * dr__);
            }
            return p;
        }

};

/// Multiplication of two functions in spatial domain.
template <typename T>
Spheric_function<spatial, T> operator*(Spheric_function<spatial, T> const& a__, Spheric_function<spatial, T> const& b__)
{
    if (a__.radial_grid().hash() != b__.radial_grid().hash()) {
        TERMINATE("wrong radial grids");
    }
    if (a__.angular_domain_size() != b__.angular_domain_size()) {
        TERMINATE("wrong angular domain sizes");
    }

    Spheric_function<spatial, T> res(a__.angular_domain_size(), a__.radial_grid());

    T const* ptr_lhs = &a__(0, 0);
    T const* ptr_rhs = &b__(0, 0);
    T* ptr_res = &res(0, 0);

    for (int i = 0; i < a__.size(); i++) {
        ptr_res[i] = ptr_lhs[i] * ptr_rhs[i];
    }

    return std::move(res);
}

/// Summation of two functions.
template <function_domain_t domain_t, typename T>
Spheric_function<domain_t, T> operator+(Spheric_function<domain_t, T> const& a__, Spheric_function<domain_t, T> const& b__)
{
    //if (a__.radial_grid().hash() != b__.radial_grid().hash()) {
    //    TERMINATE("wrong radial grids");
    //}
    if (a__.angular_domain_size() != b__.angular_domain_size()) {
        TERMINATE("wrong angular domain sizes");
    }

    Spheric_function<domain_t, T> result(a__.angular_domain_size(), a__.radial_grid());

    for (int ir = 0; ir < a__.radial_grid().num_points(); ir++) {
        for (int i = 0; i < a__.angular_domain_size(); i++) {
            result(i, ir) = a__(i, ir) + b__(i, ir);
        }
    }

    return std::move(result);
}

/// Subtraction of functions.
template <function_domain_t domain_t, typename T>
Spheric_function<domain_t, T> operator-(Spheric_function<domain_t, T> const& a__, Spheric_function<domain_t, T> const& b__)
{
    Spheric_function<domain_t, T> res(a__.angular_domain_size(), a__.radial_grid());

    T const* ptr_lhs = &a__(0, 0);
    T const* ptr_rhs = &b__(0, 0);
    T* ptr_res = &res(0, 0);

    for (size_t i = 0; i < a__.size(); i++) {
        ptr_res[i] = ptr_lhs[i] - ptr_rhs[i];
    }

    return std::move(res);
}

/// Multiply function by a scalar.
template <function_domain_t domain_t, typename T>
Spheric_function<domain_t, T> operator*(T a__, Spheric_function<domain_t, T> const& b__)
{
    Spheric_function<domain_t, T> res(b__.angular_domain_size(), b__.radial_grid());

    T const* ptr_rhs = &b__(0, 0);
    T* ptr_res = &res(0, 0);

    for (size_t i = 0; i < b__.size(); i++) {
        ptr_res[i] = a__ * ptr_rhs[i];
    }

    return std::move(res);
}
/// Multiply function by a scalar (inverse order).
template <function_domain_t domain_t, typename T>
Spheric_function<domain_t, T> operator*(Spheric_function<domain_t, T> const& b__, T a__)
{
    return std::move(a__ * b__);
}

/// Inner product of two spherical functions.
template <function_domain_t domain_t, typename T>
T inner(Spheric_function<domain_t, T> const& f1, Spheric_function<domain_t, T> const& f2)
{
    /* check radial grid */
    //if (f1.radial_grid().hash() != f2.radial_grid().hash()) {
    //    TERMINATE("radial grids don't match");
    //}

    Spline<T> s(f1.radial_grid());

    if (domain_t == spectral) {
        int lmmax = std::min(f1.angular_domain_size(), f2.angular_domain_size());
        for (int ir = 0; ir < s.num_points(); ir++) {
            for (int lm = 0; lm < lmmax; lm++) {
                s(ir) += type_wrapper<T>::bypass(std::conj(f1(lm, ir))) * f2(lm, ir);
            }
        }
    } else {
        TERMINATE_NOT_IMPLEMENTED
    }
    return s.interpolate().integrate(2);
}

/// Compute Laplacian of the spheric function.
/** Laplacian in spherical coordinates has the following expression:
 *  \f[
 *      \Delta = \frac{1}{r^2}\frac{\partial}{\partial r}\Big( r^2 \frac{\partial}{\partial r} \Big) + \frac{1}{r^2}\Delta_{\theta, \phi}
 *  \f]
 */
template <typename T>
Spheric_function<spectral, T> laplacian(Spheric_function<spectral, T> const& f__)
{
    Spheric_function<spectral, T> g;
    auto& rgrid = f__.radial_grid();
    int lmmax = f__.angular_domain_size();
    int lmax = Utils::lmax_by_lmmax(lmmax);
    g = Spheric_function<spectral, T>(lmmax, rgrid);
    
    Spline<T> s1(rgrid);
    for (int l = 0; l <= lmax; l++) {
        int ll = l * (l + 1);
        for (int m = -l; m <= l; m++) {
            int lm = Utils::lm_by_l_m(l, m);
            /* get lm component */
            auto s = f__.component(lm);
            /* compute 1st derivative */
            for (int ir = 0; ir < s.num_points(); ir++) {
                s1(ir) = s.deriv(1, ir);
            }
            s1.interpolate();
            
            for (int ir = 0; ir < s.num_points(); ir++) {
                g(lm, ir) = 2 * s1(ir) * rgrid.x_inv(ir) + s1.deriv(1, ir) - s(ir) * ll / std::pow(rgrid[ir], 2);
            }
        }
    }

    return std::move(g);
}

/// Convert from Ylm to Rlm representation.
inline Spheric_function<spectral, double> convert(Spheric_function<spectral, double_complex> const& f__)
{
    int lmax = Utils::lmax_by_lmmax(f__.angular_domain_size());

    /* cache transformation arrays */
    std::vector<double_complex> tpp(f__.angular_domain_size());
    std::vector<double_complex> tpm(f__.angular_domain_size());
    for (int l = 0; l <= lmax; l++) {
        for (int m = -l; m <= l; m++) {
            int lm = Utils::lm_by_l_m(l, m);
            tpp[lm] = SHT::rlm_dot_ylm(l, m, m);
            tpm[lm] = SHT::rlm_dot_ylm(l, m, -m);
        }
    }

    Spheric_function<spectral, double> g(f__.angular_domain_size(), f__.radial_grid());

    for (int ir = 0; ir < f__.radial_grid().num_points(); ir++) {
        int lm = 0;
        for (int l = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++) {
                if (m == 0) {
                    g(lm, ir) = std::real(f__(lm, ir));
                } else {
                    int lm1 = Utils::lm_by_l_m(l, -m);
                    g(lm, ir) = std::real(tpp[lm] * f__(lm, ir) + tpm[lm] * f__(lm1, ir));
                }
                lm++;
            }
        }
    }

    return std::move(g);
}

/// Convert from Rlm to Ylm representation.
inline Spheric_function<spectral, double_complex> convert(Spheric_function<spectral, double> const& f__)
{
    int lmax = Utils::lmax_by_lmmax(f__.angular_domain_size());

    /* cache transformation arrays */
    std::vector<double_complex> tpp(f__.angular_domain_size());
    std::vector<double_complex> tpm(f__.angular_domain_size());
    for (int l = 0; l <= lmax; l++) {
        for (int m = -l; m <= l; m++) {
            int lm = Utils::lm_by_l_m(l, m);
            tpp[lm] = SHT::ylm_dot_rlm(l, m, m);
            tpm[lm] = SHT::ylm_dot_rlm(l, m, -m);
        }
    }

    Spheric_function<spectral, double_complex> g(f__.angular_domain_size(), f__.radial_grid());

    for (int ir = 0; ir < f__.radial_grid().num_points(); ir++) {
        int lm = 0;
        for (int l = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++) {
                if (m == 0) {
                    g(lm, ir) = f__(lm, ir);
                } else {
                    int lm1 = Utils::lm_by_l_m(l, -m);
                    g(lm, ir) = tpp[lm] * f__(lm, ir) + tpm[lm] * f__(lm1, ir);
                }
                lm++;
            }
        }
    }

    return std::move(g);
}

/// Transform to spatial domain (to r, \theta, \phi coordinates).
template <typename T>
Spheric_function<spatial, T> transform(SHT* sht__, Spheric_function<spectral, T> const& f__)
{
    Spheric_function<spatial, T> g(sht__->num_points(), f__.radial_grid());
    
    sht__->backward_transform(f__.angular_domain_size(), &f__(0, 0), f__.radial_grid().num_points(), 
                              std::min(sht__->lmmax(), f__.angular_domain_size()), &g(0, 0));

    return std::move(g);
}

/// Transform to spectral domain.
template <typename T>
Spheric_function<spectral, T> transform(SHT* sht__, Spheric_function<spatial, T> const& f__)
{
    Spheric_function<spectral, T> g(sht__->lmmax(), f__.radial_grid());
    
    sht__->forward_transform(&f__(0, 0), f__.radial_grid().num_points(), sht__->lmmax(), sht__->lmmax(), &g(0, 0));

    return std::move(g);
}

/// Gradient of a spheric function.
template <function_domain_t domain_t, typename T = double_complex>
class Spheric_function_gradient
{
    private:

        Radial_grid<double> const* radial_grid_{nullptr};

        int angular_domain_size_;

        std::array<Spheric_function<domain_t, T>, 3> grad_;
    
    public:

        Spheric_function_gradient(int angular_domain_size__, Radial_grid<double> const& radial_grid__) 
            : radial_grid_(&radial_grid__)
            , angular_domain_size_(angular_domain_size__)
        {
        }

        inline Radial_grid<double> const& radial_grid() const
        {
            return *radial_grid_;
        }

        inline int angular_domain_size() const
        {
            return angular_domain_size_;
        }

        inline Spheric_function<domain_t, T>& operator[](const int x)
        {
            assert(x >= 0 && x < 3);
            return grad_[x];
        }

        inline Spheric_function<domain_t, T> const& operator[](const int x) const
        {
            assert(x >= 0 && x < 3);
            return grad_[x];
        }
};

/// Gradient of the function in complex spherical harmonics.
inline Spheric_function_gradient<spectral, double_complex> gradient(Spheric_function<spectral, double_complex>& f)
{
    Spheric_function_gradient<spectral, double_complex> g(f.angular_domain_size(), f.radial_grid());
    for (int i = 0; i < 3; i++)
    {
        g[i] = Spheric_function<spectral, double_complex>(f.angular_domain_size(), f.radial_grid());
        g[i].zero();
    }
            
    int lmax = Utils::lmax_by_lmmax(f.angular_domain_size());

    //Spline<double_complex> s(f.radial_grid());

    for (int l = 0; l <= lmax; l++)
    {
        double d1 = sqrt(double(l + 1) / double(2 * l + 3));
        double d2 = sqrt(double(l) / double(2 * l - 1));

        for (int m = -l; m <= l; m++)
        {
            int lm = Utils::lm_by_l_m(l, m);
            auto s = f.component(lm);

            for (int mu = -1; mu <= 1; mu++)
            {
                int j = (mu + 2) % 3; // map -1,0,1 to 1,2,0

                if ((l + 1) <= lmax && abs(m + mu) <= l + 1)
                {
                    int lm1 = Utils::lm_by_l_m(l + 1, m + mu); 
                    double d = d1 * SHT::clebsch_gordan(l, 1, l + 1, m, mu, m + mu);
                    for (int ir = 0; ir < f.radial_grid().num_points(); ir++)
                        g[j](lm1, ir) += (s.deriv(1, ir) - f(lm, ir) * f.radial_grid().x_inv(ir) * double(l)) * d;  
                }
                if ((l - 1) >= 0 && abs(m + mu) <= l - 1)
                {
                    int lm1 = Utils::lm_by_l_m(l - 1, m + mu); 
                    double d = d2 * SHT::clebsch_gordan(l, 1, l - 1, m, mu, m + mu); 
                    for (int ir = 0; ir < f.radial_grid().num_points(); ir++)
                        g[j](lm1, ir) -= (s.deriv(1, ir) + f(lm, ir) * f.radial_grid().x_inv(ir) * double(l + 1)) * d;
                }
            }
        }
    }

    double_complex d1(1.0 / sqrt(2.0), 0);
    double_complex d2(0, 1.0 / sqrt(2.0));

    for (int ir = 0; ir < f.radial_grid().num_points(); ir++)
    {
        for (int lm = 0; lm < f.angular_domain_size(); lm++)
        {
            double_complex g_p = g[0](lm, ir);
            double_complex g_m = g[1](lm, ir);
            g[0](lm, ir) = d1 * (g_m - g_p);
            g[1](lm, ir) = d2 * (g_m + g_p);
        }
    }

    return g;
}

/// Gradient of the function in real spherical harmonics.
inline Spheric_function_gradient<spectral, double> gradient(Spheric_function<spectral, double> const& f)
{
    int lmax = Utils::lmax_by_lmmax(f.angular_domain_size());
    SHT sht(lmax);
    auto zf = convert(f);
    auto zg = gradient(zf);
    Spheric_function_gradient<spectral, double> g(f.angular_domain_size(), f.radial_grid());
    for (int x: {0, 1, 2}) {
        g[x] = convert(zg[x]);
    }
    return g;
}

/// Dot product of two gradiensts.
inline Spheric_function<spatial, double> operator*(Spheric_function_gradient<spatial, double> const& f, 
                                                   Spheric_function_gradient<spatial, double> const& g)
{
    for (int x: {0, 1, 2}) {
        //if (f[x].radial_grid().hash() != g[x].radial_grid().hash()) {
        //    TERMINATE("wrong radial grids");
        //}
        
        if (f[x].angular_domain_size() != g[x].angular_domain_size()) {
            TERMINATE("wrong number of angular points");
        }
    }

    Spheric_function<spatial, double> result(f.angular_domain_size(), f.radial_grid());
    result.zero();

    for (int x: {0, 1, 2}) {
        for (int ir = 0; ir < f[x].radial_grid().num_points(); ir++) {
            for (int tp = 0; tp < f[x].angular_domain_size(); tp++) {
                result(tp, ir) += f[x](tp, ir) * g[x](tp, ir);
            }
        }
    }

    return result;
}

}

#endif // __SPHERIC_FUNCTION_H__
