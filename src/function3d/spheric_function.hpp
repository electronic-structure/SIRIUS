/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file spheric_function.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Spheric_function and
 *         sirius::Spheric_function_gradient classes.
 */

#ifndef __SPHERIC_FUNCTION_HPP__
#define __SPHERIC_FUNCTION_HPP__

#include <array>
#include <typeinfo>
#include "radial/spline.hpp"
#include "core/sht/sht.hpp"
#include "core/math_tools.hpp"

namespace sirius {

/// Function in spherical harmonics or spherical coordinates representation.
/** This class works in conjugation with SHT class which provides the transformation between spherical
    harmonics and spherical coordinates and also a conversion between real and complex spherical harmonics.
 */
template <function_domain_t domain_t, typename T = std::complex<double>>
class Spheric_function : public mdarray<T, 2>
{
  private:
    /// Radial grid.
    Radial_grid<double> const* radial_grid_{nullptr};

    /// Size of the angular domain.
    /** This is either a total number of spherical harmonics or a number of (theta, phi) anles covering the sphere. */
    int angular_domain_size_;

    /* copy constructir is disabled */
    Spheric_function(Spheric_function<domain_t, T> const& src__) = delete;

    /* copy assignment operator is disabled */
    Spheric_function<domain_t, T>&
    operator=(Spheric_function<domain_t, T> const& src__) = delete;

  public:
    /// Constructor of the empty function.
    Spheric_function()
    {
    }

    /// Constructor.
    Spheric_function(int angular_domain_size__, Radial_grid<double> const& radial_grid__)
        : mdarray<T, 2>({angular_domain_size__, radial_grid__.num_points()})
        , radial_grid_(&radial_grid__)
        , angular_domain_size_(angular_domain_size__)
    {
    }

    /// Constructor.
    Spheric_function(T* ptr__, int angular_domain_size__, Radial_grid<double> const& radial_grid__)
        : mdarray<T, 2>({angular_domain_size__, radial_grid__.num_points()}, ptr__)
        , radial_grid_(&radial_grid__)
        , angular_domain_size_(angular_domain_size__)
    {
    }

    /// Constructor.
    Spheric_function(memory_pool& mp__, int angular_domain_size__, Radial_grid<double> const& radial_grid__)
        : mdarray<T, 2>({angular_domain_size__, radial_grid__.num_points()}, mp__)
        , radial_grid_(&radial_grid__)
        , angular_domain_size_(angular_domain_size__)
    {
    }

    /// Move constructor.
    Spheric_function(Spheric_function<domain_t, T>&& src__)
        : mdarray<T, 2>(std::move(src__))
    {
        radial_grid_         = src__.radial_grid_;
        angular_domain_size_ = src__.angular_domain_size_;
    }

    /// Move asigment operator.
    Spheric_function<domain_t, T>&
    operator=(Spheric_function<domain_t, T>&& src__)
    {
        if (this != &src__) {
            mdarray<T, 2>::operator=(std::move(src__));
            radial_grid_         = src__.radial_grid_;
            angular_domain_size_ = src__.angular_domain_size_;
        }
        return *this;
    }

    inline Spheric_function<domain_t, T>&
    operator+=(Spheric_function<domain_t, T> const& rhs__)
    {
        for (int i1 = 0; i1 < (int)this->size(1); i1++) {
            for (int i0 = 0; i0 < (int)this->size(0); i0++) {
                (*this)(i0, i1) += rhs__(i0, i1);
            }
        }
        return *this;
    }

    inline Spheric_function<domain_t, T>&
    operator+=(Spheric_function<domain_t, T>&& rhs__)
    {
        for (int i1 = 0; i1 < (int)this->size(1); i1++) {
            for (int i0 = 0; i0 < (int)this->size(0); i0++) {
                (*this)(i0, i1) += rhs__(i0, i1);
            }
        }

        return *this;
    }

    inline Spheric_function<domain_t, T>&
    operator-=(Spheric_function<domain_t, T> const& rhs__)
    {
        for (int i1 = 0; i1 < (int)this->size(1); i1++) {
            for (int i0 = 0; i0 < (int)this->size(0); i0++) {
                (*this)(i0, i1) -= rhs__(i0, i1);
            }
        }
        return *this;
    }

    inline Spheric_function<domain_t, T>&
    operator-=(Spheric_function<domain_t, T>&& rhs__)
    {
        for (int i1 = 0; i1 < (int)this->size(1); i1++) {
            for (int i0 = 0; i0 < (int)this->size(0); i0++) {
                (*this)(i0, i1) -= rhs__(i0, i1);
            }
        }

        return *this;
    }
    /// Multiply by a constant.
    inline Spheric_function<domain_t, T>&
    operator*=(double alpha__)
    {
        for (int i1 = 0; i1 < (int)this->size(1); i1++) {
            for (int i0 = 0; i0 < (int)this->size(0); i0++) {
                (*this)(i0, i1) *= alpha__;
            }
        }

        return *this;
    }

    inline int
    angular_domain_size() const
    {
        return angular_domain_size_;
    }

    inline auto const&
    radial_grid() const
    {
        RTE_ASSERT(radial_grid_ != nullptr);
        return *radial_grid_;
    }

    auto
    component(int lm__) const
    {
        if (domain_t != function_domain_t::spectral) {
            RTE_THROW("function is not is spectral domain");
        }

        Spline<T> s(radial_grid());
        for (int ir = 0; ir < radial_grid_->num_points(); ir++) {
            s(ir) = (*this)(lm__, ir);
        }
        s.interpolate();
        return s;
    }

    T
    value(double theta__, double phi__, int jr__, double dr__) const
    {
        RTE_ASSERT(domain_t == function_domain_t::spectral);

        int lmax = sf::lmax(angular_domain_size_);
        std::vector<T> ylm(angular_domain_size_);
        sf::spherical_harmonics(lmax, theta__, phi__, &ylm[0]);
        T p = 0.0;
        for (int lm = 0; lm < angular_domain_size_; lm++) {
            double deriv = ((*this)(lm, jr__ + 1) - (*this)(lm, jr__)) / radial_grid_->dx(jr__);
            p += ylm[lm] * ((*this)(lm, jr__) + deriv * dr__);
        }
        return p;
    }
};

using Flm = Spheric_function<function_domain_t::spectral, double>;
using Ftp = Spheric_function<function_domain_t::spatial, double>;

/// 3D vector function.
template <function_domain_t domain_t, typename T = std::complex<double>>
class Spheric_vector_function : public std::array<Spheric_function<domain_t, T>, 3>
{
  private:
    Radial_grid<double> const* radial_grid_{nullptr};

    int angular_domain_size_{-1};

  public:
    /// Default constructor does nothing
    Spheric_vector_function()
    {
    }

    Spheric_vector_function(int angular_domain_size__, Radial_grid<double> const& radial_grid__)
        : radial_grid_(&radial_grid__)
        , angular_domain_size_(angular_domain_size__)
    {
        for (int x : {0, 1, 2}) {
            (*this)[x] = Spheric_function<domain_t, T>(angular_domain_size__, radial_grid__);
        }
    }

    inline Radial_grid<double> const&
    radial_grid() const
    {
        RTE_ASSERT(radial_grid_ != nullptr);
        return *radial_grid_;
    }

    inline int
    angular_domain_size() const
    {
        return angular_domain_size_;
    }
};

/// Multiplication of two functions in spatial domain.
/** The result of the operation is a scalar function in spatial domain */
template <typename T>
inline auto
operator*(Spheric_function<function_domain_t::spatial, T> const& a__,
          Spheric_function<function_domain_t::spatial, T> const& b__)
{
    if (a__.radial_grid().hash() != b__.radial_grid().hash()) {
        RTE_THROW("wrong radial grids");
    }
    if (a__.angular_domain_size() != b__.angular_domain_size()) {
        RTE_THROW("wrong angular domain sizes");
    }

    Spheric_function<function_domain_t::spatial, T> res(a__.angular_domain_size(), a__.radial_grid());

    #pragma omp parallel for schedule(static)
    for (int ir = 0; ir < res.radial_grid().num_points(); ir++) {
        for (int tp = 0; tp < res.angular_domain_size(); tp++) {
            res(tp, ir) = a__(tp, ir) * b__(tp, ir);
        }
    }

    return res;
}

/// Dot product of two gradiensts of real functions in spatial domain.
/** The result of the operation is the real scalar function in spatial domain */
inline auto
operator*(Spheric_vector_function<function_domain_t::spatial, double> const& f,
          Spheric_vector_function<function_domain_t::spatial, double> const& g)
{
    if (f.radial_grid().hash() != g.radial_grid().hash()) {
        RTE_THROW("wrong radial grids");
    }

    for (int x : {0, 1, 2}) {
        if (f[x].angular_domain_size() != g[x].angular_domain_size()) {
            RTE_THROW("wrong number of angular points");
        }
    }

    Spheric_function<function_domain_t::spatial, double> result(f.angular_domain_size(), f.radial_grid());
    result.zero();

    for (int x : {0, 1, 2}) {
        #pragma omp parallel for schedule(static)
        for (int ir = 0; ir < f.radial_grid().num_points(); ir++) {
            for (int tp = 0; tp < f.angular_domain_size(); tp++) {
                result(tp, ir) += f[x](tp, ir) * g[x](tp, ir);
            }
        }
    }

    return result;
}

/// Summation of two functions.
template <function_domain_t domain_t, typename T>
auto
operator+(Spheric_function<domain_t, T> const& a__, Spheric_function<domain_t, T> const& b__)
{
    if (a__.radial_grid().hash() != b__.radial_grid().hash()) {
        RTE_THROW("wrong radial grids");
    }
    if (a__.angular_domain_size() != b__.angular_domain_size()) {
        RTE_THROW("wrong angular domain sizes");
    }

    Spheric_function<domain_t, T> result(a__.angular_domain_size(), a__.radial_grid());

    #pragma omp parallel for schedule(static)
    for (int ir = 0; ir < a__.radial_grid().num_points(); ir++) {
        for (int i = 0; i < a__.angular_domain_size(); i++) {
            result(i, ir) = a__(i, ir) + b__(i, ir);
        }
    }

    return result;
}

/// Subtraction of functions.
template <function_domain_t domain_t, typename T>
auto
operator-(Spheric_function<domain_t, T> const& a__, Spheric_function<domain_t, T> const& b__)
{
    if (a__.radial_grid().hash() != b__.radial_grid().hash()) {
        RTE_THROW("wrong radial grids");
    }
    if (a__.angular_domain_size() != b__.angular_domain_size()) {
        RTE_THROW("wrong angular domain sizes");
    }

    Spheric_function<domain_t, T> res(a__.angular_domain_size(), a__.radial_grid());

    #pragma omp parallel for schedule(static)
    for (int ir = 0; ir < a__.radial_grid().num_points(); ir++) {
        for (int i = 0; i < a__.angular_domain_size(); i++) {
            res(i, ir) = a__(i, ir) - b__(i, ir);
        }
    }

    return res;
}

/// Multiply function by a scalar.
template <function_domain_t domain_t, typename T>
auto
operator*(T a__, Spheric_function<domain_t, T> const& b__)
{
    Spheric_function<domain_t, T> res(b__.angular_domain_size(), b__.radial_grid());

    T const* ptr_rhs = &b__(0, 0);
    T* ptr_res       = &res(0, 0);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < b__.size(); i++) {
        ptr_res[i] = a__ * ptr_rhs[i];
    }

    return res;
}
/// Multiply function by a scalar (inverse order).
template <function_domain_t domain_t, typename T>
auto
operator*(Spheric_function<domain_t, T> const& b__, T a__)
{
    return (a__ * b__);
}

/// Inner product of two spherical functions.
/** The result of the operation is a scalar value. */
template <function_domain_t domain_t, typename T>
inline auto
inner(Spheric_function<domain_t, T> const& f1, Spheric_function<domain_t, T> const& f2)
{
    Spline<T> s(f1.radial_grid());

    if (domain_t == function_domain_t::spectral) {
        int lmmax = std::min(f1.angular_domain_size(), f2.angular_domain_size());
        for (int ir = 0; ir < s.num_points(); ir++) {
            for (int lm = 0; lm < lmmax; lm++) {
                s(ir) += conj(f1(lm, ir)) * f2(lm, ir);
            }
            s(ir) *= std::pow(f1.radial_grid().x(ir), 2);
        }
    } else {
        throw std::runtime_error("not implemented");
    }
    return s.interpolate().integrate(0);
}

/// Compute Laplacian of the spheric function.
/** Laplacian in spherical coordinates has the following expression:
    \f[
    \Delta = \frac{1}{r^2}\frac{\partial}{\partial r}\Big( r^2 \frac{\partial}{\partial r} \Big) +
      \frac{1}{r^2}\Delta_{\theta, \phi}
    \f]
 */
template <typename T>
inline auto
laplacian(Spheric_function<function_domain_t::spectral, T> const& f__)
{
    Spheric_function<function_domain_t::spectral, T> g;
    auto& rgrid = f__.radial_grid();
    int lmmax   = f__.angular_domain_size();
    int lmax    = sf::lmax(lmmax);
    g           = Spheric_function<function_domain_t::spectral, T>(lmmax, rgrid);

    Spline<T> s1(rgrid);
    for (int l = 0; l <= lmax; l++) {
        int ll = l * (l + 1);
        for (int m = -l; m <= l; m++) {
            int lm = sf::lm(l, m);
            /* get lm component */
            auto s = f__.component(lm);
            /* compute 1st derivative */
            for (int ir = 0; ir < s.num_points(); ir++) {
                s1(ir) = s.deriv(1, ir);
            }
            s1.interpolate();

            for (int ir = 0; ir < s.num_points(); ir++) {
                g(lm, ir) = 2.0 * s1(ir) * rgrid.x_inv(ir) + s1.deriv(1, ir) -
                            s(ir) * static_cast<double>(ll) / std::pow(rgrid[ir], 2);
            }
        }
    }

    return g;
}

/// Convert from Ylm to Rlm representation.
inline void
convert(Spheric_function<function_domain_t::spectral, std::complex<double>> const& f__,
        Spheric_function<function_domain_t::spectral, double>& g__)
{
    int lmax = sf::lmax(f__.angular_domain_size());

    /* cache transformation arrays */
    std::vector<std::complex<double>> tpp(f__.angular_domain_size());
    std::vector<std::complex<double>> tpm(f__.angular_domain_size());
    for (int l = 0; l <= lmax; l++) {
        for (int m = -l; m <= l; m++) {
            int lm  = sf::lm(l, m);
            tpp[lm] = SHT::rlm_dot_ylm(l, m, m);
            tpm[lm] = SHT::rlm_dot_ylm(l, m, -m);
        }
    }

    for (int ir = 0; ir < f__.radial_grid().num_points(); ir++) {
        int lm = 0;
        for (int l = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++) {
                if (m == 0) {
                    g__(lm, ir) = std::real(f__(lm, ir));
                } else {
                    int lm1     = sf::lm(l, -m);
                    g__(lm, ir) = std::real(tpp[lm] * f__(lm, ir) + tpm[lm] * f__(lm1, ir));
                }
                lm++;
            }
        }
    }
}

/// Convert from Ylm to Rlm representation.
inline auto
convert(Spheric_function<function_domain_t::spectral, std::complex<double>> const& f__)
{
    Spheric_function<function_domain_t::spectral, double> g(f__.angular_domain_size(), f__.radial_grid());
    convert(f__, g);
    return g;
}

/// Convert from Rlm to Ylm representation.
inline void
convert(Spheric_function<function_domain_t::spectral, double> const& f__,
        Spheric_function<function_domain_t::spectral, std::complex<double>>& g__)
{
    int lmax = sf::lmax(f__.angular_domain_size());

    /* cache transformation arrays */
    std::vector<std::complex<double>> tpp(f__.angular_domain_size());
    std::vector<std::complex<double>> tpm(f__.angular_domain_size());
    for (int l = 0; l <= lmax; l++) {
        for (int m = -l; m <= l; m++) {
            int lm  = sf::lm(l, m);
            tpp[lm] = SHT::ylm_dot_rlm(l, m, m);
            tpm[lm] = SHT::ylm_dot_rlm(l, m, -m);
        }
    }

    for (int ir = 0; ir < f__.radial_grid().num_points(); ir++) {
        int lm = 0;
        for (int l = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++) {
                if (m == 0) {
                    g__(lm, ir) = f__(lm, ir);
                } else {
                    int lm1     = sf::lm(l, -m);
                    g__(lm, ir) = tpp[lm] * f__(lm, ir) + tpm[lm] * f__(lm1, ir);
                }
                lm++;
            }
        }
    }
}

/// Convert from Rlm to Ylm representation.
inline auto
convert(Spheric_function<function_domain_t::spectral, double> const& f__)
{
    Spheric_function<function_domain_t::spectral, std::complex<double>> g(f__.angular_domain_size(), f__.radial_grid());
    convert(f__, g);
    return g;
}

template <typename T>
inline void
transform(SHT const& sht__, Spheric_function<function_domain_t::spectral, T> const& f__,
          Spheric_function<function_domain_t::spatial, T>& g__)
{
    sht__.backward_transform(f__.angular_domain_size(), &f__(0, 0), f__.radial_grid().num_points(),
                             std::min(sht__.lmmax(), f__.angular_domain_size()), &g__(0, 0));
}

/// Transform to spatial domain (to r, \theta, \phi coordinates).
template <typename T>
inline auto
transform(SHT const& sht__, Spheric_function<function_domain_t::spectral, T> const& f__)
{
    Spheric_function<function_domain_t::spatial, T> g(sht__.num_points(), f__.radial_grid());
    transform(sht__, f__, g);
    return g;
}

template <typename T>
inline void
transform(SHT const& sht__, Spheric_function<function_domain_t::spatial, T> const& f__,
          Spheric_function<function_domain_t::spectral, T>& g__)
{
    sht__.forward_transform(&f__(0, 0), f__.radial_grid().num_points(), sht__.lmmax(), sht__.lmmax(), &g__(0, 0));
}

/// Transform to spectral domain.
template <typename T>
inline auto
transform(SHT const& sht__, Spheric_function<function_domain_t::spatial, T> const& f__)
{
    Spheric_function<function_domain_t::spectral, T> g(sht__.lmmax(), f__.radial_grid());
    transform(sht__, f__, g);
    return g;
}

/// Gradient of the function in complex spherical harmonics.
inline auto
gradient(Spheric_function<function_domain_t::spectral, std::complex<double>> const& f)
{
    Spheric_vector_function<function_domain_t::spectral, std::complex<double>> g(f.angular_domain_size(),
                                                                                 f.radial_grid());
    for (int i = 0; i < 3; i++) {
        g[i].zero();
    }

    int lmax = sf::lmax(f.angular_domain_size());

    for (int l = 0; l <= lmax; l++) {
        double d1 = std::sqrt(double(l + 1) / double(2 * l + 3));
        double d2 = std::sqrt(double(l) / double(2 * l - 1));

        for (int m = -l; m <= l; m++) {
            int lm = sf::lm(l, m);
            auto s = f.component(lm);

            for (int mu = -1; mu <= 1; mu++) {
                int j = (mu + 2) % 3; // map -1,0,1 to 1,2,0 (to y,z,x)

                if ((l + 1) <= lmax && std::abs(m + mu) <= l + 1) {
                    int lm1  = sf::lm(l + 1, m + mu);
                    double d = d1 * SHT::clebsch_gordan(l, 1, l + 1, m, mu, m + mu);
                    for (int ir = 0; ir < f.radial_grid().num_points(); ir++) {
                        g[j](lm1, ir) += (s.deriv(1, ir) - f(lm, ir) * f.radial_grid().x_inv(ir) * double(l)) * d;
                    }
                }
                if ((l - 1) >= 0 && std::abs(m + mu) <= l - 1) {
                    int lm1  = sf::lm(l - 1, m + mu);
                    double d = d2 * SHT::clebsch_gordan(l, 1, l - 1, m, mu, m + mu);
                    for (int ir = 0; ir < f.radial_grid().num_points(); ir++) {
                        g[j](lm1, ir) -= (s.deriv(1, ir) + f(lm, ir) * f.radial_grid().x_inv(ir) * double(l + 1)) * d;
                    }
                }
            }
        }
    }

    std::complex<double> d1(1.0 / std::sqrt(2.0), 0);
    std::complex<double> d2(0, 1.0 / std::sqrt(2.0));

    for (int ir = 0; ir < f.radial_grid().num_points(); ir++) {
        for (int lm = 0; lm < f.angular_domain_size(); lm++) {
            std::complex<double> g_p = g[0](lm, ir);
            std::complex<double> g_m = g[1](lm, ir);
            g[0](lm, ir)             = d1 * (g_m - g_p);
            g[1](lm, ir)             = d2 * (g_m + g_p);
        }
    }

    return g;
}

/// Gradient of the function in real spherical harmonics.
inline auto
gradient(Spheric_function<function_domain_t::spectral, double> const& f__)
{
    int lmax = sf::lmax(f__.angular_domain_size());
    SHT sht(device_t::CPU, lmax);
    auto zf = convert(f__);
    auto zg = gradient(zf);
    Spheric_vector_function<function_domain_t::spectral, double> g(f__.angular_domain_size(), f__.radial_grid());
    for (int x : {0, 1, 2}) {
        g[x] = convert(zg[x]);
    }
    return g;
}

/// Divergence of the vector function in complex spherical harmonics.
inline auto
divergence(Spheric_vector_function<function_domain_t::spectral, std::complex<double>> const& vf__)
{
    Spheric_function<function_domain_t::spectral, std::complex<double>> g(vf__.angular_domain_size(),
                                                                          vf__.radial_grid());
    g.zero();

    for (int x : {0, 1, 2}) {
        auto f = gradient(vf__[x]);
        g += f[x];
    }
    return g;
}

inline auto
divergence(Spheric_vector_function<function_domain_t::spectral, double> const& vf)
{
    Spheric_function<function_domain_t::spectral, double> g(vf.angular_domain_size(), vf.radial_grid());
    g.zero();

    for (int x : {0, 1, 2}) {
        auto zf  = convert(vf[x]);
        auto zf1 = gradient(zf);
        auto f2  = convert(zf1[x]);
        g += f2;
    }
    return g;
}

} // namespace sirius

#endif // __SPHERIC_FUNCTION_HPP__
