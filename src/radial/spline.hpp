/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file spline.hpp
 *
 *  \brief Contains definition and partial implementation of sirius::Spline class.
 */

#ifndef __SPLINE_HPP__
#define __SPLINE_HPP__

#include "radial/radial_grid.hpp"

namespace sirius {

/// Cubic spline with a not-a-knot boundary conditions.
/** The following convention for spline coefficients is used: for \f$ x \f$ in
 *  \f$ [x_i, x_{i+1}] \f$ the value of the spline is equal to
 *  \f$ a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3 \f$.
 *
 *  Suppose we have \f$ n \f$ value points \f$ y_i = f(x_i) \f$ and \f$ n - 1 \f$ segments:
 *  \f[
 *   S_i(x) = y_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3
 *  \f]
 *  Segment derivatives:
 *  \f[
 *    \begin{eqnarray}
 *      S_i'(x) &=& b_i + 2c_i(x - x_i) + 3d_i(x - x_i)^2 \\
 *      S_i''(x) &=& 2c_i + 6d_i(x-x_i) \\
 *      S_i'''(x) &=& 6d_i
 *    \end{eqnarray}
 *  \f]
 *  The following substitutions are made:
 *  \f[
 *    m_i = 2 c_i \\
 *    h_i = x_{i+1} - x_i \\
 *    y_i' = \frac{y_{i+1} - y_i}{h_i}
 *  \f]
 *  Now we can equate the derivatives at the end points of segments. From the 3rd derivative we get
 *  \f[
 *    d_i = \frac{1}{6h_i}(m_{i+1} -  m_i)
 *  \f]
 *  From the 1st derivative we get
 *  \f[
 *    b_i = y_i' - \frac{h_i}{2} m_i - \frac{h_i}{6}(m_{i+1} - m_i)
 *  \f]
 *  Using 2nd derivative condition we get
 *  \f[
 *    h_i m_i + 2(h_{i} + h_{i+1})m_{i+1} + h_{i+1}m_{i+2} = 6(y_{i+1}' - y_{i}')
 *  \f]
 *  So far we got \f$ n - 3 \f$ equations for \f$ n - 1 \f$ coefficients \f$ m_i \f$. We need two extra conditions.
 *  Not-a-knot boundary condition (counting segments and points from 1):
 *  \f[
 *    S_1'''(x_2) = S_2'''(x_2) \longrightarrow d_1 = d_2 \\
 *    S_{n-2}'''(x_{n-1}) = S_{n-1}'''(x_{n-1}) \longrightarrow d_{n-2} = d_{n-1}
 *  \f]
 */
template <typename T, typename U = double>
class Spline : public Radial_grid<U>
{
  private:
    /// Array of spline coefficients.
    mdarray<T, 2> coeffs_;
    /* forbid copy constructor */
    Spline(Spline<T, U> const& src__) = delete;
    /* forbid assignment operator */
    Spline<T, U>&
    operator=(Spline<T, U> const& src__) = delete;
    /// Solver tridiagonal system of linear equaitons.
    int
    solve(T* dl, T* d, T* du, T* b, int n)
    {
        for (int i = 0; i < n - 1; i++) {
            if (std::abs(dl[i]) == 0) {
                if (std::abs(d[i]) == 0) {
                    return i + 1;
                }
            } else if (std::abs(d[i]) >= std::abs(dl[i])) {
                T mult = dl[i] / d[i];
                d[i + 1] -= mult * du[i];
                b[i + 1] -= mult * b[i];
                if (i < n - 2) {
                    dl[i] = 0;
                }
            } else {
                T mult   = d[i] / dl[i];
                d[i]     = dl[i];
                T tmp    = d[i + 1];
                d[i + 1] = du[i] - mult * tmp;
                if (i < n - 2) {
                    dl[i]     = du[i + 1];
                    du[i + 1] = -mult * dl[i];
                }
                du[i]    = tmp;
                tmp      = b[i];
                b[i]     = b[i + 1];
                b[i + 1] = tmp - mult * b[i + 1];
            }
        }
        if (std::abs(d[n - 1]) == 0) {
            return n;
        }
        b[n - 1] /= d[n - 1];
        if (n > 1) {
            b[n - 2] = (b[n - 2] - du[n - 2] * b[n - 1]) / d[n - 2];
        }
        for (int i = n - 3; i >= 0; i--) {
            b[i] = (b[i] - du[i] * b[i + 1] - dl[i] * b[i + 2]) / d[i];
        }
        return 0;
    }
    /// Init the underlying radial grid.
    void
    init_grid(Radial_grid<U> const& radial_grid__)
    {
        /* copy the grid points */
        this->x_ = mdarray<U, 1>({radial_grid__.num_points()});
        copy(radial_grid__.x(), this->x_);
        this->init();
    }

  public:
    /// Default constructor.
    Spline()
    {
    }
    /// Constructor of a new empty spline.
    Spline(Radial_grid<U> const& radial_grid__)
    {
        init_grid(radial_grid__);
        coeffs_ = mdarray<T, 2>({this->num_points(), 4});
        coeffs_.zero();
    }
    /// Constructor of a spline from a function.
    Spline(Radial_grid<U> const& radial_grid__, std::function<T(U)> f__)
    {
        init_grid(radial_grid__);
        coeffs_ = mdarray<T, 2>({this->num_points(), 4});
        for (int i = 0; i < this->num_points(); i++) {
            coeffs_(i, 0) = f__(this->x(i));
        }
        interpolate();
    }
    /// Constructor of a spline from a list of values.
    Spline(Radial_grid<U> const& radial_grid__, std::vector<T> const& y__)
    {
        init_grid(radial_grid__);
        assert(static_cast<int>(y__.size()) <= this->num_points());
        coeffs_ = mdarray<T, 2>({this->num_points(), 4});
        coeffs_.zero();
        int i{0};
        for (auto e : y__) {
            this->coeffs_(i++, 0) = e;
        }
        interpolate();
    }

    Spline(Spline<T, U>&& src__) = default;

    Spline<T, U>&
    operator=(Spline<T, U>&& src__) = default;

    Spline<T, U>&
    operator=(std::function<T(U)> f__)
    {
        for (int ir = 0; ir < this->num_points(); ir++) {
            coeffs_(ir, 0) = f__(this->x(ir));
        }
        return interpolate();
    }

    Spline<T, U>&
    operator=(std::vector<T> const& y__)
    {
        assert(static_cast<int>(y__.size()) <= this->num_points());
        coeffs_.zero();
        int i{0};
        for (auto e : y__) {
            this->coeffs_(i++, 0) = e;
        }
        return interpolate();
    }

    /// Get the reference to a value at the point x[i].
    inline T&
    operator()(const int i)
    {
        return coeffs_(i, 0);
    }

    /// Get value at the point x[i].
    inline T const&
    operator()(const int i) const
    {
        return coeffs_(i, 0);
    }

    /// Get value at the point x[i] + dx.
    inline T
    operator()(const int i, U dx) const
    {
        assert(i >= 0);
        assert(i < this->num_points() - 1);
        assert(dx >= 0);
        return coeffs_(i, 0) + dx * (coeffs_(i, 1) + dx * (coeffs_(i, 2) + dx * coeffs_(i, 3)));
    }

    /// Compute value at any point.
    inline T
    at_point(U x) const
    {
        int j = this->index_of(x);
        if (j == -1) {
            std::stringstream s;
            s << "index of point is not found\n"
              << "  x           : " << x << "\n"
              << "  first point : " << this->first() << "\n"
              << "  last point  : " << this->last();
            RTE_THROW(s);
        }
        U dx = x - (*this)[j];
        return (*this)(j, dx);
    }

    inline T
    deriv(const int dm, const int i, const U dx) const
    {
        assert(i >= 0);
        assert(i < this->num_points() - 1);
        assert(dx >= 0);

        T result = 0;
        switch (dm) {
            case 0: {
                result = coeffs_(i, 0) + dx * (coeffs_(i, 1) + dx * (coeffs_(i, 2) + dx * coeffs_(i, 3)));
                break;
            }
            case 1: {
                result = coeffs_(i, 1) + (coeffs_(i, 2) * 2.0 + coeffs_(i, 3) * dx * 3.0) * dx;
                break;
            }
            case 2: {
                result = coeffs_(i, 2) * 2.0 + coeffs_(i, 3) * dx * 6.0;
                break;
            }
            case 3: {
                result = coeffs_(i, 3) * 6.0;
                break;
            }
            default: {
                RTE_THROW("wrong order of derivative");
                break;
            }
        }
        return result;
    }

    inline T
    deriv(int dm, int i) const
    {
        assert(i >= 0);
        assert(i < this->num_points());

        if (i == this->num_points() - 1) {
            return deriv(dm, i - 1, this->dx(i - 1));
        } else {
            return deriv(dm, i, 0);
        }
    }

    Spline<T, U>&
    interpolate()
    {
        /* number of segments; in principle we have n-1 segments, but the equations for spline are built in such a
           way that one extra m_i coefficient of a segment is necessary */
        int ns = this->num_points();
        assert(ns >= 4);
        /* lower diagonal (use coeffs as temporary storage) */
        T* dl = coeffs_.at(memory_t::host, 0, 1);
        /* main diagonal */
        T* d = coeffs_.at(memory_t::host, 0, 2);
        /* upper diagonal */
        T* du = coeffs_.at(memory_t::host, 0, 3);
        /* m_i = 2 c_i */
        std::vector<T> m(ns);
        /* derivatives of function */
        std::vector<T> dy(ns - 1);
        for (int i = 0; i < ns - 1; i++) {
            dy[i] = (coeffs_(i + 1, 0) - coeffs_(i, 0)) / this->dx(i);
        }
        /* setup "B" vector of AX=B equation */
        for (int i = 0; i < ns - 2; i++) {
            m[i + 1] = (dy[i + 1] - dy[i]) * 6.0;
        }
        /* this is derived from n-a-k boundary condition */
        m[0]      = m[1];
        m[ns - 1] = m[ns - 2];

        /* main diagonal of "A" matrix */
        for (int i = 0; i < ns - 2; i++) {
            d[i + 1] = static_cast<T>(this->x(i + 2) - this->x(i)) * 2.0;
        }
        /* subdiagonals of "A" matrix */
        for (int i = 0; i < ns - 1; i++) {
            du[i] = this->dx(i);
            dl[i] = this->dx(i);
        }

        /* last part of n-a-k boundary condition */
        U h0  = this->dx(0);
        U h1  = this->dx(1);
        d[0]  = h0 - (h1 / h0) * h1;
        du[0] = h1 * ((h1 / h0) + 1) + 2 * (h0 + h1);

        h0         = this->dx(ns - 2);
        h1         = this->dx(ns - 3);
        d[ns - 1]  = h0 - (h1 / h0) * h1;
        dl[ns - 2] = h1 * ((h1 / h0) + 1) + 2 * (h0 + h1);

        ///* this should be boundary conditions for natural spline (with zero second derivatives at boundaries) */
        // m[0] = m[ns-1] = 0;
        // du[0] = 0;
        // dl[ns - 2] = 0;
        // d[0] = d[ns-1] = 1;

        /* solve tridiagonal system */
        // int info = linalg<device_t::CPU>::gtsv(ns, 1, &dl[0], &d[0], &du[0], &m[0], ns);
        int info = solve(dl, d, du, &m[0], ns);

        if (info) {
            std::stringstream s;
            s << "error in tridiagonal solver: " << info;
            RTE_THROW(s);
        }

        for (int i = 0; i < ns - 1; i++) {
            /* this is c_i coefficient */
            coeffs_(i, 2) = m[i] / 2.0;
            /* this is why one extra segment was considered: we need m_{i+1} */
            T t = (m[i + 1] - m[i]) / 6.0;
            /* b_i coefficient */
            coeffs_(i, 1) = dy[i] - (coeffs_(i, 2) + t) * this->dx(i);
            /* d_i coefficient */
            coeffs_(i, 3) = t / this->dx(i);
        }
        coeffs_(ns - 1, 1) = 0;
        coeffs_(ns - 1, 2) = 0;
        coeffs_(ns - 1, 3) = 0;

        return *this;
    }

    /// Integrate spline with r^m prefactor.
    /**
    Derivation for r^2 prefactor is based on the following Mathematica notebook:
    \verbatim
            In[26]:= result =
         FullSimplify[
          Integrate[
           x^(2)*(a0 + a1*(x - x0) + a2*(x - x0)^2 + a3*(x - x0)^3), {x, x0,
            x1}],
          Assumptions -> {Element[{x0, x1}, Reals], x1 > x0 > 0}]


        Out[26]= 1/60 (20 a0 (-x0^3 + x1^3) +
           5 a1 (x0^4 - 4 x0 x1^3 + 3 x1^4) + (x0 -
              x1)^3 (-2 a2 (x0^2 + 3 x0 x1 + 6 x1^2) +
              a3 (x0 - x1) (x0^2 + 4 x0 x1 + 10 x1^2)))

        In[27]:= r = Expand[result] /. {x1 -> x0 + dx}

        Out[27]= -((a0 x0^3)/3) + (a1 x0^4)/12 - (a2 x0^5)/30 + (
         a3 x0^6)/60 + 1/3 a0 (dx + x0)^3 - 1/3 a1 x0 (dx + x0)^3 +
         1/3 a2 x0^2 (dx + x0)^3 - 1/3 a3 x0^3 (dx + x0)^3 +
         1/4 a1 (dx + x0)^4 - 1/2 a2 x0 (dx + x0)^4 +
         3/4 a3 x0^2 (dx + x0)^4 + 1/5 a2 (dx + x0)^5 -
         3/5 a3 x0 (dx + x0)^5 + 1/6 a3 (dx + x0)^6

        In[34]:= Collect[r, dx, Simplify]

        Out[34]= (a3 dx^6)/6 + a0 dx x0^2 + 1/2 dx^2 x0 (2 a0 + a1 x0) +
         1/5 dx^5 (a2 + 2 a3 x0) + 1/3 dx^3 (a0 + x0 (2 a1 + a2 x0)) +
         1/4 dx^4 (a1 + x0 (2 a2 + a3 x0))

        In[28]:= r1 = Collect[r/dx, dx, Simplify]

        Out[28]= (a3 dx^5)/6 + a0 x0^2 + 1/2 dx x0 (2 a0 + a1 x0) +
         1/5 dx^4 (a2 + 2 a3 x0) + 1/3 dx^2 (a0 + x0 (2 a1 + a2 x0)) +
         1/4 dx^3 (a1 + x0 (2 a2 + a3 x0))

        In[29]:= r2 = Collect[(r1 - a0*x0^2)/dx, dx, Simplify]

        Out[29]= (a3 dx^4)/6 + 1/2 x0 (2 a0 + a1 x0) +
         1/5 dx^3 (a2 + 2 a3 x0) + 1/3 dx (a0 + x0 (2 a1 + a2 x0)) +
         1/4 dx^2 (a1 + x0 (2 a2 + a3 x0))

        In[30]:= r3 = Collect[(r2 - 1/2 x0 (2 a0 + a1 x0))/dx, dx, Simplify]

        Out[30]= (a3 dx^3)/6 + 1/5 dx^2 (a2 + 2 a3 x0) +
         1/3 (a0 + x0 (2 a1 + a2 x0)) + 1/4 dx (a1 + x0 (2 a2 + a3 x0))

        In[31]:= r4 =
         Collect[(r3 - 1/3 (a0 + x0 (2 a1 + a2 x0)))/dx, dx, Simplify]

        Out[31]= (a3 dx^2)/6 + 1/5 dx (a2 + 2 a3 x0) +
         1/4 (a1 + x0 (2 a2 + a3 x0))

        In[32]:= r5 =
         Collect[(r4 - 1/4 (a1 + x0 (2 a2 + a3 x0)))/dx, dx, Simplify]

        Out[32]= (a3 dx)/6 + 1/5 (a2 + 2 a3 x0)

        In[33]:= r6 = Collect[(r5 - 1/5 (a2 + 2 a3 x0))/dx, dx, Simplify]

        Out[33]= a3/6

    \endverbatim
    */
    T
    integrate(std::vector<T>& g__, int m__) const
    {
        g__    = std::vector<T>(this->num_points());
        g__[0] = 0.0;

        switch (m__) {
            case 0: {
                T t = 1.0 / 3.0;
                for (int i = 0; i < this->num_points() - 1; i++) {
                    U dx       = this->dx(i);
                    g__[i + 1] = g__[i] +
                                 (((coeffs_(i, 3) * dx * 0.25 + coeffs_(i, 2) * t) * dx + coeffs_(i, 1) * 0.5) * dx +
                                  coeffs_(i, 0)) *
                                         dx;
                }
                break;
            }
            case 2: {
                for (int i = 0; i < this->num_points() - 1; i++) {
                    U x0 = this->x(i);
                    U dx = this->dx(i);
                    T a0 = coeffs_(i, 0);
                    T a1 = coeffs_(i, 1);
                    T a2 = coeffs_(i, 2);
                    T a3 = coeffs_(i, 3);

                    T val = dx * (dx * (dx * (dx * (dx * (dx * a3 / 6.0 + (a2 + 2.0 * a3 * x0) / 5.0) +
                                                    (a1 + x0 * (2.0 * a2 + a3 * x0)) / 4.0) +
                                              (a0 + x0 * (2.0 * a1 + a2 * x0)) / 3.0) +
                                        x0 * (2.0 * a0 + a1 * x0) / 2.0) +
                                  a0 * x0 * x0);

                    g__[i + 1] = g__[i] + val;
                }
                break;
            }
            case -1: {
                for (int i = 0; i < this->num_points() - 1; i++) {
                    U x0 = this->x(i);
                    U x1 = this->x(i + 1);
                    U dx = this->dx(i);
                    T a0 = coeffs_(i, 0);
                    T a1 = coeffs_(i, 1);
                    T a2 = coeffs_(i, 2);
                    T a3 = coeffs_(i, 3);

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(-1)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                    //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    g__[i + 1] = g__[i] +
                                 (dx / 6.0) * (6.0 * a1 + x0 * (-9.0 * a2 + 11.0 * a3 * x0) +
                                               x1 * (3.0 * a2 - 7.0 * a3 * x0 + 2.0 * a3 * x1)) +
                                 (-a0 + x0 * (a1 + x0 * (-a2 + a3 * x0))) * std::log(x0 / x1);
                }
                break;
            }
            case -2: {
                for (int i = 0; i < this->num_points() - 1; i++) {
                    U x0 = this->x(i);
                    U x1 = this->x(i + 1);
                    U dx = this->dx(i);
                    T a0 = coeffs_(i, 0);
                    T a1 = coeffs_(i, 1);
                    T a2 = coeffs_(i, 2);
                    T a3 = coeffs_(i, 3);

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(-2)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                    //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    // g__[i + 1] = g__[i] + (((x0 - x1) * (-2.0 * a0 + x0 * (2.0 * a1 - 2.0 * a2 * (x0 + x1) +
                    //             a3 * (2.0 * std::pow(x0, 2) + 5.0 * x0 * x1 - std::pow(x1, 2)))) +
                    //             2.0 * x0 * (a1 + x0 * (-2.0 * a2 + 3.0 * a3 * x0)) * x1 * std::log(x1 / x0)) /
                    //             (2.0 * x0 * x1));
                    g__[i + 1] = g__[i] +
                                 (a2 * dx - 5.0 * a3 * x0 * dx / 2.0 - a1 * (dx / x1) + a0 * (dx / x0 / x1) +
                                  (x0 / x1) * dx * (a2 - a3 * x0) + a3 * x1 * dx / 2.0) +
                                 (a1 + x0 * (-2.0 * a2 + 3.0 * a3 * x0)) * std::log(x1 / x0);
                }
                break;
            }
            case -3: {
                for (int i = 0; i < this->num_points() - 1; i++) {
                    U x0 = this->x(i);
                    U x1 = this->x(i + 1);
                    U dx = this->dx(i);
                    T a0 = coeffs_(i, 0);
                    T a1 = coeffs_(i, 1);
                    T a2 = coeffs_(i, 2);
                    T a3 = coeffs_(i, 3);

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(-3)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                    //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    // g__[i + 1] = g__[i] + (-((x0 - x1) * (a0 * (x0 + x1) + x0 * (a1 * (-x0 + x1) +
                    //             x0 * (a2 * x0 - a3 * std::pow(x0, 2) - 3.0 * a2 * x1 + 5.0 * a3 * x0 * x1 +
                    //             2.0 * a3 * std::pow(x1, 2)))) + 2.0 * std::pow(x0, 2) * (a2 - 3.0 * a3 * x0) *
                    //             std::pow(x1, 2) * std::log(x0 / x1)) / (2.0 * std::pow(x0, 2) * std::pow(x1, 2)));
                    g__[i + 1] = g__[i] +
                                 dx *
                                         (a0 * (x0 + x1) +
                                          x0 * (a1 * dx + x0 * (a2 * x0 - a3 * std::pow(x0, 2) - 3.0 * a2 * x1 +
                                                                5.0 * a3 * x0 * x1 + 2.0 * a3 * std::pow(x1, 2)))) /
                                         std::pow(x0 * x1, 2) / 2.0 +
                                 (-a2 + 3.0 * a3 * x0) * std::log(x0 / x1);
                }
                break;
            }
            case -4: {
                for (int i = 0; i < this->num_points() - 1; i++) {
                    U x0 = this->x(i);
                    U x1 = this->x(i + 1);
                    U dx = this->dx(i);
                    T a0 = coeffs_(i, 0);
                    T a1 = coeffs_(i, 1);
                    T a2 = coeffs_(i, 2);
                    T a3 = coeffs_(i, 3);

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(-4)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                    //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    // g__[i + 1] = g__[i] + ((2.0 * a0 * (-std::pow(x0, 3) + std::pow(x1, 3)) +
                    //             x0 * (x0 - x1) * (a1 * (x0 - x1) * (2.0 * x0 + x1) +
                    //             x0 * (-2.0 * a2 * std::pow(x0 - x1, 2) + a3 * x0 * (2.0 * std::pow(x0, 2) - 7.0 * x0
                    //             * x1 + 11.0 * std::pow(x1, 2)))) + 6.0 * a3 * std::pow(x0 * x1, 3) * std::log(x1 /
                    //             x0)) / (6.0 * std::pow(x0 * x1, 3)));
                    g__[i + 1] = g__[i] +
                                 (2.0 * a0 * (-std::pow(x0, 3) + std::pow(x1, 3)) -
                                  x0 * dx *
                                          (-a1 * dx * (2.0 * x0 + x1) +
                                           x0 * (-2.0 * a2 * std::pow(dx, 2) +
                                                 a3 * x0 *
                                                         (2.0 * std::pow(x0, 2) - 7.0 * x0 * x1 +
                                                          11.0 * std::pow(x1, 2))))) /
                                         std::pow(x0 * x1, 3) / 6.0 +
                                 a3 * std::log(x1 / x0);
                }
                break;
            }
            default: {
                for (int i = 0; i < this->num_points() - 1; i++) {
                    U x0 = this->x(i);
                    U x1 = this->x(i + 1);
                    T a0 = coeffs_(i, 0);
                    T a1 = coeffs_(i, 1);
                    T a2 = coeffs_(i, 2);
                    T a3 = coeffs_(i, 3);

                    // obtained with the following Mathematica code:
                    //   FullSimplify[Integrate[x^(m)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                    //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                    g__[i + 1] =
                            g__[i] +
                            (std::pow(x0, 1 + m__) *
                             (-(a0 * double((2 + m__) * (3 + m__) * (4 + m__))) +
                              x0 * (a1 * double((3 + m__) * (4 + m__)) - 2.0 * a2 * double(4 + m__) * x0 +
                                    6.0 * a3 * std::pow(x0, 2)))) /
                                    double((1 + m__) * (2 + m__) * (3 + m__) * (4 + m__)) +
                            std::pow(x1, 1 + m__) * ((a0 - x0 * (a1 + x0 * (-a2 + a3 * x0))) / double(1 + m__) +
                                                     ((a1 + x0 * (-2.0 * a2 + 3.0 * a3 * x0)) * x1) / double(2 + m__) +
                                                     ((a2 - 3.0 * a3 * x0) * std::pow(x1, 2)) / double(3 + m__) +
                                                     (a3 * std::pow(x1, 3)) / double(4 + m__));
                }
                break;
            }
        }

        return g__.back();
    }

    /// Integrate spline with r^m prefactor.
    T
    integrate(int m__) const
    {
        std::vector<T> g;
        return integrate(g, m__);
    }

    inline void
    scale(double a__)
    {
        for (int i = 0; i < this->num_points(); i++) {
            coeffs_(i, 0) *= a__;
            coeffs_(i, 1) *= a__;
            coeffs_(i, 2) *= a__;
            coeffs_(i, 3) *= a__;
        }
    }

    inline std::array<T, 4>
    coeffs(int i__) const
    {
        return {coeffs_(i__, 0), coeffs_(i__, 1), coeffs_(i__, 2), coeffs_(i__, 3)};
    }

    inline mdarray<T, 2> const&
    coeffs() const
    {
        return coeffs_;
    }

    auto
    values() const
    {
        std::vector<T> val(this->num_points());
        for (int ir = 0; ir < this->num_points(); ir++) {
            val[ir] = coeffs_(ir, 0);
        }
        return val;
    }
};

template <typename T, typename U = double>
inline Spline<T, U>
operator*(Spline<T, U> const& a__, Spline<T, U> const& b__)
{
    Spline<T> s12(reinterpret_cast<Radial_grid<U> const&>(a__));

    auto& coeffs_a = a__.coeffs();
    auto& coeffs_b = b__.coeffs();
    auto& coeffs   = const_cast<mdarray<T, 2>&>(s12.coeffs());

    for (int ir = 0; ir < a__.num_points(); ir++) {
        coeffs(ir, 0) = coeffs_a(ir, 0) * coeffs_b(ir, 0);
        coeffs(ir, 1) = coeffs_a(ir, 1) * coeffs_b(ir, 0) + coeffs_a(ir, 0) * coeffs_b(ir, 1);
        coeffs(ir, 2) = coeffs_a(ir, 2) * coeffs_b(ir, 0) + coeffs_a(ir, 1) * coeffs_b(ir, 1) +
                        coeffs_a(ir, 0) * coeffs_b(ir, 2);
        coeffs(ir, 3) = coeffs_a(ir, 3) * coeffs_b(ir, 0) + coeffs_a(ir, 2) * coeffs_b(ir, 1) +
                        coeffs_a(ir, 1) * coeffs_b(ir, 2) + coeffs_a(ir, 0) * coeffs_b(ir, 3);
    }

    return s12;
}

#ifdef SIRIUS_GPU
// extern "C" double spline_inner_product_gpu_v2(int           size__,
//                                              double const* x__,
//                                              double const* dx__,
//                                              double const* f__,
//                                              double const* g__,
//                                              double*       d_buf__,
//                                              double*       h_buf__,
//                                              int           stream_id__);
//
extern "C" void
spline_inner_product_gpu_v3(int const* idx_ri__, int num_ri__, int num_points__, double const* x__, double const* dx__,
                            double const* f__, double const* g__, double* result__);
#endif

template <typename T>
T
inner(Spline<T> const& f__, Spline<T> const& g__, int m__, int num_points__)
{
    T result = 0;

    switch (m__) {
        case 0: {
            for (int i = 0; i < num_points__ - 1; i++) {
                double dx = f__.dx(i);

                auto f = f__.coeffs(i);
                auto g = g__.coeffs(i);

                T faga = f[0] * g[0];
                T fdgd = f[3] * g[3];

                T k1 = f[0] * g[1] + f[1] * g[0];
                T k2 = f[2] * g[0] + f[1] * g[1] + f[0] * g[2];
                T k3 = f[0] * g[3] + f[1] * g[2] + f[2] * g[1] + f[3] * g[0];
                T k4 = f[1] * g[3] + f[2] * g[2] + f[3] * g[1];
                T k5 = f[2] * g[3] + f[3] * g[2];

                result += dx *
                          (faga + dx * (k1 / 2.0 +
                                        dx * (k2 / 3.0 +
                                              dx * (k3 / 4.0 + dx * (k4 / 5.0 + dx * (k5 / 6.0 + dx * fdgd / 7.0))))));
            }
            break;
        }
        case 1: {
            for (int i = 0; i < num_points__ - 1; i++) {
                double x0 = f__[i];
                double dx = f__.dx(i);

                auto f = f__.coeffs(i);
                auto g = g__.coeffs(i);

                T faga = f[0] * g[0];
                T fdgd = f[3] * g[3];

                T k1 = f[0] * g[1] + f[1] * g[0];
                T k2 = f[2] * g[0] + f[1] * g[1] + f[0] * g[2];
                T k3 = f[0] * g[3] + f[1] * g[2] + f[2] * g[1] + f[3] * g[0];
                T k4 = f[1] * g[3] + f[2] * g[2] + f[3] * g[1];
                T k5 = f[2] * g[3] + f[3] * g[2];

                result += dx * ((faga * x0) +
                                dx * ((faga + k1 * x0) / 2.0 +
                                      dx * ((k1 + k2 * x0) / 3.0 +
                                            dx * ((k2 + k3 * x0) / 4.0 +
                                                  dx * ((k3 + k4 * x0) / 5.0 +
                                                        dx * ((k4 + k5 * x0) / 6.0 +
                                                              dx * ((k5 + fdgd * x0) / 7.0 + dx * fdgd / 8.0)))))));
            }
            break;
        }
        case 2: {
            for (int i = 0; i < num_points__ - 1; i++) {
                double x0 = f__[i];
                double dx = f__.dx(i);

                auto f = f__.coeffs(i);
                auto g = g__.coeffs(i);

                T k0 = f[0] * g[0];
                T k1 = f[3] * g[1] + f[2] * g[2] + f[1] * g[3];
                T k2 = f[3] * g[0] + f[2] * g[1] + f[1] * g[2] + f[0] * g[3];
                T k3 = f[2] * g[0] + f[1] * g[1] + f[0] * g[2];
                T k4 = f[3] * g[2] + f[2] * g[3];
                T k5 = f[1] * g[0] + f[0] * g[1];
                T k6 = f[3] * g[3]; // 25 OPS

                T r1 = k4 * 0.125 + k6 * x0 * 0.25;
                T r2 = (k1 + x0 * (2.0 * k4 + k6 * x0)) * 0.14285714285714285714;
                T r3 = (k2 + x0 * (2.0 * k1 + k4 * x0)) * 0.16666666666666666667;
                T r4 = (k3 + x0 * (2.0 * k2 + k1 * x0)) * 0.2;
                T r5 = (k5 + x0 * (2.0 * k3 + k2 * x0)) * 0.25;
                T r6 = (k0 + x0 * (2.0 * k5 + k3 * x0)) * 0.33333333333333333333;
                T r7 = (x0 * (2.0 * k0 + x0 * k5)) * 0.5;

                T v = dx * k6 * 0.11111111111111111111;
                v   = dx * (r1 + v);
                v   = dx * (r2 + v);
                v   = dx * (r3 + v);
                v   = dx * (r4 + v);
                v   = dx * (r5 + v);
                v   = dx * (r6 + v);
                v   = dx * (r7 + v);

                result += dx * (k0 * x0 * x0 + v);
            }
            break;
        }
            /* canonical formula derived with Mathematica */
            /*case 2:
            {
                for (int i = 0; i < num_points__ - 1; i++)
                {
                    double x0 = f__.x(i);
                    double dx = f__.dx(i);

                    auto f = f__.coefs(i);
                    auto g = g__.coefs(i);

                    T k0 = f[0] * g[0];
                    T k1 = f[3] * g[1] + f[2] * g[2] + f[1] * g[3];
                    T k2 = f[3] * g[0] + f[2] * g[1] + f[1] * g[2] + f[0] * g[3];
                    T k3 = f[2] * g[0] + f[1] * g[1] + f[0] * g[2];
                    T k4 = f[3] * g[2] + f[2] * g[3];
                    T k5 = f[1] * g[0] + f[0] * g[1];
                    T k6 = f[3] * g[3]; // 25 OPS

                    result += dx * (k0 * x0 * x0 +
                              dx * ((x0 * (2.0 * k0 + x0 * k5)) / 2.0 +
                              dx * ((k0 + x0 * (2.0 * k5 + k3 * x0)) / 3.0 +
                              dx * ((k5 + x0 * (2.0 * k3 + k2 * x0)) / 4.0 +
                              dx * ((k3 + x0 * (2.0 * k2 + k1 * x0)) / 5.0 +
                              dx * ((k2 + x0 * (2.0 * k1 + k4 * x0)) / 6.0 +
                              dx * ((k1 + x0 * (2.0 * k4 + k6 * x0)) / 7.0 +
                              dx * ((k4 + 2.0 * k6 * x0) / 8.0 +
                              dx * k6 / 9.0))))))));
                }
                break;

            }*/

        default: {
            RTE_THROW("wrong r^m prefactor");
        }
    }
    return result;
}

template <typename T>
T
inner(Spline<T> const& f__, Spline<T> const& g__, int m__)
{
    return inner(f__, g__, m__, f__.num_points());
}

}; // namespace sirius

#endif // __SPLINE_H__
