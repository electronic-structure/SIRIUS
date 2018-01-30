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

/** \file spline.h
 *   
 *  \brief Contains definition and partial implementaiton of sirius::Spline class.
 */

#ifndef __SPLINE_H__
#define __SPLINE_H__

#include "radial_grid.h"

// TODO: add back() method like in std::vector

// TODO: [?] store radial grid, not the pointer to the grid.

namespace sirius {

/// Cubic spline with a not-a-knot boundary conditions.
/** The following convention for spline coefficients is used: between points 
 *  \f$ x_i \f$ and \f$ x_{i+1} \f$ the value of the spline is equal to 
 *  \f$ a_i + b_i(x_{i+1} - x_i) + c_i(x_{i+1}-x_i)^2 + d_i(x_{i+1}-x_i)^3 \f$. 
 */
template <typename T, typename U = double> 
class Spline
{
    private:
        
        /// Radial grid.
        Radial_grid<U> const* radial_grid_{nullptr};

        mdarray<T, 2> coeffs_;

        /* forbid copy constructor */
        Spline(Spline<T, U> const& src__) = delete;
        /* forbid assigment operator */
        Spline<T, U>& operator=(Spline<T, U> const& src__) = delete;

        /// Solver tridiagonal system of linear equaitons.
        int solve(T* dl, T* d, T* du, T* b, int n)
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
                    T mult = d[i] / dl[i];
                    d[i] = dl[i];
                    T tmp = d[i + 1];
                    d[i + 1] = du[i] - mult * tmp;
                    if (i < n - 2) {
                        dl[i] = du[i + 1];
                        du[i + 1] = -mult * dl[i];
                    }
                    du[i] = tmp;
                    tmp = b[i];
                    b[i] = b[i + 1];
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

    public:

        /// Default constructor.
        Spline()
        {
        }
        
        /// Constructor of a new empty spline.
        Spline(Radial_grid<U> const& radial_grid__) 
            : radial_grid_(&radial_grid__)
        {
            coeffs_ = mdarray<T, 2>(num_points(), 4);
            coeffs_.zero();
        }

        /// Constructor of a spline from a function.
        Spline(Radial_grid<U> const& radial_grid__, std::function<T(U)> f__)
            : radial_grid_(&radial_grid__)
        {
            coeffs_ = mdarray<T, 2>(num_points(), 4);
            for (int i = 0; i < num_points(); i++) {
                U x = (*radial_grid_)[i];
                coeffs_(i, 0) = f__(x);
            }
            interpolate();
        }

        /// Constructor of a spline from a list of values.
        Spline(Radial_grid<U> const& radial_grid__, std::vector<T> const& y__)
            : radial_grid_(&radial_grid__)
        {
            assert(static_cast<int>(y__.size()) <= num_points());
            coeffs_ = mdarray<T, 2>(num_points(), 4);
            coeffs_.zero();
            int i{0};
            for (auto e: y__) {
                this->coeffs_(i++, 0) = e;
            }
            this->interpolate();
        }

        /// Move constructor.
        Spline(Spline<T, U>&& src__)
        {
            radial_grid_ = src__.radial_grid_;
            coeffs_      = std::move(src__.coeffs_);
        }
    
        /// Move assigment operator.
        Spline<T, U>& operator=(Spline<T, U>&& src__)
        {
            if (this != &src__) {
                radial_grid_ = src__.radial_grid_;
                coeffs_      = std::move(src__.coeffs_);
            }
            return *this;
        }

        Spline<T, U>& operator=(std::function<T(U)> f__)
        {
            for (int ir = 0; ir < radial_grid_->num_points(); ir++) {
                U x = (*radial_grid_)[ir];
                coeffs_(ir, 0) = f__(x);
            }
            return this->interpolate();
        }

        void operator=(std::vector<T> const& y__)
        {
            assert(static_cast<int>(y__.size()) <= num_points());
            coeffs_.zero();
            int i{0};
            for (auto e: y__) {
                this->coeffs_(i++, 0) = e;
            }
            this->interpolate();
        }
        
        /// Integrate with r^m weight.
        T integrate(int m__) const
        {
            std::vector<T> g(num_points());
            return integrate(g, m__);
        }
        
        inline std::vector<T> values() const
        {
            std::vector<T> a(num_points());
            for (int i = 0; i < num_points(); i++) {
                a[i] = coeffs_(i, 0);
            }
            return std::move(a);
        }
        
        /// Return number of spline points.
        inline int num_points() const
        {
            return radial_grid_->num_points();
        }

        inline std::array<T, 4> coeffs(int i__) const
        {
            return {coeffs_(i__, 0), coeffs_(i__, 1), coeffs_(i__, 2), coeffs_(i__, 3)};
        }

        inline mdarray<T, 2> const& coeffs() const
        {
            return coeffs_;
        }

        inline double x(int i__) const
        {
            return (*radial_grid_)[i__];
        }

        inline double dx(int i__) const
        {
            return radial_grid_->dx(i__);
        }

        inline T operator()(U x) const
        {
            int j = radial_grid_->index_of(x);
            if (j == -1) {
                TERMINATE("point not found");
            }
            U dx = x - (*radial_grid_)[j];
            return (*this)(j, dx);
        }
        
        /// Return value at \f$ x_i \f$.
        inline T& operator[](const int i)
        {
            return coeffs_(i, 0);
        }

        inline T operator[](const int i) const
        {
            return coeffs_(i, 0);
        }

        inline T operator()(const int i, U dx) const
        {
            assert(i >= 0);
            assert(i < num_points() - 1);
            assert(dx >= 0);
            return coeffs_(i, 0) + dx * (coeffs_(i, 1) + dx * (coeffs_(i, 2) + dx * coeffs_(i, 3)));
        }
        
        inline T deriv(const int dm, const int i, const U dx) const
        {
            assert(i >= 0);
            assert(i < num_points() - 1);
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
                    TERMINATE("wrong order of derivative");
                    break;
                }
            }
            return result;
        }

        inline T deriv(int dm, int i) const
        {
            assert(i >= 0);
            assert(i < num_points());
            assert(radial_grid_ != nullptr);

            if (i == num_points() - 1) {
                return deriv(dm, i - 1, radial_grid_->dx(i - 1));
            } else {
                return deriv(dm, i, 0);
            }
        }

        inline Radial_grid<U> const& radial_grid() const
        {
            return *radial_grid_;
        }

        Spline<T, U>& interpolate()
        {
            int np = num_points();

            /* lower diagonal */
            std::vector<T> dl(np - 1);
            /* main diagonal */
            std::vector<T> d(np);
            /* upper diagonal */
            std::vector<T> du(np - 1);

            std::vector<T> x(np);
            std::vector<T> dy(np - 1);
            
            /* derivative of y */
            for (int i = 0; i < np - 1; i++) {
                dy[i] = (coeffs_(i + 1, 0) - coeffs_(i, 0)) / radial_grid_->dx(i);
            }
            
            /* setup "B" vector of AX=B equation */
            for (int i = 0; i < np - 2; i++) {
                x[i + 1] = (dy[i + 1] - dy[i]) * 6.0;
            }
            
            x[0] = -x[1];
            x[np - 1] = -x[np - 2];
            
            /* main diagonal of "A" matrix */
            for (int i = 0; i < np - 2; i++) {
                d[i + 1] = static_cast<T>(2) * (static_cast<T>(radial_grid_->dx(i)) + static_cast<T>(radial_grid_->dx(i + 1)));
            }
            U h0 = radial_grid_->dx(0);
            U h1 = radial_grid_->dx(1);
            U h2 = radial_grid_->dx(np - 2);
            U h3 = radial_grid_->dx(np - 3);
            d[0] = (h1 / h0) * h1 - h0;
            d[np - 1] = (h3 / h2) * h3 - h2;

            /* subdiagonals of "A" matrix */
            for (int i = 0; i < np - 1; i++) {
                du[i] = static_cast<T>(radial_grid_->dx(i));
                dl[i] = static_cast<T>(radial_grid_->dx(i));
            }
            du[0] = -(h1 * (1.0 + h1 / h0) + d[1]);
            dl[np - 2] = -(h3 * (1.0 + h3 / h2) + d[np - 2]); 

            /* solve tridiagonal system */
            //solve(a.data(), b.data(), c.data(), d.data(), np);
            //auto& x = d;
            //int info = linalg<CPU>::gtsv(np, 1, &a[0], &b[0], &c[0], &d[0], np);
            
            int info = solve(&dl[0], &d[0], &du[0], &x[0], np);
            
            if (info) {
                std::stringstream s;
                s << "error in tridiagonal solver: " << info;
                TERMINATE(s);
            }
            
            for (int i = 0; i < np - 1; i++) {
                coeffs_(i, 2) = x[i] / 2.0;
                T t = (x[i + 1] - x[i]) / 6.0;
                coeffs_(i, 1) = dy[i] - (coeffs_(i, 2) + t) * radial_grid_->dx(i);
                coeffs_(i, 3) = t / radial_grid_->dx(i);
            }
            coeffs_(np - 1, 1) = 0;
            coeffs_(np - 1, 2) = 0;
            coeffs_(np - 1, 3) = 0;

            return *this;
        }

        inline void scale(double a__)
        {
            for (int i = 0; i < num_points(); i++) {
                coeffs_(i, 0) *= a__;
                coeffs_(i, 1) *= a__;
                coeffs_(i, 2) *= a__;
                coeffs_(i, 3) *= a__;
            }
        }

        T integrate_simpson() const
        {
            std::vector<U> w(num_points(), 0);

            for (int i = 0; i < num_points() - 2; i++) {
                U x0 = (*radial_grid_)[i];
                U x1 = (*radial_grid_)[i + 1];
                U x2 = (*radial_grid_)[i + 2];
                w[i] += (2 * x0 + x1 - 3 * x2) * (x1 - x0) / (x0 - x2) / 6;
                w[i + 1] += (x0 - x1) * (x0 + 2 * x1 - 3 * x2) / 6 / (x2 - x1);
                w[i + 2] += std::pow(x0 - x1, 3) / 6 / (x2 - x0) / (x2 - x1);
            }
            //for (int i = 1; i < num_points() - 1; i++) {
            //    w[i] *= 0.5;
            //}

            T res{0};
            for (int i = 0; i < num_points(); i++) {
                res += w[i] * coeffs_(i, 0);
            }
            return res;
        }

        T integrate_simple() const
        {
            T res{0};
            for (int i = 0; i < num_points() - 1; i++) {
                U dx = radial_grid_->dx(i);
                res += 0.5 * (coeffs_(i, 0) + coeffs_(i + 1, 0)) * dx;
            }
            return res;
        }

        T integrate(std::vector<T>& g__, int m__) const
        {
            g__ = std::vector<T>(num_points());

            g__[0] = 0.0;

            switch (m__) {
                case 0: {
                    T t = 1.0 / 3.0;
                    for (int i = 0; i < num_points() - 1; i++) {
                        U dx = radial_grid_->dx(i);
                        g__[i + 1] = g__[i] + (((coeffs_(i, 3) * dx * 0.25 + coeffs_(i, 2) * t) * dx + coeffs_(i, 1) * 0.5) * dx + coeffs_(i, 0)) * dx;
                    }
                    break;
                }
                case 2: {
                    for (int i = 0; i < num_points() - 1; i++) {
                        U x0 = (*radial_grid_)[i];
                        U x1 = (*radial_grid_)[i + 1];
                        U dx = radial_grid_->dx(i);
                        T a0 = coeffs_(i, 0);
                        T a1 = coeffs_(i, 1);
                        T a2 = coeffs_(i, 2);
                        T a3 = coeffs_(i, 3);

                        U x0_2 = x0 * x0;
                        U x0_3 = x0_2 * x0;
                        U x1_2 = x1 * x1;
                        U x1_3 = x1_2 * x1;

                        g__[i + 1] = g__[i] + (20.0 * a0 * (x1_3 - x0_3) + 5.0 * a1 * (x0 * x0_3 + x1_3 * (3.0 * dx - x0)) - 
                                     dx * dx * dx * (-2.0 * a2 * (x0_2 + 3.0 * x0 * x1 + 6.0 * x1_2) - 
                                     a3 * dx * (x0_2 + 4.0 * x0 * x1 + 10.0 * x1_2))) / 60.0;
                    }
                    break;
                }
                case -1: {
                    for (int i = 0; i < num_points() - 1; i++) {
                        U x0 = (*radial_grid_)[i];
                        U x1 = (*radial_grid_)[i + 1];
                        U dx = radial_grid_->dx(i);
                        T a0 = coeffs_(i, 0);
                        T a1 = coeffs_(i, 1);
                        T a2 = coeffs_(i, 2);
                        T a3 = coeffs_(i, 3);

                        // obtained with the following Mathematica code:
                        //   FullSimplify[Integrate[x^(-1)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                        //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                        g__[i + 1] = g__[i] + (dx / 6.0) * (6.0 * a1 + x0 * (-9.0 * a2  + 11.0 * a3 * x0) + x1 * (3.0 * a2 - 7.0 * a3 * x0 + 2.0 * a3 * x1)) + 
                                     (-a0 + x0 * (a1 + x0 * (-a2 + a3 * x0))) * std::log(x0 / x1);
                    }
                    break;
                }
                case -2: {
                    for (int i = 0; i < num_points() - 1; i++) {
                        U x0 = (*radial_grid_)[i];
                        U x1 = (*radial_grid_)[i + 1];
                        U dx = radial_grid_->dx(i);
                        T a0 = coeffs_(i, 0);
                        T a1 = coeffs_(i, 1);
                        T a2 = coeffs_(i, 2);
                        T a3 = coeffs_(i, 3);

                        // obtained with the following Mathematica code:
                        //   FullSimplify[Integrate[x^(-2)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                        //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                        //g__[i + 1] = g__[i] + (((x0 - x1) * (-2.0 * a0 + x0 * (2.0 * a1 - 2.0 * a2 * (x0 + x1) + 
                        //             a3 * (2.0 * std::pow(x0, 2) + 5.0 * x0 * x1 - std::pow(x1, 2)))) + 
                        //             2.0 * x0 * (a1 + x0 * (-2.0 * a2 + 3.0 * a3 * x0)) * x1 * std::log(x1 / x0)) / 
                        //             (2.0 * x0 * x1));
                        g__[i + 1] = g__[i] + (a2 * dx - 5.0 * a3 * x0 * dx / 2.0 - a1 * (dx / x1) + a0 * (dx / x0 / x1)  + 
                                              (x0 / x1) * dx * (a2 - a3 * x0) + a3 * x1 * dx / 2.0) + 
                                              (a1 + x0 * (-2.0 * a2 + 3.0 * a3 * x0)) * std::log(x1 / x0);
                    }
                    break;
                }
                case -3: {
                    for (int i = 0; i < num_points() - 1; i++) {
                        U x0 = (*radial_grid_)[i];
                        U x1 = (*radial_grid_)[i + 1];
                        U dx = radial_grid_->dx(i);
                        T a0 = coeffs_(i, 0);
                        T a1 = coeffs_(i, 1);
                        T a2 = coeffs_(i, 2);
                        T a3 = coeffs_(i, 3);

                        // obtained with the following Mathematica code:
                        //   FullSimplify[Integrate[x^(-3)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                        //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                        //g__[i + 1] = g__[i] + (-((x0 - x1) * (a0 * (x0 + x1) + x0 * (a1 * (-x0 + x1) + 
                        //             x0 * (a2 * x0 - a3 * std::pow(x0, 2) - 3.0 * a2 * x1 + 5.0 * a3 * x0 * x1 + 
                        //             2.0 * a3 * std::pow(x1, 2)))) + 2.0 * std::pow(x0, 2) * (a2 - 3.0 * a3 * x0) * std::pow(x1, 2) * 
                        //             std::log(x0 / x1)) / (2.0 * std::pow(x0, 2) * std::pow(x1, 2)));
                        g__[i + 1] = g__[i] + dx * (a0 * (x0 + x1) + x0 * (a1 * dx + 
                                     x0 * (a2 * x0 - a3 * std::pow(x0, 2) - 3.0 * a2 * x1 + 5.0 * a3 * x0 * x1 + 
                                     2.0 * a3 * std::pow(x1, 2)))) / std::pow(x0 * x1, 2) / 2.0 + 
                                     (-a2 + 3.0 * a3 * x0) * std::log(x0 / x1);
                    }
                    break;
                }
                case -4: {
                    for (int i = 0; i < num_points() - 1; i++) {
                        U x0 = (*radial_grid_)[i];
                        U x1 = (*radial_grid_)[i + 1];
                        U dx = radial_grid_->dx(i);
                        T a0 = coeffs_(i, 0);
                        T a1 = coeffs_(i, 1);
                        T a2 = coeffs_(i, 2);
                        T a3 = coeffs_(i, 3);

                        // obtained with the following Mathematica code:
                        //   FullSimplify[Integrate[x^(-4)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                        //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                        //g__[i + 1] = g__[i] + ((2.0 * a0 * (-std::pow(x0, 3) + std::pow(x1, 3)) + 
                        //             x0 * (x0 - x1) * (a1 * (x0 - x1) * (2.0 * x0 + x1) + 
                        //             x0 * (-2.0 * a2 * std::pow(x0 - x1, 2) + a3 * x0 * (2.0 * std::pow(x0, 2) - 7.0 * x0 * x1 + 
                        //             11.0 * std::pow(x1, 2)))) + 6.0 * a3 * std::pow(x0 * x1, 3) * std::log(x1 / x0)) / 
                        //             (6.0 * std::pow(x0 * x1, 3)));
                        g__[i + 1] = g__[i] + (2.0 * a0 * (-std::pow(x0, 3) + std::pow(x1, 3)) -
                                     x0 * dx * (-a1 * dx * (2.0 * x0 + x1) + 
                                     x0 * (-2.0 * a2 * std::pow(dx, 2) + a3 * x0 * (2.0 * std::pow(x0, 2) - 7.0 * x0 * x1 + 
                                     11.0 * std::pow(x1, 2))))) / std::pow(x0 * x1, 3) / 6.0 + 
                                     a3 * std::log(x1 / x0);
                    }
                    break;
                }
                default: {
                    for (int i = 0; i < num_points() - 1; i++) {
                        U x0 = (*radial_grid_)[i];
                        U x1 = (*radial_grid_)[i + 1];
                        T a0 = coeffs_(i, 0);
                        T a1 = coeffs_(i, 1);
                        T a2 = coeffs_(i, 2);
                        T a3 = coeffs_(i, 3);

                        // obtained with the following Mathematica code:
                        //   FullSimplify[Integrate[x^(m)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}], 
                        //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                        g__[i + 1] = g__[i] + (std::pow(x0, 1 + m__) * (-(a0 * double((2 + m__) * (3 + m__) * (4 + m__))) + 
                                     x0 * (a1 * double((3 + m__) * (4 + m__)) - 2.0 * a2 * double(4 + m__) * x0 + 
                                     6.0 * a3 * std::pow(x0, 2)))) / double((1 + m__) * (2 + m__) * (3 + m__) * (4 + m__)) + 
                                     std::pow(x1, 1 + m__) * ((a0 - x0 * (a1 + x0 * (-a2 + a3 * x0))) / double(1 + m__) + 
                                     ((a1 + x0 * (-2.0 * a2 + 3.0 * a3 * x0)) * x1) / double(2 + m__) + 
                                     ((a2 - 3.0 * a3 * x0) * std::pow(x1, 2)) / double(3 + m__) + 
                                     (a3 * std::pow(x1, 3)) / double(4 + m__));
                    }
                    break;
                }
            }
            
            return g__[num_points() - 1];
        }

        uint64_t hash() const
        {
            return coeffs_.hash();
        }

        #ifdef __GPU
        void copy_to_device()
        {
            coeffs_.allocate_on_device();
            coeffs_.copy_to_device();
        }

        void async_copy_to_device(int thread_id__)
        {
            coeffs_.allocate_on_device();
            coeffs_.async_copy_to_device(thread_id__);
        }
        #endif
};

template <typename T>
inline Spline<T> operator*(Spline<T> const& a__, Spline<T> const& b__)
{
    //assert(a__.radial_grid().hash() == b__.radial_grid().hash());
    Spline<double> s12(a__.radial_grid());

    auto& coeffs_a = a__.coeffs();
    auto& coeffs_b = b__.coeffs();
    auto& coeffs = const_cast<mdarray<double, 2>&>(s12.coeffs());

    for (int ir = 0; ir < a__.radial_grid().num_points(); ir++) {
        coeffs(ir, 0) = coeffs_a(ir, 0) * coeffs_b(ir, 0);
        coeffs(ir, 1) = coeffs_a(ir, 1) * coeffs_b(ir, 0) + coeffs_a(ir, 0) * coeffs_b(ir, 1);
        coeffs(ir, 2) = coeffs_a(ir, 2) * coeffs_b(ir, 0) + coeffs_a(ir, 1) * coeffs_b(ir, 1) + coeffs_a(ir, 0) * coeffs_b(ir, 2);
        coeffs(ir, 3) = coeffs_a(ir, 3) * coeffs_b(ir, 0) + coeffs_a(ir, 2) * coeffs_b(ir, 1) + coeffs_a(ir, 1) * coeffs_b(ir, 2) + coeffs_a(ir, 0) * coeffs_b(ir, 3);
    }

    return std::move(s12);
}

#ifdef __GPU
extern "C" double spline_inner_product_gpu_v2(int           size__,
                                              double const* x__,
                                              double const* dx__,
                                              double const* f__,
                                              double const* g__,
                                              double*       d_buf__,
                                              double*       h_buf__,
                                              int           stream_id__);

extern "C" void spline_inner_product_gpu_v3(int const*    idx_ri__,
                                            int           num_ri__,
                                            int           num_points__,
                                            double const* x__,
                                            double const* dx__,
                                            double const* f__, 
                                            double const* g__,
                                            double*       result__);
#endif

template<typename T>
T inner(Spline<T> const& f__, Spline<T> const& g__, int m__, int num_points__)
{
    //assert(f__.radial_grid().hash() == g__.radial_grid().hash());
    
    T result = 0;

    switch (m__)
    {
        case 0:
        {
            for (int i = 0; i < num_points__ - 1; i++)
            {
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

                result += dx * (faga + 
                          dx * (k1 / 2.0 + 
                          dx * (k2 / 3.0 + 
                          dx * (k3 / 4.0 + 
                          dx * (k4 / 5.0 + 
                          dx * (k5 / 6.0 + 
                          dx * fdgd / 7.0))))));
            }
            break;
        }
        case 1:
        {
            for (int i = 0; i < num_points__ - 1; i++)
            {
                double x0 = f__.x(i);
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
                          dx * ((k5 + fdgd * x0) / 7.0 +
                          dx * fdgd / 8.0)))))));
            }
            break;
        }
        case 2:
        {
            for (int i = 0; i < num_points__ - 1; i++)
            {
                double x0 = f__.x(i);
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
                v = dx * (r1 + v);
                v = dx * (r2 + v);
                v = dx * (r3 + v);
                v = dx * (r4 + v);
                v = dx * (r5 + v); 
                v = dx * (r6 + v);
                v = dx * (r7 + v);

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
        
        default:
        {
            TERMINATE("wrong r^m prefactor");
        }
    }
    return result;
}

template<typename T>
T inner(Spline<T> const& f__, Spline<T> const& g__, int m__)
{
    return inner(f__, g__, m__, f__.num_points());
}

};

#endif // __SPLINE_H__
