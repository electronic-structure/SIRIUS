// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

#include "linalg.hpp"

// TODO: add back() method like in std::vector

// TODO: [?] store radial grid, not the pointer to the grid.

namespace sirius {

/// Cubic spline with a not-a-knot boundary conditions.
/** The following convention for spline coefficients is used: between points 
 *  \f$ x_i \f$ and \f$ x_{i+1} \f$ the value of the spline is equal to 
 *  \f$ a_i + b_i(x_{i+1} - x_i) + c_i(x_{i+1}-x_i)^2 + d_i(x_{i+1}-x_i)^3 \f$. 
 */
template <typename T> 
class Spline
{
    private:
        
        /// Radial grid.
        Radial_grid const* radial_grid_{nullptr};

        mdarray<T, 2> coeffs_;

        /* forbid copy constructor */
        Spline(Spline<T> const& src__) = delete;
        /* forbid assigment operator */
        Spline<T>& operator=(Spline<T> const& src__) = delete;

    public:

        /// Default constructor.
        Spline()
        {
        }
        
        /// Constructor of a new empty spline.
        Spline(Radial_grid const& radial_grid__) : radial_grid_(&radial_grid__)
        {
            coeffs_ = mdarray<T, 2>(num_points(), 4);
            coeffs_.zero();
        }

        /// Constructor of a spline from a function.
        Spline(Radial_grid const& radial_grid__, std::function<T(double)> f__) : radial_grid_(&radial_grid__)
        {
            coeffs_ = mdarray<T, 2>(num_points(), 4);
            for (int i = 0; i < num_points(); i++) {
                double x = (*radial_grid_)[i];
                coeffs_(i, 0) = f__(x);
            }
            interpolate();
        }

        /// Constructor of a spline from a list of values.
        Spline(Radial_grid const& radial_grid__, std::vector<T> const& y__) : radial_grid_(&radial_grid__)
        {
            assert(radial_grid_->num_points() == (int)y__.size());
            coeffs_ = mdarray<T, 2>(num_points(), 4);
            for (int i = 0; i < num_points(); i++) {
                coeffs_(i, 0) = y__[i];
            }
            interpolate();
        }

        /// Move constructor.
        Spline(Spline<T>&& src__)
        {
            radial_grid_ = src__.radial_grid_;
            coeffs_ = std::move(src__.coeffs_);
        }
    
        /// Move assigment operator.
        Spline<T>& operator=(Spline<T>&& src__)
        {
            if (this != &src__)
            {
                radial_grid_ = src__.radial_grid_;
                coeffs_ = std::move(src__.coeffs_);
            }
            return *this;
        }

        Spline<T>& operator=(std::function<T(double)> f__)
        {
            for (int ir = 0; ir < radial_grid_->num_points(); ir++)
            {
                double x = (*radial_grid_)[ir];
                coeffs_(ir, 0) = f__(x);
            }
            return this->interpolate();
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

        inline T operator()(double x) const
        {
            int np = num_points();

            assert(x <= (*radial_grid_)[np - 1]);
            
            if (x >= (*radial_grid_)[0])
            {
                int j = np - 1;
                for (int i = 0; i < np - 1; i++)
                {
                    if (x < (*radial_grid_)[i + 1])
                    {
                        j = i;
                        break;
                    }
                }
                if (j == np - 1) 
                {
                    return coeffs_(np - 1, 0);
                }
                else
                {
                    double dx = x - (*radial_grid_)[j];
                    return (*this)(j, dx);
                }
            }
            else
            {
                double dx = x - (*radial_grid_)[0];
                return (*this)(0, dx);
            }
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

        inline T operator()(const int i, double dx) const
        {
            assert(i >= 0);
            assert(i < num_points() - 1);
            return coeffs_(i, 0) + dx * (coeffs_(i, 1) + dx * (coeffs_(i, 2) + dx * coeffs_(i, 3)));
        }
        
        inline T deriv(const int dm, const int i, const double dx) const
        {
            assert(i >= 0);
            assert(i < num_points() - 1);

            switch (dm)
            {
                case 0:
                {
                    return coeffs_(i, 0) + dx * (coeffs_(i, 1) + dx * (coeffs_(i, 2) + dx * coeffs_(i, 3)));
                    break;
                }
                case 1:
                {
                    return coeffs_(i, 1) + (coeffs_(i, 2) * 2.0 + coeffs_(i, 3) * dx * 3.0) * dx;
                    break;
                }
                case 2:
                {
                    return coeffs_(i, 2) * 2.0 + coeffs_(i, 3) * dx * 6.0;
                    break;
                }
                case 3:
                {
                    return coeffs_(i, 3) * 6.0;
                    break;
                }
                default:
                {
                    TERMINATE("wrong order of derivative");
                    return 0.0; // make compiler happy
                    break;
                }
            }
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

        inline Radial_grid const& radial_grid() const
        {
            return *radial_grid_;
        }

        Spline<T>& interpolate()
        {
            int np = num_points();

            /* lower diagonal */
            std::vector<T> a(np - 1);
            /* main diagonal */
            std::vector<T> b(np);
            /* upper diagonal */
            std::vector<T> c(np - 1);

            std::vector<T> d(np);
            std::vector<T> dy(np - 1);
            
            /* derivative of y */
            for (int i = 0; i < np - 1; i++) dy[i] = (coeffs_(i + 1, 0) - coeffs_(i, 0)) / radial_grid_->dx(i);
            
            /* setup "B" vector of AX=B equation */
            for (int i = 0; i < np - 2; i++) d[i + 1] = (dy[i + 1] - dy[i]) * 6.0;
            
            d[0] = -d[1];
            d[np - 1] = -d[np - 2];
            
            /* main diagonal of "A" matrix */
            for (int i = 0; i < np - 2; i++) b[i + 1] = 2 * (radial_grid_->dx(i) + radial_grid_->dx(i + 1));
            double h0 = radial_grid_->dx(0);
            double h1 = radial_grid_->dx(1);
            double h2 = radial_grid_->dx(np - 2);
            double h3 = radial_grid_->dx(np - 3);
            b[0] = (h1 / h0) * h1 - h0;
            b[np - 1] = (h3 / h2) * h3 - h2;

            /* subdiagonals of "A" matrix */
            for (int i = 0; i < np - 1; i++)
            {
                c[i] = radial_grid_->dx(i);
                a[i] = radial_grid_->dx(i);
            }
            c[0] = -(h1 * (1 + h1 / h0) + b[1]);
            a[np - 2] = -(h3 * (1 + h3 / h2) + b[np - 2]); 

            /* solve tridiagonal system */
            int info = linalg<CPU>::gtsv(np, 1, &a[0], &b[0], &c[0], &d[0], np);
            auto x = d;
            
            if (info)
            {
                std::stringstream s;
                s << "gtsv returned " << info;
                TERMINATE(s);
            }
            
            for (int i = 0; i < np - 1; i++)
            {
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
            for (int i = 0; i < num_points(); i++)
            {
                coeffs_(i, 0) *= a__;
                coeffs_(i, 1) *= a__;
                coeffs_(i, 2) *= a__;
                coeffs_(i, 3) *= a__;
            }
        }

        T integrate(std::vector<T>& g__, int m__) const
        {
            g__ = std::vector<T>(num_points());

            g__[0] = 0.0;

            switch (m__)
            {
                case 0:
                {
                    double t = 1.0 / 3.0;
                    for (int i = 0; i < num_points() - 1; i++)
                    {
                        double dx = radial_grid_->dx(i);
                        g__[i + 1] = g__[i] + (((coeffs_(i, 3) * dx * 0.25 + coeffs_(i, 2) * t) * dx + coeffs_(i, 1) * 0.5) * dx + coeffs_(i, 0)) * dx;
                    }
                    break;
                }
                case 2:
                {
                    for (int i = 0; i < num_points() - 1; i++)
                    {
                        double x0 = (*radial_grid_)[i];
                        double x1 = (*radial_grid_)[i + 1];
                        double dx = radial_grid_->dx(i);
                        T a0 = coeffs_(i, 0);
                        T a1 = coeffs_(i, 1);
                        T a2 = coeffs_(i, 2);
                        T a3 = coeffs_(i, 3);

                        double x0_2 = x0 * x0;
                        double x0_3 = x0_2 * x0;
                        double x1_2 = x1 * x1;
                        double x1_3 = x1_2 * x1;

                        g__[i + 1] = g__[i] + (20.0 * a0 * (x1_3 - x0_3) + 5.0 * a1 * (x0 * x0_3 + x1_3 * (3.0 * dx - x0)) - 
                                     dx * dx * dx * (-2.0 * a2 * (x0_2 + 3.0 * x0 * x1 + 6.0 * x1_2) - 
                                     a3 * dx * (x0_2 + 4.0 * x0 * x1 + 10.0 * x1_2))) / 60.0;
                    }
                    break;
                }
                case -1:
                {
                    for (int i = 0; i < num_points() - 1; i++)
                    {
                        double x0 = (*radial_grid_)[i];
                        double x1 = (*radial_grid_)[i + 1];
                        T a0 = coeffs_(i, 0);
                        T a1 = coeffs_(i, 1);
                        T a2 = coeffs_(i, 2);
                        T a3 = coeffs_(i, 3);

                        // obtained with the following Mathematica code:
                        //   FullSimplify[Integrate[x^(-1)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                        //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                        g__[i + 1] = g__[i] + (-((x0 - x1) * (6.0 * a1 - 9.0 * a2 * x0 + 11.0 * a3 * std::pow(x0, 2) + 
                                     3.0 * a2 * x1 - 7.0 * a3 * x0 * x1 + 2.0 * a3 * std::pow(x1, 2))) / 6.0 + 
                                     (-a0 + x0 * (a1 - a2 * x0 + a3 * std::pow(x0, 2))) * std::log(x0 / x1));
                    }
                    break;
                }
                case -2:
                {
                    for (int i = 0; i < num_points() - 1; i++)
                    {
                        double x0 = (*radial_grid_)[i];
                        double x1 = (*radial_grid_)[i + 1];
                        T a0 = coeffs_(i, 0);
                        T a1 = coeffs_(i, 1);
                        T a2 = coeffs_(i, 2);
                        T a3 = coeffs_(i, 3);

                        // obtained with the following Mathematica code:
                        //   FullSimplify[Integrate[x^(-2)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                        //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                        g__[i + 1] = g__[i] + (((x0 - x1) * (-2.0 * a0 + x0 * (2.0 * a1 - 2.0 * a2 * (x0 + x1) + 
                                     a3 * (2.0 * std::pow(x0, 2) + 5.0 * x0 * x1 - std::pow(x1, 2)))) + 
                                     2.0 * x0 * (a1 + x0 * (-2.0 * a2 + 3.0 * a3 * x0)) * x1 * std::log(x1 / x0)) / 
                                     (2.0 * x0 * x1));
                    }
                    break;
                }
                case -3:
                {
                    for (int i = 0; i < num_points() - 1; i++)
                    {
                        double x0 = (*radial_grid_)[i];
                        double x1 = (*radial_grid_)[i + 1];
                        T a0 = coeffs_(i, 0);
                        T a1 = coeffs_(i, 1);
                        T a2 = coeffs_(i, 2);
                        T a3 = coeffs_(i, 3);

                        // obtained with the following Mathematica code:
                        //   FullSimplify[Integrate[x^(-3)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                        //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                        g__[i + 1] = g__[i] + (-((x0 - x1) * (a0 * (x0 + x1) + x0 * (a1 * (-x0 + x1) + 
                                     x0 * (a2 * x0 - a3 * std::pow(x0, 2) - 3.0 * a2 * x1 + 5.0 * a3 * x0 * x1 + 
                                     2.0 * a3 * std::pow(x1, 2)))) + 2.0 * std::pow(x0, 2) * (a2 - 3.0 * a3 * x0) * std::pow(x1, 2) * 
                                     std::log(x0 / x1)) / (2.0 * std::pow(x0, 2) * std::pow(x1, 2)));
                    }
                    break;
                }
                case -4:
                {
                    for (int i = 0; i < num_points() - 1; i++)
                    {
                        double x0 = (*radial_grid_)[i];
                        double x1 = (*radial_grid_)[i + 1];
                        T a0 = coeffs_(i, 0);
                        T a1 = coeffs_(i, 1);
                        T a2 = coeffs_(i, 2);
                        T a3 = coeffs_(i, 3);

                        // obtained with the following Mathematica code:
                        //   FullSimplify[Integrate[x^(-4)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                        //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                        g__[i + 1] = g__[i] + ((2.0 * a0 * (-std::pow(x0, 3) + std::pow(x1, 3)) + 
                                     x0 * (x0 - x1) * (a1 * (x0 - x1) * (2.0 * x0 + x1) + 
                                     x0 * (-2.0 * a2 * std::pow(x0 - x1, 2) + a3 * x0 * (2.0 * std::pow(x0, 2) - 7.0 * x0 * x1 + 
                                     11.0 * std::pow(x1, 2)))) + 6.0 * a3 * std::pow(x0 * x1, 3) * std::log(x1 / x0)) / 
                                     (6.0 * std::pow(x0 * x1, 3)));
                    }
                    break;
                }
                default:
                {
                    for (int i = 0; i < num_points() - 1; i++)
                    {
                        double x0 = (*radial_grid_)[i];
                        double x1 = (*radial_grid_)[i + 1];
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
    assert(a__.radial_grid().hash() == b__.radial_grid().hash());
    Spline<double> s12(a__.radial_grid());

    auto& coeffs_a = a__.coeffs();
    auto& coeffs_b = b__.coeffs();
    auto& coeffs = const_cast<mdarray<double, 2>&>(s12.coeffs());

    for (int ir = 0; ir < a__.radial_grid().num_points(); ir++)
    {
        coeffs(ir, 0) = coeffs_a(ir, 0) * coeffs_b(ir, 0);
        coeffs(ir, 1) = coeffs_a(ir, 1) * coeffs_b(ir, 0) + coeffs_a(ir, 0) * coeffs_b(ir, 1);
        coeffs(ir, 2) = coeffs_a(ir, 2) * coeffs_b(ir, 0) + coeffs_a(ir, 1) * coeffs_b(ir, 1) + coeffs_a(ir, 0) * coeffs_b(ir, 2);
        coeffs(ir, 3) = coeffs_a(ir, 3) * coeffs_b(ir, 0) + coeffs_a(ir, 2) * coeffs_b(ir, 1) + coeffs_a(ir, 1) * coeffs_b(ir, 2) + coeffs_a(ir, 0) * coeffs_b(ir, 3);
    }

    return std::move(s12);
}

extern "C" double spline_inner_product_gpu_v2(int size__, double const* x__, double const* dx__, double const* f__, 
                                              double const* g__, double* d_buf__, double* h_buf__, int stream_id__);

template<typename T>
T inner(Spline<T> const& f__, Spline<T> const& g__, int m__, int num_points__)
{
    assert(f__.radial_grid().hash() == g__.radial_grid().hash());
    
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
