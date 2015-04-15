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

/** \file spline.h
 *   
 *  \brief Contains definition and partial implementaiton of sirius::Spline class.
 */

#ifndef __SPLINE_H__
#define __SPLINE_H__

#include "linalg.h"

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
        Radial_grid const* radial_grid_;

        mdarray<T, 2> coefs_;

        Spline(Spline<T> const& src__) = delete;

        Spline<T>& operator=(Spline<T> const& src__) = delete;

    public:

        /// Default constructor.
        Spline() : radial_grid_(nullptr)
        {
        }
        
        /// Constructor of a new uninitialized spline.
        Spline(Radial_grid const& radial_grid__) : radial_grid_(&radial_grid__)
        {
            coefs_ = mdarray<T, 2>(radial_grid_->num_points(), 4);
        }

        /// Constructor of a constant value spline.
        Spline(Radial_grid const& radial_grid__, T val__) : radial_grid_(&radial_grid__)
        {
            int np = num_points();
            coefs_ = mdarray<T, 2>(np, 4);
            for (int i = 0; i < np; i++) coefs_(i, 0) = val__;
        }
        
        /// Constructor of a spline from a list of values.
        Spline(Radial_grid const& radial_grid__, std::vector<T>& y__) : radial_grid_(&radial_grid__)
        {
            assert(radial_grid_.num_points() == (int)y__.size());
            int np = num_points();
            coefs_ = mdarray<T, 2>(np, 4);
            for (int i = 0; i < np; i++) coefs_(i, 0) = y__[i];
            interpolate();
        }

        Spline(Spline<T>&& src__)
        {
            radial_grid_ = std::move(src__.radial_grid_);
            coefs_ = std::move(src__.coefs_);
        }

        Spline<T>& operator=(Spline<T>&& src__)
        {
            if (this != &src__)
            {
                radial_grid_ = std::move(src__.radial_grid_);
                coefs_ = std::move(src__.coefs_);
            }
            return *this;
        }
    
        template <typename U> 
        friend class Spline;
        
        /// Integrate with r^m weight.
        T integrate(int m__)
        {
            std::vector<T> g(num_points());
            return integrate(g, m__);
        }
        
        inline std::vector<T> values()
        {
            std::vector<T> a(num_points());
            for (int i = 0; i < num_points(); i++) a[i] = coefs_(i, 0);
            return a;
        }
        
        /// Return number of spline points.
        inline int num_points()
        {
            return radial_grid_->num_points();
        }

        inline std::array<T, 4> coefs(int i__)
        {
            return {coefs_(i__, 0), coefs_(i__, 1), coefs_(i__, 2), coefs_(i__, 3)};
        }

        inline mdarray<T, 2>& coefs()
        {
            return coefs_;
        }

        inline double x(int i__)
        {
            return (*radial_grid_)[i__];
        }

        inline double dx(int i__)
        {
            return radial_grid_->dx(i__);
        }












        inline T operator()(double x)
        {
            int np = num_points();

            assert(x <= radial_grid_[np - 1]);
            
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
                    return coefs_(np - 1, 0);
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
            assert(i >= 0 && i < (int)a.size());
            return coefs_(i, 0);
        }

        inline T operator()(const int i, double dx)
        {
            assert(i >= 0);
            assert(i < (int)a.size() - 1);
            assert(i < (int)b.size());
            assert(i < (int)c.size());
            assert(i < (int)d.size());
            return coefs_(i, 0) + dx * (coefs_(i, 1) + dx * (coefs_(i, 2) + dx * coefs_(i, 3)));
        }
        
        inline T deriv(const int dm, const int i, const double dx)
        {
            assert(i < (int)a.size() - 1);
            assert(i < (int)b.size());
            assert(i < (int)c.size());
            assert(i < (int)d.size());

            switch (dm)
            {
                case 0:
                {
                    return coefs_(i, 0) + dx * (coefs_(i, 1) + dx * (coefs_(i, 2) + dx * coefs_(i, 3)));
                    break;
                }
                case 1:
                {
                    return coefs_(i, 1) + (coefs_(i, 2) * 2.0 + coefs_(i, 3) * dx * 3.0) * dx;
                    break;
                }
                case 2:
                {
                    return coefs_(i, 2) * 2.0 + coefs_(i, 3) * dx * 6.0;
                    break;
                }
                case 3:
                {
                    return coefs_(i, 3) * 6.0;
                    break;
                }
                default:
                {
                    error_local(__FILE__, __LINE__, "wrong order of derivative");
                    return 0.0; // make compiler happy
                    break;
                }
            }
        }

        inline T deriv(const int dm, const int i)
        {
            if (i == num_points() - 1) 
            {
                return deriv(dm, i - 1, radial_grid_->dx(i - 1));
            }
            else 
            {
                return deriv(dm, i, 0);
            }
        }

        inline Radial_grid const& radial_grid() const
        {
            return *radial_grid_;
        }

        //== inline double radial_grid(int ir__)
        //== {
        //==     return (*radial_grid_)[ir__];
        //== }

        Spline<T>& interpolate()
        {
            return this->template interpolate<CPU>();
        }

        template <processing_unit_t pu>
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
            for (int i = 0; i < np - 1; i++) dy[i] = (coefs_(i + 1, 0) - coefs_(i, 0)) / radial_grid_->dx(i);
            
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
                error_local(__FILE__, __LINE__, s);
            }
            
            //std::vector<T> c1(np - 1);
            //std::vector<T> d1(np);
            //
            //std::vector<T> x(np);

            //c1[0] = c[0] / b[0];
            //for (int i = 1; i < np - 1; i++) c1[i] = c[i] / (b[i] - a[i - 1] * c1[i - 1]);

            //d1[0] = d[0] / b[0];
            //for (int i = 1; i < np; i++) d1[i] = (d[i] - a[i - 1] * d1[i - 1]) / (b[i] - a[i - 1] * c1[i - 1]);

            //x[np - 1] = d1[np - 1];
            //for (int i = np - 2; i >= 0; i--) x[i] = d1[i] - c1[i] * x[i + 1];

            for (int i = 0; i < np - 1; i++)
            {
                coefs_(i, 2) = x[i] / 2.0;
                T t = (x[i + 1] - x[i]) / 6.0;
                coefs_(i, 1) = dy[i] - (coefs_(i, 2) + t) * radial_grid_->dx(i);
                coefs_(i, 3) = t / radial_grid_->dx(i);
            }
            return *this;
        }

        T integrate(std::vector<T>& g__, int m__);

        /// Integrate two splines with r^1 or r^2 weight
        template <typename U>
        static T integrate(Spline<T>* f, Spline<U>* g, int m);

        /// Integrate two splines with r^1 or r^2 weight up to a given number of points
        template <typename U>
        static T integrate(Spline<T>* f, Spline<U>* g, int m, int num_points);

        uint64_t hash()
        {
            int np = num_points();
            coefs_(np - 1, 1) = 0;
            coefs_(np - 1, 2) = 0;
            coefs_(np - 1, 3) = 0;
            return coefs_.hash();
        }

        #ifdef _GPU_
        void copy_to_device()
        {
            coefs_.allocate_on_device();
            coefs_.copy_to_device();
        }

        void async_copy_to_device(int thread_id__)
        {
            coefs_.allocate_on_device();
            coefs_.async_copy_to_device(thread_id__);
        }
        #endif

        //template<processing_unit_t pu>
        //T* at()
        //{
        //    return coefs_.at<pu>();
        //}
};

//extern "C" double spline_inner_product_gpu_v2(int size__, double const* x__, double const* dx__, double* f__, double* g__, int stream_id__);
//

extern "C" double spline_inner_product_gpu_v2(int size__, double const* x__, double const* dx__, double const* f__, 
                                              double const* g__, double* d_buf__, double* h_buf__, int stream_id__);


template<processing_unit_t pu, typename T>
T inner(Spline<T>& f__, Spline<T>& g__, int m__)
{
    assert(f__.radial_grid_hash() == g__.radial_grid_hash());
    
    T result = 0;

    int thread_id = Platform::thread_id();

    switch (pu)
    {
        case GPU:
        {
            //return spline_inner_product_gpu_v2(f__.num_points(), f__.radial_grid().x().template at<GPU>(), f__.radial_grid().dx().template at<GPU>(), 
            //                                   f__.coefs().template at<GPU>(), g__.coefs().template at<GPU>(), thread_id);
            break;
        }
        case CPU:
        {
            switch (m__)
            {
                //case 0:
                //{
                //    for (int i = 0; i < num_points - 1; i++)
                //    {
                //        double dx = f->radial_grid_.dx(i);
                //        
                //        T faga = f->a[i] * g->a[i];
                //        T fdgd = f->d[i] * g->d[i];

                //        T k1 = f->a[i] * g->b[i] + f->b[i] * g->a[i];
                //        T k2 = f->c[i] * g->a[i] + f->b[i] * g->b[i] + f->a[i] * g->c[i];
                //        T k3 = f->a[i] * g->d[i] + f->b[i] * g->c[i] + f->c[i] * g->b[i] + f->d[i] * g->a[i];
                //        T k4 = f->b[i] * g->d[i] + f->c[i] * g->c[i] + f->d[i] * g->b[i];
                //        T k5 = f->c[i] * g->d[i] + f->d[i] * g->c[i];

                //        result += dx * (faga + 
                //                  dx * (k1 / 2.0 + 
                //                  dx * (k2 / 3.0 + 
                //                  dx * (k3 / 4.0 + 
                //                  dx * (k4 / 5.0 + 
                //                  dx * (k5 / 6.0 + 
                //                  dx * fdgd / 7.0))))));
                //    }
                //    break;

                //}
                case 1:
                {
                    for (int i = 0; i < f__.num_points() - 1; i++)
                    {
                        double x0 = f__.x(i);
                        double dx = f__.dx(i);
                        
                        auto f = f__.coefs(i);
                        auto g = g__.coefs(i);

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
                    for (int i = 0; i < f__.num_points() - 1; i++)
                    {
                        double x0 = f__.x(i);
                        double dx = f__.dx(i);

                        auto f = f__.coefs(i);
                        auto g = g__.coefs(i);

                        T faga = f[0] * g[0];
                        T fdgd = f[3] * g[3];
                        
                        T k1 = f[3] * g[1] + f[2] * g[2] + f[1] * g[3];
                        T k2 = f[3] * g[0] + f[2] * g[1] + f[1] * g[2] + f[0] * g[3];
                        T k3 = f[2] * g[0] + f[1] * g[1] + f[0] * g[2];
                        T k4 = f[3] * g[2] + f[2] * g[3];
                        T k5 = f[1] * g[0] + f[0] * g[1];

                        result += dx * ((faga * x0 * x0) + 
                                  dx * ((x0 * (2.0 * faga + x0 * k5)) / 2.0 +
                                  dx * ((faga + x0 * (2.0 * k5 + k3 * x0)) / 3.0 + 
                                  dx * ((k5 + x0 * (2.0 * k3 + k2 * x0)) / 4.0 +
                                  dx * ((k3 + x0 * (2.0 * k2 + k1 * x0)) / 5.0 + 
                                  dx * ((k2 + x0 * (2.0 * k1 + k4 * x0)) / 6.0 + 
                                  dx * ((k1 + x0 * (2.0 * k4 + fdgd * x0)) / 7.0 + 
                                  dx * ((k4 + 2.0 * fdgd * x0) / 8.0 + 
                                  dx * fdgd / 9.0)))))))); 
                    }
                    break;
                }
                default:
                {
                    TERMINATE("wrong r^m prefactor");
                }
            }
        }
    }
    return result;
}

#include "spline.hpp"

};

#endif // __SPLINE_H__
