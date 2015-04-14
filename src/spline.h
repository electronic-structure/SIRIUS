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
        Radial_grid radial_grid_;
        
        /// Spline "a" coefficients.
        std::vector<T> a;
        
        /// Spline "b" coefficients.
        std::vector<T> b;
        
        /// Spline "c" coefficients.
        std::vector<T> c;
        
        /// Spline "d" coefficients.
        std::vector<T> d;

        mdarray<T, 2> packed_spline_data_;

        // TODO: maybe add x-coordinate as an oprator. we know the radial grid and we can return x here

    public:

        Spline(Spline<T> const& src__) = delete;

        Spline(Spline<T>&& src__)
        {
            radial_grid_ = std::move(src__.radial_grid_);
            a = std::move(src__.a);
            b = std::move(src__.b);
            c = std::move(src__.c);
            d = std::move(src__.d);
            packed_spline_data_ = std::move(src__.packed_spline_data_);
        }

        Spline<T>& operator=(Spline<T> const& src__) = delete;

        Spline<T>& operator=(Spline<T>&& src__)
        {
            if (this != &src__)
            {
                radial_grid_ = std::move(src__.radial_grid_);
                a = std::move(src__.a);
                b = std::move(src__.b);
                c = std::move(src__.c);
                d = std::move(src__.d);
                packed_spline_data_ = std::move(src__.packed_spline_data_);
            }
            return *this;
        }
    
        template <typename U> 
        friend class Spline;

        /// Default constructor.
        Spline()
        {
        }
        
        /// Constructor of a new uninitialized spline.
        Spline(Radial_grid radial_grid__) : radial_grid_(radial_grid__)
        {
            a.resize(num_points());
        }

        /// Constructor of a constant value spline.
        Spline(Radial_grid radial_grid__, T val__) : radial_grid_(radial_grid__)
        {
            int np = num_points();
            a.resize(np);
            for (int i = 0; i < np; i++) a[i] = val__;
        }
        
        /// Constructor of a spline.
        Spline(Radial_grid& radial_grid__, std::vector<T>& y__) : radial_grid_(radial_grid__)
        {
            interpolate(y__);
        }
        
        inline Spline<T>& interpolate(std::vector<T>& y__)
        {
            assert(radial_grid_.num_points() == (int)y__.size());
            a = y__;
            return interpolate();
        }
        
        /// Integrate with r^m weight.
        T integrate(int m__)
        {
            std::vector<T> g(num_points());
            return integrate(g, m__);
        }
        
        inline std::vector<T>& values()
        {
            return a;
        }
        
        /// Return number of spline points.
        inline int num_points()
        {
            return radial_grid_.num_points();
        }

        inline Spline<T>& operator=(std::vector<T>& y)
        {
            a = y;
            return *this;
        }

        inline T operator()(double x)
        {
            int np = num_points();

            assert(x <= radial_grid_[np - 1]);
            
            if (x >= radial_grid_[0])
            {
                int j = np - 1;
                for (int i = 0; i < np - 1; i++)
                {
                    if (x < radial_grid_[i + 1])
                    {
                        j = i;
                        break;
                    }
                }
                if (j == np - 1) 
                {
                    return a[np - 1];
                }
                else
                {
                    double dx = x - radial_grid_[j];
                    //return a[j] + dx * (b[j] + dx * (c[j] + dx * d[j]));
                    return (*this)(j, dx);
                }
            }
            else
            {
                double dx = x - radial_grid_[0];
                return a[0] + dx * (b[0] + dx * (c[0] + dx * d[0]));
            }
        }
        
        /// Return value at \f$ x_i \f$.
        inline T& operator[](const int i)
        {
            assert(i >= 0 && i < (int)a.size());
            return a[i];
        }

        inline T operator()(const int i, double dx)
        {
            assert(i >= 0);
            assert(i < (int)a.size() - 1);
            assert(i < (int)b.size());
            assert(i < (int)c.size());
            assert(i < (int)d.size());
            return a[i] + dx * (b[i] + dx * (c[i] + dx * d[i]));
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
                    return a[i] + dx * (b[i] + dx * (c[i] + dx * d[i]));
                    break;
                }
                case 1:
                {
                    return b[i] + dx * (c[i] * 2.0 + d[i] * dx * 3.0);
                    break;
                }
                case 2:
                {
                    return c[i] * 2.0 + d[i] * dx * 6.0;
                    break;
                }
                case 3:
                {
                    return d[i] * 6.0;
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
                return deriv(dm, i - 1, radial_grid_.dx(i - 1));
            }
            else 
            {
                return deriv(dm, i, 0);
            }
        }

        inline Radial_grid radial_grid() const
        {
            return radial_grid_;
        }

        inline double radial_grid(int ir__)
        {
            return radial_grid_[ir__];
        }

        Spline<T>& interpolate();

        void get_coefs(T* array, int lda);

        T integrate(std::vector<T>& g, int m);

        /// Integrate two splines with r^1 or r^2 weight
        template <typename U>
        static T integrate(Spline<T>* f, Spline<U>* g, int m);

        /// Integrate two splines with r^1 or r^2 weight up to a given number of points
        template <typename U>
        static T integrate(Spline<T>* f, Spline<U>* g, int m, int num_points);

        uint64_t hash()
        {
            mdarray<T, 1> v(4 * num_points() - 3);
            int n = 0;
            for (int i = 0; i < num_points(); i++)
            {
                v(n++) = a[i];
            }
            for (int i = 0; i < num_points() - 1; i++)
            {
                v(n++) = b[i];
                v(n++) = c[i];
                v(n++) = d[i];
            }
            return v.hash();
        }

        void pack()
        {
            packed_spline_data_ = mdarray<T, 2>(num_points(), 6);
            for (int i = 0; i < num_points(); i++)
            {
                packed_spline_data_(i, 0) = radial_grid_[i];
                packed_spline_data_(i, 2) = a[i];
            }
            for (int i = 0; i < num_points() - 1; i++)
            {
                packed_spline_data_(i, 1) = radial_grid_.dx(i);
                packed_spline_data_(i, 3) = b[i];
                packed_spline_data_(i, 4) = c[i];
                packed_spline_data_(i, 5) = d[i];
            }
        }

        #ifdef _GPU_
        void copy_to_device()
        {
            packed_spline_data_.allocate_on_device();
            packed_spline_data_.copy_to_device();
        }
        #endif

        template<processing_unit_t pu>
        T* at()
        {
            return packed_spline_data_.at<pu>();
        }
};

extern "C" double spline_inner_product_gpu_v2(int size__, double* x_dx__, double* f__, double* g__);

template<processing_unit_t pu, typename T>
T inner(Spline<T>& f__, Spline<T>& g__, int m__)
{
    if (pu == GPU)
    {
        T* x_dx = f__.template at<GPU>();
        T* f = &x_dx[2 * f__.num_points()];
        T* g = &(g__.template at<GPU>()[2 * f__.num_points()]);
        return spline_inner_product_gpu_v2(f__.num_points(), x_dx, f, g);
    }
    else
    {
        STOP();
    }
}

#include "spline.hpp"

};

#endif // __SPLINE_H__
