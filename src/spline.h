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

#ifndef __SPLINE_H__
#define __SPLINE_H__

/** \file spline.h
 *   
 *  \brief Contains definition and partial implementaiton of sirius::Spline class.
 */

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

        // TODO: maybe add x-coordinate as an oprator. we know the radial grid and we can return x here

    public:
    
        template <typename U> 
        friend class Spline;

        /// Constructor of an empty spline.
        Spline()
        {
        }
        
        /// Constructor of a zero valued spline.
        Spline(Radial_grid radial_grid__) : radial_grid_(radial_grid__)
        {
            int np = num_points();

            a = std::vector<T>(np);
            b = std::vector<T>(np - 1);
            c = std::vector<T>(np - 1);
            d = std::vector<T>(np - 1);

            memset(&a[0], 0, np * sizeof(T));
            memset(&b[0], 0, (np - 1) * sizeof(T));
            memset(&c[0], 0, (np - 1) * sizeof(T));
            memset(&d[0], 0, (np - 1) * sizeof(T));
        }
        
        /// Constructor of a spline.
        Spline(Radial_grid& radial_grid__, std::vector<T>& y) : radial_grid_(radial_grid__)
        {
            interpolate(y);
        }
        
        inline Spline<T>& interpolate(std::vector<T>& y)
        {
            assert(radial_grid_.num_points() == (int)y.size());
            a = y;
            return interpolate();
        }
        
        T integrate(int m = 0)
        {
            std::vector<T> g(num_points());
            return integrate(g, m);
        }
        
        T integrate(int n, int m)
        {
            std::vector<T> g(num_points());
            integrate(g, m);
            return g[n];
        }

        inline std::vector<T>& values()
        {
            return a;
        }
        
        inline int num_points()
        {
            return radial_grid_.num_points();
        }

        inline Spline<T>& operator=(std::vector<T>& y)
        {
            a = y;
            return *this;
        }

        // TODO: check spline iterpolation between different grids. Can it be done analytically?
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
                    return a[j] + dx * (b[j] + dx * (c[j] + dx * d[j]));
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
            assert(i >= 0 && i < (int)a.size() - 1);
            return a[i] + dx * (b[i] + dx * (c[i] + dx * d[i]));
        }
        
        inline T deriv(const int dm, const int i, const double dx)
        {
            assert(i < num_points() - 1);

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
                return deriv(dm, i, 0.0);
            }
        }

        Spline<T>& interpolate();

        void get_coefs(T* array, int lda);

        T integrate(std::vector<T>& g, int m = 0);

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
};

#include "spline.hpp"

};

#endif // __SPLINE_H__
