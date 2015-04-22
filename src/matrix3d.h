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

/** \file matrix3d.h
 *   
 *  \brief Contains declaration and implementation of matrix3d class.
 */

#ifndef __MATRIX3D_H__
#define __MATRIX3D_H__

#include <string.h>
#include "vector3d.h"
#include "typedefs.h"
#include "error_handling.h"

/// Handling of a 3x3 matrix of numerical data types.
template <typename T>
class matrix3d
{
    private:

        T mtrx_[3][3];

    public:
    
        template <typename U> 
        friend class matrix3d;
        
        matrix3d()
        {
            memset(&mtrx_[0][0], 0, 9 * sizeof(T));
        }
        
        matrix3d(T mtrx__[3][3])
        {
            memcpy(&mtrx_[0][0], &mtrx__[0][0], 9 * sizeof(T));
        }

        matrix3d(const matrix3d<T>& src)
        {
            memcpy(&mtrx_[0][0], &src.mtrx_[0][0], 9 * sizeof(T));
        }
        
        template <typename U>
        matrix3d(const matrix3d<U>& src)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++) mtrx_[i][j] = src.mtrx_[i][j];
            }
        }

        matrix3d<T>& operator=(const matrix3d<T>& rhs)
        {
            if (this != &rhs) memcpy(&this->mtrx_[0][0], &rhs.mtrx_[0][0], 9 * sizeof(T));
            return *this;
        }

        inline T& operator()(const int i, const int j)
        {
            return mtrx_[i][j];
        }
        
        /// Multiply two matrices.
        inline matrix3d<T> operator*(matrix3d<T> b)
        {
            matrix3d<T> c;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 3; k++) c(i, j) += (*this)(i, k) * b(k, j);
                }
            }
            return c;
        }

        /// Matrix-vector multiplication.
        template <typename U>
        inline vector3d<decltype(T{} * U{})> operator*(vector3d<U> const& b)
        {
            vector3d<decltype(T{} * U{})> a;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++) a[i] += (*this)(i, j) * b[j];
            }
            return a;
        }

        /// Multiply matrix by a scalar number.
        template <typename U>
        inline matrix3d<T> operator*(U p)
        {
            matrix3d<T> c;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++) c(i, j) = (*this)(i, j) * p;
            }
            return c;
        }
        
        /// Return determinant of a matrix.
        inline T det()
        {
            return (mtrx_[0][2] * (mtrx_[1][0] * mtrx_[2][1] - mtrx_[1][1] * mtrx_[2][0]) + 
                    mtrx_[0][1] * (mtrx_[1][2] * mtrx_[2][0] - mtrx_[1][0] * mtrx_[2][2]) + 
                    mtrx_[0][0] * (mtrx_[1][1] * mtrx_[2][2] - mtrx_[1][2] * mtrx_[2][1]));
        }

};

/// Return transpose of the matrix.
template <typename T>
matrix3d<T> transpose(matrix3d<T> src)
{
    matrix3d<T> mtrx;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++) mtrx(i, j) = src(j, i);
    }
    return mtrx;
}

/// Return inverse of the matrix.
template <typename T>
matrix3d<T> inverse(matrix3d<T> src)
{
    matrix3d<T> mtrx;
    
    T t1 = src.det();
    
    if (std::abs(t1) < 1e-10) error_local(__FILE__, __LINE__, "matix is degenerate");
    
    t1 = 1.0 / t1;

    mtrx(0, 0) = t1 * (src(1, 1) * src(2, 2) - src(1, 2) * src(2, 1));
    mtrx(0, 1) = t1 * (src(0, 2) * src(2, 1) - src(0, 1) * src(2, 2));
    mtrx(0, 2) = t1 * (src(0, 1) * src(1, 2) - src(0, 2) * src(1, 1));
    mtrx(1, 0) = t1 * (src(1, 2) * src(2, 0) - src(1, 0) * src(2, 2));
    mtrx(1, 1) = t1 * (src(0, 0) * src(2, 2) - src(0, 2) * src(2, 0));
    mtrx(1, 2) = t1 * (src(0, 2) * src(1, 0) - src(0, 0) * src(1, 2));
    mtrx(2, 0) = t1 * (src(1, 0) * src(2, 1) - src(1, 1) * src(2, 0));
    mtrx(2, 1) = t1 * (src(0, 1) * src(2, 0) - src(0, 0) * src(2, 1));
    mtrx(2, 2) = t1 * (src(0, 0) * src(1, 1) - src(0, 1) * src(1, 0));

    return mtrx;
}

#endif // __MATRIX3D_H__
