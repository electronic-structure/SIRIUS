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

/** \file geometry3d.hpp
 *
 *  \brief Simple classes and functions to work with the 3D vectors and matrices of the crystal lattice.
 */

#ifndef __GEOMETRY3D_HPP__
#define __GEOMETRY3D_HPP__

#include <assert.h>
#include <cmath>
#include <array>
#include <vector>
#include <ostream>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <sstream>

namespace geometry3d {

/// Simple implementation of 3d vector.
template <typename T>
class vector3d : public std::array<T, 3>
{
  public:
    /// Create zero vector
    vector3d()
    {
        (*this) = {0, 0, 0};
    }

    /// Create arbitrary vector.
    vector3d(T x, T y, T z)
    {
        (*this) = {x, y, z};
    }

    /// Create from std::initializer_list.
    vector3d(std::initializer_list<T> v__)
    {
        assert(v__.size() == 3);
        for (int x : {0, 1, 2}) {
            (*this)[x] = v__.begin()[x];
        }
    }

    /// Create from std::vector.
    vector3d(std::vector<T> v__)
    {
        assert(v__.size() == 3);
        for (int x : {0, 1, 2}) {
            (*this)[x] = v__[x];
        }
    }

    /// Create from raw pointer.
    vector3d(T const* ptr__)
    {
        for (int x : {0, 1, 2}) {
            (*this)[x] = ptr__[x];
        }
    }

    /// Create from array.
    vector3d(std::array<T, 3> v__)
    {
        for (int x : {0, 1, 2}) {
            (*this)[x] = v__[x];
        }
    }

    /// Copy constructor.
    vector3d(vector3d<T> const& vec__)
    {
        for (int x : {0, 1, 2}) {
            (*this)[x] = vec__[x];
        }
    }

    /// Return L1 norm of the vector.
    inline T l1norm() const
    {
        return std::abs((*this)[0]) + std::abs((*this)[1]) + std::abs((*this)[2]);
    }

    /// Return vector length (L2 norm).
    inline double length() const
    {
        return std::sqrt(this->length2());
    }

    /// Return square length of the vector.
    inline double length2() const
    {
        return static_cast<double>(std::pow((*this)[0], 2) + std::pow((*this)[1], 2) + std::pow((*this)[2], 2));
    }

    template <typename U>
    inline vector3d<decltype(T{} + U{})> operator+(vector3d<U> const& b) const
    {
        vector3d<decltype(T{} + U{})> a = *this;
        for (int x : {0, 1, 2}) {
            a[x] += b[x];
        }
        return a;
    }

    template <typename U>
    inline vector3d<decltype(T{} - U{})> operator-(vector3d<U> const& b) const
    {
        vector3d<decltype(T{} - U{})> a = *this;
        for (int x : {0, 1, 2}) {
            a[x] -= b[x];
        }
        return a;
    }

    inline vector3d<T>& operator+=(vector3d<T> const& b)
    {
        for (int x : {0, 1, 2}) {
            (*this)[x] += b[x];
        }
        return *this;
    }

    inline vector3d<T>& operator-=(vector3d<T> const& b)
    {
        for (int x : {0, 1, 2}) {
            (*this)[x] -= b[x];
        }
        return *this;
    }

    template <typename U>
    inline friend vector3d<decltype(T{} * U{})> operator*(vector3d<T> vec, U p)
    {
        vector3d<decltype(T{} * U{})> a;
        for (int x : {0, 1, 2}) {
            a[x] = vec[x] * p;
        }
        return a;
    }

    template <typename U>
    inline friend vector3d<decltype(T{} * U{})> operator*(U p, vector3d<T> vec)
    {
        return vec * p;
    }

    template <typename U>
    inline friend vector3d<decltype(T{} * U{})> operator/(vector3d<T> vec, U p)
    {
        vector3d<decltype(T{} * U{})> a;
        for (int x : {0, 1, 2}) {
            a[x] = vec[x] / p;
        }
        return a;
    }
};

template <typename T, typename U>
inline auto dot(vector3d<T> const a, vector3d<U> const b) -> decltype(T{} * U{})
{
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

template <typename T>
inline vector3d<T> cross(vector3d<T> const a, vector3d<T> const b)
{
    vector3d<T> res;
    res[0] = a[1] * b[2] - a[2] * b[1];
    res[1] = a[2] * b[0] - a[0] * b[2];
    res[2] = a[0] * b[1] - a[1] * b[0];
    return res;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, vector3d<T> const& v)
{
    out << v[0] << " " << v[1] << " " << v[2];
    return out;
}

/// Handling of a 3x3 matrix of numerical data types.
template <typename T>
class matrix3d
{
  private:
    /// Store matrix \f$ M_{ij} \f$ as <tt>mtrx[i][j]</tt>.
    T mtrx_[3][3];

  public:
    template <typename U>
    friend class matrix3d;

    /// Construct a zero matrix.
    matrix3d()
    {
        std::memset(&mtrx_[0][0], 0, 9 * sizeof(T));
    }

    /// Construct matrix form plain 3x3 array.
    matrix3d(T mtrx__[3][3])
    {
        std::memcpy(&mtrx_[0][0], &mtrx__[0][0], 9 * sizeof(T));
    }

    /// Construct matrix from std::vector.
    matrix3d(std::vector<std::vector<T>> src__)
    {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                mtrx_[i][j] = src__[i][j];
            }
        }
    }

    /// Copy constructor.
    template <typename U>
    matrix3d(const matrix3d<U>& src__)
    {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                mtrx_[i][j] = src__.mtrx_[i][j];
            }
        }
    }

    /// Construct matrix form std::initializer_list.
    matrix3d(std::initializer_list<std::initializer_list<T>> mtrx__)
    {
        for (int i : {0, 1, 2}) {
            for (int j : {0, 1, 2}) {
                mtrx_[i][j] = mtrx__.begin()[i].begin()[j];
            }
        }
    }

    /// Assigment operator.
    matrix3d<T>& operator=(const matrix3d<T>& rhs)
    {
        if (this != &rhs) {
            std::memcpy(&this->mtrx_[0][0], &rhs.mtrx_[0][0], 9 * sizeof(T));
        }
        return *this;
    }

    inline T& operator()(const int i, const int j)
    {
        return mtrx_[i][j];
    }

    inline T const& operator()(const int i, const int j) const
    {
        return mtrx_[i][j];
    }

    /// Multiply two matrices.
    template <typename U>
    inline matrix3d<decltype(T{} * U{})> operator*(matrix3d<U> const& b) const
    {
        matrix3d<decltype(T{} * U{})> c;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    c(i, j) += (*this)(i, k) * b(k, j);
                }
            }
        }
        return c;
    }

    /// Matrix-vector multiplication.
    template <typename U>
    inline vector3d<decltype(T{} * U{})> operator*(vector3d<U> const& b) const
    {
        vector3d<decltype(T{} * U{})> a;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                a[i] += (*this)(i, j) * b[j];
            }
        }
        return a;
    }

    /// Sum of two matrices.
    template <typename U>
    inline matrix3d<decltype(T{} + U{})> operator+(matrix3d<U> const& b) const
    {
        matrix3d<decltype(T{} + U{})> a;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                a(i, j) = (*this)(i, j) + b(i, j);
            }
        }
        return a;
    }

    /// += operator
    template <typename U>
    inline matrix3d<T> operator+=(matrix3d<U> const& b)
    {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                (*this)(i, j) += b(i, j);
            }
        }
        return *this;
    }

    /// Multiply matrix by a scalar number.
    template <typename U>
    inline matrix3d<decltype(T{} * U{})> operator*(U p) const
    {
        matrix3d<decltype(T{} * U{})> c;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                c(i, j) = (*this)(i, j) * p;
            }
        }
        return c;
    }

    /// Multiply matrix by a scalar number.
    template <typename U>
    inline matrix3d<T>& operator*=(U p)
    {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                (*this)(i, j) *= p;
            }
        }
        return *this;
    }

    /// Return determinant of a matrix.
    inline T det() const
    {
        return (mtrx_[0][2] * (mtrx_[1][0] * mtrx_[2][1] - mtrx_[1][1] * mtrx_[2][0]) +
                mtrx_[0][1] * (mtrx_[1][2] * mtrx_[2][0] - mtrx_[1][0] * mtrx_[2][2]) +
                mtrx_[0][0] * (mtrx_[1][1] * mtrx_[2][2] - mtrx_[1][2] * mtrx_[2][1]));
    }

    inline void zero()
    {
        std::fill(&mtrx_[0][0], &mtrx_[0][0] + 9, 0);
    }
};

/// Return transpose of the matrix.
template <typename T>
inline matrix3d<T> transpose(matrix3d<T> src)
{
    matrix3d<T> mtrx;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mtrx(i, j) = src(j, i);
        }
    }
    return mtrx;
}

template <typename T>
inline matrix3d<T> inverse_aux(matrix3d<T> src)
{
    matrix3d<T> mtrx;

    mtrx(0, 0) = (src(1, 1) * src(2, 2) - src(1, 2) * src(2, 1));
    mtrx(0, 1) = (src(0, 2) * src(2, 1) - src(0, 1) * src(2, 2));
    mtrx(0, 2) = (src(0, 1) * src(1, 2) - src(0, 2) * src(1, 1));
    mtrx(1, 0) = (src(1, 2) * src(2, 0) - src(1, 0) * src(2, 2));
    mtrx(1, 1) = (src(0, 0) * src(2, 2) - src(0, 2) * src(2, 0));
    mtrx(1, 2) = (src(0, 2) * src(1, 0) - src(0, 0) * src(1, 2));
    mtrx(2, 0) = (src(1, 0) * src(2, 1) - src(1, 1) * src(2, 0));
    mtrx(2, 1) = (src(0, 1) * src(2, 0) - src(0, 0) * src(2, 1));
    mtrx(2, 2) = (src(0, 0) * src(1, 1) - src(0, 1) * src(1, 0));

    return mtrx;
}

/// Return inverse of the integer matrix
inline matrix3d<int> inverse(matrix3d<int> src)
{
    int t1 = src.det();
    if (std::abs(t1) != 1) {
        throw std::runtime_error("integer matrix can't be inverted");
    }
    return inverse_aux(src) * t1;
}

/// Return inverse of the matrix.
template <typename T>
inline matrix3d<T> inverse(matrix3d<T> src)
{
    T t1 = src.det();

    if (std::abs(t1) < 1e-10) {
        throw std::runtime_error("matrix is degenerate");
    }

    return inverse_aux(src) * (1.0 / t1);
}

template <typename T>
inline std::ostream& operator<<(std::ostream& out, matrix3d<T>& v)
{
    out << "{";
    for (int i = 0; i < 3; i++) {
        out << "{";
        for (int j = 0; j < 3; j++) {
            out << v(i, j);
            if (j != 2) {
                out << ", ";
            }
        }
        out << "}";
        if (i != 2) {
            out << ",";
        } else {
            out << "}";
        }
    }
    return out;
}

inline std::pair<vector3d<double>, vector3d<int>> reduce_coordinates(vector3d<double> coord)
{
    const double eps{1e-9};

    std::pair<vector3d<double>, vector3d<int>> v;

    v.first = coord;
    for (int i = 0; i < 3; i++) {
        v.second[i] = (int)floor(v.first[i]);
        v.first[i] -= v.second[i];
        if (v.first[i] < -eps || v.first[i] > 1.0 + eps) {
            std::stringstream s;
            s << "wrong fractional coordinates" << std::endl
              << v.first[0] << " " << v.first[1] << " " << v.first[2];
            throw std::runtime_error(s.str());
        }
        if (v.first[i] < 0) {
            v.first[i] = 0;
        }
        if (v.first[i] >= (1 - eps)) {
            v.first[i] = 0;
            v.second[i] += 1;
        }
        if (v.first[i] < 0 || v.first[i] >= 1) {
            std::stringstream s;
            s << "wrong fractional coordinates" << std::endl
              << v.first[0] << " " << v.first[1] << " " << v.first[2];
            throw std::runtime_error(s.str());
        }
    }
    for (int x : {0, 1, 2}) {
        if (std::abs(coord[x] - (v.first[x] + v.second[x])) > eps) {
            std::stringstream s;
            s << "wrong coordinate reduction" << std::endl
              << "  original coord: " << coord << std::endl
              << "  reduced coord: " << v.first << std::endl
              << "  T: " << v.second;
            throw std::runtime_error(s.str());
        }
    }
    return v;
}

/// Find supercell that circumscribes the sphere with a given radius.
/** Serach for the translation limits (N1, N2, N3) such that the resulting supercell with the lattice
 *  vectors a1 * N1, a2 * N2, a3 * N3 fully contains the sphere with a given radius. This is done
 *  by equating the expressions for the volume of the supercell:
 *   Volume = |(A1 x A2) * A3| = N1 * N2 * N3 * |(a1 x a2) * a3|
 *   Volume = h * S = 2 * R * |a_i x a_j| * N_i * N_j */
inline vector3d<int> find_translations(double radius__, matrix3d<double> const& lattice_vectors__)
{
    vector3d<double> a0(lattice_vectors__(0, 0), lattice_vectors__(1, 0), lattice_vectors__(2, 0));
    vector3d<double> a1(lattice_vectors__(0, 1), lattice_vectors__(1, 1), lattice_vectors__(2, 1));
    vector3d<double> a2(lattice_vectors__(0, 2), lattice_vectors__(1, 2), lattice_vectors__(2, 2));

    double det = std::abs(lattice_vectors__.det());

    vector3d<int> limits;

    limits[0] = static_cast<int>(2 * radius__ * cross(a1, a2).length() / det) + 1;
    limits[1] = static_cast<int>(2 * radius__ * cross(a0, a2).length() / det) + 1;
    limits[2] = static_cast<int>(2 * radius__ * cross(a0, a1).length() / det) + 1;

    return {limits[0], limits[1], limits[2]};
}

/// Transform Cartesian coordinates [x,y,z] to spherical coordinates [r,theta,phi]
inline vector3d<double> spherical_coordinates(vector3d<double> vc)
{
    geometry3d::vector3d<double> vs;

    const double eps{1e-12};

    const double twopi = 6.2831853071795864769;

    vs[0] = vc.length();

    if (vs[0] <= eps) {
        vs[1] = 0.0;
        vs[2] = 0.0;
    } else {
        vs[1] = std::acos(vc[2] / vs[0]); // theta = cos^{-1}(z/r)

        if (std::abs(vc[0]) > eps || std::abs(vc[1]) > eps) {
            vs[2] = std::atan2(vc[1], vc[0]); // phi = tan^{-1}(y/x)
            if (vs[2] < 0.0) {
                vs[2] += twopi;
            }
        } else {
            vs[2] = 0.0;
        }
    }

    return vs;
}

} // namespace geometry3d

#endif // __GEOMETRY3D_HPP__
