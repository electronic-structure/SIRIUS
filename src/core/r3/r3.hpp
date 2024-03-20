/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file r3.hpp
 *
 *  \brief Simple classes and functions to work with vectors and matrices of the R^3 space.
 */

#ifndef __R3_HPP__
#define __R3_HPP__

#include <cassert>
#include <cmath>
#include <array>
#include <vector>
#include <ostream>
#include <initializer_list>
#include <stdexcept>
#include <sstream>

namespace sirius {

/// Work with 3D vectors and matrices.
namespace r3 {

/// Simple implementation of 3d vector.
template <typename T>
class vector : public std::array<T, 3>
{
  public:
    /// Create zero vector
    vector()
    {
        (*this) = {0, 0, 0};
    }

    /// Create arbitrary vector.
    vector(T x, T y, T z)
    {
        (*this) = {x, y, z};
    }

    /// Create from std::initializer_list.
    vector(std::initializer_list<T> v__)
    {
        assert(v__.size() == 3);
        for (int x : {0, 1, 2}) {
            (*this)[x] = v__.begin()[x];
        }
    }

    vector&
    operator=(std::initializer_list<T> v__)
    {
        assert(v__.size() == 3);
        for (int x : {0, 1, 2}) {
            (*this)[x] = v__.begin()[x];
        }
        return *this;
    }

    /// Create from std::vector.
    vector(std::vector<T> const& v__)
    {
        assert(v__.size() == 3);
        for (int x : {0, 1, 2}) {
            (*this)[x] = v__[x];
        }
    }

    vector&
    operator=(std::vector<T> const& v__)
    {
        assert(v__.size() == 3);
        for (int x : {0, 1, 2}) {
            (*this)[x] = v__[x];
        }
        return *this;
    }

    /// Create from raw pointer.
    vector(T const* ptr__)
    {
        for (int x : {0, 1, 2}) {
            (*this)[x] = ptr__[x];
        }
    }

    /// Create from array.
    vector(std::array<T, 3> v__)
    {
        for (int x : {0, 1, 2}) {
            (*this)[x] = v__[x];
        }
    }

    /// Copy constructor.
    vector(vector<T> const& vec__)
    {
        for (int x : {0, 1, 2}) {
            (*this)[x] = vec__[x];
        }
    }

    /// Return L1 norm of the vector.
    inline T
    l1norm() const
    {
        return std::abs((*this)[0]) + std::abs((*this)[1]) + std::abs((*this)[2]);
    }

    /// Return vector length (L2 norm).
    inline double
    length() const
    {
        return std::sqrt(this->length2());
    }

    /// Return square length of the vector.
    inline double
    length2() const
    {
        return static_cast<double>(std::pow((*this)[0], 2) + std::pow((*this)[1], 2) + std::pow((*this)[2], 2));
    }

    inline vector<T>&
    operator+=(vector<T> const& b)
    {
        for (int x : {0, 1, 2}) {
            (*this)[x] += b[x];
        }
        return *this;
    }

    inline vector<T>&
    operator-=(vector<T> const& b)
    {
        for (int x : {0, 1, 2}) {
            (*this)[x] -= b[x];
        }
        return *this;
    }
};

template <typename T, typename U>
inline vector<decltype(T{} + U{})>
operator+(vector<T> const& a, vector<U> const& b)
{
    vector<decltype(T{} + U{})> c;
    for (int x : {0, 1, 2}) {
        c[x] = a[x] + b[x];
    }
    return c;
}

template <typename T, typename U>
inline vector<decltype(T{} - U{})>
operator-(vector<T> const& a, vector<U> const& b)
{
    vector<decltype(T{} - U{})> c;
    for (int x : {0, 1, 2}) {
        c[x] = a[x] - b[x];
    }
    return c;
}

template <typename T, typename U>
inline std::enable_if_t<std::is_scalar<U>::value, vector<decltype(T{} * U{})>>
operator*(vector<T> const& vec, U p)
{
    vector<decltype(T{} * U{})> a;
    for (int x : {0, 1, 2}) {
        a[x] = vec[x] * p;
    }
    return a;
}

template <typename T, typename U>
inline std::enable_if_t<std::is_scalar<U>::value, vector<decltype(T{} * U{})>>
operator*(U p, vector<T> const& vec)
{
    return vec * p;
}

template <typename T, typename U>
inline std::enable_if_t<std::is_scalar<U>::value, vector<decltype(T{} * U{})>>
operator/(vector<T> const& vec, U p)
{
    vector<decltype(T{} * U{})> a;
    for (int x : {0, 1, 2}) {
        a[x] = vec[x] / p;
    }
    return a;
}

template <typename T, typename U>
inline auto
dot(vector<T> const a, vector<U> const b) -> decltype(T{} * U{})
{
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

template <typename T>
inline auto
cross(vector<T> const a, vector<T> const b)
{
    vector<T> res;
    res[0] = a[1] * b[2] - a[2] * b[1];
    res[1] = a[2] * b[0] - a[0] * b[2];
    res[2] = a[0] * b[1] - a[1] * b[0];
    return res;
}

template <typename T>
std::ostream&
operator<<(std::ostream& out, r3::vector<T> const& v)
{
    out << "{" << v[0] << ", " << v[1] << ", " << v[2] << "}";
    return out;
}

/// Handling of a 3x3 matrix of numerical data types.
template <typename T>
class matrix
{
  private:
    /// Store matrix \f$ M_{ij} \f$ as <tt>mtrx[i][j]</tt>.
    T mtrx_[3][3];

  public:
    template <typename U>
    friend class matrix;

    /// Construct a zero matrix.
    matrix()
    {
        this->zero();
    }

    /// Construct matrix form plain 3x3 array.
    matrix(T mtrx__[3][3])
    {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                mtrx_[i][j] = mtrx__[i][j];
            }
        }
    }

    /// Construct matrix from std::vector.
    matrix(std::vector<std::vector<T>> src__)
    {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                mtrx_[i][j] = src__[i][j];
            }
        }
    }

    matrix(std::array<std::array<T, 3>, 3> src__)
    {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                mtrx_[i][j] = src__[i][j];
            }
        }
    }

    /// Copy constructor.
    template <typename U>
    matrix(matrix<U> const& src__)
    {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                mtrx_[i][j] = src__.mtrx_[i][j];
            }
        }
    }

    /// Construct matrix form std::initializer_list.
    matrix(std::initializer_list<std::initializer_list<T>> mtrx__)
    {
        for (int i : {0, 1, 2}) {
            for (int j : {0, 1, 2}) {
                mtrx_[i][j] = mtrx__.begin()[i].begin()[j];
            }
        }
    }

    /// Assignment operator.
    matrix<T>&
    operator=(matrix<T> const& rhs)
    {
        if (this != &rhs) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    this->mtrx_[i][j] = rhs.mtrx_[i][j];
                }
            }
        }
        return *this;
    }

    inline T&
    operator()(const int i, const int j)
    {
        return mtrx_[i][j];
    }

    inline T const&
    operator()(const int i, const int j) const
    {
        return mtrx_[i][j];
    }

    /// Sum of two matrices.
    template <typename U>
    inline auto
    operator+(matrix<U> const& b) const
    {
        matrix<decltype(T{} + U{})> a;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                a(i, j) = (*this)(i, j) + b(i, j);
            }
        }
        return a;
    }

    /// += operator
    template <typename U>
    inline auto&
    operator+=(matrix<U> const& b)
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
    inline auto&
    operator*=(U p)
    {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                (*this)(i, j) *= p;
            }
        }
        return *this;
    }

    /// Return determinant of a matrix.
    inline T
    det() const
    {
        return (mtrx_[0][2] * (mtrx_[1][0] * mtrx_[2][1] - mtrx_[1][1] * mtrx_[2][0]) +
                mtrx_[0][1] * (mtrx_[1][2] * mtrx_[2][0] - mtrx_[1][0] * mtrx_[2][2]) +
                mtrx_[0][0] * (mtrx_[1][1] * mtrx_[2][2] - mtrx_[1][2] * mtrx_[2][1]));
    }

    inline void
    zero()
    {
        std::fill(&mtrx_[0][0], &mtrx_[0][0] + 9, 0);
    }
};

/// Multiply matrix by a scalar number.
template <typename T, typename U>
inline std::enable_if_t<std::is_scalar<U>::value, matrix<decltype(T{} * U{})>>
operator*(matrix<T> const& a__, U p__)
{
    matrix<decltype(T{} * U{})> c;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            c(i, j) = a__(i, j) * p__;
        }
    }
    return c;
}

template <typename T, typename U>
inline std::enable_if_t<std::is_scalar<U>::value, matrix<decltype(T{} * U{})>>
operator*(U p__, matrix<T> const& a__)
{
    return a__ * p__;
}

inline bool
operator==(matrix<int> const& a__, matrix<int> const& b__)
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (a__(i, j) != b__(i, j)) {
                return false;
            }
        }
    }
    return true;
}

/// Multiply two matrices.
template <typename T, typename U>
inline auto
dot(matrix<T> const& a__, matrix<U> const& b__)
{
    matrix<decltype(T{} * U{})> c;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                c(i, j) += a__(i, k) * b__(k, j);
            }
        }
    }
    return c;
}

/// Matrix-vector multiplication.
template <typename T, typename U>
inline auto
dot(matrix<T> const& m__, vector<U> const& b__)
{
    vector<decltype(T{} * U{})> a;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            a[i] += m__(i, j) * b__[j];
        }
    }
    return a;
}

/// Vector-matrix multiplication.
template <typename T, typename U>
inline auto
dot(vector<U> const& b__, matrix<T> const& m__)
{
    vector<decltype(T{} * U{})> a;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            a[i] += b__[j] * m__(j, i);
        }
    }
    return a;
}

/// Return transpose of the matrix.
template <typename T>
inline auto
transpose(matrix<T> src)
{
    matrix<T> mtrx;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mtrx(i, j) = src(j, i);
        }
    }
    return mtrx;
}

template <typename T>
inline auto
inverse_aux(matrix<T> src)
{
    matrix<T> mtrx;

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
inline auto
inverse(matrix<int> src)
{
    int t1 = src.det();
    if (std::abs(t1) != 1) {
        throw std::runtime_error("integer matrix can't be inverted");
    }
    return inverse_aux(src) * t1;
}

/// Return inverse of the matrix.
template <typename T>
inline auto
inverse(matrix<T> src)
{
    T t1 = src.det();

    if (std::abs(t1) < 1e-10) {
        throw std::runtime_error("matrix is degenerate");
    }

    return inverse_aux(src) * (1.0 / t1);
}

template <typename T>
inline std::ostream&
operator<<(std::ostream& out, matrix<T> const& v)
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

/// Reduce the coordinates to the first unit cell.
/** Split the input vector in lattice coordinates to the sum r0 + T, where T is the lattice translation
 *  vector (three integers) and r0 is the vector within the first unit cell with coordinates in [0, 1) range. */
inline auto
reduce_coordinates(vector<double> coord__)
{
    const double eps{1e-9};

    std::pair<vector<double>, vector<int>> v;

    v.first = coord__;
    for (int i = 0; i < 3; i++) {
        v.second[i] = (int)floor(v.first[i]);
        v.first[i] -= v.second[i];
        if (v.first[i] < -eps || v.first[i] > 1.0 + eps) {
            std::stringstream s;
            s << "wrong fractional coordinates" << std::endl << v.first[0] << " " << v.first[1] << " " << v.first[2];
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
            s << "wrong fractional coordinates" << std::endl << v.first[0] << " " << v.first[1] << " " << v.first[2];
            throw std::runtime_error(s.str());
        }
    }
    for (int x : {0, 1, 2}) {
        if (std::abs(coord__[x] - (v.first[x] + v.second[x])) > eps) {
            std::stringstream s;
            s << "wrong coordinate reduction" << std::endl
              << "  original coord: " << coord__ << std::endl
              << "  reduced coord: " << v.first << std::endl
              << "  T: " << v.second;
            throw std::runtime_error(s.str());
        }
    }
    return v;
}

/// Find supercell that circumscribes the sphere with a given radius.
/** Search for the translation limits (N1, N2, N3) such that the resulting supercell with the lattice
 *  vectors a1 * N1, a2 * N2, a3 * N3 fully contains the sphere with a given radius. This is done
 *  by equating the expressions for the volume of the supercell:
 *   Volume = |(A1 x A2) * A3| = N1 * N2 * N3 * |(a1 x a2) * a3|
 *   Volume = h * S = 2 * R * |a_i x a_j| * N_i * N_j */
inline auto
find_translations(double radius__, matrix<double> const& lattice_vectors__)
{
    vector<double> a0(lattice_vectors__(0, 0), lattice_vectors__(1, 0), lattice_vectors__(2, 0));
    vector<double> a1(lattice_vectors__(0, 1), lattice_vectors__(1, 1), lattice_vectors__(2, 1));
    vector<double> a2(lattice_vectors__(0, 2), lattice_vectors__(1, 2), lattice_vectors__(2, 2));

    double det = std::abs(lattice_vectors__.det());

    vector<int> limits;

    limits[0] = static_cast<int>(2 * radius__ * cross(a1, a2).length() / det) + 1;
    limits[1] = static_cast<int>(2 * radius__ * cross(a0, a2).length() / det) + 1;
    limits[2] = static_cast<int>(2 * radius__ * cross(a0, a1).length() / det) + 1;

    return limits;
}

/// Transform Cartesian coordinates [x,y,z] to spherical coordinates [r,theta,phi]
inline auto
spherical_coordinates(vector<double> vc)
{
    r3::vector<double> vs;

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

} // namespace r3

} // namespace sirius

#endif // __R3_HPP__
