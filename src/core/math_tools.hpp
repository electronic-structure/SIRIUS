// Copyright (c) 2013-2023 Anton Kozhevnikov, Thomas Schulthess
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

/** \file math_tools.hpp
 *
 *  \brief Math helper functions.
 */

#ifndef __MATH_TOOLS_HPP__
#define __MATH_TOOLS_HPP__

#include <cmath>
#include <complex>
#include <iomanip>
#include "core/rte/rte.hpp"

namespace sirius {

inline auto
confined_polynomial(double r, double R, int p1, int p2, int dm)
{
    double t = 1.0 - std::pow(r / R, 2);
    switch (dm) {
        case 0: {
            return (std::pow(r, p1) * std::pow(t, p2));
        }
        case 2: {
            return (-4 * p1 * p2 * std::pow(r, p1) * std::pow(t, p2 - 1) / std::pow(R, 2) +
                    p1 * (p1 - 1) * std::pow(r, p1 - 2) * std::pow(t, p2) +
                    std::pow(r, p1) * (4 * (p2 - 1) * p2 * std::pow(r, 2) * std::pow(t, p2 - 2) / std::pow(R, 4) -
                                       2 * p2 * std::pow(t, p2 - 1) / std::pow(R, 2)));
        }
        default: {
            RTE_THROW("wrong derivative order");
            return 0.0;
        }
    }
}

/// Sign of the variable.
template <typename T>
inline int
sign(T val)
{
    return (T(0) < val) - (val < T(0));
}

/// Checks if number is integer with a given tolerance.
template <typename T>
inline bool
is_int(T val__, T eps__)
{
    if (std::abs(std::round(val__) - val__) > eps__) {
        return false;
    } else {
        return true;
    }
}

/// Compute a factorial.
template <typename T>
inline T
factorial(int n)
{
    RTE_ASSERT(n >= 0);

    T result{1};
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

inline auto
round(double a__, int n__)
{
    std::stringstream s;
    s << std::setprecision(n__) << std::fixed << a__;
    double result;
    s >> result;
    return result;
    // double a0 = std::floor(a__);
    // double b  = std::round((a__ - a0) * std::pow(10, n__)) / std::pow(10, n__);
    // return a0 + b;
}

inline auto
round(std::complex<double> a__, int n__)
{
    return std::complex<double>(round(a__.real(), n__), round(a__.imag(), n__));
}

/// Simple hash function.
/** Example: std::printf("hash: %16llX\n", hash()); */
inline auto
hash(void const* buff, size_t size, uint64_t h = 5381)
{
    unsigned char const* p = static_cast<unsigned char const*>(buff);
    for (size_t i = 0; i < size; i++) {
        h = ((h << 5) + h) + p[i];
    }
    return h;
}

/// Simple random number generator.
inline uint32_t
random_uint32(bool reset = false)
{
    static uint32_t a = 123456;
    if (reset) {
        a = 123456;
    }
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

template <typename T>
inline T
random();

template <>
inline int
random<int>()
{
    return static_cast<int>(random_uint32());
}

template <>
inline double
random<double>()
{
    return static_cast<double>(random_uint32()) / std::numeric_limits<uint32_t>::max();
}

template <>
inline std::complex<double>
random<std::complex<double>>()
{
    return std::complex<double>(random<double>(), random<double>());
}

template <>
inline float
random<float>()
{
    return static_cast<float>(random<double>());
}

template <>
inline std::complex<float>
random<std::complex<float>>()
{
    return std::complex<float>(random<float>(), random<float>());
}

template <typename T>
auto
abs_diff(T a, T b)
{
    return std::abs(a - b);
}

template <typename T>
auto
rel_diff(T a, T b)
{
    return std::abs(a - b) / (std::abs(a) + std::abs(b) + 1e-13);
}

/// Return complex conjugate of a number. For a real value this is the number itself.
inline auto
conj(double x__)
{
    /* std::conj() will return complex for a double value input; this is not what we want */
    return x__;
}

/// Return complex conjugate of a number.
inline auto
conj(std::complex<double> x__)
{
    return std::conj(x__);
}

template <typename T>
inline T
zero_if_not_complex(T x__)
{
    return x__;
};

template <typename T>
inline T
zero_if_not_complex(std::complex<T> x__)
{
    return 0;
};

} // namespace sirius

#endif
