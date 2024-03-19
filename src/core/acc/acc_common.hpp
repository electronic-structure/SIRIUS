/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file acc_common.hpp
 *
 *  \brief Common device functions used by GPU kernels.
 */

#ifndef __ACC_COMMON_HPP__
#define __ACC_COMMON_HPP__

#include <stdio.h>
#include "acc.hpp"
#include "acc_runtime.hpp"
#include "core/typedefs.hpp"

namespace sirius {

template <>
struct Real<gpu_complex_type<float>>
{
    using type = float;
};

template <>
struct Real<gpu_complex_type<double>>
{
    using type = double;
};

template <typename T>
using real_type = typename Real<T>::type;

namespace acc {

const double twopi = 6.2831853071795864769;

inline __device__ size_t
array2D_offset(int i0, int i1, int ld0)
{
    return i0 + i1 * ld0;
}

inline __device__ size_t
array3D_offset(int i0, int i1, int i2, int ld0, int ld1)
{
    return i0 + ld0 * (i1 + i2 * ld1);
}

inline __device__ size_t
array4D_offset(int i0, int i1, int i2, int i3, int ld0, int ld1, int ld2)
{
    return i0 + ld0 * (i1 + ld1 * (i2 + i3 * ld2));
}

inline __host__ __device__ int
num_blocks(int length, int block_size)
{
    return (length / block_size) + ((length % block_size) ? 1 : 0);
}

inline __device__ auto
add_accNumbers(double x, double y)
{
    return x + y;
}

inline __device__ auto
add_accNumbers(float x, float y)
{
    return x + y;
}

inline __device__ auto
add_accNumbers(gpu_complex_type<double> x, gpu_complex_type<double> y)
{
    return accCadd(x, y);
}

inline __device__ auto
add_accNumbers(gpu_complex_type<float> x, gpu_complex_type<float> y)
{
    return accCaddf(x, y);
}

inline __device__ auto
sub_accNumbers(double x, double y)
{
    return x - y;
}

inline __device__ auto
sub_accNumbers(float x, float y)
{
    return x - y;
}

inline __device__ auto
sub_accNumbers(gpu_complex_type<double> x, gpu_complex_type<double> y)
{
    return accCsub(x, y);
}

inline __device__ auto
sub_accNumbers(gpu_complex_type<float> x, gpu_complex_type<float> y)
{
    return accCsubf(x, y);
}

inline __device__ auto
make_accComplex(float x, float y)
{
    return make_accFloatComplex(x, y);
}

inline __device__ auto
make_accComplex(double x, double y)
{
    return make_accDoubleComplex(x, y);
}

inline __device__ auto
mul_accNumbers(gpu_complex_type<double> x, gpu_complex_type<double> y)
{
    return accCmul(x, y);
}

inline __device__ auto
mul_accNumbers(double x, gpu_complex_type<double> y)
{
    return make_accComplex(x * y.x, x * y.y);
}

inline __device__ auto
mul_accNumbers(gpu_complex_type<float> x, gpu_complex_type<float> y)
{
    return accCmulf(x, y);
}

inline __device__ auto
mul_accNumbers(float x, gpu_complex_type<float> y)
{
    return make_accComplex(x * y.x, x * y.y);
}

template <typename T>
inline __device__ auto
accZero();

template <>
inline __device__ auto
accZero<double>()
{
    return 0;
}

template <>
inline __device__ auto
accZero<float>()
{
    return 0;
}

template <>
inline __device__ auto
accZero<gpu_complex_type<double>>()
{
    return make_accComplex(double{0}, double{0});
}

template <>
inline __device__ auto
accZero<gpu_complex_type<float>>()
{
    return make_accComplex(float{0}, float{0});
}

inline bool __device__
is_zero(gpu_complex_type<float> x)
{
    return (x.x == 0.0) && (x.y == 0);
}

inline bool __device__
is_zero(gpu_complex_type<double> x)
{
    return (x.x == 0.0) && (x.y == 0);
}

inline bool __device__
is_zero(float x)
{
    return x == 0.0;
}

inline bool __device__
is_zero(double x)
{
    return x == 0.0;
}

} // namespace acc
} // namespace sirius

#endif
