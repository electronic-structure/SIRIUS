/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file omp.hpp
 *
 *  \brief Add or substitute OMP functions.
 */

#ifndef __OMP_HPP__
#define __OMP_HPP__

#include <complex>

#if defined(_OPENMP)

#include <omp.h>

/* NVC++ does not support user-defined OpenMP reductions ("declare reduction for user defined
 * reductions" is not part of the "NVIDIA subset" of OpenMP); guard it out for that compiler. */
#if !defined(__NVCOMPILER)
#pragma omp declare reduction( \
    complex_double_plus : std::complex<double> : omp_out += omp_in \
) initializer(omp_priv = std::complex<double>{0.0, 0.0})
#endif

#else
inline int
omp_get_max_threads()
{
    return 1;
}

inline int
omp_get_thread_num()
{
    return 0;
}

inline void
omp_set_nested(int i)
{
}

inline int
omp_get_num_threads()
{
    return 1;
}

inline double
omp_get_wtime()
{
    return 0;
}
#endif

#endif
