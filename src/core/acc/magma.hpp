/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file magma.hpp
 *
 *  \brief Interface to some of the MAGMA functions.
 */

#ifndef __MAGMA_HPP__
#define __MAGMA_HPP__

#include <stdio.h>
#include <assert.h>
#include <magma.h>
#include <magma_z.h>
#include <magma_d.h>
#include <cstring>

#include "magma_threadsetting.h"

namespace sirius {

/// Interface to MAGMA functions.
namespace magma {

inline void
init()
{
    magma_init();
}

inline void
finalize()
{
    magma_finalize();
}

inline int
spotrf(char uplo, int n, float* A, int lda)
{
    if (!(uplo == 'U' || uplo == 'L')) {
        printf("magma_spotrf_wrapper: wrong uplo\n");
        exit(-1);
    }
    magma_uplo_t magma_uplo = (uplo == 'U') ? MagmaUpper : MagmaLower;
    magma_int_t info;
    magma_spotrf_gpu(magma_uplo, n, A, lda, &info);
    return info;
}

inline int
dpotrf(char uplo, int n, double* A, int lda)
{
    if (!(uplo == 'U' || uplo == 'L')) {
        printf("magma_dpotrf_wrapper: wrong uplo\n");
        exit(-1);
    }
    magma_uplo_t magma_uplo = (uplo == 'U') ? MagmaUpper : MagmaLower;
    magma_int_t info;
    magma_dpotrf_gpu(magma_uplo, n, A, lda, &info);
    return info;
}

inline int
cpotrf(char uplo, int n, magmaFloatComplex* A, int lda)
{
    if (!(uplo == 'U' || uplo == 'L')) {
        printf("magma_cpotrf_wrapper: wrong uplo\n");
        exit(-1);
    }
    magma_uplo_t magma_uplo = (uplo == 'U') ? MagmaUpper : MagmaLower;
    magma_int_t info;
    magma_cpotrf_gpu(magma_uplo, n, A, lda, &info);
    return info;
}

inline int
zpotrf(char uplo, int n, magmaDoubleComplex* A, int lda)
{
    if (!(uplo == 'U' || uplo == 'L')) {
        printf("magma_zpotrf_wrapper: wrong uplo\n");
        exit(-1);
    }
    magma_uplo_t magma_uplo = (uplo == 'U') ? MagmaUpper : MagmaLower;
    magma_int_t info;
    magma_zpotrf_gpu(magma_uplo, n, A, lda, &info);
    return info;
}

inline int
strtri(char uplo, int n, float* A, int lda)
{
    if (!(uplo == 'U' || uplo == 'L')) {
        printf("magma_strtri_wrapper: wrong uplo\n");
        exit(-1);
    }
    magma_uplo_t magma_uplo = (uplo == 'U') ? MagmaUpper : MagmaLower;
    magma_int_t info;
    magma_strtri_gpu(magma_uplo, MagmaNonUnit, n, A, lda, &info);
    return info;
}

inline int
dtrtri(char uplo, int n, double* A, int lda)
{
    if (!(uplo == 'U' || uplo == 'L')) {
        printf("magma_dtrtri_wrapper: wrong uplo\n");
        exit(-1);
    }
    magma_uplo_t magma_uplo = (uplo == 'U') ? MagmaUpper : MagmaLower;
    magma_int_t info;
    magma_dtrtri_gpu(magma_uplo, MagmaNonUnit, n, A, lda, &info);
    return info;
}

inline int
ctrtri(char uplo, int n, magmaFloatComplex* A, int lda)
{
    if (!(uplo == 'U' || uplo == 'L')) {
        printf("magma_ctrtri_wrapper: wrong uplo\n");
        exit(-1);
    }
    magma_uplo_t magma_uplo = (uplo == 'U') ? MagmaUpper : MagmaLower;
    magma_int_t info;
    magma_ctrtri_gpu(magma_uplo, MagmaNonUnit, n, A, lda, &info);
    return info;
}

inline int
ztrtri(char uplo, int n, magmaDoubleComplex* A, int lda)
{
    if (!(uplo == 'U' || uplo == 'L')) {
        printf("magma_ztrtri_wrapper: wrong uplo\n");
        exit(-1);
    }
    magma_uplo_t magma_uplo = (uplo == 'U') ? MagmaUpper : MagmaLower;
    magma_int_t info;
    magma_ztrtri_gpu(magma_uplo, MagmaNonUnit, n, A, lda, &info);
    return info;
}

} // namespace magma

} // namespace sirius

#endif
