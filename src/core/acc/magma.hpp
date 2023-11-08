// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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
