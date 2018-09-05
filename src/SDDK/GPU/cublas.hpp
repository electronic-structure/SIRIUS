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

/** \file cublas.hpp
 *   
 *  \brief Interface to cuBLAS related functions.
 */

#ifndef __CUBLAS__HPP__
#define __CUBLAS__HPP__

#include <unistd.h>
//#include <cublas.h>
//#include <cublas_v2.h>
#include "cuda.hpp"

extern "C" cublasStatus_t cublasGetError();

namespace cublas {

inline void error_message(cublasStatus_t status)
{
    switch (status) {
        case CUBLAS_STATUS_NOT_INITIALIZED: {
            printf("the library was not initialized\n");
            break;
        }
        case CUBLAS_STATUS_INVALID_VALUE: {
            printf("the parameters m,n,k<0\n");
            break;
        }
        case CUBLAS_STATUS_ARCH_MISMATCH: {
            printf("the device does not support double-precision\n");
            break;
        }
        case CUBLAS_STATUS_EXECUTION_FAILED: {
            printf("the function failed to launch on the GPU\n");
            break;
        }
        default: {
            printf("cublas status unknown");
        }
    }
}


#ifdef NDEBUG
#define CALL_CUBLAS(func__, args__)                                                 \
{                                                                                   \
    cublasStatus_t status;                                                          \
    if ((status = func__ args__) != CUBLAS_STATUS_SUCCESS) {                        \
        error_message(status);                                                      \
        char nm[1024];                                                              \
        gethostname(nm, 1024);                                                      \
        printf("hostname: %s\n", nm);                                               \
        printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
        stack_backtrace();                                                          \
    }                                                                               \
}
#else
#define CALL_CUBLAS(func__, args__)                                                 \
{                                                                                   \
    cublasStatus_t status;                                                          \
    func__ args__;                                                                  \
    cudaDeviceSynchronize();                                                        \
    status = cublasGetError();                                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                                          \
        error_message(status);                                                      \
        char nm[1024];                                                              \
        gethostname(nm, 1024);                                                      \
        printf("hostname: %s\n", nm);                                               \
        printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
        stack_backtrace();                                                          \
    }                                                                               \
}
#endif

inline cublasHandle_t& null_stream_handle()
{
    static cublasHandle_t null_stream_handle_;
    return null_stream_handle_;
}

inline std::vector<cublasHandle_t>& stream_handles()
{
    static std::vector<cublasHandle_t> stream_handles_;
    return stream_handles_;
}

inline void create_stream_handles()
{
    acc::set_device();
    CALL_CUBLAS(cublasCreate, (&null_stream_handle()));
    
    stream_handles() = std::vector<cublasHandle_t>(acc::num_streams());
    for (int i = 0; i < acc::num_streams(); i++) {
        CALL_CUBLAS(cublasCreate, (&stream_handles()[i]));

        CALL_CUBLAS(cublasSetStream, (stream_handles()[i], acc::stream(i)));
    }
}

inline void destroy_stream_handles()
{
    acc::set_device();
    CALL_CUBLAS(cublasDestroy, (null_stream_handle()));
    for (int i = 0; i < acc::num_streams(); i++) {
        CALL_CUBLAS(cublasDestroy, (stream_handles()[i]));
    }
}

inline cublasHandle_t stream_handle(int id__)
{
    return (id__ == -1) ? null_stream_handle() : stream_handles()[id__];
}

inline void zgemv(int transa, int32_t m, int32_t n, cuDoubleComplex* alpha, cuDoubleComplex* a, int32_t lda, 
                  cuDoubleComplex* x, int32_t incx, cuDoubleComplex* beta, cuDoubleComplex* y, int32_t incy, int stream_id)
{
    const cublasOperation_t trans[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
    acc::set_device();
    CALL_CUBLAS(cublasZgemv, (stream_handle(stream_id), trans[transa], m, n, alpha, a, lda, x, incx, beta, y, incy));
}

inline void zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                  cuDoubleComplex* alpha, cuDoubleComplex* a, int32_t lda, cuDoubleComplex* b, 
                  int32_t ldb, cuDoubleComplex* beta, cuDoubleComplex* c, int32_t ldc, int stream_id)
{
    const cublasOperation_t trans[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
    acc::set_device();
    CALL_CUBLAS(cublasZgemm, (stream_handle(stream_id), trans[transa], trans[transb], m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

inline void dgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                  double const* alpha, double const* a, int32_t lda, double const* b, 
                  int32_t ldb, double const* beta, double* c, int32_t ldc, int stream_id)
{
    const cublasOperation_t trans[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
    acc::set_device();
    CALL_CUBLAS(cublasDgemm, (stream_handle(stream_id), trans[transa], trans[transb], m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

inline void dtrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__,
                  double* alpha__, double* A__, int lda__, double* B__, int ldb__)
{
    if (!(side__ == 'L' || side__ == 'R')) {
        printf("cublas_dtrmm: wrong side\n");
        exit(-1);
    }
    cublasSideMode_t side = (side__ == 'L') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

    if (!(uplo__ == 'U' || uplo__ == 'L')) {
        printf("cublas_dtrmm: wrong uplo\n");
        exit(-1);
    }
    cublasFillMode_t uplo = (uplo__ == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

    if (!(transa__ == 'N' || transa__ == 'T' || transa__ == 'C')) {
        printf("cublas_dtrmm: wrong transa\n");
        exit(-1);
    }
    cublasOperation_t transa = CUBLAS_OP_N;
    if (transa__ == 'T') {
        transa = CUBLAS_OP_T;
    }
    if (transa__ == 'C') {
        transa = CUBLAS_OP_C;
    }

    if (!(diag__ == 'N' || diag__ == 'U')) {
        printf("cublas_dtrmm: wrong diag\n");
        exit(-1);
    }
    cublasDiagType_t diag = (diag__ == 'N') ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;
    acc::set_device();
    CALL_CUBLAS(cublasDtrmm, (null_stream_handle(), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__, B__, ldb__));
}

inline void ztrmm(char             side__,
                  char             uplo__,
                  char             transa__,
                  char             diag__,
                  int              m__,
                  int              n__,
                  cuDoubleComplex* alpha__,
                  cuDoubleComplex* A__,
                  int              lda__,
                  cuDoubleComplex* B__,
                  int              ldb__)
{
    if (!(side__ == 'L' || side__ == 'R')) {
        printf("cublas_ztrmm: wrong side\n");
        exit(-1);
    }
    cublasSideMode_t side = (side__ == 'L') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

    if (!(uplo__ == 'U' || uplo__ == 'L')) {
        printf("cublas_ztrmm: wrong uplo\n");
        exit(-1);
    }
    cublasFillMode_t uplo = (uplo__ == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

    if (!(transa__ == 'N' || transa__ == 'T' || transa__ == 'C')) {
        printf("cublas_ztrmm: wrong transa\n");
        exit(-1);
    }
    cublasOperation_t transa = CUBLAS_OP_N;
    if (transa__ == 'T') {
        transa = CUBLAS_OP_T;
    }
    if (transa__ == 'C') {
        transa = CUBLAS_OP_C;
    }

    if (!(diag__ == 'N' || diag__ == 'U')) {
        printf("cublas_ztrmm: wrong diag\n");
        exit(-1);
    }
    cublasDiagType_t diag = (diag__ == 'N') ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;
    acc::set_device();
    CALL_CUBLAS(cublasZtrmm, (null_stream_handle(), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__, B__, ldb__));
}

inline void dger(int           m,
                 int           n,
                 double const* alpha,
                 double*       x,
                 int           incx,
                 double*       y,
                 int           incy,
                 double*       A,
                 int           lda,
                 int           stream_id)
{
    acc::set_device();
    CALL_CUBLAS(cublasDger, (stream_handle(stream_id), m, n, alpha, x, incx, y, incy, A, lda));
}

inline void zgeru(int                    m,
                  int                    n,
                  cuDoubleComplex const* alpha,
                  cuDoubleComplex const* x,
                  int                    incx,
                  cuDoubleComplex const* y,
                  int                    incy,
                  cuDoubleComplex*       A,
                  int                    lda, 
                  int                    stream_id)
{
    acc::set_device();
    CALL_CUBLAS(cublasZgeru, (stream_handle(stream_id), m, n, alpha, x, incx, y, incy, A, lda));
}

inline void zaxpy(int                    n__,
                  cuDoubleComplex const* alpha__,
                  cuDoubleComplex const* x__,
                  int                    incx__,
                  cuDoubleComplex*       y__,
                  int                    incy__)
{
    acc::set_device();
    CALL_CUBLAS(cublasZaxpy, (null_stream_handle(), n__, alpha__, x__, incx__, y__, incy__));
}

} // namespace cublas

#endif
