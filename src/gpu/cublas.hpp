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
#include "acc.hpp"

extern "C" cublasStatus_t cublasGetError();

namespace cublas {

inline void error_message(cublasStatus_t status)
{
    switch (status) {
        case CUBLAS_STATUS_NOT_INITIALIZED: {
            std::printf("the library was not initialized\n");
            break;
        }
        case CUBLAS_STATUS_INVALID_VALUE: {
            std::printf("the parameters m,n,k<0\n");
            break;
        }
        case CUBLAS_STATUS_ARCH_MISMATCH: {
            std::printf("the device does not support double-precision\n");
            break;
        }
        case CUBLAS_STATUS_EXECUTION_FAILED: {
            std::printf("the function failed to launch on the GPU\n");
            break;
        }
        default: {
            std::printf("cublas status unknown");
        }
    }
}

inline cublasOperation_t get_cublasOperation_t(char c)
{
    switch (c) {
        case 'n':
        case 'N': {
            return CUBLAS_OP_N;
        }
        case 't':
        case 'T': {
            return CUBLAS_OP_T;
        }
        case 'c':
        case 'C': {
            return CUBLAS_OP_C;
        }
        default: {
            throw std::runtime_error("get_cublasOperation_t(): wrong operation");
        }
    }
    return CUBLAS_OP_N; // make compiler happy
}

inline cublasSideMode_t get_cublasSideMode_t(char c)
{
    switch (c) {
        case 'l':
        case 'L': {
            return CUBLAS_SIDE_LEFT;
        }
        case 'r':
        case 'R': {
            return CUBLAS_SIDE_RIGHT;
        }
        default: {
            throw std::runtime_error("get_cublasSideMode_t(): wrong side");
        }
    }
    return CUBLAS_SIDE_LEFT; //make compiler happy
}

inline cublasFillMode_t get_cublasFillMode_t(char c)
{
    switch (c) {
        case 'u':
        case 'U': {
            return CUBLAS_FILL_MODE_UPPER;
        }
        case 'l':
        case 'L': {
            return CUBLAS_FILL_MODE_LOWER;
        }
        default: {
            throw std::runtime_error("get_cublasFillMode_t(): wrong mode");
        }
    }
    return CUBLAS_FILL_MODE_UPPER; // make compiler happy
}

inline cublasDiagType_t get_cublasDiagType_t(char c)
{
    switch (c) {
        case 'n':
        case 'N': {
            return CUBLAS_DIAG_NON_UNIT;
        }
        case 'u':
        case 'U': {
            return CUBLAS_DIAG_UNIT;
        }
        default: {
            throw std::runtime_error("get_cublasDiagType_t(): wrong diagonal type");
        }
    }
    return CUBLAS_DIAG_NON_UNIT; // make compiler happy
}

#ifdef NDEBUG
#define CALL_CUBLAS(func__, args__)                                                 \
{                                                                                   \
    cublasStatus_t status;                                                          \
    if ((status = func__ args__) != CUBLAS_STATUS_SUCCESS) {                        \
        error_message(status);                                                      \
        char nm[1024];                                                              \
        gethostname(nm, 1024);                                                      \
        std::printf("hostname: %s\n", nm);                                               \
        std::printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
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
        std::printf("hostname: %s\n", nm);                                               \
        std::printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
        stack_backtrace();                                                          \
    }                                                                               \
}
#endif

/// Store the default (null) stream handler.
inline cublasHandle_t& null_stream_handle()
{
    static cublasHandle_t null_stream_handle_;
    return null_stream_handle_;
}

/// Store the cublas handlers associated with cuda streams.
inline std::vector<cublasHandle_t>& stream_handles()
{
    static std::vector<cublasHandle_t> stream_handles_;
    return stream_handles_;
}

inline void create_stream_handles()
{
    //acc::set_device();
    CALL_CUBLAS(cublasCreate, (&null_stream_handle()));

    stream_handles() = std::vector<cublasHandle_t>(acc::num_streams());
    for (int i = 0; i < acc::num_streams(); i++) {
        CALL_CUBLAS(cublasCreate, (&stream_handles()[i]));

        CALL_CUBLAS(cublasSetStream, (stream_handles()[i], acc::stream(stream_id(i))));
    }
}

inline void destroy_stream_handles()
{
    //acc::set_device();
    CALL_CUBLAS(cublasDestroy, (null_stream_handle()));
    for (int i = 0; i < acc::num_streams(); i++) {
        CALL_CUBLAS(cublasDestroy, (stream_handles()[i]));
    }
}

inline cublasHandle_t stream_handle(int id__)
{
    return (id__ == -1) ? null_stream_handle() : stream_handles()[id__];
}

inline void zgemv(char transa, int32_t m, int32_t n, cuDoubleComplex* alpha, cuDoubleComplex* a, int32_t lda, 
                  cuDoubleComplex* x, int32_t incx, cuDoubleComplex* beta, cuDoubleComplex* y, int32_t incy, int stream_id)
{
    //acc::set_device();
    CALL_CUBLAS(cublasZgemv, (stream_handle(stream_id), get_cublasOperation_t(transa), m, n, alpha, a, lda, x, incx, beta, y, incy));
}

inline void zgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, 
                  cuDoubleComplex const* alpha, cuDoubleComplex const* a, int32_t lda, cuDoubleComplex const* b, 
                  int32_t ldb, cuDoubleComplex const* beta, cuDoubleComplex* c, int32_t ldc, int stream_id)
{
    //acc::set_device();
    CALL_CUBLAS(cublasZgemm, (stream_handle(stream_id), get_cublasOperation_t(transa), get_cublasOperation_t(transb),
                              m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

inline void dgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, 
                  double const* alpha, double const* a, int32_t lda, double const* b, 
                  int32_t ldb, double const* beta, double* c, int32_t ldc, int stream_id)
{
    //acc::set_device();
    CALL_CUBLAS(cublasDgemm, (stream_handle(stream_id), get_cublasOperation_t(transa), get_cublasOperation_t(transb), 
                              m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

inline void dtrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__,
                  double const* alpha__, double const* A__, int lda__, double* B__, int ldb__, int stream_id)
{
    cublasSideMode_t side = get_cublasSideMode_t(side__);
    cublasFillMode_t uplo = get_cublasFillMode_t(uplo__);
    cublasOperation_t transa = get_cublasOperation_t(transa__);
    cublasDiagType_t diag = get_cublasDiagType_t(diag__);
    //acc::set_device();
    CALL_CUBLAS(cublasDtrmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__, B__, ldb__));
}

inline void ztrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__,
                  cuDoubleComplex const* alpha__, cuDoubleComplex const* A__, int lda__, cuDoubleComplex* B__,
                  int ldb__, int stream_id)
{
    cublasSideMode_t side = get_cublasSideMode_t(side__);
    cublasFillMode_t uplo = get_cublasFillMode_t(uplo__);
    cublasOperation_t transa = get_cublasOperation_t(transa__);
    cublasDiagType_t diag = get_cublasDiagType_t(diag__);
    //acc::set_device();
    CALL_CUBLAS(cublasZtrmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__, B__, ldb__));
}

inline void dger(int           m,
                 int           n,
                 double const* alpha,
                 double const* x,
                 int           incx,
                 double const* y,
                 int           incy,
                 double*       A,
                 int           lda,
                 int           stream_id)
{
    //acc::set_device();
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
    //acc::set_device();
    CALL_CUBLAS(cublasZgeru, (stream_handle(stream_id), m, n, alpha, x, incx, y, incy, A, lda));
}

inline void zaxpy(int                    n__,
                  cuDoubleComplex const* alpha__,
                  cuDoubleComplex const* x__,
                  int                    incx__,
                  cuDoubleComplex*       y__,
                  int                    incy__)
{
    //acc::set_device();
    CALL_CUBLAS(cublasZaxpy, (null_stream_handle(), n__, alpha__, x__, incx__, y__, incy__));
}

namespace xt {

inline cublasXtHandle_t& cublasxt_handle()
{
    static cublasXtHandle_t handle;
    return handle;
}

inline void create_handle()
{
    int device_id[1];
    device_id[0] = acc::get_device_id();
    CALL_CUBLAS(cublasXtCreate, (&cublasxt_handle()));
    CALL_CUBLAS(cublasXtDeviceSelect, (cublasxt_handle(), 1, device_id));
    CALL_CUBLAS(cublasXtSetBlockDim, (cublasxt_handle(), 4096));
}

inline void destroy_handle()
{
    CALL_CUBLAS(cublasXtDestroy, (cublasxt_handle()));
}

inline void zgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, 
                  cuDoubleComplex const* alpha, cuDoubleComplex const* a, int32_t lda, cuDoubleComplex const* b,
                  int32_t ldb, cuDoubleComplex const* beta, cuDoubleComplex* c, int32_t ldc)
{
    //acc::set_device();
    CALL_CUBLAS(cublasXtZgemm, (cublasxt_handle(), get_cublasOperation_t(transa), get_cublasOperation_t(transb),
                                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

inline void dgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, 
                  double const* alpha, double const* a, int32_t lda, double const* b, 
                  int32_t ldb, double const* beta, double* c, int32_t ldc)
{
    //acc::set_device();
    CALL_CUBLAS(cublasXtDgemm, (cublasxt_handle(), get_cublasOperation_t(transa), get_cublasOperation_t(transb),
                                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

inline void dtrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__,
                  double const* alpha__, double const* A__, int lda__, double* B__, int ldb__)
{
    cublasSideMode_t side = get_cublasSideMode_t(side__);
    cublasFillMode_t uplo = get_cublasFillMode_t(uplo__);
    cublasOperation_t transa = get_cublasOperation_t(transa__);
    cublasDiagType_t diag = get_cublasDiagType_t(diag__);
    //acc::set_device();
    CALL_CUBLAS(cublasXtDtrmm, (cublasxt_handle(), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__, B__, ldb__));
}

inline void ztrmm(char             side__,
                  char             uplo__,
                  char             transa__,
                  char             diag__,
                  int              m__,
                  int              n__,
                  cuDoubleComplex const* alpha__,
                  cuDoubleComplex const* A__,
                  int              lda__,
                  cuDoubleComplex* B__,
                  int              ldb__)
{
    cublasSideMode_t side = get_cublasSideMode_t(side__);
    cublasFillMode_t uplo = get_cublasFillMode_t(uplo__);
    cublasOperation_t transa = get_cublasOperation_t(transa__);
    cublasDiagType_t diag = get_cublasDiagType_t(diag__);
    //acc::set_device();
    CALL_CUBLAS(cublasXtZtrmm, (cublasxt_handle(), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__, B__, ldb__));
}

} // namespace xt

} // namespace cublas

#endif
