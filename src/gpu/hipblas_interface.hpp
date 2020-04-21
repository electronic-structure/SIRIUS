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

/** \file hipblas.hpp
 *
 *  \brief Interface to hipblas related functions.
 */

#ifndef __HIP_BLAS_INTERFACE_HPP__
#define __HIP_BLAS_INTERFACE_HPP__

#include <unistd.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#include <vector>
#include <hipblas.h>
#include "acc.hpp"
#include "hipblas_port.h"
// #include "blas_lapack.h"

namespace hipblas {

inline void error_message(hipblasStatus_t status)
{
    switch (status) {
        case HIPBLAS_STATUS_NOT_INITIALIZED: {
            std::printf("the library was not initialized\n");
            break;
        }
        case HIPBLAS_STATUS_INVALID_VALUE: {
            std::printf("the parameters m,n,k<0\n");
            break;
        }
        case HIPBLAS_STATUS_ARCH_MISMATCH: {
            std::printf("the device does not support double-precision\n");
            break;
        }
        case HIPBLAS_STATUS_EXECUTION_FAILED: {
            std::printf("the function failed to launch on the GPU\n");
            break;
        }
        default: {
            std::printf("hipblas status unknown");
        }
    }
}

inline hipblasOperation_t get_hipblasOperation_t(char c)
{
    switch (c) {
        case 'n':
        case 'N': {
            return HIPBLAS_OP_N;
        }
        case 't':
        case 'T': {
            return HIPBLAS_OP_T;
        }
        case 'c':
        case 'C': {
            return HIPBLAS_OP_C;
        }
        default: {
            throw std::runtime_error("get_hipblasOperation_t(): wrong operation");
        }
    }
    return HIPBLAS_OP_N; // make compiler happy
}

inline hipblasSideMode_t get_hipblasSideMode_t(char c)
{
    switch (c) {
        case 'l':
        case 'L': {
            return HIPBLAS_SIDE_LEFT;
        }
        case 'r':
        case 'R': {
            return HIPBLAS_SIDE_RIGHT;
        }
        default: {
            throw std::runtime_error("get_hipblasSideMode_t(): wrong side");
        }
    }
    return HIPBLAS_SIDE_LEFT; //make compiler happy
}

inline hipblasFillMode_t get_hipblasFillMode_t(char c)
{
    switch (c) {
        case 'u':
        case 'U': {
            return HIPBLAS_FILL_MODE_UPPER;
        }
        case 'l':
        case 'L': {
            return HIPBLAS_FILL_MODE_LOWER;
        }
        default: {
            throw std::runtime_error("get_hipblasFillMode_t(): wrong mode");
        }
    }
    return HIPBLAS_FILL_MODE_UPPER; // make compiler happy
}

inline hipblasDiagType_t get_hipblasDiagType_t(char c)
{
    switch (c) {
        case 'n':
        case 'N': {
            return HIPBLAS_DIAG_NON_UNIT;
        }
        case 'u':
        case 'U': {
            return HIPBLAS_DIAG_UNIT;
        }
        default: {
            throw std::runtime_error("get_hipblasDiagType_t(): wrong diagonal type");
        }
    }
    return HIPBLAS_DIAG_NON_UNIT; // make compiler happy
}

#ifdef NDEBUG
#define CALL_HIPBLAS(func__, args__)                                                                                   \
    {                                                                                                                  \
        hipblasStatus_t status;                                                                                        \
        if ((status = func__ args__) != HIPBLAS_STATUS_SUCCESS) {                                                      \
            error_message(status);                                                                                     \
            char nm[1024];                                                                                             \
            gethostname(nm, 1024);                                                                                     \
            std::printf("hostname: %s\n", nm);                                                                              \
            std::printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__);                                \
            stack_backtrace();                                                                                         \
        }                                                                                                              \
        hipDeviceSynchronize();                                                                                        \
    }
#else
#define CALL_HIPBLAS(func__, args__)                                                                                   \
    {                                                                                                                  \
        hipblasStatus_t status;                                                                                        \
        if ((status = func__ args__) != HIPBLAS_STATUS_SUCCESS) {                                                      \
            error_message(status);                                                                                     \
            char nm[1024];                                                                                             \
            gethostname(nm, 1024);                                                                                     \
            std::printf("hostname: %s\n", nm);                                                                              \
            std::printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__);                                \
            stack_backtrace();                                                                                         \
        }                                                                                                              \
        hipDeviceSynchronize();                                                                                        \
    }
#endif

/// Store the default (null) stream handler.
inline hipblasHandle_t& null_stream_handle()
{
    static hipblasHandle_t null_stream_handle_;
    return null_stream_handle_;
}

/// Store the hipblas handlers associated with hip streams.
inline std::vector<hipblasHandle_t>& stream_handles()
{
    static std::vector<hipblasHandle_t> stream_handles_;
    return stream_handles_;
}

inline void create_stream_handles()
{
    CALL_HIPBLAS(hipblasCreate, (&null_stream_handle()));

    stream_handles() = std::vector<hipblasHandle_t>(acc::num_streams());
    for (int i = 0; i < acc::num_streams(); i++) {
        CALL_HIPBLAS(hipblasCreate, (&stream_handles()[i]));

        CALL_HIPBLAS(hipblasSetStream, (stream_handles()[i], acc::stream(stream_id(i))));
    }
}

inline void destroy_stream_handles()
{
    CALL_HIPBLAS(hipblasDestroy, (null_stream_handle()));
    for (int i = 0; i < acc::num_streams(); i++) {
        CALL_HIPBLAS(hipblasDestroy, (stream_handles()[i]));
    }
}

inline hipblasHandle_t stream_handle(int id)
{
    return (id == -1) ? null_stream_handle() : stream_handles()[id];
}

inline void zgemv(char transa, int32_t m, int32_t n, hipDoubleComplex* alpha, hipDoubleComplex* a, int32_t lda,
                  hipDoubleComplex* x, int32_t incx, hipDoubleComplex* beta, hipDoubleComplex* y, int32_t incy,
                  int stream_id)
{
    CALL_HIPBLAS(hipblas_port_Zgemv, (stream_handle(stream_id), get_hipblasOperation_t(transa), m, n, alpha, a, lda, x,
                                      incx, beta, y, incy));
}

inline void zgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, hipDoubleComplex const* alpha,
                  hipDoubleComplex const* a, int32_t lda, hipDoubleComplex const* b, int32_t ldb,
                  hipDoubleComplex const* beta, hipDoubleComplex* c, int32_t ldc, int stream_id)
{
    CALL_HIPBLAS(hipblas_port_Zgemm, (stream_handle(stream_id), get_hipblasOperation_t(transa),
                                      get_hipblasOperation_t(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

inline void dgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, double const* alpha, double const* a,
                  int32_t lda, double const* b, int32_t ldb, double const* beta, double* c, int32_t ldc, int stream_id)
{
    CALL_HIPBLAS(hipblasDgemm, (stream_handle(stream_id), get_hipblasOperation_t(transa),
                                get_hipblasOperation_t(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

inline void dtrmm(char side, char uplo, char transa, char diag, int m, int n, double const* alpha,
                  double const* A, int lda, double* B, int ldb)
{
    hipblasSideMode_t side_gpu    = get_hipblasSideMode_t(side);
    hipblasFillMode_t uplo_gpu    = get_hipblasFillMode_t(uplo);
    hipblasOperation_t transa_gpu = get_hipblasOperation_t(transa);
    hipblasDiagType_t diag_gpu    = get_hipblasDiagType_t(diag);
    CALL_HIPBLAS(hipblas_port_Dtrmm,
                 (null_stream_handle(), side_gpu, uplo_gpu, transa_gpu, diag_gpu, m, n, alpha, A, lda, B, ldb, B, ldb));
}

inline void ztrmm(char side, char uplo, char transa, char diag, int m, int n,
                  hipDoubleComplex const* alpha, hipDoubleComplex const* A, int lda, hipDoubleComplex* B,
                  int ldb)
{
    hipblasSideMode_t side_gpu    = get_hipblasSideMode_t(side);
    hipblasFillMode_t uplo_gpu    = get_hipblasFillMode_t(uplo);
    hipblasOperation_t transa_gpu = get_hipblasOperation_t(transa);
    hipblasDiagType_t diag_gpu    = get_hipblasDiagType_t(diag);
    CALL_HIPBLAS(hipblas_port_Ztrmm,
                 (null_stream_handle(), side_gpu, uplo_gpu, transa_gpu, diag_gpu, m, n, alpha, A, lda, B, ldb, B, ldb));

    // copy to host, calculate, copy back
    // int size_A, size_B;
    // size_B = n * ldb;
    // if (side == 'l' || side == 'L') {
    //     if (transa == 'n' || transa == 'N')
    //         size_A = m * lda;
    //     else
    //         size_A = n * lda;
    // } else {
    //     if (transa == 'n' || transa == 'N')
    //         size_A = n * lda;
    //     else
    //         size_A = m * lda;
    // }
    // std::vector<hipDoubleComplex> A_host(size_A);
    // std::vector<hipDoubleComplex> B_host(size_B);
    // acc::copyout(A_host.data(), A, A_host.size());
    // acc::copyout(B_host.data(), B, B_host.size());
    // ftn_int mf = m;
    // ftn_int nf = n;
    // ftn_int ldaf = lda;
    // ftn_int ldbf = ldb;
    // FORTRAN(dtrmm)
    // (&side, &uplo, &transa, "N", &mf, &nf, const_cast<ftn_double*>((const ftn_double*)alpha),
    //  ((ftn_double*)A_host.data()), &ldaf, ((ftn_double*)B_host.data()), &ldbf, (ftn_len)1,
    //  (ftn_len)1, (ftn_len)1, (ftn_len)1);
    // acc::copyin(const_cast<hipDoubleComplex*>(B), B_host.data(), B_host.size());
}

inline void dger(int m, int n, double const* alpha, double const* x, int incx, double const* y, int incy, double* A,
                 int lda, int stream_id)
{
    CALL_HIPBLAS(hipblasDger, (stream_handle(stream_id), m, n, alpha, x, incx, y, incy, A, lda));
}

inline void zgeru(int m, int n, hipDoubleComplex const* alpha, hipDoubleComplex const* x, int incx,
                  hipDoubleComplex const* y, int incy, hipDoubleComplex* A, int lda, int stream_id)
{
    CALL_HIPBLAS(hipblas_port_Zgeru, (stream_handle(stream_id), m, n, alpha, x, incx, y, incy, A, lda));
}

inline void zaxpy(int n, hipDoubleComplex const* alpha, hipDoubleComplex const* x, int incx,
                  hipDoubleComplex* y, int incy)
{
    CALL_HIPBLAS(hipblas_port_Zaxpy, (null_stream_handle(), n, alpha, x, incx, y, incy));
}

} // namespace hipblas

#endif
