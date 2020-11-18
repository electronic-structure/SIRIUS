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

/** \file acc_blas.hpp
 *
 *  \brief Blas functions for execution on GPUs.
 */

#ifndef __ACCBLAS_HPP__
#define __ACCBLAS_HPP__

#include <unistd.h>
#include <vector>
#include "acc_blas_api.hpp"
#include "acc.hpp"

namespace accblas {

#ifdef SIRIUS_CUDA
inline const char*
error_message(::acc::blas::status_t status)
{
    switch (status) {
        case CUBLAS_STATUS_NOT_INITIALIZED: {
            return "the library was not initialized";
            break;
        }
        case CUBLAS_STATUS_INVALID_VALUE: {
            return "the parameters m,n,k<0";
            break;
        }
        case CUBLAS_STATUS_ARCH_MISMATCH: {
            return "the device does not support double-precision";
            break;
        }
        case CUBLAS_STATUS_EXECUTION_FAILED: {
            return "the function failed to launch on the GPU";
            break;
        }
        default: {
            return "gpublas status unknown";
        }
    }
}
#else
inline const char*
error_message(::acc::blas::status_t status)
{
    return rocblas_status_to_string(status);
}
#endif

inline ::acc::blas::operation_t
get_gpublasOperation_t(char c)
{
    switch (c) {
        case 'n':
        case 'N': {
            return ::acc::blas::operation::None;
        }
        case 't':
        case 'T': {
            return ::acc::blas::operation::Transpose;
        }
        case 'c':
        case 'C': {
            return ::acc::blas::operation::ConjugateTranspose;
        }
        default: {
            throw std::runtime_error("get_gpublasOperation_t(): wrong operation");
        }
    }
    return ::acc::blas::operation::None; // make compiler happy
}

inline ::acc::blas::side_mode_t
get_gpublasSideMode_t(char c)
{
    switch (c) {
        case 'l':
        case 'L': {
            return ::acc::blas::side::Left;
        }
        case 'r':
        case 'R': {
            return ::acc::blas::side::Right;
        }
        default: {
            throw std::runtime_error("get_gpublasSideMode_t(): wrong side");
        }
    }
    return ::acc::blas::side::Left; // make compiler happy
}

inline ::acc::blas::fill_mode_t
get_gpublasFillMode_t(char c)
{
    switch (c) {
        case 'u':
        case 'U': {
            return ::acc::blas::fill::Upper;
        }
        case 'l':
        case 'L': {
            return ::acc::blas::fill::Lower;
        }
        default: {
            throw std::runtime_error("get_gpublasFillMode_t(): wrong mode");
        }
    }
    return ::acc::blas::fill::Upper; // make compiler happy
}

inline ::acc::blas::diagonal_t
get_gpublasDiagonal_t(char c)
{
    switch (c) {
        case 'n':
        case 'N': {
            return ::acc::blas::diagonal::NonUnit;
        }
        case 'u':
        case 'U': {
            return ::acc::blas::diagonal::Unit;
        }
        default: {
            throw std::runtime_error("get_gpublasDiagonal_t(): wrong diagonal type");
        }
    }
    return ::acc::blas::diagonal::NonUnit; // make compiler happy
}

#define CALL_GPU_BLAS(func__, args__)                                                                                  \
    {                                                                                                                  \
        ::acc::blas::status_t status;                                                                                  \
        if ((status = func__ args__) != ::acc::blas::status::Success) {                                                \
            error_message(status);                                                                                     \
            char nm[1024];                                                                                             \
            gethostname(nm, 1024);                                                                                     \
            std::printf("hostname: %s\n", nm);                                                                         \
            std::printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__);                           \
            stack_backtrace();                                                                                         \
        }                                                                                                              \
    }

/// Store the default (null) stream handler.
::acc::blas::handle_t& null_stream_handle();


/// Store the gpublas handlers associated with acc streams.
std::vector<::acc::blas::handle_t>& stream_handles();

inline void
create_stream_handles()
{
    // acc::set_device();
    CALL_GPU_BLAS(::acc::blas::create, (&null_stream_handle()));

    stream_handles() = std::vector<::acc::blas::handle_t>(acc::num_streams());
    for (int i = 0; i < acc::num_streams(); i++) {
        CALL_GPU_BLAS(::acc::blas::create, (&stream_handles()[i]));

        CALL_GPU_BLAS(::acc::blas::set_stream, (stream_handles()[i], acc::stream(stream_id(i))));
    }
}

inline void
destroy_stream_handles()
{
    // acc::set_device();
    CALL_GPU_BLAS(::acc::blas::destroy, (null_stream_handle()));
    for (int i = 0; i < acc::num_streams(); i++) {
        CALL_GPU_BLAS(::acc::blas::destroy, (stream_handles()[i]));
    }
}

inline ::acc::blas::handle_t
stream_handle(int id__)
{
    return (id__ == -1) ? null_stream_handle() : stream_handles()[id__];
}

inline void
zgemv(char transa, int32_t m, int32_t n, acc_complex_double_t* alpha, acc_complex_double_t* a, int32_t lda,
      acc_complex_double_t* x, int32_t incx, acc_complex_double_t* beta, acc_complex_double_t* y, int32_t incy,
      int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(::acc::blas::zgemv, (stream_handle(stream_id), get_gpublasOperation_t(transa), m, n,
                                       reinterpret_cast<const ::acc::blas::complex_double_t*>(alpha),
                                       reinterpret_cast<const ::acc::blas::complex_double_t*>(a), lda,
                                       reinterpret_cast<const ::acc::blas::complex_double_t*>(x), incx,
                                       reinterpret_cast<const ::acc::blas::complex_double_t*>(beta),
                                       reinterpret_cast<::acc::blas::complex_double_t*>(y), incy));
}

inline void
zgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, acc_complex_double_t const* alpha,
      acc_complex_double_t const* a, int32_t lda, acc_complex_double_t const* b, int32_t ldb,
      acc_complex_double_t const* beta, acc_complex_double_t* c, int32_t ldc, int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(::acc::blas::zgemm,
                  (stream_handle(stream_id), get_gpublasOperation_t(transa), get_gpublasOperation_t(transb), m, n, k,
                   reinterpret_cast<const ::acc::blas::complex_double_t*>(alpha),
                   reinterpret_cast<const ::acc::blas::complex_double_t*>(a), lda,
                   reinterpret_cast<const ::acc::blas::complex_double_t*>(b), ldb,
                   reinterpret_cast<const ::acc::blas::complex_double_t*>(beta),
                   reinterpret_cast<::acc::blas::complex_double_t*>(c), ldc));
}

inline void
dgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, double const* alpha, double const* a, int32_t lda,
      double const* b, int32_t ldb, double const* beta, double* c, int32_t ldc, int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(::acc::blas::dgemm, (stream_handle(stream_id), get_gpublasOperation_t(transa),
                                       get_gpublasOperation_t(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

inline void
dtrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, double const* alpha__, double const* A__,
      int lda__, double* B__, int ldb__, int stream_id)
{
    ::acc::blas::side_mode_t side   = get_gpublasSideMode_t(side__);
    ::acc::blas::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    ::acc::blas::operation_t transa = get_gpublasOperation_t(transa__);
    ::acc::blas::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
#ifdef SIRIUS_CUDA
    CALL_GPU_BLAS(::acc::blas::dtrmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__, alpha__, A__,
                                       lda__, B__, ldb__, B__, ldb__));
#else
    // rocblas trmm function does not take three matrices
    CALL_GPU_BLAS(::acc::blas::dtrmm,
                  (stream_handle(stream_id), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__));
#endif
}

inline void
ztrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, acc_complex_double_t const* alpha__,
      acc_complex_double_t const* A__, int lda__, acc_complex_double_t* B__, int ldb__, int stream_id)
{
    ::acc::blas::side_mode_t side   = get_gpublasSideMode_t(side__);
    ::acc::blas::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    ::acc::blas::operation_t transa = get_gpublasOperation_t(transa__);
    ::acc::blas::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
#ifdef SIRIUS_CUDA
    CALL_GPU_BLAS(::acc::blas::ztrmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__,
                                       reinterpret_cast<const ::acc::blas::complex_double_t*>(alpha__),
                                       reinterpret_cast<const ::acc::blas::complex_double_t*>(A__), lda__,
                                       reinterpret_cast<::acc::blas::complex_double_t*>(B__), ldb__,
                                       reinterpret_cast<::acc::blas::complex_double_t*>(B__), ldb__));
#else
    // rocblas trmm function does not take three matrices
    CALL_GPU_BLAS(::acc::blas::ztrmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__,
                                       reinterpret_cast<const ::acc::blas::complex_double_t*>(alpha__),
                                       reinterpret_cast<const ::acc::blas::complex_double_t*>(A__), lda__,
                                       reinterpret_cast<::acc::blas::complex_double_t*>(B__), ldb__));
#endif
}

inline void
dger(int m, int n, double const* alpha, double const* x, int incx, double const* y, int incy, double* A, int lda,
     int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(::acc::blas::dger, (stream_handle(stream_id), m, n, alpha, x, incx, y, incy, A, lda));
}

inline void
zgeru(int m, int n, acc_complex_double_t const* alpha, acc_complex_double_t const* x, int incx,
      acc_complex_double_t const* y, int incy, acc_complex_double_t* A, int lda, int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(::acc::blas::zgeru,
                  (stream_handle(stream_id), m, n, reinterpret_cast<const ::acc::blas::complex_double_t*>(alpha),
                   reinterpret_cast<const ::acc::blas::complex_double_t*>(x), incx,
                   reinterpret_cast<const ::acc::blas::complex_double_t*>(y), incy,
                   reinterpret_cast<::acc::blas::complex_double_t*>(A), lda));
}

inline void
zaxpy(int n__, acc_complex_double_t const* alpha__, acc_complex_double_t const* x__, int incx__,
      acc_complex_double_t* y__, int incy__)
{
    // acc::set_device();
    CALL_GPU_BLAS(::acc::blas::zaxpy,
                  (null_stream_handle(), n__, reinterpret_cast<const ::acc::blas::complex_double_t*>(alpha__),
                   reinterpret_cast<const ::acc::blas::complex_double_t*>(x__), incx__,
                   reinterpret_cast<::acc::blas::complex_double_t*>(y__), incy__));
}

inline void
dscal(int n__, double const* alpha__, double * x__, int incx__)
{
    // acc::set_device();
    CALL_GPU_BLAS(::acc::blas::dscal,
                  (null_stream_handle(), n__, alpha__, x__, incx__));
}

#if defined(SIRIUS_CUDA)
namespace xt {

cublasXtHandle_t& cublasxt_handle();

inline void
create_handle()
{
    int device_id[1];
    device_id[0] = acc::get_device_id();
    CALL_GPU_BLAS(cublasXtCreate, (&cublasxt_handle()));
    CALL_GPU_BLAS(cublasXtDeviceSelect, (cublasxt_handle(), 1, device_id));
    CALL_GPU_BLAS(cublasXtSetBlockDim, (cublasxt_handle(), 4096));
}

inline void
destroy_handle()
{
    CALL_GPU_BLAS(cublasXtDestroy, (cublasxt_handle()));
}

inline void
zgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, acc_complex_double_t const* alpha,
      acc_complex_double_t const* a, int32_t lda, acc_complex_double_t const* b, int32_t ldb,
      acc_complex_double_t const* beta, acc_complex_double_t* c, int32_t ldc)
{
    // acc::set_device();
    CALL_GPU_BLAS(cublasXtZgemm, (cublasxt_handle(), get_gpublasOperation_t(transa), get_gpublasOperation_t(transb), m,
                                  n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

inline void
dgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, double const* alpha, double const* a, int32_t lda,
      double const* b, int32_t ldb, double const* beta, double* c, int32_t ldc)
{
    // acc::set_device();
    CALL_GPU_BLAS(cublasXtDgemm, (cublasxt_handle(), get_gpublasOperation_t(transa), get_gpublasOperation_t(transb), m,
                                  n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

inline void
dtrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, double const* alpha__, double const* A__,
      int lda__, double* B__, int ldb__)
{
    ::acc::blas::side_mode_t side   = get_gpublasSideMode_t(side__);
    ::acc::blas::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    ::acc::blas::operation_t transa = get_gpublasOperation_t(transa__);
    ::acc::blas::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
    CALL_GPU_BLAS(cublasXtDtrmm,
                  (cublasxt_handle(), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__, B__, ldb__));
}

inline void
ztrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, acc_complex_double_t const* alpha__,
      acc_complex_double_t const* A__, int lda__, acc_complex_double_t* B__, int ldb__)
{
    ::acc::blas::side_mode_t side   = get_gpublasSideMode_t(side__);
    ::acc::blas::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    ::acc::blas::operation_t transa = get_gpublasOperation_t(transa__);
    ::acc::blas::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
    CALL_GPU_BLAS(cublasXtZtrmm,
                  (cublasxt_handle(), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__, B__, ldb__));
}

} // namespace xt
#endif

} // namespace gpublas

#endif
