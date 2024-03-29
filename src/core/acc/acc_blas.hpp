/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file acc_blas.hpp
 *
 *  \brief Blas functions for execution on GPUs.
 */

#ifndef __ACC_BLAS_HPP__
#define __ACC_BLAS_HPP__

#include <unistd.h>
#include <vector>
#include "acc_blas_api.hpp"
#include "acc.hpp"

namespace sirius {

namespace acc {

/// User facing interface to GPU blas functions.
namespace blas {

#ifdef SIRIUS_CUDA
inline const char*
error_message(acc::blas_api::status_t status)
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
error_message(acc::blas_api::status_t status)
{
    return rocblas_status_to_string(status);
}
#endif

inline acc::blas_api::operation_t
get_gpublasOperation_t(char c)
{
    switch (c) {
        case 'n':
        case 'N': {
            return acc::blas_api::operation::None;
        }
        case 't':
        case 'T': {
            return acc::blas_api::operation::Transpose;
        }
        case 'c':
        case 'C': {
            return acc::blas_api::operation::ConjugateTranspose;
        }
        default: {
            throw std::runtime_error("get_gpublasOperation_t(): wrong operation");
        }
    }
    return acc::blas_api::operation::None; // make compiler happy
}

inline acc::blas_api::side_mode_t
get_gpublasSideMode_t(char c)
{
    switch (c) {
        case 'l':
        case 'L': {
            return acc::blas_api::side::Left;
        }
        case 'r':
        case 'R': {
            return acc::blas_api::side::Right;
        }
        default: {
            throw std::runtime_error("get_gpublasSideMode_t(): wrong side");
        }
    }
    return acc::blas_api::side::Left; // make compiler happy
}

inline acc::blas_api::fill_mode_t
get_gpublasFillMode_t(char c)
{
    switch (c) {
        case 'u':
        case 'U': {
            return acc::blas_api::fill::Upper;
        }
        case 'l':
        case 'L': {
            return acc::blas_api::fill::Lower;
        }
        default: {
            throw std::runtime_error("get_gpublasFillMode_t(): wrong mode");
        }
    }
    return acc::blas_api::fill::Upper; // make compiler happy
}

inline acc::blas_api::diagonal_t
get_gpublasDiagonal_t(char c)
{
    switch (c) {
        case 'n':
        case 'N': {
            return acc::blas_api::diagonal::NonUnit;
        }
        case 'u':
        case 'U': {
            return acc::blas_api::diagonal::Unit;
        }
        default: {
            throw std::runtime_error("get_gpublasDiagonal_t(): wrong diagonal type");
        }
    }
    return acc::blas_api::diagonal::NonUnit; // make compiler happy
}

#define CALL_GPU_BLAS(func__, args__)                                                                                  \
    {                                                                                                                  \
        acc::blas_api::status_t status;                                                                                \
        if ((status = func__ args__) != acc::blas_api::status::Success) {                                              \
            error_message(status);                                                                                     \
            char nm[1024];                                                                                             \
            gethostname(nm, 1024);                                                                                     \
            std::printf("hostname: %s\n", nm);                                                                         \
            std::printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__);                           \
            acc::stack_backtrace();                                                                                    \
        }                                                                                                              \
    }

/// Store the default (null) stream handler.
acc::blas_api::handle_t&
null_stream_handle();

/// Store the gpublas handlers associated with acc streams.
std::vector<acc::blas_api::handle_t>&
stream_handles();

inline void
create_stream_handles()
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::create, (&null_stream_handle()));

    stream_handles() = std::vector<acc::blas_api::handle_t>(acc::num_streams());
    for (int i = 0; i < acc::num_streams(); i++) {
        CALL_GPU_BLAS(acc::blas_api::create, (&stream_handles()[i]));

        CALL_GPU_BLAS(acc::blas_api::set_stream, (stream_handles()[i], acc::stream(acc::stream_id(i))));
    }
}

inline void
destroy_stream_handles()
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::destroy, (null_stream_handle()));
    for (int i = 0; i < acc::num_streams(); i++) {
        CALL_GPU_BLAS(acc::blas_api::destroy, (stream_handles()[i]));
    }
}

inline acc::blas_api::handle_t
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
    CALL_GPU_BLAS(acc::blas_api::zgemv, (stream_handle(stream_id), get_gpublasOperation_t(transa), m, n,
                                         reinterpret_cast<const acc::blas_api::complex_double_t*>(alpha),
                                         reinterpret_cast<const acc::blas_api::complex_double_t*>(a), lda,
                                         reinterpret_cast<const acc::blas_api::complex_double_t*>(x), incx,
                                         reinterpret_cast<const acc::blas_api::complex_double_t*>(beta),
                                         reinterpret_cast<acc::blas_api::complex_double_t*>(y), incy));
}

inline void
cgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, acc_complex_float_t const* alpha,
      acc_complex_float_t const* a, int32_t lda, acc_complex_float_t const* b, int32_t ldb,
      acc_complex_float_t const* beta, acc_complex_float_t* c, int32_t ldc, int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::cgemm,
                  (stream_handle(stream_id), get_gpublasOperation_t(transa), get_gpublasOperation_t(transb), m, n, k,
                   reinterpret_cast<const acc::blas_api::complex_float_t*>(alpha),
                   reinterpret_cast<const acc::blas_api::complex_float_t*>(a), lda,
                   reinterpret_cast<const acc::blas_api::complex_float_t*>(b), ldb,
                   reinterpret_cast<const acc::blas_api::complex_float_t*>(beta),
                   reinterpret_cast<acc::blas_api::complex_float_t*>(c), ldc));
}

inline void
zgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, acc_complex_double_t const* alpha,
      acc_complex_double_t const* a, int32_t lda, acc_complex_double_t const* b, int32_t ldb,
      acc_complex_double_t const* beta, acc_complex_double_t* c, int32_t ldc, int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::zgemm,
                  (stream_handle(stream_id), get_gpublasOperation_t(transa), get_gpublasOperation_t(transb), m, n, k,
                   reinterpret_cast<const acc::blas_api::complex_double_t*>(alpha),
                   reinterpret_cast<const acc::blas_api::complex_double_t*>(a), lda,
                   reinterpret_cast<const acc::blas_api::complex_double_t*>(b), ldb,
                   reinterpret_cast<const acc::blas_api::complex_double_t*>(beta),
                   reinterpret_cast<acc::blas_api::complex_double_t*>(c), ldc));
}

inline void
sgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, float const* alpha, float const* a, int32_t lda,
      float const* b, int32_t ldb, float const* beta, float* c, int32_t ldc, int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::sgemm, (stream_handle(stream_id), get_gpublasOperation_t(transa),
                                         get_gpublasOperation_t(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

inline void
dgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, double const* alpha, double const* a, int32_t lda,
      double const* b, int32_t ldb, double const* beta, double* c, int32_t ldc, int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::dgemm, (stream_handle(stream_id), get_gpublasOperation_t(transa),
                                         get_gpublasOperation_t(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

inline void
strmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, float const* alpha__, float const* A__,
      int lda__, float* B__, int ldb__, int stream_id)
{
    acc::blas_api::side_mode_t side   = get_gpublasSideMode_t(side__);
    acc::blas_api::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    acc::blas_api::operation_t transa = get_gpublasOperation_t(transa__);
    acc::blas_api::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
#ifdef SIRIUS_CUDA
    CALL_GPU_BLAS(acc::blas_api::strmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__, alpha__, A__,
                                         lda__, B__, ldb__, B__, ldb__));
#else
    // rocblas trmm function does not take three matrices
    CALL_GPU_BLAS(acc::blas_api::strmm,
                  (stream_handle(stream_id), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__));
#endif
}

inline void
dtrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, double const* alpha__, double const* A__,
      int lda__, double* B__, int ldb__, int stream_id)
{
    acc::blas_api::side_mode_t side   = get_gpublasSideMode_t(side__);
    acc::blas_api::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    acc::blas_api::operation_t transa = get_gpublasOperation_t(transa__);
    acc::blas_api::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
#ifdef SIRIUS_CUDA
    CALL_GPU_BLAS(acc::blas_api::dtrmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__, alpha__, A__,
                                         lda__, B__, ldb__, B__, ldb__));
#else
    // rocblas trmm function does not take three matrices
    CALL_GPU_BLAS(acc::blas_api::dtrmm,
                  (stream_handle(stream_id), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__));
#endif
}

inline void
ctrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, acc_complex_float_t const* alpha__,
      acc_complex_float_t const* A__, int lda__, acc_complex_float_t* B__, int ldb__, int stream_id)
{
    acc::blas_api::side_mode_t side   = get_gpublasSideMode_t(side__);
    acc::blas_api::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    acc::blas_api::operation_t transa = get_gpublasOperation_t(transa__);
    acc::blas_api::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
#ifdef SIRIUS_CUDA
    CALL_GPU_BLAS(acc::blas_api::ctrmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__,
                                         reinterpret_cast<const acc::blas_api::complex_float_t*>(alpha__),
                                         reinterpret_cast<const acc::blas_api::complex_float_t*>(A__), lda__,
                                         reinterpret_cast<acc::blas_api::complex_float_t*>(B__), ldb__,
                                         reinterpret_cast<acc::blas_api::complex_float_t*>(B__), ldb__));
#else
    // rocblas trmm function does not take three matrices
    CALL_GPU_BLAS(acc::blas_api::ctrmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__,
                                         reinterpret_cast<const acc::blas_api::complex_float_t*>(alpha__),
                                         reinterpret_cast<const acc::blas_api::complex_float_t*>(A__), lda__,
                                         reinterpret_cast<acc::blas_api::complex_float_t*>(B__), ldb__));
#endif
}

inline void
ztrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, acc_complex_double_t const* alpha__,
      acc_complex_double_t const* A__, int lda__, acc_complex_double_t* B__, int ldb__, int stream_id)
{
    acc::blas_api::side_mode_t side   = get_gpublasSideMode_t(side__);
    acc::blas_api::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    acc::blas_api::operation_t transa = get_gpublasOperation_t(transa__);
    acc::blas_api::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
#ifdef SIRIUS_CUDA
    CALL_GPU_BLAS(acc::blas_api::ztrmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__,
                                         reinterpret_cast<const acc::blas_api::complex_double_t*>(alpha__),
                                         reinterpret_cast<const acc::blas_api::complex_double_t*>(A__), lda__,
                                         reinterpret_cast<acc::blas_api::complex_double_t*>(B__), ldb__,
                                         reinterpret_cast<acc::blas_api::complex_double_t*>(B__), ldb__));
#else
    // rocblas trmm function does not take three matrices
    CALL_GPU_BLAS(acc::blas_api::ztrmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__,
                                         reinterpret_cast<const acc::blas_api::complex_double_t*>(alpha__),
                                         reinterpret_cast<const acc::blas_api::complex_double_t*>(A__), lda__,
                                         reinterpret_cast<acc::blas_api::complex_double_t*>(B__), ldb__));
#endif
}

inline void
sger(int m, int n, float const* alpha, float const* x, int incx, float const* y, int incy, float* A, int lda,
     int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::sger, (stream_handle(stream_id), m, n, alpha, x, incx, y, incy, A, lda));
}

inline void
dger(int m, int n, double const* alpha, double const* x, int incx, double const* y, int incy, double* A, int lda,
     int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::dger, (stream_handle(stream_id), m, n, alpha, x, incx, y, incy, A, lda));
}

inline void
cgeru(int m, int n, acc_complex_float_t const* alpha, acc_complex_float_t const* x, int incx,
      acc_complex_float_t const* y, int incy, acc_complex_float_t* A, int lda, int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::cgeru,
                  (stream_handle(stream_id), m, n, reinterpret_cast<const acc::blas_api::complex_float_t*>(alpha),
                   reinterpret_cast<const acc::blas_api::complex_float_t*>(x), incx,
                   reinterpret_cast<const acc::blas_api::complex_float_t*>(y), incy,
                   reinterpret_cast<acc::blas_api::complex_float_t*>(A), lda));
}

inline void
zgeru(int m, int n, acc_complex_double_t const* alpha, acc_complex_double_t const* x, int incx,
      acc_complex_double_t const* y, int incy, acc_complex_double_t* A, int lda, int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::zgeru,
                  (stream_handle(stream_id), m, n, reinterpret_cast<const acc::blas_api::complex_double_t*>(alpha),
                   reinterpret_cast<const acc::blas_api::complex_double_t*>(x), incx,
                   reinterpret_cast<const acc::blas_api::complex_double_t*>(y), incy,
                   reinterpret_cast<acc::blas_api::complex_double_t*>(A), lda));
}

inline void
zaxpy(int n__, acc_complex_double_t const* alpha__, acc_complex_double_t const* x__, int incx__,
      acc_complex_double_t* y__, int incy__)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::zaxpy,
                  (null_stream_handle(), n__, reinterpret_cast<const acc::blas_api::complex_double_t*>(alpha__),
                   reinterpret_cast<const acc::blas_api::complex_double_t*>(x__), incx__,
                   reinterpret_cast<acc::blas_api::complex_double_t*>(y__), incy__));
}

inline void
dscal(int n__, double const* alpha__, double* x__, int incx__)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::dscal, (null_stream_handle(), n__, alpha__, x__, incx__));
}

inline void
sscal(int n__, float const* alpha__, float* x__, int incx__)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::sscal, (null_stream_handle(), n__, alpha__, x__, incx__));
}

#if defined(SIRIUS_CUDA)
/// Interface to cuBlasXt functions
namespace xt {

cublasXtHandle_t&
cublasxt_handle();

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
cgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, acc_complex_float_t const* alpha,
      acc_complex_float_t const* a, int32_t lda, acc_complex_float_t const* b, int32_t ldb,
      acc_complex_float_t const* beta, acc_complex_float_t* c, int32_t ldc)
{
    // acc::set_device();
    CALL_GPU_BLAS(cublasXtCgemm, (cublasxt_handle(), get_gpublasOperation_t(transa), get_gpublasOperation_t(transb), m,
                                  n, k, alpha, a, lda, b, ldb, beta, c, ldc));
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
sgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, float const* alpha, float const* a, int32_t lda,
      float const* b, int32_t ldb, float const* beta, float* c, int32_t ldc)
{
    // acc::set_device();
    CALL_GPU_BLAS(cublasXtSgemm, (cublasxt_handle(), get_gpublasOperation_t(transa), get_gpublasOperation_t(transb), m,
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
strmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, float const* alpha__, float const* A__,
      int lda__, float* B__, int ldb__)
{
    acc::blas_api::side_mode_t side   = get_gpublasSideMode_t(side__);
    acc::blas_api::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    acc::blas_api::operation_t transa = get_gpublasOperation_t(transa__);
    acc::blas_api::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
    CALL_GPU_BLAS(cublasXtStrmm,
                  (cublasxt_handle(), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__, B__, ldb__));
}

inline void
dtrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, double const* alpha__, double const* A__,
      int lda__, double* B__, int ldb__)
{
    acc::blas_api::side_mode_t side   = get_gpublasSideMode_t(side__);
    acc::blas_api::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    acc::blas_api::operation_t transa = get_gpublasOperation_t(transa__);
    acc::blas_api::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
    CALL_GPU_BLAS(cublasXtDtrmm,
                  (cublasxt_handle(), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__, B__, ldb__));
}

inline void
ctrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, acc_complex_float_t const* alpha__,
      acc_complex_float_t const* A__, int lda__, acc_complex_float_t* B__, int ldb__)
{
    acc::blas_api::side_mode_t side   = get_gpublasSideMode_t(side__);
    acc::blas_api::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    acc::blas_api::operation_t transa = get_gpublasOperation_t(transa__);
    acc::blas_api::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
    CALL_GPU_BLAS(cublasXtCtrmm,
                  (cublasxt_handle(), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__, B__, ldb__));
}

inline void
ztrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, acc_complex_double_t const* alpha__,
      acc_complex_double_t const* A__, int lda__, acc_complex_double_t* B__, int ldb__)
{
    acc::blas_api::side_mode_t side   = get_gpublasSideMode_t(side__);
    acc::blas_api::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    acc::blas_api::operation_t transa = get_gpublasOperation_t(transa__);
    acc::blas_api::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
    CALL_GPU_BLAS(cublasXtZtrmm,
                  (cublasxt_handle(), side, uplo, transa, diag, m__, n__, alpha__, A__, lda__, B__, ldb__, B__, ldb__));
}

} // namespace xt
#endif

} // namespace blas

} // namespace acc

} // namespace sirius

#endif
