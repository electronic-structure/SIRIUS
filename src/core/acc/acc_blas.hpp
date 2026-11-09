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

#include <cstdint>
#include "acc.hpp"

namespace sirius {

namespace acc {

/// User facing interface to GPU blas functions.
namespace blas {

void
create_stream_handles();

void
destroy_stream_handles();

void
zgemv(char transa, int32_t m, int32_t n, acc_complex_double_t* alpha, acc_complex_double_t* a, int32_t lda,
      acc_complex_double_t* x, int32_t incx, acc_complex_double_t* beta, acc_complex_double_t* y, int32_t incy,
      int stream_id);

void
cgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, acc_complex_float_t const* alpha,
      acc_complex_float_t const* a, int32_t lda, acc_complex_float_t const* b, int32_t ldb,
      acc_complex_float_t const* beta, acc_complex_float_t* c, int32_t ldc, int stream_id);

void
zgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, acc_complex_double_t const* alpha,
      acc_complex_double_t const* a, int32_t lda, acc_complex_double_t const* b, int32_t ldb,
      acc_complex_double_t const* beta, acc_complex_double_t* c, int32_t ldc, int stream_id);

void
sgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, float const* alpha, float const* a, int32_t lda,
      float const* b, int32_t ldb, float const* beta, float* c, int32_t ldc, int stream_id);

void
dgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, double const* alpha, double const* a, int32_t lda,
      double const* b, int32_t ldb, double const* beta, double* c, int32_t ldc, int stream_id);

void
strmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, float const* alpha__, float const* A__,
      int lda__, float* B__, int ldb__, int stream_id);

void
dtrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, double const* alpha__, double const* A__,
      int lda__, double* B__, int ldb__, int stream_id);

void
ctrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, acc_complex_float_t const* alpha__,
      acc_complex_float_t const* A__, int lda__, acc_complex_float_t* B__, int ldb__, int stream_id);

void
ztrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, acc_complex_double_t const* alpha__,
      acc_complex_double_t const* A__, int lda__, acc_complex_double_t* B__, int ldb__, int stream_id);

void
sger(int m, int n, float const* alpha, float const* x, int incx, float const* y, int incy, float* A, int lda,
     int stream_id);

void
dger(int m, int n, double const* alpha, double const* x, int incx, double const* y, int incy, double* A, int lda,
     int stream_id);

void
cgeru(int m, int n, acc_complex_float_t const* alpha, acc_complex_float_t const* x, int incx,
      acc_complex_float_t const* y, int incy, acc_complex_float_t* A, int lda, int stream_id);

void
zgeru(int m, int n, acc_complex_double_t const* alpha, acc_complex_double_t const* x, int incx,
      acc_complex_double_t const* y, int incy, acc_complex_double_t* A, int lda, int stream_id);

void
zaxpy(int n__, acc_complex_double_t const* alpha__, acc_complex_double_t const* x__, int incx__,
      acc_complex_double_t* y__, int incy__);

void
dscal(int n__, double const* alpha__, double* x__, int incx__);

void
sscal(int n__, float const* alpha__, float* x__, int incx__);

/// Batched dot products result[i] = x_i^H y_i, where x_i = x + i * stridex and y_i = y + i * stridey.
/** x and y are in device memory, result is in host memory. The call is blocking. */
void
zdotc_strided_batched(int n__, acc_complex_double_t const* x__, int stridex__, acc_complex_double_t const* y__,
                      int stridey__, int batch_count__, acc_complex_double_t* result__, int stream_id);

/// Batched dot products result[i] = x_i^H y_i, where x_i = x + i * stridex and y_i = y + i * stridey.
/** x and y are in device memory, result is in host memory. The call is blocking. */
void
cdotc_strided_batched(int n__, acc_complex_float_t const* x__, int stridex__, acc_complex_float_t const* y__,
                      int stridey__, int batch_count__, acc_complex_float_t* result__, int stream_id);

#if defined(SIRIUS_CUDA)
/// Interface to cuBlasXt functions
namespace xt {

void
create_handle();

void
destroy_handle();

void
cgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, acc_complex_float_t const* alpha,
      acc_complex_float_t const* a, int32_t lda, acc_complex_float_t const* b, int32_t ldb,
      acc_complex_float_t const* beta, acc_complex_float_t* c, int32_t ldc);

void
zgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, acc_complex_double_t const* alpha,
      acc_complex_double_t const* a, int32_t lda, acc_complex_double_t const* b, int32_t ldb,
      acc_complex_double_t const* beta, acc_complex_double_t* c, int32_t ldc);

void
sgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, float const* alpha, float const* a, int32_t lda,
      float const* b, int32_t ldb, float const* beta, float* c, int32_t ldc);

void
dgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, double const* alpha, double const* a, int32_t lda,
      double const* b, int32_t ldb, double const* beta, double* c, int32_t ldc);

void
strmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, float const* alpha__, float const* A__,
      int lda__, float* B__, int ldb__);

void
dtrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, double const* alpha__, double const* A__,
      int lda__, double* B__, int ldb__);

void
ctrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, acc_complex_float_t const* alpha__,
      acc_complex_float_t const* A__, int lda__, acc_complex_float_t* B__, int ldb__);

void
ztrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, acc_complex_double_t const* alpha__,
      acc_complex_double_t const* A__, int lda__, acc_complex_double_t* B__, int ldb__);

} // namespace xt
#endif

} // namespace blas

} // namespace acc

} // namespace sirius

#endif
