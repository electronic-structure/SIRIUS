/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#if defined(SIRIUS_CUDA) || defined(SIRIUS_ROCM)
#include "acc_blas.hpp"
#include "acc_blas_api.hpp"

namespace sirius {

namespace acc {

namespace blas {

acc::blas_api::handle_t&
null_stream_handle()
{
    static acc::blas_api::handle_t null_stream_handle_;
    return null_stream_handle_;
}

std::vector<acc::blas_api::handle_t>&
stream_handles()
{
    static std::vector<acc::blas_api::handle_t> stream_handles_;
    return stream_handles_;
}

#if defined(SIRIUS_CUDA)
namespace xt {
cublasXtHandle_t&
cublasxt_handle()
{
    static cublasXtHandle_t handle;
    return handle;
}
} // namespace xt
#endif
void
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

void
destroy_stream_handles()
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::destroy, (null_stream_handle()));
    for (int i = 0; i < acc::num_streams(); i++) {
        CALL_GPU_BLAS(acc::blas_api::destroy, (stream_handles()[i]));
    }
}

void
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

void
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

void
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

void
sgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, float const* alpha, float const* a, int32_t lda,
      float const* b, int32_t ldb, float const* beta, float* c, int32_t ldc, int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::sgemm, (stream_handle(stream_id), get_gpublasOperation_t(transa),
                                         get_gpublasOperation_t(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

void
dgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, double const* alpha, double const* a, int32_t lda,
      double const* b, int32_t ldb, double const* beta, double* c, int32_t ldc, int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::dgemm, (stream_handle(stream_id), get_gpublasOperation_t(transa),
                                         get_gpublasOperation_t(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

void
strmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, float const* alpha__, float const* A__,
      int lda__, float* B__, int ldb__, int stream_id)
{
    acc::blas_api::side_mode_t side   = get_gpublasSideMode_t(side__);
    acc::blas_api::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    acc::blas_api::operation_t transa = get_gpublasOperation_t(transa__);
    acc::blas_api::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::strmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__, alpha__, A__,
                                         lda__, B__, ldb__, B__, ldb__));
}

void
dtrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, double const* alpha__, double const* A__,
      int lda__, double* B__, int ldb__, int stream_id)
{
    acc::blas_api::side_mode_t side   = get_gpublasSideMode_t(side__);
    acc::blas_api::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    acc::blas_api::operation_t transa = get_gpublasOperation_t(transa__);
    acc::blas_api::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::dtrmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__, alpha__, A__,
                                         lda__, B__, ldb__, B__, ldb__));
}

void
ctrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, acc_complex_float_t const* alpha__,
      acc_complex_float_t const* A__, int lda__, acc_complex_float_t* B__, int ldb__, int stream_id)
{
    acc::blas_api::side_mode_t side   = get_gpublasSideMode_t(side__);
    acc::blas_api::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    acc::blas_api::operation_t transa = get_gpublasOperation_t(transa__);
    acc::blas_api::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::ctrmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__,
                                         reinterpret_cast<const acc::blas_api::complex_float_t*>(alpha__),
                                         reinterpret_cast<const acc::blas_api::complex_float_t*>(A__), lda__,
                                         reinterpret_cast<acc::blas_api::complex_float_t*>(B__), ldb__,
                                         reinterpret_cast<acc::blas_api::complex_float_t*>(B__), ldb__));
}

void
ztrmm(char side__, char uplo__, char transa__, char diag__, int m__, int n__, acc_complex_double_t const* alpha__,
      acc_complex_double_t const* A__, int lda__, acc_complex_double_t* B__, int ldb__, int stream_id)
{
    acc::blas_api::side_mode_t side   = get_gpublasSideMode_t(side__);
    acc::blas_api::fill_mode_t uplo   = get_gpublasFillMode_t(uplo__);
    acc::blas_api::operation_t transa = get_gpublasOperation_t(transa__);
    acc::blas_api::diagonal_t diag    = get_gpublasDiagonal_t(diag__);
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::ztrmm, (stream_handle(stream_id), side, uplo, transa, diag, m__, n__,
                                         reinterpret_cast<const acc::blas_api::complex_double_t*>(alpha__),
                                         reinterpret_cast<const acc::blas_api::complex_double_t*>(A__), lda__,
                                         reinterpret_cast<acc::blas_api::complex_double_t*>(B__), ldb__,
                                         reinterpret_cast<acc::blas_api::complex_double_t*>(B__), ldb__));
}

void
sger(int m, int n, float const* alpha, float const* x, int incx, float const* y, int incy, float* A, int lda,
     int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::sger, (stream_handle(stream_id), m, n, alpha, x, incx, y, incy, A, lda));
}

void
dger(int m, int n, double const* alpha, double const* x, int incx, double const* y, int incy, double* A, int lda,
     int stream_id)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::dger, (stream_handle(stream_id), m, n, alpha, x, incx, y, incy, A, lda));
}

void
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

void
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

void
zaxpy(int n__, acc_complex_double_t const* alpha__, acc_complex_double_t const* x__, int incx__,
      acc_complex_double_t* y__, int incy__)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::zaxpy,
                  (null_stream_handle(), n__, reinterpret_cast<const acc::blas_api::complex_double_t*>(alpha__),
                   reinterpret_cast<const acc::blas_api::complex_double_t*>(x__), incx__,
                   reinterpret_cast<acc::blas_api::complex_double_t*>(y__), incy__));
}

void
dscal(int n__, double const* alpha__, double* x__, int incx__)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::dscal, (null_stream_handle(), n__, alpha__, x__, incx__));
}

void
sscal(int n__, float const* alpha__, float* x__, int incx__)
{
    // acc::set_device();
    CALL_GPU_BLAS(acc::blas_api::sscal, (null_stream_handle(), n__, alpha__, x__, incx__));
}

void
zdotc_strided_batched(int n__, acc_complex_double_t const* x__, int stridex__, acc_complex_double_t const* y__,
                      int stridey__, int batch_count__, acc_complex_double_t* result__, int stream_id)
{
#if defined(SIRIUS_ROCM)
    /* the handle is in the default host pointer mode: rocBLAS copies the result to the host and synchronizes */
    CALL_GPU_BLAS(rocblas_zdotc_strided_batched,
                  (stream_handle(stream_id), n__, reinterpret_cast<const acc::blas_api::complex_double_t*>(x__), 1,
                   stridex__, reinterpret_cast<const acc::blas_api::complex_double_t*>(y__), 1, stridey__,
                   batch_count__, reinterpret_cast<acc::blas_api::complex_double_t*>(result__)));
#elif defined(SIRIUS_CUDA)
    /* cuBLAS has no batched dot product; use 1x1 gemms, which need the output in device memory */
    acc::blas_api::complex_double_t alpha{1, 0};
    acc::blas_api::complex_double_t beta{0, 0};
    auto d_result = acc::allocate<acc_complex_double_t>(batch_count__);
    CALL_GPU_BLAS(cublasZgemmStridedBatched,
                  (stream_handle(stream_id), CUBLAS_OP_C, CUBLAS_OP_N, 1, 1, n__, &alpha,
                   reinterpret_cast<const acc::blas_api::complex_double_t*>(x__), n__, stridex__,
                   reinterpret_cast<const acc::blas_api::complex_double_t*>(y__), n__, stridey__, &beta,
                   reinterpret_cast<acc::blas_api::complex_double_t*>(d_result), 1, 1, batch_count__));
    acc::copyout(result__, d_result, batch_count__, acc::stream_id(stream_id));
    acc::sync_stream(acc::stream_id(stream_id));
    acc::deallocate(d_result);
#endif
}

void
cdotc_strided_batched(int n__, acc_complex_float_t const* x__, int stridex__, acc_complex_float_t const* y__,
                      int stridey__, int batch_count__, acc_complex_float_t* result__, int stream_id)
{
#if defined(SIRIUS_ROCM)
    /* the handle is in the default host pointer mode: rocBLAS copies the result to the host and synchronizes */
    CALL_GPU_BLAS(rocblas_cdotc_strided_batched,
                  (stream_handle(stream_id), n__, reinterpret_cast<const acc::blas_api::complex_float_t*>(x__), 1,
                   stridex__, reinterpret_cast<const acc::blas_api::complex_float_t*>(y__), 1, stridey__, batch_count__,
                   reinterpret_cast<acc::blas_api::complex_float_t*>(result__)));
#elif defined(SIRIUS_CUDA)
    /* cuBLAS has no batched dot product; use 1x1 gemms, which need the output in device memory */
    acc::blas_api::complex_float_t alpha{1, 0};
    acc::blas_api::complex_float_t beta{0, 0};
    auto d_result = acc::allocate<acc_complex_float_t>(batch_count__);
    CALL_GPU_BLAS(cublasCgemmStridedBatched,
                  (stream_handle(stream_id), CUBLAS_OP_C, CUBLAS_OP_N, 1, 1, n__, &alpha,
                   reinterpret_cast<const acc::blas_api::complex_float_t*>(x__), n__, stridex__,
                   reinterpret_cast<const acc::blas_api::complex_float_t*>(y__), n__, stridey__, &beta,
                   reinterpret_cast<acc::blas_api::complex_float_t*>(d_result), 1, 1, batch_count__));
    acc::copyout(result__, d_result, batch_count__, acc::stream_id(stream_id));
    acc::sync_stream(acc::stream_id(stream_id));
    acc::deallocate(d_result);
#endif
}

#if defined(SIRIUS_CUDA)
namespace xt {

void
create_handle()
{
    int device_id[1];
    device_id[0] = acc::get_device_id();
    CALL_GPU_BLAS(cublasXtCreate, (&cublasxt_handle()));
    CALL_GPU_BLAS(cublasXtDeviceSelect, (cublasxt_handle(), 1, device_id));
    CALL_GPU_BLAS(cublasXtSetBlockDim, (cublasxt_handle(), 4096));
}

void
destroy_handle()
{
    CALL_GPU_BLAS(cublasXtDestroy, (cublasxt_handle()));
}

void
cgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, acc_complex_float_t const* alpha,
      acc_complex_float_t const* a, int32_t lda, acc_complex_float_t const* b, int32_t ldb,
      acc_complex_float_t const* beta, acc_complex_float_t* c, int32_t ldc)
{
    // acc::set_device();
    CALL_GPU_BLAS(cublasXtCgemm, (cublasxt_handle(), get_gpublasOperation_t(transa), get_gpublasOperation_t(transb), m,
                                  n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

void
zgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, acc_complex_double_t const* alpha,
      acc_complex_double_t const* a, int32_t lda, acc_complex_double_t const* b, int32_t ldb,
      acc_complex_double_t const* beta, acc_complex_double_t* c, int32_t ldc)
{
    // acc::set_device();
    CALL_GPU_BLAS(cublasXtZgemm, (cublasxt_handle(), get_gpublasOperation_t(transa), get_gpublasOperation_t(transb), m,
                                  n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

void
sgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, float const* alpha, float const* a, int32_t lda,
      float const* b, int32_t ldb, float const* beta, float* c, int32_t ldc)
{
    // acc::set_device();
    CALL_GPU_BLAS(cublasXtSgemm, (cublasxt_handle(), get_gpublasOperation_t(transa), get_gpublasOperation_t(transb), m,
                                  n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

void
dgemm(char transa, char transb, int32_t m, int32_t n, int32_t k, double const* alpha, double const* a, int32_t lda,
      double const* b, int32_t ldb, double const* beta, double* c, int32_t ldc)
{
    // acc::set_device();
    CALL_GPU_BLAS(cublasXtDgemm, (cublasxt_handle(), get_gpublasOperation_t(transa), get_gpublasOperation_t(transb), m,
                                  n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

void
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

void
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

void
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

void
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
