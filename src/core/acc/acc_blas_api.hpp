/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file acc_blas_api.hpp
 *
 *  \brief Interface to cuBLAS / rocblas related functions.
 */

#ifndef __ACC_BLAS_API_HPP__
#define __ACC_BLAS_API_HPP__

#include <utility>

#if defined(SIRIUS_CUDA)
#include <cublas_v2.h>

#elif defined(SIRIUS_ROCM)
#include <rocblas.h>

#else
#error Either SIRIUS_CUDA or SIRIUS_ROCM must be defined!
#endif

namespace sirius {

namespace acc {
/// Internal interface to accelerated BLAS functions (CUDA or ROCM).
namespace blas_api {

#if defined(SIRIUS_CUDA)
using handle_t         = cublasHandle_t;
using status_t         = cublasStatus_t;
using operation_t      = cublasOperation_t;
using side_mode_t      = cublasSideMode_t;
using diagonal_t       = cublasDiagType_t;
using fill_mode_t      = cublasFillMode_t;
using complex_float_t  = cuComplex;
using complex_double_t = cuDoubleComplex;
#endif

#if defined(SIRIUS_ROCM)
using handle_t         = rocblas_handle;
using status_t         = rocblas_status;
using operation_t      = rocblas_operation;
using side_mode_t      = rocblas_side;
using diagonal_t       = rocblas_diagonal;
using fill_mode_t      = rocblas_fill;
using complex_float_t  = rocblas_float_complex;
using complex_double_t = rocblas_double_complex;
#endif

namespace operation {
#if defined(SIRIUS_CUDA)
constexpr auto None               = CUBLAS_OP_N;
constexpr auto Transpose          = CUBLAS_OP_T;
constexpr auto ConjugateTranspose = CUBLAS_OP_C;
#endif

#if defined(SIRIUS_ROCM)
constexpr auto None               = rocblas_operation_none;
constexpr auto Transpose          = rocblas_operation_transpose;
constexpr auto ConjugateTranspose = rocblas_operation_conjugate_transpose;
#endif
} // namespace operation

namespace side {
#if defined(SIRIUS_CUDA)
constexpr auto Left  = CUBLAS_SIDE_LEFT;
constexpr auto Right = CUBLAS_SIDE_RIGHT;
#endif

#if defined(SIRIUS_ROCM)
constexpr auto Left  = rocblas_side_left;
constexpr auto Right = rocblas_side_right;
#endif
} // namespace side

namespace diagonal {
#if defined(SIRIUS_CUDA)
constexpr auto NonUnit = CUBLAS_DIAG_NON_UNIT;
constexpr auto Unit    = CUBLAS_DIAG_UNIT;
#endif

#if defined(SIRIUS_ROCM)
constexpr auto NonUnit = rocblas_diagonal_non_unit;
constexpr auto Unit    = rocblas_diagonal_unit;
#endif
} // namespace diagonal

namespace fill {
#if defined(SIRIUS_CUDA)
constexpr auto Upper = CUBLAS_FILL_MODE_UPPER;
constexpr auto Lower = CUBLAS_FILL_MODE_LOWER;
#endif

#if defined(SIRIUS_ROCM)
constexpr auto Upper = rocblas_fill_upper;
constexpr auto Lower = rocblas_fill_lower;
#endif
} // namespace fill

namespace status {
#if defined(SIRIUS_CUDA)
constexpr auto Success = CUBLAS_STATUS_SUCCESS;
#endif

#if defined(SIRIUS_ROCM)
constexpr auto Success = rocblas_status_success;
#endif
} // namespace status

// =======================================
// Forwarding functions of to GPU BLAS API
// =======================================
template <typename... ARGS>
inline auto
create(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_create_handle(std::forward<ARGS>(args)...);
#else
    return cublasCreate(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto
destroy(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_destroy_handle(std::forward<ARGS>(args)...);
#else
    return cublasDestroy(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto
set_stream(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_set_stream(std::forward<ARGS>(args)...);
#else
    return cublasSetStream(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto
get_stream(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_get_stream(std::forward<ARGS>(args)...);
#else
    return cublasGetStream(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto
sgemm(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_sgemm(std::forward<ARGS>(args)...);
#else
    return cublasSgemm(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
dgemm(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_dgemm(std::forward<ARGS>(args)...);
#else
    return cublasDgemm(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
cgemm(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_cgemm(std::forward<ARGS>(args)...);
#else
    return cublasCgemm(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
zgemm(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_zgemm(std::forward<ARGS>(args)...);
#else
    return cublasZgemm(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
dgemv(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_dgemv(std::forward<ARGS>(args)...);
#else
    return cublasDgemv(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
zgemv(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_zgemv(std::forward<ARGS>(args)...);
#else
    return cublasZgemv(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
strmm(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_strmm(std::forward<ARGS>(args)...);
#else
    return cublasStrmm(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
dtrmm(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_dtrmm(std::forward<ARGS>(args)...);
#else
    return cublasDtrmm(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
ctrmm(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_ctrmm(std::forward<ARGS>(args)...);
#else
    return cublasCtrmm(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
ztrmm(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_ztrmm(std::forward<ARGS>(args)...);
#else
    return cublasZtrmm(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
sger(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_sger(std::forward<ARGS>(args)...);
#else
    return cublasSger(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
dger(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_dger(std::forward<ARGS>(args)...);
#else
    return cublasDger(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
cgeru(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_cgeru(std::forward<ARGS>(args)...);
#else
    return cublasCgeru(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
zgeru(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_zgeru(std::forward<ARGS>(args)...);
#else
    return cublasZgeru(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
zaxpy(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_zaxpy(std::forward<ARGS>(args)...);
#else
    return cublasZaxpy(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
dscal(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_dscal(std::forward<ARGS>(args)...);
#else
    return cublasDscal(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

template <typename... ARGS>
inline auto
sscal(ARGS&&... args) -> status_t
{
#if defined(SIRIUS_ROCM)
    return rocblas_sscal(std::forward<ARGS>(args)...);
#else
    return cublasSscal(std::forward<ARGS>(args)...);
#endif // SIRIUS_ROCM
}

} // namespace blas_api
} // namespace acc
} // namespace sirius

#endif
