/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <utility>
#include <tuple>
#include "rocblas_port/rocblas-types.h"
#include "rocblas_port/status.h"
#include "rocblas_port/definitions.h"
#include "rocblas_port/handle.h"
#include "rocblas_port/utility.h"
#include "rocblas_port/reduction.h"
#include "rocblas_port/port_helper_func.h"

namespace {


template<typename T>
struct CreateReal {
    template<typename U>
    __device__ __host__ static inline T eval(const U& val) {
        return T(val);
    }
};

template<>
struct CreateReal<hipFloatComplex> {
    template<typename U>
    __device__ __host__ static inline hipFloatComplex eval(const U& val) {
        return hipFloatComplex((float)val, 0.f);
    }
};

template<>
struct CreateReal<hipDoubleComplex> {
    template<typename U>
    __device__ __host__ static inline hipDoubleComplex eval(const U& val) {
        return hipDoubleComplex((double)val, 0.);
    }
};



/*
 * Load matrix value
 */
// transposed
template<rocblas_operation OP>
struct MatrixLoadGemm {
    template <typename T>
    __device__ static inline T eval(const T* M, const rocblas_int ld, const rocblas_int row,
                                    const rocblas_int col)
    {
        return M[col + row * ld];
    }
};

// Normal
template<>
struct MatrixLoadGemm<rocblas_operation_none> {
    template <typename T>
    __device__ static inline T eval(const T* M, const rocblas_int ld, const rocblas_int row, const rocblas_int col)
    {
        return M[row + col * ld];
    }
};



template <rocblas_operation OP>
struct MatrixRowsGemm {
    template <typename T, typename U>
    __host__ __device__ static inline T eval(const T rows, const U cols) {
        // transpose or hermitian.
        return cols;
    }
};

template <>
struct MatrixRowsGemm<rocblas_operation_none> {
    template <typename T, typename U>
    __host__ __device__ static inline T eval(const T rows, const U cols) {
        // transpose or hermitian.
        return rows;
    }
};

template <rocblas_operation OP>
struct MatrixColsGemm {
    template <typename T, typename U>
    __host__ __device__ static inline T eval(const T rows, const U cols) {
        // transpose or hermitian.
        return rows;
    }
};

template <>
struct MatrixColsGemm<rocblas_operation_none> {
    template <typename T, typename U>
    __host__ __device__ static inline T eval(const T rows, const U cols) {
        // transpose or hermitian.
        return cols;
    }
};


/*
 *
 */
template <rocblas_operation OP_A, rocblas_operation OP_B, typename U, typename T>
__global__ void gemm_kernel(rocblas_int m, rocblas_int n, rocblas_int k, U alpha_device_host,
                                   const T* __restrict__ A, rocblas_int lda, const T* __restrict__ B, rocblas_int ldb,
                                   U beta_device_host, T* C, rocblas_int ldc)
{

    const int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    if (tx >= m)
        return;

    const auto alpha = load_scalar(alpha_device_host);
    const auto beta = load_scalar(beta_device_host);
    const int row_C = tx;
    const int col_C = hipBlockIdx_y;

    const int cols_A = k;

    T res = CreateReal<T>::eval(0);
    for (int col = 0; col < cols_A; ++col) {
        res += ConjOp<OP_A, T>::eval(MatrixLoadGemm<OP_A>::eval(A, lda, row_C, col)) *
               ConjOp<OP_B, T>::eval(MatrixLoadGemm<OP_B>::eval(B, ldb, col, col_C));
    }

    C[row_C + col_C * ldc] = alpha * res + beta * C[row_C + col_C * ldc];
}

template <rocblas_operation OP_A, rocblas_operation OP_B, typename T>
rocblas_status rocblas_gemm(rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k, const T* alpha,
                            const T* A, rocblas_int lda, const T* B, rocblas_int ldb, const T* beta, T* C,
                            rocblas_int ldc)
{
    if (!handle)
        return rocblas_status_invalid_handle;
    if (!alpha)
        return rocblas_status_invalid_pointer;

    if (!A || !B || !C)
        return rocblas_status_invalid_pointer;

    if (!m || !n || !k)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    dim3 threads(256);
    dim3 grid((m + threads.x - 1) / threads.x, n);

    if (handle->pointer_mode == rocblas_pointer_mode_device) {
        hipLaunchKernelGGL(gemm_kernel<OP_A, OP_B>,
                           grid, threads, 0, rocblas_stream, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        if (rb_port_cmp_and_real_only(*alpha, 0))
            return rocblas_status_success;
        hipLaunchKernelGGL(gemm_kernel<OP_A, OP_B>,
                           grid, threads, 0, rocblas_stream, m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    }
    return rocblas_status_success;
}

template <rocblas_operation OP_A, typename T>
rocblas_status rocblas_select_op_b(rocblas_handle handle, rocblas_operation transb,
                                        rocblas_int m, rocblas_int n, rocblas_int k, const T* alpha, const T* A,
                                        rocblas_int lda, const T* B, rocblas_int ldb, const T* beta, T* C,
                                        rocblas_int ldc)
{
    if (transb == rocblas_operation_none)
        return rocblas_gemm<OP_A, rocblas_operation_none>(handle, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else if (transb == rocblas_operation_transpose)
        return rocblas_gemm<OP_A, rocblas_operation_transpose>(handle, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                                               ldc);
    else if (transb == rocblas_operation_conjugate_transpose)
        return rocblas_gemm<OP_A, rocblas_operation_conjugate_transpose>(handle, m, n, k, alpha, A, lda, B, ldb,
                                                                         beta, C, ldc);
    else
        return rocblas_status_not_implemented;
}

template <typename T>
rocblas_status rocblas_select(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                        rocblas_int m, rocblas_int n, rocblas_int k, const T* alpha, const T* A,
                                        rocblas_int lda, const T* B, rocblas_int ldb, const T* beta, T* C,
                                        rocblas_int ldc)
{
    if (transa == rocblas_operation_none)
        return rocblas_select_op_b<rocblas_operation_none>(handle, transb, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                                           ldc);
    else if (transa == rocblas_operation_transpose)
        return rocblas_select_op_b<rocblas_operation_transpose>(handle, transb, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                                                ldc);
    else if (transa == rocblas_operation_conjugate_transpose)
        return rocblas_select_op_b<rocblas_operation_conjugate_transpose>(handle, transb, m, n, k, alpha, A, lda, B,
                                                                          ldb, beta, C, ldc);
    else
        return rocblas_status_not_implemented;
}

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_port_sgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                  rocblas_int m, rocblas_int n, rocblas_int k, const float* alpha, const float* A,
                                  rocblas_int lda, const float* B, rocblas_int ldb, const float* beta, float* C,
                                  rocblas_int ldc)
{
    return rocblas_select(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

rocblas_status rocblas_port_dgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                  rocblas_int m, rocblas_int n, rocblas_int k, const double* alpha, const double* A,
                                  rocblas_int lda, const double* B, rocblas_int ldb, const double* beta, double* C,
                                  rocblas_int ldc)
{
    return rocblas_select(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

rocblas_status rocblas_port_cgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                  rocblas_int m, rocblas_int n, rocblas_int k, const hipFloatComplex* alpha,
                                  const hipFloatComplex* A, rocblas_int lda, const hipFloatComplex* B, rocblas_int ldb,
                                  const hipFloatComplex* beta, hipFloatComplex* C, rocblas_int ldc)
{
    return rocblas_select(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

rocblas_status rocblas_port_zgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                  rocblas_int m, rocblas_int n, rocblas_int k, const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A, rocblas_int lda, const hipDoubleComplex* B,
                                  rocblas_int ldb, const hipDoubleComplex* beta, hipDoubleComplex* C, rocblas_int ldc)
{
    return rocblas_select(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
} // extern "C"
