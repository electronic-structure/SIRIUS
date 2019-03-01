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
#include "rocblas_port/rocblas-types.h"
#include "rocblas_port/status.h"
#include "rocblas_port/definitions.h"
#include "rocblas_port/handle.h"
#include "rocblas_port/utility.h"
#include "rocblas_port/reduction.h"
#include "rocblas_port/port_helper_func.h"

namespace {

template <rocblas_int DIM_X, rocblas_int DIM_Y, rocblas_operation OP_B, typename T, typename U>
__global__ void gemmn_kernel(rocblas_int m, rocblas_int n, rocblas_int k, U alpha_device_host, const T* __restrict__ A,
                             rocblas_int lda, const T* __restrict__ B, rocblas_int ldb, U beta_device_host, T* C,
                             rocblas_int ldc)
{
    auto alpha              = load_scalar(alpha_device_host);
    auto beta               = load_scalar(beta_device_host);
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;

    if (DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind;

    __shared__ T sdata[DIM_X * 4 * DIM_Y];

    T res_A[4]; // micor tile is 4 * 4
    T res_B[4];

    res_A[0] = res_B[0] = T(0.0);
    res_A[1] = res_B[0] = T(0.0);
    res_A[2] = res_B[0] = T(0.0);
    res_A[3] = res_B[0] = T(0.0);

    ind = hipBlockIdx_x * DIM_X * 4 + tx;

    rocblas_int k_tail = k % (4 * DIM_Y);
    rocblas_int col    = ty * 4;
    rocblas_int col_B  = hipBlockIdx_y;

    B += col_B * MatrixDim<OP_B>::ld(ldb, 1);

    for (col = ty * 4; col < (k - k_tail); col += 4 * DIM_Y) {
        res_B[0] = rb_port_conj_op<OP_B, T>::eval(B[(col + 0) * MatrixDim<OP_B>::inc(ldb, 1)]);
        res_B[1] = rb_port_conj_op<OP_B, T>::eval(B[(col + 1) * MatrixDim<OP_B>::inc(ldb, 1)]);
        res_B[2] = rb_port_conj_op<OP_B, T>::eval(B[(col + 2) * MatrixDim<OP_B>::inc(ldb, 1)]);
        res_B[3] = rb_port_conj_op<OP_B, T>::eval(B[(col + 3) * MatrixDim<OP_B>::inc(ldb, 1)]);

        if (ind < m) {
            res_A[0] += A[ind + (col + 0) * lda] * res_B[0];
            res_A[0] += A[ind + (col + 1) * lda] * res_B[1];
            res_A[0] += A[ind + (col + 2) * lda] * res_B[2];
            res_A[0] += A[ind + (col + 3) * lda] * res_B[3];
        }

        if (ind + DIM_X < m) {
            res_A[1] += A[ind + DIM_X + (col + 0) * lda] * res_B[0];
            res_A[1] += A[ind + DIM_X + (col + 1) * lda] * res_B[1];
            res_A[1] += A[ind + DIM_X + (col + 2) * lda] * res_B[2];
            res_A[1] += A[ind + DIM_X + (col + 3) * lda] * res_B[3];
        }

        if (ind + 2 * DIM_X < m) {
            res_A[2] += A[ind + 2 * DIM_X + (col + 0) * lda] * res_B[0];
            res_A[2] += A[ind + 2 * DIM_X + (col + 1) * lda] * res_B[1];
            res_A[2] += A[ind + 2 * DIM_X + (col + 2) * lda] * res_B[2];
            res_A[2] += A[ind + 2 * DIM_X + (col + 3) * lda] * res_B[3];
        }

        if (ind + 3 * DIM_X < m) {
            res_A[3] += A[ind + 3 * DIM_X + (col + 0) * lda] * res_B[0];
            res_A[3] += A[ind + 3 * DIM_X + (col + 1) * lda] * res_B[1];
            res_A[3] += A[ind + 3 * DIM_X + (col + 2) * lda] * res_B[2];
            res_A[3] += A[ind + 3 * DIM_X + (col + 3) * lda] * res_B[3];
        }
    }

    // if n  is not multiple of (DIM_Y * 4)
    if (k_tail > 0) {
        res_B[0] =
            (col + 0 < k) ? rb_port_conj_op<OP_B, T>::eval(B[(col + 0) * MatrixDim<OP_B>::inc(ldb, 1)]) : T(0);
        res_B[1] =
            (col + 1 < k) ? rb_port_conj_op<OP_B, T>::eval(B[(col + 1) * MatrixDim<OP_B>::inc(ldb, 1)]) : T(0);
        res_B[2] =
            (col + 2 < k) ? rb_port_conj_op<OP_B, T>::eval(B[(col + 2) * MatrixDim<OP_B>::inc(ldb, 1)]) : T(0);
        res_B[3] =
            (col + 3 < k) ? rb_port_conj_op<OP_B, T>::eval(B[(col + 3) * MatrixDim<OP_B>::inc(ldb, 1)]) : T(0);

        if (ind < m) {
            res_A[0] += A[ind + (col + 0) * lda * (col + 0 < k)] * res_B[0];
            res_A[0] += A[ind + (col + 1) * lda * (col + 1 < k)] * res_B[1];
            res_A[0] += A[ind + (col + 2) * lda * (col + 2 < k)] * res_B[2];
            res_A[0] += A[ind + (col + 3) * lda * (col + 3 < k)] * res_B[3];
        }

        if (ind + DIM_X < m) {
            res_A[1] += A[ind + DIM_X + (col + 0) * lda * (col + 0 < k)] * res_B[0];
            res_A[1] += A[ind + DIM_X + (col + 1) * lda * (col + 1 < k)] * res_B[1];
            res_A[1] += A[ind + DIM_X + (col + 2) * lda * (col + 2 < k)] * res_B[2];
            res_A[1] += A[ind + DIM_X + (col + 3) * lda * (col + 3 < k)] * res_B[3];
        }

        if (ind + 2 * DIM_X < m) {
            res_A[2] += A[ind + 2 * DIM_X + (col + 0) * lda * (col + 0 < k)] * res_B[0];
            res_A[2] += A[ind + 2 * DIM_X + (col + 1) * lda * (col + 1 < k)] * res_B[1];
            res_A[2] += A[ind + 2 * DIM_X + (col + 2) * lda * (col + 2 < k)] * res_B[2];
            res_A[2] += A[ind + 2 * DIM_X + (col + 3) * lda * (col + 3 < k)] * res_B[3];
        }

        if (ind + 3 * DIM_X < m) {
            res_A[3] += A[ind + 3 * DIM_X + (col + 0) * lda * (col + 0 < k)] * res_B[0];
            res_A[3] += A[ind + 3 * DIM_X + (col + 1) * lda * (col + 1 < k)] * res_B[1];
            res_A[3] += A[ind + 3 * DIM_X + (col + 2) * lda * (col + 2 < k)] * res_B[2];
            res_A[3] += A[ind + 3 * DIM_X + (col + 3) * lda * (col + 3 < k)] * res_B[3];
        }
    }

    sdata[tx + ty * DIM_X * 4]             = res_A[0];
    sdata[tx + DIM_X + ty * DIM_X * 4]     = res_A[1];
    sdata[tx + 2 * DIM_X + ty * DIM_X * 4] = res_A[2];
    sdata[tx + 3 * DIM_X + ty * DIM_X * 4] = res_A[3];

    __syncthreads();

    ind = hipBlockIdx_x * DIM_X * 4 + thread_id;
    if (thread_id < DIM_X * 4) {
        for (rocblas_int i = 1; i < DIM_Y; i++)
            sdata[thread_id] += sdata[thread_id + DIM_X * 4 * i];

        if (ind < m)
            C[ind + col_B * ldc] = alpha * sdata[thread_id] + beta * C[ind + col_B * ldc];
    }
}

template <rocblas_int NB_X, rocblas_operation OP_A, rocblas_operation OP_B, typename T, typename U>
__global__ void gemmc_kernel(rocblas_int cols_AT, U alpha_device_host, const T* __restrict__ A, rocblas_int lda,
                             const T* __restrict__ B, rocblas_int ldb, U beta_device_host, T* C, rocblas_int ldc)
{
    auto alpha     = load_scalar(alpha_device_host);
    auto beta      = load_scalar(beta_device_host);
    rocblas_int tx = hipThreadIdx_x;

    if (tx < cols_AT)
        A += tx;

    rocblas_int col_A = hipBlockIdx_x;
    rocblas_int col_B = hipBlockIdx_y;
    A += col_A * lda;
    B += col_B * MatrixDim<OP_B>::ld(ldb, 1);

    T res(0);

    __shared__ T sdata[NB_X];

    // partial sums
    rocblas_int cols_AT_full = (cols_AT / NB_X) * NB_X;

    for (rocblas_int i = 0; i < cols_AT_full; i += NB_X)
        res += rb_port_conj_op<OP_A, T>::eval(A[i]) * rb_port_conj_op<OP_B, T>::eval(B[(tx + i) * MatrixDim<OP_B>::inc(ldb, 1)]);

    if (tx + cols_AT_full < cols_AT)
        res += rb_port_conj_op<OP_A, T>::eval(A[cols_AT_full]) * rb_port_conj_op<OP_B, T>::eval(B[(tx + cols_AT_full) * MatrixDim<OP_B>::inc(ldb, 1)]);

    sdata[tx] = res;

    // tree reduction of partial sums,
    if (NB_X > 16) {
        rocblas_sum_reduce<NB_X>(tx, sdata);
    } else {
        __syncthreads();

        if (tx == 0) {
            for (rocblas_int i = 1; i < cols_AT && i < NB_X; i++)
                sdata[0] += sdata[i];
        }

        __syncthreads();
    }

    if (tx == 0)
        C[col_A + col_B * ldc] = alpha * sdata[0] + beta * C[col_A + col_B * ldc];
}

template <typename>
constexpr char rocblas_gemm_name[] = "unknown";
template <>
constexpr char rocblas_gemm_name<float>[] = "rocblas_sgemm";
template <>
constexpr char rocblas_gemm_name<double>[] = "rocblas_dgemm";

/*! \brief BLAS Level 2 API

    \details
    xGEMM performs one of the matrix-vector operations

        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    trans     rocblas_operation
    @param[in]
    m         rocblas_int
    @param[in]
    n         rocblas_int
    @param[in]
    alpha
              specifies the scalar alpha.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      specifies the increment for the elements of x.
    @param[in]
    beta      specifies the scalar beta.
    @param[out]
    y         pointer storing vector y on the GPU.
    @param[in]
    incy      rocblas_int
              specifies the increment for the elements of y.

    ********************************************************************/

template <rocblas_operation OP_B, typename T>
rocblas_status rocblas_gemm(rocblas_handle handle, rocblas_operation transa, rocblas_int m,
                            rocblas_int n, rocblas_int k, const T* alpha, const T* A, rocblas_int lda, const T* B,
                            rocblas_int ldb, const T* beta, T* C, rocblas_int ldc)
{
    if (!handle)
        return rocblas_status_invalid_handle;
    if (!alpha || !beta)
        return rocblas_status_invalid_pointer;

    if (!A || !B || !C)
        return rocblas_status_invalid_pointer;

    if (m < 0 || n < 0 || k < 0 || lda < m || lda < 1 || ldb < k || ldb < 1 || ldc < m || ldc < 1)
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible. Not Argument error
     */
    if (!m || !n || !k)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    if (transa == rocblas_operation_none) {
        // GEMMN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
        static constexpr int GEMMN_DIM_X = 32;
        static constexpr int GEMMN_DIM_Y = 16;
        rocblas_int blocks               = (m - 1) / (GEMMN_DIM_X * 4) + 1;

        dim3 gemmn_grid(blocks, n);
        dim3 gemmn_threads(GEMMN_DIM_X, GEMMN_DIM_Y);

        if (handle->pointer_mode == rocblas_pointer_mode_device) {
            hipLaunchKernelGGL((gemmn_kernel<GEMMN_DIM_X, GEMMN_DIM_Y, OP_B>), gemmn_grid, gemmn_threads, 0, rocblas_stream,
                               m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        } else {
            if (rb_port_cmp_and_real_only(*alpha, 0.0) && rb_port_cmp_and_real_only(*beta, 1))
                return rocblas_status_success;

            hipLaunchKernelGGL((gemmn_kernel<GEMMN_DIM_X, GEMMN_DIM_Y, OP_B>), gemmn_grid, gemmn_threads, 0, rocblas_stream,
                               m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
        }
    } else {
        // transpose
        // number of columns on the y-dim of the grid, using gemmc because gemmt(transpose) is a
        // instance of gemmc (conjugate)
        static constexpr int NB = 256;
        dim3 gemmc_grid(m, n);
        dim3 gemmc_threads(NB);

        if (handle->pointer_mode == rocblas_pointer_mode_device) {
            if (transa == rocblas_operation_transpose)
                hipLaunchKernelGGL(gemmc_kernel<NB, rocblas_operation_transpose, OP_B>, gemmc_grid, gemmc_threads, 0,
                                   rocblas_stream, k, alpha, A, lda, B, ldb, beta, C, ldc);
            else
                hipLaunchKernelGGL(gemmc_kernel<NB, rocblas_operation_conjugate_transpose, OP_B>, gemmc_grid,
                                   gemmc_threads, 0, rocblas_stream, k, alpha, A, lda, B, ldb, beta, C, ldc);
        } else {
            if (rb_port_cmp_and_real_only(*alpha, 0) && rb_port_cmp_and_real_only(*beta, 1))
                return rocblas_status_success;

            if (transa == rocblas_operation_transpose)
                hipLaunchKernelGGL(gemmc_kernel<NB, rocblas_operation_transpose, OP_B>, gemmc_grid, gemmc_threads, 0,
                                   rocblas_stream, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
            else
                hipLaunchKernelGGL(gemmc_kernel<NB, rocblas_operation_conjugate_transpose, OP_B>, gemmc_grid,
                                   gemmc_threads, 0, rocblas_stream, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
        }
    }
    return rocblas_status_success;
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
    if (transb == rocblas_operation_none)
        return rocblas_gemm<rocblas_operation_none>(handle, transa, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else if (transb == rocblas_operation_transpose)
        return rocblas_gemm<rocblas_operation_transpose>(handle, transa, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else if (transb == rocblas_operation_conjugate_transpose)
        return rocblas_gemm<rocblas_operation_conjugate_transpose>(handle, transa, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        return rocblas_status_not_implemented;
}

rocblas_status rocblas_port_dgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                  rocblas_int m, rocblas_int n, rocblas_int k, const double* alpha, const double* A,
                                  rocblas_int lda, const double* B, rocblas_int ldb, const double* beta, double* C,
                                  rocblas_int ldc)
{
    if (transb == rocblas_operation_none)
        return rocblas_gemm<rocblas_operation_none>(handle, transa, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else if (transb == rocblas_operation_transpose)
        return rocblas_gemm<rocblas_operation_transpose>(handle, transa, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else if (transb == rocblas_operation_conjugate_transpose)
        return rocblas_gemm<rocblas_operation_conjugate_transpose>(handle, transa, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        return rocblas_status_not_implemented;
}

rocblas_status rocblas_port_cgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                  rocblas_int m, rocblas_int n, rocblas_int k, const hipFloatComplex* alpha,
                                  const hipFloatComplex* A, rocblas_int lda, const hipFloatComplex* B, rocblas_int ldb,
                                  const hipFloatComplex* beta, hipFloatComplex* C, rocblas_int ldc)
{
    if (transb == rocblas_operation_none)
        return rocblas_gemm<rocblas_operation_none>(handle, transa, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else if (transb == rocblas_operation_transpose)
        return rocblas_gemm<rocblas_operation_transpose>(handle, transa, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else if (transb == rocblas_operation_conjugate_transpose)
        return rocblas_gemm<rocblas_operation_conjugate_transpose>(handle, transa, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        return rocblas_status_not_implemented;
}

rocblas_status rocblas_port_zgemm(rocblas_handle handle, rocblas_operation transa, rocblas_operation transb,
                                  rocblas_int m, rocblas_int n, rocblas_int k, const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A, rocblas_int lda, const hipDoubleComplex* B,
                                  rocblas_int ldb, const hipDoubleComplex* beta, hipDoubleComplex* C, rocblas_int ldc)
{
    if (transb == rocblas_operation_none)
        return rocblas_gemm<rocblas_operation_none>(handle, transa, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else if (transb == rocblas_operation_transpose)
        return rocblas_gemm<rocblas_operation_transpose>(handle, transa, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else if (transb == rocblas_operation_conjugate_transpose)
        return rocblas_gemm<rocblas_operation_conjugate_transpose>(handle, transa, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        return rocblas_status_not_implemented;
}
} // extern "C"
