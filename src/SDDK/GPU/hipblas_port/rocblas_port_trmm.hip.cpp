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


template<rocblas_fill MATRIX_TYPE, rocblas_diagonal DIAG_TYPE, rocblas_operation OP>
struct MatrixLoad;


/*
 * FULL Matrix
 */
template<rocblas_diagonal DIAG_TYPE>
struct MatrixLoad<rocblas_fill_full, DIAG_TYPE, rocblas_operation_none> {
    template <typename T>
    __device__ static inline T eval(const T* M, const rocblas_int ld, const rocblas_int inc, const rocblas_int row,
                                    const rocblas_int col)
    {
        return M[row * inc + col * ld];
    }
};

//transposed
template<rocblas_diagonal DIAG_TYPE, rocblas_operation OP>
struct MatrixLoad<rocblas_fill_full, DIAG_TYPE, OP> {
    template <typename T>
    __device__ static inline T eval(const T* M, const rocblas_int ld, const rocblas_int inc, const rocblas_int row,
                                    const rocblas_int col)
    {
        return M[col * inc + row * ld];
    }
};

/*
 * Lower Tri Matrix
 */
// non-unit diag
template<>
struct MatrixLoad<rocblas_fill_lower, rocblas_diagonal_non_unit, rocblas_operation_none> {
    template <typename T>
    __device__ static inline T eval(const T* M, const rocblas_int ld, const rocblas_int inc, const rocblas_int row,
                                    const rocblas_int col)
    {
        if (col > row) return CreateReal<T>::eval(0);
        return M[row * inc + col * ld];
    }
};

// transposed non-unit diag
template<rocblas_operation OP>
struct MatrixLoad<rocblas_fill_lower, rocblas_diagonal_non_unit, OP> {
    template <typename T>
    __device__ static inline T eval(const T* M, const rocblas_int ld, const rocblas_int inc, const rocblas_int row,
                                    const rocblas_int col)
    {
        if (row > col) return CreateReal<T>::eval(0);
        return M[col * inc + row * ld];
    }
};

// unit diag
template<>
struct MatrixLoad<rocblas_fill_lower, rocblas_diagonal_unit, rocblas_operation_none> {
    template <typename T>
    __device__ static inline T eval(const T* M, const rocblas_int ld, const rocblas_int inc, const rocblas_int row,
                                    const rocblas_int col)
    {
        if (col == row) return CreateReal<T>::eval(1);
        if (col > row) return CreateReal<T>::eval(0);
        return M[row * inc + col * ld];
    }
};

// transposed unit diag
template<rocblas_operation OP>
struct MatrixLoad<rocblas_fill_lower, rocblas_diagonal_unit, OP> {
    template <typename T>
    __device__ static inline T eval(const T* M, const rocblas_int ld, const rocblas_int inc, const rocblas_int row,
                                    const rocblas_int col)
    {
        if (col == row) return CreateReal<T>::eval(1);
        if (row > col) return CreateReal<T>::eval(0);
        return M[col * inc + row * ld];
    }
};

/*
 * Upper Tri Matrix
 */
// non-unit diag
template<>
struct MatrixLoad<rocblas_fill_upper, rocblas_diagonal_non_unit, rocblas_operation_none> {
    template <typename T>
    __device__ static inline T eval(const T* M, const rocblas_int ld, const rocblas_int inc, const rocblas_int row,
                                    const rocblas_int col)
    {
        if (col < row) return CreateReal<T>::eval(0);
        return M[row * inc + col * ld];
    }
};
// transposed non-unit diag
template<rocblas_operation OP>
struct MatrixLoad<rocblas_fill_upper, rocblas_diagonal_non_unit, OP> {
    template <typename T>
    __device__ static inline T eval(const T* M, const rocblas_int ld, const rocblas_int inc, const rocblas_int row,
                                    const rocblas_int col)
    {
        if (row < col) return CreateReal<T>::eval(0);
        return M[col * inc + row * ld];
    }
};

// unit diag
template <>
struct MatrixLoad<rocblas_fill_upper, rocblas_diagonal_unit, rocblas_operation_none>
{
    template <typename T>
    __device__ static inline T eval(const T* M, const rocblas_int ld, const rocblas_int inc, const rocblas_int row,
                                    const rocblas_int col)
    {
        if (col == row)
            return CreateReal<T>::eval(1);
        if (col < row)
            return CreateReal<T>::eval(0);
        return M[row * inc + col * ld];
    }
};

// transposed unit diag
template <rocblas_operation OP>
struct MatrixLoad<rocblas_fill_upper, rocblas_diagonal_unit, OP>
{
    template <typename T>
    __device__ static inline T eval(const T* M, const rocblas_int ld, const rocblas_int inc, const rocblas_int row,
                                    const rocblas_int col)
    {
        if (col == row)
            return CreateReal<T>::eval(1);
        if (row < col)
            return CreateReal<T>::eval(0);
        return M[col * inc + row * ld];
    }
};

/*
 * A*B and A*B^t and A*B^H
 */
template <rocblas_int DIM_X, rocblas_int DIM_Y, rocblas_operation OP_B, rocblas_fill FILL_A, rocblas_fill FILL_B,
          rocblas_diagonal DIAG_A, rocblas_diagonal DIAG_B, typename T, typename U>
__global__ void trmmn_kernel(rocblas_int m, rocblas_int n, U alpha_device_host, const T* __restrict__ A,
                             rocblas_int lda, const T* __restrict__ B, rocblas_int ldb, T* C, rocblas_int ldc)
{
    auto alpha              = load_scalar(alpha_device_host);
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

    rocblas_int n_tail = n % (4 * DIM_Y);
    rocblas_int col    = ty * 4;
    rocblas_int col_B  = hipBlockIdx_y;

    // B += col_B * ldb;

    for (col = ty * 4; col < (n - n_tail); col += 4 * DIM_Y) {
        res_B[0] = MatrixLoad<FILL_B, DIAG_B, OP_B>::eval(B, ldb, 1, col + 0, col_B);
        res_B[1] = MatrixLoad<FILL_B, DIAG_B, OP_B>::eval(B, ldb, 1, col + 1, col_B);
        res_B[2] = MatrixLoad<FILL_B, DIAG_B, OP_B>::eval(B, ldb, 1, col + 2, col_B);
        res_B[3] = MatrixLoad<FILL_B, DIAG_B, OP_B>::eval(B, ldb, 1, col + 3, col_B);

        res_B[0] = rb_port_conj_op<OP_B, T>::eval(res_B[0]);
        res_B[1] = rb_port_conj_op<OP_B, T>::eval(res_B[1]);
        res_B[2] = rb_port_conj_op<OP_B, T>::eval(res_B[2]);
        res_B[3] = rb_port_conj_op<OP_B, T>::eval(res_B[3]);

        if (ind < m) {
            res_A[0] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind, (col + 0)) * res_B[0];
            res_A[0] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind, (col + 1)) * res_B[1];
            res_A[0] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind, (col + 2)) * res_B[2];
            res_A[0] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind, (col + 3)) * res_B[3];
        }

        if (ind + DIM_X < m) {
            res_A[1] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + DIM_X, (col + 0)) * res_B[0];
            res_A[1] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + DIM_X, (col + 1)) * res_B[1];
            res_A[1] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + DIM_X, (col + 2)) * res_B[2];
            res_A[1] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + DIM_X, (col + 3)) * res_B[3];
        }

        if (ind + 2 * DIM_X < m) {
            res_A[2] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 2 * DIM_X, (col + 0)) *
                res_B[0];
            res_A[2] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 2 * DIM_X, (col + 1)) *
                res_B[1];
            res_A[2] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 2 * DIM_X, (col + 2)) *
                res_B[2];
            res_A[2] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 2 * DIM_X, (col + 3)) *
                res_B[3];
        }

        if (ind + 3 * DIM_X < m) {
            res_A[3] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 3 * DIM_X, (col + 0)) *
                res_B[0];
            res_A[3] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 3 * DIM_X, (col + 1)) *
                res_B[1];
            res_A[3] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 3 * DIM_X, (col + 2)) *
                res_B[2];
            res_A[3] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 3 * DIM_X, (col + 3)) *
                res_B[3];
        }
    }

    // if n  is not multiple of (DIM_Y * 4)
    if (n_tail > 0) {
        res_B[0] = T(0);
        res_B[1] = T(0);
        res_B[2] = T(0);
        res_B[3] = T(0);

        if (col + 0 < n)
            res_B[0] = MatrixLoad<FILL_B, DIAG_B, OP_B>::eval(B, ldb, 1, col + 0, col_B);
        if (col + 1 < n)
            res_B[1] = MatrixLoad<FILL_B, DIAG_B, OP_B>::eval(B, ldb, 1, col + 1, col_B);
        if (col + 2 < n)
            res_B[2] = MatrixLoad<FILL_B, DIAG_B, OP_B>::eval(B, ldb, 1, col + 2, col_B);
        if (col + 3 < n)
            res_B[3] = MatrixLoad<FILL_B, DIAG_B, OP_B>::eval(B, ldb, 1, col + 3, col_B);

        res_B[0] = rb_port_conj_op<OP_B, T>::eval(res_B[0]);
        res_B[1] = rb_port_conj_op<OP_B, T>::eval(res_B[1]);
        res_B[2] = rb_port_conj_op<OP_B, T>::eval(res_B[2]);
        res_B[3] = rb_port_conj_op<OP_B, T>::eval(res_B[3]);

        if (ind < m) {
            res_A[0] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind, (col + 0) * (col + 0 < n)) *
                res_B[0];
            res_A[0] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind, (col + 1) * (col + 1 < n)) *
                res_B[1];
            res_A[0] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind, (col + 2) * (col + 2 < n)) *
                res_B[2];
            res_A[0] +=
                MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind, (col + 3) * (col + 3 < n)) *
                res_B[3];
        }

        if (ind + DIM_X < m) {
            res_A[1] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + DIM_X,
                                                                                 (col + 0) * (col + 0 < n)) *
                        res_B[0];
            res_A[1] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + DIM_X,
                                                                                 (col + 1) * (col + 1 < n)) *
                        res_B[1];
            res_A[1] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + DIM_X,
                                                                                 (col + 2) * (col + 2 < n)) *
                        res_B[2];
            res_A[1] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + DIM_X,
                                                                                 (col + 3) * (col + 3 < n)) *
                        res_B[3];
        }

        if (ind + 2 * DIM_X < m) {
            res_A[2] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 2 * DIM_X,
                                                                                 (col + 0) * (col + 0 < n)) *
                        res_B[0];
            res_A[2] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 2 * DIM_X,
                                                                                 (col + 1) * (col + 1 < n)) *
                        res_B[1];
            res_A[2] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 2 * DIM_X,
                                                                                 (col + 2) * (col + 2 < n)) *
                        res_B[2];
            res_A[2] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 2 * DIM_X,
                                                                                 (col + 3) * (col + 3 < n)) *
                        res_B[3];
        }

        if (ind + 3 * DIM_X < m) {
            res_A[3] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 3 * DIM_X,
                                                                                 (col + 0) * (col + 0 < n)) *
                        res_B[0];
            res_A[3] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 3 * DIM_X,
                                                                                 (col + 1) * (col + 1 < n)) *
                        res_B[1];
            res_A[3] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 3 * DIM_X,
                                                                                 (col + 2) * (col + 2 < n)) *
                        res_B[2];
            res_A[3] += MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, ind + 3 * DIM_X,
                                                                                 (col + 3) * (col + 3 < n)) *
                        res_B[3];
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
            C[ind + col_B * ldc] = alpha * sdata[thread_id];
    }
}

template <rocblas_int NB_X, rocblas_operation OP_A, rocblas_operation OP_B, rocblas_fill FILL_A, rocblas_fill FILL_B,
          rocblas_diagonal DIAG_A, rocblas_diagonal DIAG_B, typename T, typename U>
__global__ void trmmc_kernel(rocblas_int m, rocblas_int n, U alpha_device_host, const T* __restrict__ A,
                             rocblas_int lda, const T* __restrict__ B, rocblas_int ldb, T* C, rocblas_int ldc)
{

    auto alpha     = load_scalar(alpha_device_host);
    rocblas_int tx = hipThreadIdx_x;

    // if (tx < m)
    //     A += tx;
    rocblas_int tx_load_id = 0;
    if (tx < m)
        tx_load_id = tx;

    rocblas_int col_A = hipBlockIdx_x;
    rocblas_int col_B = hipBlockIdx_y;
    // A += col_A * lda;
    // B += col_B * ldb;

    T res(0);

    __shared__ T sdata[NB_X];

    // partial sums
    rocblas_int m_full = (m / NB_X) * NB_X;

    for (rocblas_int i = 0; i < m_full; i += NB_X)
        res +=
            rb_port_conj_op<OP_A, T>::eval(MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, i + tx_load_id, col_A)) *
            rb_port_conj_op<OP_B, T>::eval(MatrixLoad<FILL_B, DIAG_B, OP_B>::eval(B, ldb, 1, (tx + i), col_B));

    if (tx + m_full < m)
        res += rb_port_conj_op<OP_A, T>::eval(
                   MatrixLoad<FILL_A, DIAG_A, rocblas_operation_none>::eval(A, lda, 1, m_full + tx_load_id, col_A)) *
               rb_port_conj_op<OP_B, T>::eval(MatrixLoad<FILL_B, DIAG_B, OP_B>::eval(B, ldb, 1, (tx + m_full), col_B));

    sdata[tx] = res;

    // tree reduction of partial sums,
    if (NB_X > 16) {
        rocblas_sum_reduce<NB_X>(tx, sdata);
    } else {
        __syncthreads();

        if (tx == 0) {
            for (rocblas_int i = 1; i < m && i < NB_X; i++)
                sdata[0] += sdata[i];
        }

        __syncthreads();
    }

    if (tx == 0)
        C[col_A + col_B * ldc] = alpha * sdata[0];
}

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

template <rocblas_fill FILL_A, rocblas_fill FILL_B, rocblas_diagonal DIAG_A, rocblas_diagonal DIAG_B,
          rocblas_operation OP_B, typename T>
rocblas_status rocblas_trmm(rocblas_handle handle, rocblas_operation transa, rocblas_int rows_A, rocblas_int rows_B,
                            rocblas_int cols_B, const T* alpha, const T* A, rocblas_int lda, const T* B,
                            rocblas_int ldb, T* C, rocblas_int ldc)
{
    if (!handle)
        return rocblas_status_invalid_handle;
    if (!alpha)
        return rocblas_status_invalid_pointer;

    if (!A || !B || !C)
        return rocblas_status_invalid_pointer;

    if (rows_A < 0 || rows_B < 0 || cols_B < 0 || lda < rows_A || lda < 1 || ldb < rows_B || ldb < 1 || ldc < rows_A || ldc < 1)
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible. Not Argument error
     */
    if (!rows_A || !rows_B || !cols_B)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    if (transa == rocblas_operation_none) {
        // GEMMN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
        static constexpr int GEMMN_DIM_X = 32;
        static constexpr int GEMMN_DIM_Y = 16;
        rocblas_int blocks               = (rows_A - 1) / (GEMMN_DIM_X * 4) + 1;

        dim3 trmmn_grid(blocks, cols_B);
        dim3 trmmn_threads(GEMMN_DIM_X, GEMMN_DIM_Y);

        if (handle->pointer_mode == rocblas_pointer_mode_device) {
            hipLaunchKernelGGL((trmmn_kernel<GEMMN_DIM_X, GEMMN_DIM_Y, OP_B, FILL_A, FILL_B, DIAG_A, DIAG_B>),
                               trmmn_grid, trmmn_threads, 0, rocblas_stream, rows_A, rows_B, alpha, A, lda, B, ldb, C, ldc);
        } else {
            if (rb_port_cmp_and_real_only(*alpha, 0.0))
                return rocblas_status_success;

            hipLaunchKernelGGL((trmmn_kernel<GEMMN_DIM_X, GEMMN_DIM_Y, OP_B, FILL_A, FILL_B, DIAG_A, DIAG_B>),
                               trmmn_grid, trmmn_threads, 0, rocblas_stream, rows_A, rows_B, *alpha, A, lda, B, ldb, C, ldc);
        }
    } else {
        // transpose
        // number of columns on the y-dim of the grid, using trmmc because trmmt(transpose) is a
        // instance of trmmc (conjugate)
        static constexpr int NB = 256;
        dim3 trmmc_grid(rows_B, cols_B);
        dim3 trmmc_threads(NB);

        if (handle->pointer_mode == rocblas_pointer_mode_device) {
            if (transa == rocblas_operation_transpose)
                hipLaunchKernelGGL(trmmc_kernel<NB, rocblas_operation_transpose, OP_B, FILL_A, FILL_B, DIAG_A, DIAG_B>,
                                   trmmc_grid, trmmc_threads, 0, rocblas_stream, rows_A, rows_B, alpha, A, lda, B, ldb, C, ldc);
            else
                hipLaunchKernelGGL(
                    trmmc_kernel<NB, rocblas_operation_conjugate_transpose, OP_B, FILL_A, FILL_B, DIAG_A, DIAG_B>,
                    trmmc_grid, trmmc_threads, 0, rocblas_stream, rows_A, rows_B, alpha, A, lda, B, ldb, C, ldc);
        } else {
            if (rb_port_cmp_and_real_only(*alpha, 0))
                return rocblas_status_success;

            if (transa == rocblas_operation_transpose)
                hipLaunchKernelGGL(trmmc_kernel<NB, rocblas_operation_transpose, OP_B, FILL_A, FILL_B, DIAG_A, DIAG_B>,
                                   trmmc_grid, trmmc_threads, 0, rocblas_stream, rows_A, rows_B, *alpha, A, lda, B, ldb, C, ldc);
            else
                hipLaunchKernelGGL(
                    trmmc_kernel<NB, rocblas_operation_conjugate_transpose, OP_B, FILL_A, FILL_B, DIAG_A, DIAG_B>,
                    trmmc_grid, trmmc_threads, 0, rocblas_stream, rows_A, rows_B, *alpha, A, lda, B, ldb, C, ldc);
        }
    }
    return rocblas_status_success;
}

/*
 * ===========================================================================
 *    Helper functions to select template parameters
 * ===========================================================================
 */
template <rocblas_fill FILL_A, rocblas_fill FILL_B, rocblas_diagonal DIAG_A, rocblas_diagonal DIAG_B, typename T>
rocblas_status rocblas_trmm_select_op_b(
    std::tuple<rocblas_fill, rocblas_fill, rocblas_diagonal, rocblas_diagonal, rocblas_operation> templ_param,
    rocblas_handle handle, rocblas_operation transa, rocblas_int rows_A, rocblas_int rows_B, rocblas_int cols_B,
    const T* alpha, const T* A, rocblas_int lda, const T* B, rocblas_int ldb, T* C, rocblas_int ldc)
{
    if (std::get<4>(templ_param) == rocblas_operation_none) {
        return rocblas_trmm<FILL_A, FILL_B, DIAG_A, DIAG_B, rocblas_operation_none, T>(
            handle, transa, rows_A, rows_B, cols_B, alpha, A, lda, B, ldb, C, ldc);
    } else if (std::get<4>(templ_param) == rocblas_operation_transpose) {
        return rocblas_trmm<FILL_A, FILL_B, DIAG_A, DIAG_B, rocblas_operation_transpose, T>(
            handle, transa, rows_A, rows_B, cols_B, alpha, A, lda, B, ldb, C, ldc);
    } else {
        return rocblas_trmm<FILL_A, FILL_B, DIAG_A, DIAG_B, rocblas_operation_conjugate_transpose, T>(
            handle, transa, rows_A, rows_B, cols_B, alpha, A, lda, B, ldb, C, ldc);
    }
}

template <rocblas_fill FILL_A, rocblas_fill FILL_B, rocblas_diagonal DIAG_A, typename T>
rocblas_status rocblas_trmm_select_diag2(
    std::tuple<rocblas_fill, rocblas_fill, rocblas_diagonal, rocblas_diagonal, rocblas_operation> templ_param,
    rocblas_handle handle, rocblas_operation transa, rocblas_int rows_A, rocblas_int rows_B, rocblas_int cols_B,
    const T* alpha, const T* A, rocblas_int lda, const T* B, rocblas_int ldb, T* C, rocblas_int ldc)
{
    if (std::get<3>(templ_param) == rocblas_diagonal_unit) {
        return rocblas_trmm_select_op_b<FILL_A, FILL_B, DIAG_A, rocblas_diagonal_unit>(
            templ_param, handle, transa, rows_A, rows_B, cols_B, alpha, A, lda, B, ldb, C, ldc);
    } else {
        return rocblas_trmm_select_op_b<FILL_A, FILL_B, DIAG_A, rocblas_diagonal_non_unit>(
            templ_param, handle, transa, rows_A, rows_B, cols_B, alpha, A, lda, B, ldb, C, ldc);
    }
}

template <rocblas_fill FILL_A, rocblas_fill FILL_B, typename T>
rocblas_status rocblas_trmm_select_diag1(
    std::tuple<rocblas_fill, rocblas_fill, rocblas_diagonal, rocblas_diagonal, rocblas_operation> templ_param,
    rocblas_handle handle, rocblas_operation transa, rocblas_int rows_A, rocblas_int rows_B, rocblas_int cols_B,
    const T* alpha, const T* A, rocblas_int lda, const T* B, rocblas_int ldb, T* C, rocblas_int ldc)
{
    if (std::get<2>(templ_param) == rocblas_diagonal_unit) {
        return rocblas_trmm_select_diag2<FILL_A, FILL_B, rocblas_diagonal_unit>(
            templ_param, handle, transa, rows_A, rows_B, cols_B, alpha, A, lda, B, ldb, C, ldc);
    } else {
        return rocblas_trmm_select_diag2<FILL_A, FILL_B, rocblas_diagonal_non_unit>(
            templ_param, handle, transa, rows_A, rows_B, cols_B, alpha, A, lda, B, ldb, C, ldc);
    }
}

template <rocblas_fill FILL_A, typename T>
rocblas_status rocblas_trmm_select_fill2(
    std::tuple<rocblas_fill, rocblas_fill, rocblas_diagonal, rocblas_diagonal, rocblas_operation> templ_param,
    rocblas_handle handle, rocblas_operation transa, rocblas_int rows_A, rocblas_int rows_B, rocblas_int cols_B,
    const T* alpha, const T* A, rocblas_int lda, const T* B, rocblas_int ldb, T* C, rocblas_int ldc)
{
    if (std::get<1>(templ_param) == rocblas_fill_lower) {
        return rocblas_trmm_select_diag1<FILL_A, rocblas_fill_lower>(templ_param, handle, transa, rows_A, rows_B,
                                                                     cols_B, alpha, A, lda, B, ldb, C, ldc);
    } else if (std::get<1>(templ_param) == rocblas_fill_upper) {
        return rocblas_trmm_select_diag1<FILL_A, rocblas_fill_upper>(templ_param, handle, transa, rows_A, rows_B,
                                                                     cols_B, alpha, A, lda, B, ldb, C, ldc);
    } else {
        return rocblas_trmm_select_diag1<FILL_A, rocblas_fill_full>(templ_param, handle, transa, rows_A, rows_B, cols_B,
                                                                    alpha, A, lda, B, ldb, C, ldc);
    }
}

template <typename T>
rocblas_status rocblas_trmm_select_fill1(
    std::tuple<rocblas_fill, rocblas_fill, rocblas_diagonal, rocblas_diagonal, rocblas_operation> templ_param,
    rocblas_handle handle, rocblas_operation transa, rocblas_int rows_A, rocblas_int rows_B, rocblas_int cols_B,
    const T* alpha, const T* A, rocblas_int lda, const T* B, rocblas_int ldb, T* C, rocblas_int ldc)
{
    if (std::get<0>(templ_param) == rocblas_fill_lower) {
        return rocblas_trmm_select_fill2<rocblas_fill_lower>(templ_param, handle, transa, rows_A, rows_B, cols_B, alpha,
                                                             A, lda, B, ldb, C, ldc);
    } else if (std::get<0>(templ_param) == rocblas_fill_upper) {
        return rocblas_trmm_select_fill2<rocblas_fill_upper>(templ_param, handle, transa, rows_A, rows_B, cols_B, alpha,
                                                             A, lda, B, ldb, C, ldc);
    } else {
        return rocblas_trmm_select_fill2<rocblas_fill_full>(templ_param, handle, transa, rows_A, rows_B, cols_B, alpha,
                                                            A, lda, B, ldb, C, ldc);
    }
}

template <typename T>
rocblas_status rocblas_trmm_select(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                                   rocblas_diagonal diag, rocblas_int m, rocblas_int n, const T* alpha, const T* A,
                                   rocblas_int lda, const T* B, rocblas_int ldb, T* C, rocblas_int ldc)
{
    rocblas_operation transA, transB;
    rocblas_fill uploA, uploB;
    rocblas_diagonal diagA, diagB;
    rocblas_int rows_A, rows_B, cols_B;

    // create parameters according to multiplication order
    if (side == rocblas_side_right) {
        rows_A = m;
        rows_B = n;
        cols_B = n;
        std::swap(A, B);
        std::swap(lda, ldb);
        transA = rocblas_operation_none;
        transB = trans;
        uploA  = rocblas_fill_full;
        uploB  = uplo;
        diagA = rocblas_diagonal_non_unit;
        diagB = diag;
    } else {
        rows_A = m;
        rows_B = m;
        cols_B = n;
        transB = rocblas_operation_none;
        transA = trans;
        uploB= rocblas_fill_full;
        uploA = uplo;
        diagB = rocblas_diagonal_non_unit;
        diagA = diag;

    }

    auto templ_param = std::make_tuple(uploA, uploB, diagA, diagB, transB);

    return rocblas_trmm_select_fill1(templ_param, handle, transA, rows_A, rows_B, cols_B, alpha, A, lda, B, ldb, C, ldc);
}

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_port_strmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                                  rocblas_diagonal diag, rocblas_int m, rocblas_int n, const float* alpha,
                                  const float* A, rocblas_int lda, const float* B, rocblas_int ldb, float* C,
                                  rocblas_int ldc)
{
    return rocblas_trmm_select<float>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

rocblas_status rocblas_port_dtrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                                  rocblas_diagonal diag, rocblas_int m, rocblas_int n, const double* alpha,
                                  const double* A, rocblas_int lda, const double* B, rocblas_int ldb, double* C,
                                  rocblas_int ldc)
{
    return rocblas_trmm_select<double>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

rocblas_status rocblas_port_ctrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                                  rocblas_diagonal diag, rocblas_int m, rocblas_int n, const hipFloatComplex* alpha,
                                  const hipFloatComplex* A, rocblas_int lda, const hipFloatComplex* B, rocblas_int ldb,
                                  hipFloatComplex* C, rocblas_int ldc)
{
    return rocblas_trmm_select<hipFloatComplex>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

rocblas_status rocblas_port_ztrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                                  rocblas_diagonal diag, rocblas_int m, rocblas_int n, const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A, rocblas_int lda, const hipDoubleComplex* B,
                                  rocblas_int ldb, hipDoubleComplex* C, rocblas_int ldc)
{
    return rocblas_trmm_select<hipDoubleComplex>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

} // extern "C"
