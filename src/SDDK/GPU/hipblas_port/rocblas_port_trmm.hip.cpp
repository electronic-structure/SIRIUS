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
#include <exception>

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


template <rocblas_operation OP>
struct MatrixRows {
    template <typename T, typename U>
    __host__ __device__ static inline T eval(const T rows, const U cols) {
        // transpose or hermitian.
        return cols;
    }
};

template <>
struct MatrixRows<rocblas_operation_none> {
    template <typename T, typename U>
    __host__ __device__ static inline T eval(const T rows, const U cols) {
        // transpose or hermitian.
        return rows;
    }
};

template <rocblas_operation OP>
struct MatrixCols {
    template <typename T, typename U>
    __host__ __device__ static inline T eval(const T rows, const U cols) {
        // transpose or hermitian.
        return rows;
    }
};

template <>
struct MatrixCols<rocblas_operation_none> {
    template <typename T, typename U>
    __host__ __device__ static inline T eval(const T rows, const U cols) {
        // transpose or hermitian.
        return cols;
    }
};

template <rocblas_operation OP>
struct MatrixStore
{
    template <typename T, typename U>
    __device__ static inline T eval(T* M, const rocblas_int ld, const rocblas_int inc, const rocblas_int row,
                                    const rocblas_int col, const U& val)
    {
        return M[col * inc + row * ld] = val;;
    }
};

template <>
struct MatrixStore<rocblas_operation_none>
{
    template <typename T, typename U>
    __device__ static inline void eval(T* M, const rocblas_int ld, const rocblas_int inc, const rocblas_int row,
                                    const rocblas_int col, const U& val)
    {
        M[row * inc + col * ld] = val;
    }
};

/*
 *
 */
template <rocblas_operation OP_A_ELEMENT, rocblas_operation OP_A, rocblas_operation OP_B, rocblas_operation OP_C,
          rocblas_fill FILL_A, rocblas_fill FILL_B, rocblas_diagonal DIAG_A, rocblas_diagonal DIAG_B, typename U,
          typename T>
__global__ void trmmn_kernel_a_t_h(rocblas_int m, rocblas_int n, U alpha_device_host, const T* __restrict__ A,
                                   rocblas_int lda, const T* __restrict__ B, rocblas_int ldb, T* C, rocblas_int ldc)
{

    const int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    if (tx >= m)
        return;

    const auto alpha     = load_scalar(alpha_device_host);
    const int row_op_C = MatrixRows<OP_C>::eval(tx, hipBlockIdx_y);
    const int col_op_C = MatrixCols<OP_C>::eval(tx, hipBlockIdx_y);

    const int rows_B = MatrixRows<OP_B>::eval(m, n);

    T res(0);
    for (int col = 0; col < rows_B; ++col) {
        res += ConjOp<OP_A_ELEMENT, T>::eval(MatrixLoad<FILL_A, DIAG_A, OP_A>::eval(A, lda, 1, row_op_C, col)) *
               MatrixLoad<FILL_B, DIAG_B, OP_B>::eval(B, ldb, 1, col, col_op_C);
    }

    MatrixStore<OP_C>::eval(C, ldc, 1, row_op_C, col_op_C, res*alpha);
}



template <rocblas_operation OP_A_ELEMENT,rocblas_operation OP_A, rocblas_operation OP_B, rocblas_operation OP_C, rocblas_fill FILL_A,
          rocblas_diagonal DIAG_A, typename T>
rocblas_status rocblas_trmm(rocblas_handle handle, rocblas_int m, rocblas_int n, const T* alpha, const T* A,
                            rocblas_int lda, const T* B, rocblas_int ldb, T* C, rocblas_int ldc)
{
    if (!handle)
        return rocblas_status_invalid_handle;
    if (!alpha)
        return rocblas_status_invalid_pointer;

    if (!A || !B || !C)
        return rocblas_status_invalid_pointer;

    if (!m || !n)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    dim3 threads(256);
    dim3 grid((m + threads.x - 1) / threads.x, n);

    if (handle->pointer_mode == rocblas_pointer_mode_device) {
        hipLaunchKernelGGL(trmmn_kernel_a_t_h<OP_A_ELEMENT, OP_A, OP_B, OP_C, FILL_A, rocblas_fill_full, DIAG_A,
                                              rocblas_diagonal_non_unit>,
                           grid, threads, 0, rocblas_stream, m, n, alpha, A, lda, B, ldb, C, ldc);
    } else {
        if (rb_port_cmp_and_real_only(*alpha, 0))
            return rocblas_status_success;
        hipLaunchKernelGGL(trmmn_kernel_a_t_h<OP_A_ELEMENT, OP_A, OP_B, OP_C, FILL_A, rocblas_fill_full, DIAG_A,
                                              rocblas_diagonal_non_unit>,
                           grid, threads, 0, rocblas_stream, m, n, *alpha, A, lda, B, ldb, C, ldc);
    }
    return rocblas_status_success;
}

template <rocblas_operation OP_A_ELEMENT, rocblas_operation OP_A, rocblas_operation OP_B, rocblas_operation OP_C,
          rocblas_fill FILL_A, typename T>
rocblas_status rocblas_trmm_select_diag(rocblas_handle handle, rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                                        const T* alpha, const T* A, rocblas_int lda, const T* B, rocblas_int ldb, T* C,
                                        rocblas_int ldc)
{
    if (diag == rocblas_diagonal_unit) {
        return rocblas_trmm<OP_A_ELEMENT, OP_A, OP_B, OP_C, FILL_A, rocblas_diagonal_unit>(handle, m, n, alpha, A, lda,
                                                                                           B, ldb, C, ldc);
    } else {
        return rocblas_trmm<OP_A_ELEMENT, OP_A, OP_B, OP_C, FILL_A, rocblas_diagonal_non_unit>(handle, m, n, alpha, A,
                                                                                               lda, B, ldb, C, ldc);
    }
}

template <rocblas_operation OP_A_ELEMENT, rocblas_operation OP_A, rocblas_operation OP_B, rocblas_operation OP_C,
          typename T>
rocblas_status rocblas_trmm_select_fill(rocblas_handle handle, rocblas_fill uplo, rocblas_diagonal diag, rocblas_int m,
                                        rocblas_int n, const T* alpha, const T* A, rocblas_int lda, const T* B,
                                        rocblas_int ldb, T* C, rocblas_int ldc)
{
    if (uplo == rocblas_fill_lower) {
        return rocblas_trmm_select_diag<OP_A_ELEMENT, OP_A, OP_B, OP_C, rocblas_fill_lower>(handle, diag, m, n, alpha,
                                                                                            A, lda, B, ldb, C, ldc);
    } else if (uplo == rocblas_fill_upper) {
        return rocblas_trmm_select_diag<OP_A_ELEMENT, OP_A, OP_B, OP_C, rocblas_fill_upper>(handle, diag, m, n, alpha,
                                                                                            A, lda, B, ldb, C, ldc);
    } else {
        return rocblas_trmm_select_diag<OP_A_ELEMENT, OP_A, OP_B, OP_C, rocblas_fill_full>(handle, diag, m, n, alpha,
                                                                                            A, lda, B, ldb, C, ldc);
    }
}

template <typename T>
rocblas_status rocblas_trmm_select(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
                                   rocblas_diagonal diag, rocblas_int m, rocblas_int n, const T* alpha, const T* A,
                                   rocblas_int lda, const T* B, rocblas_int ldb, T* C, rocblas_int ldc)
{
    if (side == rocblas_side_left) {
        if (trans == rocblas_operation_none) {
            return rocblas_trmm_select_fill<rocblas_operation_none, rocblas_operation_none, rocblas_operation_none,
                                            rocblas_operation_none>(handle, uplo, diag, m, n, alpha, A, lda, B, ldb, C,
                                                                    ldc);
        } else if (trans == rocblas_operation_transpose) {
            return rocblas_trmm_select_fill<rocblas_operation_none, rocblas_operation_transpose, rocblas_operation_none,
                                            rocblas_operation_none>(handle, uplo, diag, m, n, alpha, A, lda, B, ldb, C,
                                                                    ldc);
        } else {
            return rocblas_trmm_select_fill<rocblas_operation_conjugate_transpose,
                                            rocblas_operation_conjugate_transpose, rocblas_operation_none,
                                            rocblas_operation_none>(handle, uplo, diag, m, n, alpha, A, lda, B, ldb, C,
                                                                    ldc);
        }
    } else {
        // Use the following identities:
        // B*A = (AT*BT)T
        // B*AT = (A*BT)T
        if (trans == rocblas_operation_none) {
            return rocblas_trmm_select_fill<rocblas_operation_none, rocblas_operation_transpose,
                                            rocblas_operation_transpose, rocblas_operation_transpose>(
                handle, uplo, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
        } else if (trans == rocblas_operation_transpose) {
            return rocblas_trmm_select_fill<rocblas_operation_none, rocblas_operation_none, rocblas_operation_transpose,
                                            rocblas_operation_transpose>(handle, uplo, diag, m, n, alpha, A, lda, B,
                                                                         ldb, C, ldc);
        } else {
            return rocblas_trmm_select_fill<rocblas_operation_conjugate_transpose, rocblas_operation_none,
                                            rocblas_operation_transpose, rocblas_operation_transpose>(
                handle, uplo, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
        }
    }
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
