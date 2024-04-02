/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file scale_matrix.cu
 *
 *  \brief Contains implementation of CUDA kernels to scale matrix elements (rows or columns).
 */

#include "core/acc/acc_common.hpp"
#include "core/acc/acc_runtime.hpp"

using namespace sirius;
using namespace sirius::acc;

template <typename T>
__global__ void scale_matrix_columns_gpu_kernel(int nrow, gpu_complex_type<T>* mtrx, T* a);

template <>
__global__ void scale_matrix_columns_gpu_kernel<double>
(
    int nrow,
    acc_complex_double_t* mtrx,
    double* a
)
{
    int icol = blockIdx.y;
    int irow = blockIdx.x * blockDim.x + threadIdx.x;
    if (irow < nrow) {
        mtrx[array2D_offset(irow, icol, nrow)] =
            accCmul(mtrx[array2D_offset(irow, icol, nrow)], make_accDoubleComplex(a[icol], 0));
    }
}

template <>
__global__ void scale_matrix_columns_gpu_kernel<float>
    (
        int nrow,
        acc_complex_float_t* mtrx,
        float* a
    )
{
    int icol = blockIdx.y;
    int irow = blockIdx.x * blockDim.x + threadIdx.x;
    if (irow < nrow) {
        mtrx[array2D_offset(irow, icol, nrow)] =
            accCmulf(mtrx[array2D_offset(irow, icol, nrow)], make_accFloatComplex(a[icol], 0));
    }
}

// scale each column of the matrix by a column-dependent constant
extern "C" void scale_matrix_columns_gpu_double(int nrow,
                                                 int ncol,
                                                 acc_complex_double_t* mtrx,
                                                 double* a)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(nrow, grid_t.x), ncol);

    accLaunchKernel((scale_matrix_columns_gpu_kernel<double>), dim3(grid_b), dim3(grid_t), 0, 0, nrow, mtrx, a);
}

extern "C" void scale_matrix_columns_gpu_float(int nrow,
                                                int ncol,
                                                acc_complex_float_t* mtrx,
                                                float* a)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(nrow, grid_t.x), ncol);

    accLaunchKernel((scale_matrix_columns_gpu_kernel<float>), dim3(grid_b), dim3(grid_t), 0, 0, nrow, mtrx, a);
}

__global__ void scale_matrix_rows_gpu_kernel
(
    int nrow__,
    acc_complex_double_t* mtrx__,
    double const* v__
)
{
    int icol = blockIdx.y;
    int irow = blockDim.x * blockIdx.x + threadIdx.x;
    if (irow < nrow__) {
        acc_complex_double_t z = mtrx__[array2D_offset(irow, icol, nrow__)];
        mtrx__[array2D_offset(irow, icol, nrow__)] = make_accDoubleComplex(z.x * v__[irow], z.y * v__[irow]);
    }
}

// scale each row of the matrix by a row-dependent constant
extern "C" void scale_matrix_rows_gpu(int nrow__,
                                      int ncol__,
                                      acc_complex_double_t* mtrx__,
                                      double const* v__)
{
    dim3 grid_t(256);
    dim3 grid_b(num_blocks(nrow__, grid_t.x), ncol__);

    accLaunchKernel((scale_matrix_rows_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
        nrow__,
        mtrx__,
        v__
    );
}

__global__ void scale_matrix_elements_gpu_kernel
(
    acc_complex_double_t* mtrx__,
    int ld__,
    int nrow__,
    double beta__
)
{
    int icol = blockIdx.y;
    int irow = blockDim.x * blockIdx.x + threadIdx.x;
    if (irow < nrow__) {
        acc_complex_double_t z = mtrx__[array2D_offset(irow, icol, ld__)];
        mtrx__[array2D_offset(irow, icol, ld__)] = make_accDoubleComplex(z.x * beta__, z.y * beta__);
    }
}

extern "C" void scale_matrix_elements_gpu(acc_complex_double_t* ptr__,
                                          int ld__,
                                          int nrow__,
                                          int ncol__,
                                          double beta__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(nrow__, grid_t.x), ncol__);

    accLaunchKernel((scale_matrix_elements_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
        ptr__,
        ld__,
        nrow__,
        beta__
    );
}
