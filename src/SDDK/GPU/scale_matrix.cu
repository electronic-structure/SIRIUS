// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file scale_matrix.cu
 *
 *  \brief Contains implementaiton of CUDA kernels to scale matrix elements (rows or columns).
 */
#include "cuda_common.hpp"
#include "acc_runtime.hpp"

__global__ void scale_matrix_columns_gpu_kernel
(
    int nrow,
    acc_complex_double_t* mtrx,
    double* a
)
{
    int icol = blockIdx.y;
    int irow = blockIdx.x * blockDim.x + threadIdx.x;
    if (irow < nrow) 
    {
        mtrx[array2D_offset(irow, icol, nrow)] =
            accCmul(mtrx[array2D_offset(irow, icol, nrow)], make_accDoubleComplex(a[icol], 0));
    }
}

// scale each column of the matrix by a column-dependent constant
extern "C" void scale_matrix_columns_gpu(int nrow,
                                         int ncol,
                                         acc_complex_double_t* mtrx,
                                         double* a)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(nrow, grid_t.x), ncol);

    accLaunchKernel((scale_matrix_columns_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
        nrow,
        mtrx,
        a
    );
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
