/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file diag_mm.cu
 *
 *  \brief
 */

#include "diag_mm.hpp"
#include "acc_runtime.hpp"
#include "acc.hpp"

template <class T>
__global__ std::enable_if_t<!std::is_same<acc_complex_double_t, T>::value>
diag_mm(const T* diag, int n, const T* X, int lda_x, int ncols, T* Y, int lda_y, T alpha)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < ncols && row < n) {
        T X_elem                 = *(X + lda_x * col + row);
        T D                      = diag[row];
        *(Y + lda_y * col + row) = alpha * D * X_elem;
    }
}

template <class T>
__global__ std::enable_if_t<std::is_same<acc_complex_double_t, T>::value>
diag_mm(const T* diag, int n, const T* X, int lda_x, int ncols, T* Y, int lda_y, T alpha)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < ncols && row < n) {
        acc_complex_double_t X_elem = *(X + lda_x * col + row);
        acc_complex_double_t D      = diag[row];
        *(Y + lda_y * col + row)    = accCmul(accCmul(alpha, D), X_elem);
    }
}

template <class T>
void
call_diagmm(const T* diag, int n, const T* X, int lda_x, int ncols, T* Y, int lda_y, T alpha)
{
    int numthreads = 32;
    dim3 threadsPerBlock(numthreads, numthreads);

    int num_block_rows = (n + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int num_block_cols = (ncols + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 numBlocks(num_block_rows, num_block_cols);

    diag_mm<<<numBlocks, threadsPerBlock>>>(diag, n, X, lda_x, ncols, Y, lda_y, alpha);
}

extern "C" {
void
ddiagmm(const double* diag, int n, const double* X, int lda_x, int ncols, double* Y, int lda_y, double alpha)
{
    call_diagmm(diag, n, X, lda_x, ncols, Y, lda_y, alpha);
}

void
sdiagmm(const float* diag, int n, const float* X, int lda_x, int ncols, float* Y, int lda_y, float alpha)
{
    call_diagmm(diag, n, X, lda_x, ncols, Y, lda_y, alpha);
}
void
zdiagmm(const std::complex<double>* diag, int n, const std::complex<double>* X, int lda_x, int ncols,
        std::complex<double>* Y, int lda_y, std::complex<double> alpha)
{
    call_diagmm(reinterpret_cast<const acc_complex_double_t*>(diag), n,
                reinterpret_cast<const acc_complex_double_t*>(X), lda_x, ncols,
                reinterpret_cast<acc_complex_double_t*>(Y), lda_y, acc_complex_double_t{alpha.real(), alpha.imag()});
}
}
