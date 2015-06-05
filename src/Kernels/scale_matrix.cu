#include "kernels_common.h"

__global__ void scale_matrix_columns_gpu_kernel
(
    int nrow,
    cuDoubleComplex* mtrx,
    double* a
)
{
    int icol = blockIdx.y;
    int irow = blockIdx.x * blockDim.x + threadIdx.x;
    if (irow < nrow) 
    {
        mtrx[array2D_offset(irow, icol, nrow)] =
            cuCmul(mtrx[array2D_offset(irow, icol, nrow)], make_cuDoubleComplex(a[icol], 0));
    }
}

// scale each column of the matrix by a column-dependent constant
extern "C" void scale_matrix_columns_gpu(int nrow,
                                         int ncol,
                                         cuDoubleComplex* mtrx,
                                         double* a)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(nrow, grid_t.x), ncol);

    scale_matrix_columns_gpu_kernel <<<grid_b, grid_t>>>
    (
        nrow,
        mtrx,
        a
    );
}

__global__ void scale_matrix_rows_gpu_kernel
(
    int nrow,
    cuDoubleComplex* mtrx,
    double* v
)
{
    int icol = blockIdx.y;
    int irow = blockDim.x * blockIdx.x + threadIdx.x;
    if (irow < nrow) 
    {
        mtrx[array2D_offset(irow, icol, nrow)] = 
            cuCmul(mtrx[array2D_offset(irow, icol, nrow)], make_cuDoubleComplex(v[irow], 0));
    }
}

// scale each row of the matrix by a row-dependent constant
extern "C" void scale_matrix_rows_gpu(int nrow,
                                      int ncol,
                                      cuDoubleComplex* mtrx,
                                      double* v)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(nrow, grid_t.x), ncol);

    scale_matrix_rows_gpu_kernel <<<grid_b, grid_t>>>
    (
        nrow,
        mtrx,
        v
    );
}

