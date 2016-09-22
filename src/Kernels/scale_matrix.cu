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
    int nrow__,
    cuDoubleComplex* mtrx__,
    double const* v__
)
{
    int icol = blockIdx.y;
    int irow = blockDim.x * blockIdx.x + threadIdx.x;
    if (irow < nrow__) 
    {
        cuDoubleComplex z = mtrx__[array2D_offset(irow, icol, nrow__)];
        mtrx__[array2D_offset(irow, icol, nrow__)] = make_cuDoubleComplex(z.x * v__[irow], z.y * v__[irow]);
        //mtrx[array2D_offset(irow, icol, nrow)] = 
        //    cuCmul(mtrx[array2D_offset(irow, icol, nrow)], make_cuDoubleComplex(v[irow], 0));
    }
}

// scale each row of the matrix by a row-dependent constant
extern "C" void scale_matrix_rows_gpu(int nrow__,
                                      int ncol__,
                                      cuDoubleComplex* mtrx__,
                                      double const* v__)
{
    dim3 grid_t(256);
    dim3 grid_b(num_blocks(nrow__, grid_t.x), ncol__);

    scale_matrix_rows_gpu_kernel <<<grid_b, grid_t>>>
    (
        nrow__,
        mtrx__,
        v__
    );
}

__global__ void scale_matrix_elements_gpu_kernel
(
    cuDoubleComplex* mtrx__,
    int ld__,
    int nrow__,
    double beta__
)
{
    int icol = blockIdx.y;
    int irow = blockDim.x * blockIdx.x + threadIdx.x;
    if (irow < nrow__) {
        cuDoubleComplex z = mtrx__[array2D_offset(irow, icol, nrow__)];
        mtrx__[array2D_offset(irow, icol, nrow__)] = make_cuDoubleComplex(z.x * beta__, z.y * beta__);
    }
}

extern "C" void scale_matrix_elements_gpu(cuDoubleComplex* ptr__,
                                          int ld__,
                                          int nrow__,
                                          int ncol__,
                                          double beta__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(nrow__, grid_t.x), ncol__);

    scale_matrix_elements_gpu_kernel <<<grid_b, grid_t>>>
    (
        ptr__,
        ld__,
        nrow__,
        beta__
    );
}
