#include "kernels_common.hpp"

__global__ void update_apw_coeffs_gpu_kernel(cuDoubleComplex* apw_coeffs__,
                                             int ld__,
                                             cuDoubleComplex* v__,
                                             cuDoubleComplex* alm__,
                                             int nrow__)
{ 
    int icol = blockIdx.y;
    int irow = blockDim.x * blockIdx.x + threadIdx.x;
    if (irow < nrow__) {
        apw_coeffs__[array2D_offset(irow, icol, ld__)] += v__[icol] * alm__[irow];
    }
}


extern "C" void update_apw_coeffs_gpu(cuDoubleComplex* apw_coeffs__,
                                      int ld__,
                                      cuDoubleComplex* v__,
                                      cuDoubleComplex* alm__,
                                      int nrow__,
                                      int ncol__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(nrow__, grid_t.x), ncol__);

    update_apw_coeffs_gpu_kernel <<<grid_b, grid_t>>>
    (


    );
}
