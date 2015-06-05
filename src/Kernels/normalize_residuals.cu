#include "kernels_common.h"

__global__ void normalize_residuals_gpu_kernel
(
    int const num_gkvec_row,
    int const* res_idx,
    double const* norm2,
    cuDoubleComplex* res
)
{
    int igk = blockIdx.x * blockDim.x + threadIdx.x;
    int ibnd = res_idx[blockIdx.y];

    if (igk < num_gkvec_row)
    {
        int k = array2D_offset(igk, blockIdx.y, num_gkvec_row);
        res[k] = cuCdiv(res[k], make_cuDoubleComplex(sqrt(norm2[ibnd]), 0.0));
    }
}

extern "C" void normalize_residuals_gpu(int num_gkvec_row,
                                        int num_res_local,
                                        int const* res_idx,
                                        double* norm2,
                                        cuDoubleComplex* res)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gkvec_row, grid_t.x), num_res_local);

    normalize_residuals_gpu_kernel <<<grid_b, grid_t>>>
    (
        num_gkvec_row,
        res_idx,
        norm2,
        res
    );
}


