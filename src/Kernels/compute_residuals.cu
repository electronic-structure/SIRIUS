#include "kernels_common.h"

__global__ void compute_residuals_gpu_kernel
(
    int const num_gkvec_row,
    int const* res_idx,
    double const* eval,
    cuDoubleComplex const* hpsi,
    cuDoubleComplex const* opsi,
    cuDoubleComplex* res
)
{
    int igk = blockIdx.x * blockDim.x + threadIdx.x;
    int ibnd = res_idx[blockIdx.y];

    if (igk < num_gkvec_row)
    {
        int k = array2D_offset(igk, blockIdx.y, num_gkvec_row);
        /* res = hpsi_j - e_j * opsi_j */
        res[k] = cuCsub(hpsi[k], cuCmul(make_cuDoubleComplex(eval[ibnd], 0), opsi[k]));
    }
}

__global__ void compute_residuals_norm_gpu_kernel
(
    int num_gkvec_row,
    int* res_idx,
    cuDoubleComplex const* res,
    double* res_norm
)
{
    int N = num_blocks(num_gkvec_row, blockDim.x);

    extern __shared__ char sdata_ptr[];
    double* sdata = (double*)&sdata_ptr[0];

    sdata[threadIdx.x] = 0.0;

    for (int n = 0; n < N; n++)
    {
        int igk = n * blockDim.x + threadIdx.x;
        if (igk < num_gkvec_row)
        {
            int k = array2D_offset(igk, blockIdx.x, num_gkvec_row);
            sdata[threadIdx.x] += res[k].x * res[k].x + res[k].y * res[k].y;
        }
    }

    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) 
    {
        if (threadIdx.x % (2 * s) == 0) sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + s];
        __syncthreads();
    }
    
    res_norm[res_idx[blockIdx.x]] = sdata[0];
}

extern "C" void compute_residuals_gpu(int num_gkvec_row,
                                      int num_res_local,
                                      int* res_idx,
                                      double* eval,
                                      cuDoubleComplex const* hpsi,
                                      cuDoubleComplex const* opsi,
                                      cuDoubleComplex* res,
                                      double* res_norm)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gkvec_row, grid_t.x), num_res_local);

    compute_residuals_gpu_kernel <<<grid_b, grid_t>>>
    (
        num_gkvec_row,
        res_idx,
        eval,
        hpsi,
        opsi,
        res
    );

    grid_b = dim3(num_res_local);
    compute_residuals_norm_gpu_kernel <<<grid_b, grid_t, grid_t.x * sizeof(double)>>>
    (
        num_gkvec_row,
        res_idx,
        res,
        res_norm
    );
}

