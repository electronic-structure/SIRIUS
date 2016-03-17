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
    double* res_norm,
    int reduced,
    int mpi_rank
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
    
    if (!reduced)
    {
        res_norm[res_idx[blockIdx.x]] = sdata[0];
    }
    else
    {
        if (mpi_rank == 0)
        {
            double x = res[array2D_offset(0, blockIdx.x, num_gkvec_row)].x;
            res_norm[res_idx[blockIdx.x]] = 2 * sdata[0] - x * x;
        }
        else
        {
            res_norm[res_idx[blockIdx.x]] = 2 * sdata[0];
        }
    }
}

__global__ void apply_preconditioner_gpu_kernel
(
    int const num_gkvec_row,
    int const* res_idx,
    double const* eval,
    double const* h_diag,
    double const* o_diag,
    cuDoubleComplex* res
)
{
    int igk = blockIdx.x * blockDim.x + threadIdx.x;
    int ibnd = res_idx[blockIdx.y];

    if (igk < num_gkvec_row)
    {
        double p = (h_diag[igk] - eval[ibnd] * o_diag[igk]); // QE formula is in Ry; here we convert to Ha
        p = 0.5 * (1 + p + sqrt(1.0 + (p - 1) * (p - 1)));
        int k = array2D_offset(igk, blockIdx.y, num_gkvec_row);
        res[k] = cuCdiv(res[k], make_cuDoubleComplex(p, 0.0));
    }
}


extern "C" void residuals_aux_gpu(int num_gvec_loc__,
                                  int num_res_local__,
                                  int* res_idx__,
                                  double* eval__,
                                  cuDoubleComplex const* hpsi__,
                                  cuDoubleComplex const* opsi__,
                                  double const* h_diag__,
                                  double const* o_diag__,
                                  cuDoubleComplex* res__,
                                  double* res_norm__,
                                  double* p_norm__,
                                  int gkvec_reduced__,
                                  int mpi_rank__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec_loc__, grid_t.x), num_res_local__);

    compute_residuals_gpu_kernel <<<grid_b, grid_t>>>
    (
        num_gvec_loc__,
        res_idx__,
        eval__,
        hpsi__,
        opsi__,
        res__
    );

    grid_b = dim3(num_res_local__);

    compute_residuals_norm_gpu_kernel <<<grid_b, grid_t, grid_t.x * sizeof(double)>>>
    (
        num_gvec_loc__,
        res_idx__,
        res__,
        res_norm__,
        gkvec_reduced__,
        mpi_rank__
    );

    grid_b = dim3(num_blocks(num_gvec_loc__, grid_t.x), num_res_local__);

    apply_preconditioner_gpu_kernel <<<grid_b, grid_t>>>
    (
        num_gvec_loc__,
        res_idx__,
        eval__,
        h_diag__,
        o_diag__,
        res__
    );

    grid_b = dim3(num_res_local__);

    compute_residuals_norm_gpu_kernel <<<grid_b, grid_t, grid_t.x * sizeof(double)>>>
    (
        num_gvec_loc__,
        res_idx__,
        res__,
        p_norm__,
        gkvec_reduced__,
        mpi_rank__
    );
}

