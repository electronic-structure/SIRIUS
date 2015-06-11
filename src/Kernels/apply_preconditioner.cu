#include "kernels_common.h"

__global__ void compute_residuals_norm_gpu_kernel
(
    int num_gkvec_row,
    int* res_idx,
    cuDoubleComplex const* res,
    double* res_norm
);

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
        double p = 2 * (h_diag[igk] - eval[ibnd] * o_diag[igk]); // QE formula is in Ry; here we convert to Ha
        p = 0.25 * (1 + p + sqrt(1.0 + (p - 1) * (p - 1)));
        int k = array2D_offset(igk, blockIdx.y, num_gkvec_row);
        res[k] = cuCdiv(res[k], make_cuDoubleComplex(p, 0.0));
    }
}

extern "C" void apply_preconditioner_gpu(int num_gkvec_row,
                                         int num_res_local,
                                         int* res_idx,
                                         double* eval,
                                         double const* h_diag,
                                         double const* o_diag,
                                         cuDoubleComplex* res,
                                         double* res_norm)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gkvec_row, grid_t.x), num_res_local);

    apply_preconditioner_gpu_kernel <<<grid_b, grid_t>>>
    (
        num_gkvec_row,
        res_idx,
        eval,
        h_diag,
        o_diag,
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

