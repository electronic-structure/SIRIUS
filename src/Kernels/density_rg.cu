#include "../SDDK/GPU/cuda_common.hpp"

__global__ void update_density_rg_1_gpu_kernel(int size__,
                                               cuDoubleComplex const* psi_rg__,
                                               double wt__,
                                               double* density_rg__)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < size__)
    {
        cuDoubleComplex z = psi_rg__[ir];
        density_rg__[ir] += (z.x * z.x + z.y * z.y) * wt__;
    }
}

extern "C" void update_density_rg_1_gpu(int size__, 
                                        cuDoubleComplex const* psi_rg__, 
                                        double wt__, 
                                        double* density_rg__)
{
    //CUDA_timer t("update_density_rg_1_gpu");

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(size__, grid_t.x));

    update_density_rg_1_gpu_kernel <<<grid_b, grid_t>>>
    (
        size__,
        psi_rg__,
        wt__,
        density_rg__
    );
}

__global__ void update_density_rg_2_gpu_kernel(int size__,
                                               cuDoubleComplex const* psi_up_rg__,
                                               cuDoubleComplex const* psi_dn_rg__,
                                               double wt__,
                                               double* density_x_rg__,
                                               double* density_y_rg__)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < size__) {
        cuDoubleComplex z = cuCmul(psi_up_rg__[ir], cuConj(psi_dn_rg__[ir]));
        density_x_rg__[ir] += 2 * z.x * wt__;
        density_y_rg__[ir] -= 2 * z.y * wt__;
    }
}

extern "C" void update_density_rg_2_gpu(int size__, 
                                        cuDoubleComplex const* psi_up_rg__, 
                                        cuDoubleComplex const* psi_dn_rg__, 
                                        double wt__, 
                                        double* density_x_rg__,
                                        double* density_y_rg__)
{
    //CUDA_timer t("update_density_rg_1_gpu");

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(size__, grid_t.x));

    update_density_rg_2_gpu_kernel <<<grid_b, grid_t>>>
    (
        size__,
        psi_up_rg__,
        psi_dn_rg__,
        wt__,
        density_x_rg__,
        density_y_rg__
    );
}


