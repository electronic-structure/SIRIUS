#include "kernels_common.h"

__global__ void update_it_density_matrix_1_gpu_kernel(int fft_size,
                                                      int ispn,
                                                      cuDoubleComplex const* psi_it,
                                                      double wt,
                                                      double* it_density_matrix)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < fft_size)
    {
        cuDoubleComplex z = psi_it[ir];
        it_density_matrix[array2D_offset(ir, ispn, fft_size)] += (z.x * z.x + z.y * z.y) * wt;
    }
}

extern "C" void update_it_density_matrix_1_gpu(int fft_size, 
                                               int ispin,
                                               cuDoubleComplex const* psi_it, 
                                               double wt, 
                                               double* it_density_matrix)
{
    CUDA_timer t("update_it_density_matrix_gpu");

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(fft_size, grid_t.x));

    update_it_density_matrix_1_gpu_kernel <<<grid_b, grid_t>>>
    (
        fft_size,
        ispin,
        psi_it,
        wt,
        it_density_matrix
    );
}

