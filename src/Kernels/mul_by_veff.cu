#include "../SDDK/GPU/cuda_common.h"
#include "../SDDK/GPU/cuda.hpp"

__global__ void mul_by_veff0_gpu_kernel(int                    size__,
                                        double const*          veff__,
                                        cuDoubleComplex*       buf__)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size__) {
        cuDoubleComplex z = buf__[i];
        double v0 = veff__[array2D_offset(i, 0, size__)];
        buf__[i] = make_cuDoubleComplex(z.x * v0, z.y * v0);
    }
}

__global__ void mul_by_veff1_gpu_kernel(int                    size__,
                                        double const*          veff__,
                                        cuDoubleComplex*       buf__)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size__) {
        cuDoubleComplex z = buf__[i];
        double v1 = veff__[array2D_offset(i, 1, size__)];
        buf__[i] = make_cuDoubleComplex(z.x * v1, z.y * v1);
    }
}

__global__ void mul_by_veff2_gpu_kernel(int                    size__,
                                        double const*          veff__,
                                        cuDoubleComplex*       buf__)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size__) {
        cuDoubleComplex z = buf__[i];
        cuDoubleComplex v = make_cuDoubleComplex(veff__[array2D_offset(i, 2, size__)],
                                                -veff__[array2D_offset(i, 3, size__)]);  
        buf__[i] = cuCmul(z, v);
    }
}

__global__ void mul_by_veff3_gpu_kernel(int                    size__,
                                        double const*          veff__,
                                        cuDoubleComplex*       buf__)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size__) {
        cuDoubleComplex z = buf__[i];
        cuDoubleComplex v = make_cuDoubleComplex(veff__[array2D_offset(i, 2, size__)],
                                                 veff__[array2D_offset(i, 3, size__)]);  
        buf__[i] = cuCmul(z, v);
    }
}

extern "C" void mul_by_veff_gpu(int ispn__, int size__, double const* veff__, cuDoubleComplex* buf__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(size__, grid_t.x));

    switch (ispn__) {
        case 0: {
            mul_by_veff0_gpu_kernel<<<grid_b, grid_t>>>(size__, veff__, buf__);
            break;
        }
        case 1: {
            mul_by_veff1_gpu_kernel<<<grid_b, grid_t>>>(size__, veff__, buf__);
            break;
        }
        case 2: {
            mul_by_veff2_gpu_kernel<<<grid_b, grid_t>>>(size__, veff__, buf__);
            break;
        }

        case 3: {
            mul_by_veff3_gpu_kernel<<<grid_b, grid_t>>>(size__, veff__, buf__);
            break;
        }
    }
}
