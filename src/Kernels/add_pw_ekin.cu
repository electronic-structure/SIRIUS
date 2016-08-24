#include "kernels_common.h"

__global__ void add_pw_ekin_gpu_kernel(int num_gvec__,
                                       double const* pw_ekin__,
                                       cuDoubleComplex const* vphi__,
                                       cuDoubleComplex* hphi__)
{
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < num_gvec__) {
        hphi__[ig] = cuCadd(vphi__[ig], make_cuDoubleComplex(hphi__[ig].x * pw_ekin__[ig], hphi__[ig].y * pw_ekin__[ig]));
    }
}

extern "C" void add_pw_ekin_gpu(int num_gvec__,
                                double const* pw_ekin__,
                                cuDoubleComplex const* vphi__,
                                cuDoubleComplex* hphi__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec__, grid_t.x));

    add_pw_ekin_gpu_kernel <<<grid_b, grid_t>>>
    (
        num_gvec__,
        pw_ekin__,
        vphi__,
        hphi__
    );

}
