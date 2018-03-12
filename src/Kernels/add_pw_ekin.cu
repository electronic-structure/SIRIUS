/** \file add_pw_ekin.cu
 *   
 *  \brief CUDA kernel for the hphi update.
 */

#include "../SDDK/GPU/cuda_common.hpp"

__global__ void add_pw_ekin_gpu_kernel(int num_gvec__,
                                       double alpha__,
                                       double const* pw_ekin__,
                                       cuDoubleComplex const* phi__,
                                       cuDoubleComplex const* vphi__,
                                       cuDoubleComplex* hphi__)
{
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < num_gvec__) {
        cuDoubleComplex z1 = cuCadd(vphi__[ig], make_cuDoubleComplex(alpha__ * pw_ekin__[ig] * phi__[ig].x, 
                                                                     alpha__ * pw_ekin__[ig] * phi__[ig].y));
        hphi__[ig] = cuCadd(hphi__[ig], z1);
    }
}

/// Update the hphi wave functions.
/** The following operation is performed:
 *    hphi[ig] += (alpha *  pw_ekin[ig] * phi[ig] + vphi[ig])
 */
extern "C" void add_pw_ekin_gpu(int num_gvec__,
                                double alpha__,
                                double const* pw_ekin__,
                                cuDoubleComplex const* phi__,
                                cuDoubleComplex const* vphi__,
                                cuDoubleComplex* hphi__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec__, grid_t.x));

    add_pw_ekin_gpu_kernel <<<grid_b, grid_t>>>
    (
        num_gvec__,
        alpha__,
        pw_ekin__,
        phi__,
        vphi__,
        hphi__
    );

}
