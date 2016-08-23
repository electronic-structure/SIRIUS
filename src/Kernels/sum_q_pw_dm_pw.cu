#include "kernels_common.h"

__global__ void sum_q_pw_dm_pw_gpu_kernel
(
    int num_gvec_loc__,
    int nbf__,
    double const* q_pw__,
    double const* dm_pw__,
    cuDoubleComplex* rho_pw__
)
{
    int ld = nbf__ * (nbf__ + 1) / 2;

    int igloc = blockIdx.x * blockDim.x + threadIdx.x;
    if (igloc < num_gvec_loc__) {
        cuDoubleComplex zsum = make_cuDoubleComplex(0, 0);

        for (int i = 0; i < ld; i++) {
            cuDoubleComplex z1 = make_cuDoubleComplex(2.0 * q_pw__[array2D_offset(i, 2 * igloc, ld)],
                                                      2.0 * q_pw__[array2D_offset(i, 2 * igloc + 1, ld)]);
            cuDoubleComplex z2 = make_cuDoubleComplex(dm_pw__[array2D_offset(i, 2 * igloc, ld)],
                                                      dm_pw__[array2D_offset(i, 2 * igloc + 1, ld)]);
            zsum = cuCadd(zsum, cuCmul(z1, z2));
        }

        /* remove one diagonal contribution which was double-counted */
        for (int xi = 0; xi < nbf__; xi++) {
            int i = xi * (xi + 1) / 2 + xi;

            cuDoubleComplex z1 = make_cuDoubleComplex(q_pw__[array2D_offset(i, 2 * igloc, ld)],
                                                      q_pw__[array2D_offset(i, 2 * igloc + 1, ld)]);
            cuDoubleComplex z2 = make_cuDoubleComplex(dm_pw__[array2D_offset(i, 2 * igloc, ld)],
                                                      dm_pw__[array2D_offset(i, 2 * igloc + 1, ld)]);
            zsum = cuCsub(zsum, cuCmul(z1, z2));
        }

        rho_pw__[igloc] = cuCadd(rho_pw__[igloc], zsum);
    }
}

extern "C" void sum_q_pw_dm_pw_gpu(int num_gvec_loc__,
                                   int nbf__,
                                   double const* q_pw__,
                                   double const* dm_pw__,
                                   cuDoubleComplex* rho_pw__)
{
    CUDA_timer t("sum_q_pw_dm_pw_gpu");

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec_loc__, grid_t.x));
    
    sum_q_pw_dm_pw_gpu_kernel <<<grid_b, grid_t>>>
    (
        num_gvec_loc__, 
        nbf__, 
        q_pw__, 
        dm_pw__, 
        rho_pw__
    );
}


