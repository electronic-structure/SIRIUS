#include "kernels_common.h"

__global__ void sum_q_pw_dm_pw_gpu_kernel
(
    int num_gvec_loc__,
    int nbf__,
    cuDoubleComplex const* q_pw_t__,
    cuDoubleComplex const* dm_pw__,
    cuDoubleComplex* rho_pw__
)
{
    int ld1 = nbf__ * (nbf__ + 1) / 2;
    int ld2 = nbf__ * nbf__;

    int igloc = blockIdx.x * blockDim.x + threadIdx.x;
    if (igloc < num_gvec_loc__)
    {
        cuDoubleComplex zval = make_cuDoubleComplex(0.0, 0.0);

        // \sum_{xi1, xi2} D_{xi2,xi1} * Q(G)_{xi1, xi2}
        for (int xi2 = 0; xi2 < nbf__; xi2++)
        {
            int idx12 = xi2 * (xi2 + 1) / 2;

            // add diagonal term
            zval = cuCadd(zval, cuCmul(dm_pw__[array2D_offset(xi2 * nbf__ + xi2, igloc, ld2)], 
                                       q_pw_t__[array2D_offset(idx12 + xi2, igloc, ld1)]));

            // add non-diagonal terms
            for (int xi1 = 0; xi1 < xi2; xi1++, idx12++)
            {
                cuDoubleComplex q = q_pw_t__[array2D_offset(idx12, igloc, ld1)];
                cuDoubleComplex d1 = dm_pw__[array2D_offset(xi2 * nbf__ + xi1, igloc, ld2)];
                cuDoubleComplex d2 = dm_pw__[array2D_offset(xi1 * nbf__ + xi2, igloc, ld2)];

                zval = cuCadd(zval, cuCmul(q, d1));
                zval = cuCadd(zval, cuCmul(cuConj(q), d2));
            }
        }
        rho_pw__[igloc] = cuCadd(rho_pw__[igloc], zval);
    }
}

extern "C" void sum_q_pw_dm_pw_gpu(int num_gvec_loc__,
                                   int nbf__,
                                   cuDoubleComplex const* q_pw_t__,
                                   cuDoubleComplex const* dm_pw__,
                                   cuDoubleComplex* rho_pw__)
{
    CUDA_timer t("sum_q_pw_dm_pw_gpu");

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec_loc__, grid_t.x));
    
    sum_q_pw_dm_pw_gpu_kernel <<<grid_b, grid_t>>>
    (
        num_gvec_loc__, 
        nbf__, 
        q_pw_t__, 
        dm_pw__, 
        rho_pw__
    );
}


