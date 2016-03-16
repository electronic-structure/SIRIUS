#include "kernels_common.h"

__global__ void sum_q_pw_dm_pw_gpu_kernel
(
    int num_gvec_loc__,
    int nbf__,
    cuDoubleComplex const* q_pw_t__,
    double const* dm_pw__,
    cuDoubleComplex* rho_pw__
)
{
    int ld = nbf__ * (nbf__ + 1) / 2;

    int igloc = blockIdx.x * blockDim.x + threadIdx.x;
    if (igloc < num_gvec_loc__)
    {
        double ar = 0;
        double ai = 0;

        for (int i = 0; i < ld; i++)
        {
            double q = 2.0 * q_pw_t__[array2D_offset(i, igloc, ld)].x;

            /* D_{xi2,xi1} * Q(G)_{xi1, xi2} + D_{xi1,xi2} * Q(G)_{xix, xi1}^{+} */
            ar += dm_pw__[array2D_offset(i, 2 * igloc,     ld)] * q;
            ai += dm_pw__[array2D_offset(i, 2 * igloc + 1, ld)] * q;
        }

        /* remove one diagonal contribution which was double-counted */
        for (int xi = 0; xi < nbf__; xi++)
        {
            int i = xi * (xi + 1) / 2 + xi;
            double q = q_pw_t__[array2D_offset(i, igloc, ld)].x;

            /* D_{xi2,xi1} * Q(G)_{xi1, xi2} + D_{xi1,xi2} * Q(G)_{xix, xi1}^{+} */
            ar -= dm_pw__[array2D_offset(i, 2 * igloc,     ld)] * q;
            ai -= dm_pw__[array2D_offset(i, 2 * igloc + 1, ld)] * q;
        }

        rho_pw__[igloc] = cuCadd(rho_pw__[igloc], make_cuDoubleComplex(ar, ai));
    }
}

extern "C" void sum_q_pw_dm_pw_gpu(int num_gvec_loc__,
                                   int nbf__,
                                   cuDoubleComplex const* q_pw_t__,
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
        q_pw_t__, 
        dm_pw__, 
        rho_pw__
    );
}


