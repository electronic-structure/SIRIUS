#include "kernels_common.h"

__global__ void sum_q_pw_dm_pw_gpu_kernel
(
    int nbf__,
    double const* q_pw__,
    double const* dm_pw__,
    double const* sym_weight__,
    cuDoubleComplex* rho_pw__
)
{
    extern __shared__ char sdata_ptr[];
    double* rho_re = (double*)&sdata_ptr[0];
    double* rho_im = (double*)&sdata_ptr[sizeof(double) * blockDim.x];

    int igloc = blockIdx.x;

    rho_re[threadIdx.x] = 0;
    rho_im[threadIdx.x] = 0;

    int ld = nbf__ * (nbf__ + 1) / 2;

    int N = num_blocks(ld, blockDim.x);

    for (int n = 0; n < N; n++) {
        int i = n * blockDim.x + threadIdx.x;
        if (i < ld) {
            double qx =  q_pw__[array2D_offset(i, 2 * igloc,     ld)];
            double qy =  q_pw__[array2D_offset(i, 2 * igloc + 1, ld)];
            double dx = dm_pw__[array2D_offset(i, 2 * igloc,     ld)];
            double dy = dm_pw__[array2D_offset(i, 2 * igloc + 1, ld)];

            rho_re[threadIdx.x] += sym_weight__[i] * (dx * qx - dy * qy);
            rho_im[threadIdx.x] += sym_weight__[i] * (dy * qx + dx * qy);
        }
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            rho_re[threadIdx.x] = rho_re[threadIdx.x] + rho_re[threadIdx.x + s];
            rho_im[threadIdx.x] = rho_im[threadIdx.x] + rho_im[threadIdx.x + s];
        }
        __syncthreads();
    }

    //== if (igloc == 0 && threadIdx.x == 0) {
    //==     printf("sum_q_pw_dm_pw_gpu_kernel: %18.12f %18.12f\n", rho_re[0], rho_im[0]);
    //== }

    rho_pw__[igloc] = cuCadd(rho_pw__[igloc], make_cuDoubleComplex(rho_re[0], rho_im[0]));
}

extern "C" void sum_q_pw_dm_pw_gpu(int num_gvec_loc__,
                                   int nbf__,
                                   double const* q_pw__,
                                   double const* dm_pw__,
                                   double const* sym_weight__,
                                   cuDoubleComplex* rho_pw__)
{
    CUDA_timer t("sum_q_pw_dm_pw_gpu");

    dim3 grid_t(64);
    dim3 grid_b(num_gvec_loc__);
    
    sum_q_pw_dm_pw_gpu_kernel <<<grid_b, grid_t, 2 * grid_t.x * sizeof(double)>>>
    (
        nbf__, 
        q_pw__, 
        dm_pw__,
        sym_weight__,
        rho_pw__
    );
}


