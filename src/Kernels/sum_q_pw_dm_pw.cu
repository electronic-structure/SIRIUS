// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file sum_q_pw_dm_pw.cu
 *
 *  \brief CUDA kernel to perform a summation over xi,xi' indices for the charge density augmentation.
 */

#include "../SDDK/GPU/cuda_common.hpp"
#include "../SDDK/GPU/acc_runtime.hpp"

#ifdef __CUDA
#include "../SDDK/GPU/cuda_timer.hpp"
#endif

__global__ void sum_q_pw_dm_pw_gpu_kernel
(
    int nbf__,
    double const* q_pw__,
    double const* dm_pw__,
    double const* sym_weight__,
    acc_complex_double_t* rho_pw__
)
{
    ACC_DYNAMIC_SHARED( char, sdata_ptr)
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
    if (threadIdx.x == 0) {
        rho_pw__[igloc] = accCadd(rho_pw__[igloc], make_accDoubleComplex(rho_re[0], rho_im[0]));
    }
}

extern "C" void sum_q_pw_dm_pw_gpu(int num_gvec_loc__,
                                   int nbf__,
                                   double const* q_pw__,
                                   double const* dm_pw__,
                                   double const* sym_weight__,
                                   acc_complex_double_t* rho_pw__,
                                   int stream_id__)
{
#ifdef __CUDA
    CUDA_timer t("sum_q_pw_dm_pw_gpu");
#endif

    acc_stream_t stream = (acc_stream_t)acc::stream(stream_id(stream_id__));

    dim3 grid_t(64);
    dim3 grid_b(num_gvec_loc__);
    
    accLaunchKernel((sum_q_pw_dm_pw_gpu_kernel), dim3(grid_b), dim3(grid_t), 2 * grid_t.x * sizeof(double), stream, 
        nbf__, 
        q_pw__, 
        dm_pw__,
        sym_weight__,
        rho_pw__
    );
}


