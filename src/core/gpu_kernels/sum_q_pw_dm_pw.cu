/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file sum_q_pw_dm_pw.cu
 *
 *  \brief CUDA kernel to perform a summation over xi,xi' indices for the charge density augmentation.
 */

#include "core/acc/acc_common.hpp"
#include "core/acc/acc_runtime.hpp"

#ifdef SIRIUS_CUDA
#include "core/acc/cuda_timer.hpp"
#endif

using namespace sirius;
using namespace sirius::acc;

__global__ void
sum_q_pw_dm_pw_gpu_kernel(int nbf__, double const* q_pw__, int ldq__, double const* dm_pw__, int ldd__,
    double const* sym_weight__, acc_complex_double_t* rho_pw__)
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
            double qx =  q_pw__[array2D_offset(i, 2 * igloc,     ldq__)];
            double qy =  q_pw__[array2D_offset(i, 2 * igloc + 1, ldq__)];
            double dx = dm_pw__[array2D_offset(i, 2 * igloc,     ldd__)];
            double dy = dm_pw__[array2D_offset(i, 2 * igloc + 1, ldd__)];

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

extern "C" void sum_q_pw_dm_pw_gpu(int num_gvec_loc__, int nbf__, double const* q_pw__, int ldq__,
                                   double const* dm_pw__, int ldd__, double const* sym_weight__,
                                   acc_complex_double_t* rho_pw__, int stream_id__)
{
#ifdef SIRIUS_CUDA
    CUDA_timer t("sum_q_pw_dm_pw_gpu");
#endif

    acc_stream_t stream = (acc_stream_t)acc::stream(stream_id(stream_id__));

    dim3 grid_t(64);
    dim3 grid_b(num_gvec_loc__);

    accLaunchKernel((sum_q_pw_dm_pw_gpu_kernel), dim3(grid_b), dim3(grid_t), 2 * grid_t.x * sizeof(double), stream, 
        nbf__, q_pw__, ldq__, dm_pw__, ldd__, sym_weight__, rho_pw__);
}


