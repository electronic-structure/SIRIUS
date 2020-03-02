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

/** \file density_rg.cu
 *
 *  \brief CUDA kernel to update density on the regular FFT grid.
 */

#include "../SDDK/GPU/cuda_common.hpp"
#include "../SDDK/GPU/acc_runtime.hpp"

__global__ void update_density_rg_1_complex_gpu_kernel(int size__,
                                                       acc_complex_double_t const* psi_rg__,
                                                       double wt__,
                                                       double* density_rg__)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < size__) {
        acc_complex_double_t z = psi_rg__[ir];
        density_rg__[ir] += (z.x * z.x + z.y * z.y) * wt__;
    }
}

/* Update one density component from one complex wave-function */
extern "C" void update_density_rg_1_complex_gpu(int size__, 
                                                acc_complex_double_t const* psi_rg__, 
                                                double wt__, 
                                                double* density_rg__)
{
    //CUDA_timer t("update_density_rg_1_gpu");

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(size__, grid_t.x));

    accLaunchKernel((update_density_rg_1_complex_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
        size__,
        psi_rg__,
        wt__,
        density_rg__
    );
}

__global__ void update_density_rg_1_real_gpu_kernel(int size__,
                                                    double const* psi_rg__,
                                                    double wt__,
                                                    double* density_rg__)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < size__) {
        double p = psi_rg__[ir];
        density_rg__[ir] += p * p * wt__;
    }
}

/* Update one density component from one real wave-function */
extern "C" void update_density_rg_1_real_gpu(int size__,
                                             double const* psi_rg__,
                                             double wt__, 
                                             double* density_rg__)
{
    //CUDA_timer t("update_density_rg_1_gpu");

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(size__, grid_t.x));

    accLaunchKernel((update_density_rg_1_real_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
        size__,
        psi_rg__,
        wt__,
        density_rg__
    );
}

__global__ void update_density_rg_2_gpu_kernel(int size__,
                                               acc_complex_double_t const* psi_up_rg__,
                                               acc_complex_double_t const* psi_dn_rg__,
                                               double wt__,
                                               double* density_x_rg__,
                                               double* density_y_rg__)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < size__) {
        acc_complex_double_t z = accCmul(psi_up_rg__[ir], accConj(psi_dn_rg__[ir]));
        density_x_rg__[ir] += 2 * z.x * wt__;
        density_y_rg__[ir] -= 2 * z.y * wt__;
    }
}

/* Update off-diagonal density component in non-collinear case */
extern "C" void update_density_rg_2_gpu(int size__,
                                        acc_complex_double_t const* psi_up_rg__,
                                        acc_complex_double_t const* psi_dn_rg__,
                                        double wt__,
                                        double* density_x_rg__,
                                        double* density_y_rg__)
{
    //CUDA_timer t("update_density_rg_1_gpu");

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(size__, grid_t.x));

    accLaunchKernel((update_density_rg_2_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0,
        size__,
        psi_up_rg__,
        psi_dn_rg__,
        wt__,
        density_x_rg__,
        density_y_rg__
    );
}


