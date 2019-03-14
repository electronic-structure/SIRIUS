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

/** \file create_beta_gk.cu
 *
 *  \brief CUDA kernel for the generation of beta(G+k) projectors.
 */

#include "../SDDK/GPU/cuda_common.hpp"
#include "../SDDK/GPU/acc_runtime.hpp"

#ifdef __CUDA
#include "../SDDK/GPU/cuda_timer.hpp"
#endif

__global__ void create_beta_gk_gpu_kernel
(
    int num_gkvec__, 
    int const* beta_desc__,
    acc_complex_double_t const* beta_gk_t, 
    double const* gkvec, 
    double const* atom_pos,
    acc_complex_double_t* beta_gk
)
{
    int ia = blockIdx.y;
    int igk = blockDim.x * blockIdx.x + threadIdx.x;

    int nbf              = beta_desc__[array2D_offset(0, ia, 4)];
    int offset_beta_gk   = beta_desc__[array2D_offset(1, ia, 4)];
    int offset_beta_gk_t = beta_desc__[array2D_offset(2, ia, 4)];

    if (igk < num_gkvec__) {
        double p = 0;
        for (int x = 0; x < 3; x++) {
            p += atom_pos[array2D_offset(x, ia, 3)] * gkvec[array2D_offset(x, igk, 3)];
        }
        p *= twopi;

        double sinp = sin(p);
        double cosp = cos(p);

        for (int xi = 0; xi < nbf; xi++) {
            beta_gk[array2D_offset(igk, offset_beta_gk + xi, num_gkvec__)] =
                accCmul(beta_gk_t[array2D_offset(igk, offset_beta_gk_t + xi, num_gkvec__)],
                       make_accDoubleComplex(cosp, -sinp));
        }
    }
}

extern "C" void create_beta_gk_gpu(int num_atoms,
                                   int num_gkvec,
                                   int const* beta_desc,
                                   acc_complex_double_t const* beta_gk_t,
                                   double const* gkvec,
                                   double const* atom_pos,
                                   acc_complex_double_t* beta_gk)
{
#ifdef __CUDA
    CUDA_timer t("create_beta_gk_gpu");
#endif

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gkvec, grid_t.x), num_atoms);

    accLaunchKernel((create_beta_gk_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
        num_gkvec,
        beta_desc,
        beta_gk_t,
        gkvec,
        atom_pos,
        beta_gk
    );
}

