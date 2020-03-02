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

/** \file add_pw_ekin.cu
 *
 *  \brief CUDA kernel for the hphi update.
 */

#include "../SDDK/GPU/cuda_common.hpp"
#include "../SDDK/GPU/acc_runtime.hpp"

__global__ void add_pw_ekin_gpu_kernel(int num_gvec__,
                                       double alpha__,
                                       double const* pw_ekin__,
                                       acc_complex_double_t const* phi__,
                                       acc_complex_double_t const* vphi__,
                                       acc_complex_double_t* hphi__)
{
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < num_gvec__) {
        acc_complex_double_t z1 = accCadd(vphi__[ig], make_accDoubleComplex(alpha__ * pw_ekin__[ig] * phi__[ig].x, 
                                                                     alpha__ * pw_ekin__[ig] * phi__[ig].y));
        hphi__[ig] = accCadd(hphi__[ig], z1);
    }
}

/// Update the hphi wave functions.
/** The following operation is performed:
 *    hphi[ig] += (alpha *  pw_ekin[ig] * phi[ig] + vphi[ig])
 */
extern "C" void add_pw_ekin_gpu(int num_gvec__,
                                double alpha__,
                                double const* pw_ekin__,
                                acc_complex_double_t const* phi__,
                                acc_complex_double_t const* vphi__,
                                acc_complex_double_t* hphi__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec__, grid_t.x));

    accLaunchKernel((add_pw_ekin_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
        num_gvec__,
        alpha__,
        pw_ekin__,
        phi__,
        vphi__,
        hphi__
    );

}
