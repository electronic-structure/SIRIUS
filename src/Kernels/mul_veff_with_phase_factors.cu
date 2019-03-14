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

/** \file mul_veff_with_phase_factors.cu
 *
 *  \brief CUDA kernel to multiply effective potential by the phase factors.
 */

#include "../SDDK/GPU/cuda_common.hpp"
#include "../SDDK/GPU/acc_runtime.hpp"

__global__ void mul_veff_with_phase_factors_gpu_kernel(int num_gvec_loc__,
                                                       acc_complex_double_t const* veff__, 
                                                       int const* gvx__, 
                                                       int const* gvy__, 
                                                       int const* gvz__, 
                                                       int num_atoms__,
                                                       double const* atom_pos__, 
                                                       acc_complex_double_t* veff_a__,
                                                       int ld__)
{
    int ia = blockIdx.y;
    double ax = atom_pos__[array2D_offset(ia, 0, num_atoms__)];
    double ay = atom_pos__[array2D_offset(ia, 1, num_atoms__)];
    double az = atom_pos__[array2D_offset(ia, 2, num_atoms__)];

    int igloc = blockDim.x * blockIdx.x + threadIdx.x;
    if (igloc < num_gvec_loc__) {
        int gvx = gvx__[igloc];
        int gvy = gvy__[igloc];
        int gvz = gvz__[igloc];

        double p = twopi * (ax * gvx + ay * gvy + az * gvz);

        veff_a__[array2D_offset(igloc, ia, ld__)] = accCmul(veff__[igloc], make_accDoubleComplex(cos(p), sin(p)));
    }
}
 
extern "C" void mul_veff_with_phase_factors_gpu(int num_atoms__,
                                                int num_gvec_loc__, 
                                                acc_complex_double_t const* veff__, 
                                                int const* gvx__, 
                                                int const* gvy__, 
                                                int const* gvz__, 
                                                double const* atom_pos__,
                                                double* veff_a__,
                                                int ld__,
                                                int stream_id__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec_loc__, grid_t.x), num_atoms__);

    acc_stream_t stream = (acc_stream_t)acc::stream(stream_id(stream_id__));

    accLaunchKernel((mul_veff_with_phase_factors_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, stream, 
        num_gvec_loc__,
        veff__,
        gvx__,
        gvy__,
        gvz__,
        num_atoms__,
        atom_pos__,
        (acc_complex_double_t*)veff_a__,
        ld__
    );
}
