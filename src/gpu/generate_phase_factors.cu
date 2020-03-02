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

/** \file generate_phase_factors.cu
 *
 *  \brief CUDA kernel to generate plane-wave atomic phase factors.
 */

#include "../SDDK/GPU/cuda_common.hpp"
#include "../SDDK/GPU/acc_runtime.hpp"

__global__ void generate_phase_factors_gpu_kernel
(
    int num_gvec_loc, 
    int num_atoms,
    double const* atom_pos, 
    int const* gvec, 
    acc_complex_double_t* phase_factors
)
{
    int ia = blockIdx.y;
    int igloc = blockIdx.x * blockDim.x + threadIdx.x;

    if (igloc < num_gvec_loc) {
        int gvx = gvec[array2D_offset(igloc, 0, num_gvec_loc)];
        int gvy = gvec[array2D_offset(igloc, 1, num_gvec_loc)];
        int gvz = gvec[array2D_offset(igloc, 2, num_gvec_loc)];
    
        double ax = atom_pos[array2D_offset(ia, 0, num_atoms)];
        double ay = atom_pos[array2D_offset(ia, 1, num_atoms)];
        double az = atom_pos[array2D_offset(ia, 2, num_atoms)];

        double p = twopi * (ax * gvx + ay * gvy + az * gvz);

        double sinp = sin(p);
        double cosp = cos(p);

        phase_factors[array2D_offset(igloc, ia, num_gvec_loc)] = make_accDoubleComplex(cosp, sinp);
    }
}


extern "C" void generate_phase_factors_gpu(int num_gvec_loc__,
                                           int num_atoms__,
                                           int const* gvec__,
                                           double const* atom_pos__,
                                           acc_complex_double_t* phase_factors__)

{
    dim3 grid_t(32);
    dim3 grid_b(num_blocks(num_gvec_loc__, grid_t.x), num_atoms__);

    accLaunchKernel((generate_phase_factors_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
        num_gvec_loc__, 
        num_atoms__,
        atom_pos__, 
        gvec__, 
        phase_factors__
    );
}
