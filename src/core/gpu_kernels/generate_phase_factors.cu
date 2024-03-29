/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file generate_phase_factors.cu
 *
 *  \brief CUDA kernel to generate plane-wave atomic phase factors.
 */

#include "core/acc/acc_common.hpp"
#include "core/acc//acc_runtime.hpp"

using namespace sirius;
using namespace sirius::acc;

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
