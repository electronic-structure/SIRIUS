/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file mul_veff_with_phase_factors.cu
 *
 *  \brief CUDA kernel to multiply effective potential by the phase factors.
 */

#include "core/acc/acc_common.hpp"
#include "core/acc/acc_runtime.hpp"

using namespace sirius;
using namespace sirius::acc;

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
