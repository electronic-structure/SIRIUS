/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file generate_dm_pw.cu
 *
 *  \brief CUDA kernel to generate a product of phase-factors and density matrix.
 */

#include "core/acc/acc_common.hpp"
#include "core/acc/acc_runtime.hpp"
#include "core/acc/acc_blas.hpp"

using namespace sirius;
using namespace sirius::acc;

__global__ void generate_phase_factors_conj_gpu_kernel
(
    int num_gvec_loc__, 
    int num_atoms__, 
    double const* atom_pos__, 
    int const* gvx__, 
    int const* gvy__, 
    int const* gvz__, 
    acc_complex_double_t* phase_factors__
)
{
    int ia = blockIdx.y;
    double ax = atom_pos__[array2D_offset(ia, 0, num_atoms__)];
    double ay = atom_pos__[array2D_offset(ia, 1, num_atoms__)];
    double az = atom_pos__[array2D_offset(ia, 2, num_atoms__)];

    int igloc = blockIdx.x * blockDim.x + threadIdx.x;

    if (igloc < num_gvec_loc__) {
        int gvx = gvx__[igloc];
        int gvy = gvy__[igloc];
        int gvz = gvz__[igloc];

        double p = twopi * (ax * gvx + ay * gvy + az * gvz);
        phase_factors__[array2D_offset(igloc, ia, num_gvec_loc__)] = make_accDoubleComplex(cos(p), -sin(p));
    }
}

extern "C" void generate_dm_pw_gpu(int num_atoms__,
                                   int num_gvec_loc__,
                                   int nbf__,
                                   double const* atom_pos__,
                                   int const* gvx__,
                                   int const* gvy__,
                                   int const* gvz__,
                                   double* phase_factors__, 
                                   double const* dm__,
                                   double* dm_pw__,
                                   int stream_id__)
{
    //CUDA_timer t("generate_dm_pw_gpu");

    acc_stream_t stream = (acc_stream_t)acc::stream(stream_id(stream_id__));

    dim3 grid_t(32);
    dim3 grid_b(num_blocks(num_gvec_loc__, grid_t.x), num_atoms__);

    accLaunchKernel((generate_phase_factors_conj_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, stream, 
        num_gvec_loc__, 
        num_atoms__, 
        atom_pos__, 
        gvx__, 
        gvy__, 
        gvz__, 
        (acc_complex_double_t*)phase_factors__
    );

    double alpha = 1;
    double beta = 0;

    blas::dgemm('N', 'T', nbf__ * (nbf__ + 1) / 2, num_gvec_loc__ * 2, num_atoms__,
                &alpha,
                dm__, nbf__ * (nbf__ + 1) / 2,
                phase_factors__, num_gvec_loc__ * 2,
                &beta,
                dm_pw__, nbf__ * (nbf__ + 1) / 2,
                stream_id__);
   acc::sync_stream(stream_id(stream_id__));
}

