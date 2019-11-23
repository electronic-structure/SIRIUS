// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
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

/** \file spherical_harmonics.cu
 *
 *  \brief CUDA kernels to generate spherical harminics.
 */

#include "../SDDK/GPU/cuda_common.hpp"
#include "../SDDK/GPU/acc_runtime.hpp"

inline __device__ int lmidx(int l, int m)
{
    return l * l + l + m;
}

__global__ void spherical_harmonics_ylm_gpu_kernel(int lmax__, int ntp__, double const* tp__,
                                                   acc_complex_double_t* ylm__, int ld__)
{
    int itp = blockDim.x * blockIdx.x + threadIdx.x;

    if (itp < ntp__) {
        double theta = tp__[2 * itp];
        double phi   = tp__[2 * itp + 1];
        double sint = sin(theta);
        double cost = cos(theta);

        acc_complex_double_t* ylm = &ylm__[array2D_offset(0, itp, ld__)];

        ylm[0].x = 1.0 / sqrt(2 * twopi);
        ylm[0].y = 0;

        for (int l = 1; l <= lmax__; l++) {
            ylm[lmidx(l, l)] = -sqrt(1 + 1.0 / 2 / l) * sint * ylm[lmidx(l - 1, l - 1)];
        }
        for (int l = 0; l < lmax__; l++) {
            ylm[lmidx(l + 1, l)] = sqrt(2.0 * l + 3) * cost * ylm[lmidx(l, l)];
        }
        for (int m = 0; m <= lmax__ - 2; m++) {
            for (int l = m + 2; l <= lmax__; l++) {
                double alm = std::sqrt(static_cast<double>((2 * l - 1) * (2 * l + 1)) / (l * l - m * m));
                double blm = std::sqrt(static_cast<double>((l - 1 - m) * (l - 1 + m)) / ((2 * l - 3) * (2 * l - 1)));
                ylm[lmidx(l, m)] = alm * (cost * ylm[lmidx(l - 1, m)] - blm * ylm[lmidx(l - 2, m)]);
            }
        }
    }
}

extern "C" spherical_harmonics_ylm_gpu(int lmax__, int ntp__, double const* tp__, acc_complex_double_t* ylm__, int ld__)
{
    dim3 grid_t(32);
    dim3 grid_b(num_blocks(ntp, grid_t.x));
    accLaunchKernel((spherical_harmonics_ylm_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0,
        lmax__, ntp__, tp__, ylm__, ld__);
}
