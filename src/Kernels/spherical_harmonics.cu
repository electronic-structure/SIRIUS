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
            ylm[lmidx(l, l)].x = -sqrt(1 + 1.0 / 2 / l) * sint * ylm[lmidx(l - 1, l - 1)].x;
            ylm[lmidx(l, l)].y = 0;
        }
        for (int l = 0; l < lmax__; l++) {
            ylm[lmidx(l + 1, l)].x = sqrt(2.0 * l + 3) * cost * ylm[lmidx(l, l)].x;
            ylm[lmidx(l + 1, l)].y = 0;
        }
        for (int m = 0; m <= lmax__ - 2; m++) {
            for (int l = m + 2; l <= lmax__; l++) {
                double alm = std::sqrt(static_cast<double>((2 * l - 1) * (2 * l + 1)) / (l * l - m * m));
                double blm = std::sqrt(static_cast<double>((l - 1 - m) * (l - 1 + m)) / ((2 * l - 3) * (2 * l - 1)));
                ylm[lmidx(l, m)].x = alm * (cost * ylm[lmidx(l - 1, m)].x - blm * ylm[lmidx(l - 2, m)].x);
                ylm[lmidx(l, m)].y = 0;
            }
        }

        //for (int m = 0; m <= lmax__; m++) {
        //    acc_double_complex z = std::exp(double_complex(0.0, m * phi)) * std::pow(-1, m);
        //    for (int l = m; l <= lmax; l++) {
        //        ylm[utils::lm(l, m)] = result_array[gsl_sf_legendre_array_index(l, m)] *  z;
        //        if (m && m % 2) {
        //            ylm[utils::lm(l, -m)] = -std::conj(ylm[utils::lm(l, m)]);
        //        } else {
        //            ylm[utils::lm(l, -m)] = std::conj(ylm[utils::lm(l, m)]);
        //        }
        //    }
        //}


    }
}

extern "C" void spherical_harmonics_ylm_gpu(int lmax__, int ntp__, double const* tp__, acc_complex_double_t* ylm__, int ld__)
{
    dim3 grid_t(32);
    dim3 grid_b(num_blocks(ntp__, grid_t.x));
    accLaunchKernel((spherical_harmonics_ylm_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0,
        lmax__, ntp__, tp__, ylm__, ld__);
}


__global__ void spherical_harmonics_rlm_gpu_kernel(int lmax__, int ntp__, double const* tp__,
                                                   double* rlm__, int ld__)
{
    int itp = blockDim.x * blockIdx.x + threadIdx.x;

    if (itp < ntp__) {
        double theta = tp__[itp];
        double phi   = tp__[ntp__ + itp];
        double sint = sin(theta);
        double cost = cos(theta);

        double* rlm = &rlm__[array2D_offset(0, itp, ld__)];

        rlm[0] = 1.0 / sqrt(2 * twopi);

        for (int l = 1; l <= lmax__; l++) {
            rlm[lmidx(l, l)] = -sqrt(1 + 1.0 / 2 / l) * sint * rlm[lmidx(l - 1, l - 1)];
        }
        for (int l = 0; l < lmax__; l++) {
            rlm[lmidx(l + 1, l)] = sqrt(2.0 * l + 3) * cost * rlm[lmidx(l, l)];
        }
        for (int m = 0; m <= lmax__ - 2; m++) {
            for (int l = m + 2; l <= lmax__; l++) {
                double alm = std::sqrt(static_cast<double>((2 * l - 1) * (2 * l + 1)) / (l * l - m * m));
                double blm = std::sqrt(static_cast<double>((l - 1 - m) * (l - 1 + m)) / ((2 * l - 3) * (2 * l - 1)));
                rlm[lmidx(l, m)] = alm * (cost * rlm[lmidx(l - 1, m)] - blm * rlm[lmidx(l - 2, m)]);
            }
        }

        double c0 = std::cos(phi);
        double c1 = 1;
        double s0 = -std::sin(phi);
        double s1 = 0;
        double c2 = 2 * c0;

        double const t = std::sqrt(2.0);

        for (int m = 1; m <= lmax__; m++) {
            double c = c2 * c1 - c0;
            c0 = c1;
            c1 = c;
            double s = c2 * s1 - s0;
            s0 = s1;
            s1 = s;
            for (int l = m; l <= lmax__; l++) {
                double p = rlm[lmidx(l, m)];
                rlm[lmidx(l, m)] = t * p * c;
                if (m % 2) {
                    rlm[lmidx(l, -m)] = t * p * s;
                } else {
                    rlm[lmidx(l, -m)] = -t * p * s;
                }
            }
        }
    }
}

extern "C" void spherical_harmonics_rlm_gpu(int lmax__, int ntp__, double const* tp__, double* rlm__, int ld__)
{
    dim3 grid_t(32);
    dim3 grid_b(num_blocks(ntp__, grid_t.x));
    accLaunchKernel((spherical_harmonics_rlm_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0,
        lmax__, ntp__, tp__, rlm__, ld__);
}
