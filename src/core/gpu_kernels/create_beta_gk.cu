/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file create_beta_gk.cu
 *
 *  \brief CUDA kernel for the generation of beta(G+k) projectors.
 */

#include "core/acc/acc_common.hpp"
#include "core/acc/acc_runtime.hpp"

#ifdef SIRIUS_CUDA
#include "core/acc/cuda_timer.hpp"
#endif

using namespace sirius;
using namespace sirius::acc;

template <typename T>
__global__ void create_beta_gk_gpu_kernel
(
    int num_gkvec__,
    int const* beta_desc__,
    gpu_complex_type<T> const* beta_gk_t,
    double const* gkvec,
    double const* atom_pos,
    gpu_complex_type<T>* beta_gk
);

template <>
__global__ void create_beta_gk_gpu_kernel<float>
(
    int num_gkvec__,
    int const* beta_desc__,
    acc_complex_float_t const* beta_gk_t,
    double const* gkvec,
    double const* atom_pos,
    acc_complex_float_t* beta_gk
)
{
    int ia = blockIdx.y;
    int igk = blockDim.x * blockIdx.x + threadIdx.x;

    int nbf              = beta_desc__[array2D_offset(0, ia, 4)];
    int offset_beta_gk   = beta_desc__[array2D_offset(1, ia, 4)];
    int offset_beta_gk_t = beta_desc__[array2D_offset(2, ia, 4)];

    if (igk < num_gkvec__) {
        float p = 0;
        for (int x = 0; x < 3; x++) {
            p += atom_pos[array2D_offset(x, ia, 3)] * gkvec[array2D_offset(x, igk, 3)];
        }
        p *= twopi;

        float sinp = sin(p);
        float cosp = cos(p);

        for (int xi = 0; xi < nbf; xi++) {
            beta_gk[array2D_offset(igk, offset_beta_gk + xi, num_gkvec__)] =
                accCmulf(beta_gk_t[array2D_offset(igk, offset_beta_gk_t + xi, num_gkvec__)],
                        make_accFloatComplex(cosp, -sinp));
        }
    }
}

template <>
__global__ void create_beta_gk_gpu_kernel<double>
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

extern "C" void create_beta_gk_gpu_float(int num_atoms,
                                         int num_gkvec,
                                         int const* beta_desc,
                                         acc_complex_float_t const* beta_gk_t,
                                         double const* gkvec,
                                         double const* atom_pos,
                                         acc_complex_float_t* beta_gk)
{
#ifdef SIRIUS_CUDA
    CUDA_timer t("create_beta_gk_gpu");
#endif

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gkvec, grid_t.x), num_atoms);

    accLaunchKernel((create_beta_gk_gpu_kernel<float>), dim3(grid_b), dim3(grid_t), 0, 0,
                    num_gkvec,
                    beta_desc,
                    beta_gk_t,
                    gkvec,
                    atom_pos,
                    beta_gk
    );
}

extern "C" void create_beta_gk_gpu_double(int num_atoms,
                                          int num_gkvec,
                                          int const* beta_desc,
                                          acc_complex_double_t const* beta_gk_t,
                                          double const* gkvec,
                                          double const* atom_pos,
                                          acc_complex_double_t* beta_gk)
{
#ifdef SIRIUS_CUDA
    CUDA_timer t("create_beta_gk_gpu");
#endif

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gkvec, grid_t.x), num_atoms);

    accLaunchKernel((create_beta_gk_gpu_kernel<double>), dim3(grid_b), dim3(grid_t), 0, 0,
        num_gkvec,
        beta_desc,
        beta_gk_t,
        gkvec,
        atom_pos,
        beta_gk
    );
}

