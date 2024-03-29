/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file augmentation_operator.cu
 *
 *  \brief CUDA kernels to generate augmentation operator and its derivative.
 */

#include "core/acc/acc.hpp"
#include "core/acc/acc_common.hpp"
#include "core/acc/acc_runtime.hpp"

using namespace sirius;
using namespace sirius::acc;

__global__ void aug_op_pw_coeffs_gpu_kernel(int ngvec__, int const* gvec_shell__, int const* idx__, int idxmax__,
                                            acc_complex_double_t const* zilm__, int const* l_by_lm__, int lmmax__,
                                            double const* gc__, int ld0__, int ld1__,
                                            double const* gvec_rlm__, int ld2__,
                                            double const* ri_values__, int ld3__, int ld4__,
                                            double* q_pw__, int ld5__, double fourpi_omega__)

{
    int igloc = blockDim.x * blockIdx.x + threadIdx.x;
    int idx12 = blockIdx.y;
    int idxsh = gvec_shell__[igloc];

    if (igloc < ngvec__) {
        int lm1     = idx__[array2D_offset(0, idx12, 3)];
        int lm2     = idx__[array2D_offset(1, idx12, 3)];
        int idxrf12 = idx__[array2D_offset(2, idx12, 3)];

        acc_complex_double_t z = make_accDoubleComplex(0, 0);
        for (int lm = 0; lm < lmmax__; lm++) {
            double d = gvec_rlm__[array2D_offset(lm, igloc, ld2__)] *
                ri_values__[array3D_offset(idxrf12, l_by_lm__[lm], idxsh, ld3__, ld4__)] *
                gc__[array3D_offset(lm, lm2, lm1, ld0__, ld1__)];
            z.x += d * zilm__[lm].x;
            z.y -= d * zilm__[lm].y;
        }
        q_pw__[array2D_offset(idx12, 2 * igloc,     ld5__)] = z.x * fourpi_omega__;
        q_pw__[array2D_offset(idx12, 2 * igloc + 1, ld5__)] = z.y * fourpi_omega__;
    }
}

extern "C" void aug_op_pw_coeffs_gpu(int ngvec__, int const* gvec_shell__, int const* idx__, int idxmax__,
                                     acc_complex_double_t const* zilm__, int const* l_by_lm__, int lmmax__,
                                     double const* gc__, int ld0__, int ld1__,
                                     double const* gvec_rlm__, int ld2__,
                                     double const* ri_values__, int ld3__, int ld4__,
                                     double* q_pw__, int ld5__, double fourpi_omega__)
{
    dim3 grid_t(32);
    dim3 grid_b(num_blocks(ngvec__, grid_t.x), idxmax__);

    accLaunchKernel((aug_op_pw_coeffs_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0,
        ngvec__, gvec_shell__, idx__, idxmax__, zilm__, l_by_lm__, lmmax__, gc__, ld0__, ld1__, gvec_rlm__, ld2__,
        ri_values__, ld3__, ld4__, q_pw__, ld5__, fourpi_omega__);
}

__global__ void aug_op_pw_coeffs_deriv_gpu_kernel(int ngvec__, int const* gvec_shell__, double const* gvec_cart__,
                                                  int const* idx__, int idxmax__,
                                                  double const* gc__, int ld0__, int ld1__,
                                                  double const* rlm__, double const* rlm_dg__, int ld2__,
                                                  double const* ri_values__, double const* ri_dg_values__, int ld3__,
                                                  int ld4__, double* q_pw__, int ld5__, double fourpi__, int nu__,
                                                  int lmax_q__)

{
    int igloc = blockDim.x * blockIdx.x + threadIdx.x;
    int idx12 = blockIdx.y;
    int idxsh = gvec_shell__[igloc];

    if (igloc < ngvec__) {
        int lm1     = idx__[array2D_offset(0, idx12, 3)];
        int lm2     = idx__[array2D_offset(1, idx12, 3)];
        int idxrf12 = idx__[array2D_offset(2, idx12, 3)];
        double gvc_nu = gvec_cart__[array2D_offset(nu__, igloc, 3)];

        acc_complex_double_t z = make_accDoubleComplex(0, 0);
        acc_complex_double_t phase = make_accDoubleComplex(1, 0);
        int lm = 0;
        for (int l = 0; l <= lmax_q__; l++) {
            double d1 = 0;
            double d2 = 0;
            for (int m = -l; m <= l; m++, lm++) {
                double gc = gc__[array3D_offset(lm, lm2, lm1, ld0__, ld1__)];
                d1 += rlm_dg__[array3D_offset(lm, nu__, igloc, ld2__, 3)] * gc;
                d2 += rlm__[array2D_offset(lm, igloc, ld2__)] * gc;
            }
            double d = d1 * ri_values__[array3D_offset(l, idxrf12, idxsh, ld3__, ld4__)] +
                       d2 * ri_dg_values__[array3D_offset(l, idxrf12, idxsh, ld3__, ld4__)] * gvc_nu;
            z.x += d * phase.x;
            z.y -= d * phase.y;
            /* i^l */
            phase = accCmul(phase, make_accDoubleComplex(0, 1));
        }
        q_pw__[array2D_offset(idx12, 2 * igloc,     ld5__)] = z.x * fourpi__;
        q_pw__[array2D_offset(idx12, 2 * igloc + 1, ld5__)] = z.y * fourpi__;
    }
}

extern "C" void aug_op_pw_coeffs_deriv_gpu(int ngvec__, int const* gvec_shell__, double const* gvec_cart__,
                                           int const* idx__, int idxmax__,
                                           double const* gc__, int ld0__, int ld1__,
                                           double const* rlm__, double const* rlm_dg__, int ld2__,
                                           double const* ri_values__, double const* ri_dg_values__, int ld3__, int ld4__,
                                           double* q_pw__, int ld5__, double fourpi__, int nu__, int lmax_q__)
{
    dim3 grid_t(32);
    dim3 grid_b(num_blocks(ngvec__, grid_t.x), idxmax__);

    accLaunchKernel((aug_op_pw_coeffs_deriv_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0,
        ngvec__, gvec_shell__, gvec_cart__, idx__, idxmax__, gc__, ld0__, ld1__,
        rlm__, rlm_dg__, ld2__, ri_values__, ri_dg_values__, ld3__, ld4__, q_pw__, ld5__, fourpi__, nu__, lmax_q__);
}

