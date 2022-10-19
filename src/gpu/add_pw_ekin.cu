// Copyright (c) 2013-2022 Anton Kozhevnikov, Thomas Schulthess
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
 *  \brief GPU kernel for the hphi update.
 */

#include "gpu/acc_common.hpp"
#include "gpu/acc_runtime.hpp"

template <typename T>
__global__ void
add_to_hphi_pw_gpu_kernel(int num_gvec__, gpu_complex_type<T> const* vphi__, gpu_complex_type<T>* hphi__)
{
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < num_gvec__) {
        hphi__[ig] = add_accNumbers(hphi__[ig], vphi__[ig]);
    }
}

template <typename T>
__global__ void
add_to_hphi_pw_gpu_kernel(int num_gvec__, T const* pw_ekin__, gpu_complex_type<T> const* phi__,
        gpu_complex_type<T> const* vphi__, gpu_complex_type<T>* hphi__)
{
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < num_gvec__) {
        auto z1    = add_accNumbers(vphi__[ig], mul_accNumbers(pw_ekin__[ig], phi__[ig]));
        hphi__[ig] = add_accNumbers(hphi__[ig], z1);
    }
}

template <typename T>
__global__ void
add_to_hphi_lapw_gpu_kernel(int num_gvec__, gpu_complex_type<T>* const p__, T const* gkvec_cart__,
        gpu_complex_type<T>* hphi__)
{
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < num_gvec__) {
        /* hphi[ig] = hphi[ig] + 1/2 p_{x,y,z}[ig] * G_{x,y,z}[ig] */
        hphi__[ig] = add_accNumbers(hphi__[ig], mul_accNumbers(0.5 * gkvec_cart__[ig], p__[ig]));
    }
}


/// Update the hphi wave functions.
/** The following operation is performed:
 *    hphi[ig] += (alpha *  pw_ekin[ig] * phi[ig] + vphi[ig])
 */
extern "C" {

void
add_to_hphi_pw_gpu_float(int num_gvec__, int add_ekin__, float const* pw_ekin__, gpu_complex_type<float> const* phi__,
    gpu_complex_type<float> const* vphi__, gpu_complex_type<float>* hphi__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec__, grid_t.x));

    if (add_ekin__) {
        accLaunchKernel((add_to_hphi_pw_gpu_kernel<float>), dim3(grid_b), dim3(grid_t), 0, 0, num_gvec__, pw_ekin__,
                        phi__, vphi__, hphi__);
    } else {
        accLaunchKernel((add_to_hphi_pw_gpu_kernel<float>), dim3(grid_b), dim3(grid_t), 0, 0, num_gvec__,
                        vphi__, hphi__);
    }
}

void
add_to_hphi_pw_gpu_double(int num_gvec__, int add_ekin__, double const* pw_ekin__, gpu_complex_type<double> const* phi__,
    gpu_complex_type<double> const* vphi__, gpu_complex_type<double>* hphi__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec__, grid_t.x));

    if (add_ekin__) {
        accLaunchKernel((add_to_hphi_pw_gpu_kernel<double>), dim3(grid_b), dim3(grid_t), 0, 0, num_gvec__, pw_ekin__,
                        phi__, vphi__, hphi__);
    } else {
        accLaunchKernel((add_to_hphi_pw_gpu_kernel<double>), dim3(grid_b), dim3(grid_t), 0, 0, num_gvec__,
                        vphi__, hphi__);
    }
}

void
add_to_hphi_lapw_gpu_float(int num_gvec__, gpu_complex_type<float>* const p__, float const* gkvec_cart__,
        gpu_complex_type<float>* hphi__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec__, grid_t.x));

    accLaunchKernel((add_to_hphi_lapw_gpu_kernel<float>), dim3(grid_b), dim3(grid_t), 0, 0, num_gvec__,
                     p__, gkvec_cart__, hphi__);
}

void
add_to_hphi_lapw_gpu_double(int num_gvec__, gpu_complex_type<double>* const p__, double const* gkvec_cart__,
        gpu_complex_type<double>* hphi__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec__, grid_t.x));

    accLaunchKernel((add_to_hphi_lapw_gpu_kernel<double>), dim3(grid_b), dim3(grid_t), 0, 0, num_gvec__,
                     p__, gkvec_cart__, hphi__);
}

}
