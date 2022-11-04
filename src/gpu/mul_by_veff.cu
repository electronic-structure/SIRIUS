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

/** \file mul_by_veff.cu
 *
 *  \brief CUDA kernels to multiply wave-functions by effective potential.
 */

//#include "gpu/acc_common.hpp"
//#include "gpu/acc_runtime.hpp"
//
//template <typename T>
//__global__ void
//mul_by_veff_real_real_gpu_kernel(int nr__, T const* in__,T const* veff__, T* out__)
//{
//    int i = blockDim.x * blockIdx.x + threadIdx.x;
//    if (i < nr__) {
//        out__[i] = in__[i] * veff__[i];
//    }
//}
//
//template <typename T>
//__global__ void
//mul_by_veff_complex_real_gpu_kernel(int nr__, gpu_complex_type<T> const* in__, T const* veff__,
//        gpu_complex_type<T>* out__)
//{
//    int i = blockDim.x * blockIdx.x + threadIdx.x;
//    if (i < nr__) {
//        out__[i] = mul_accNumbers(veff__[i], in__[i]);
//    }
//}
//
//template <typename T>
//__global__ void
//mul_by_veff_complex_complex_gpu_kernel(int nr__, gpu_complex_type<T> const* in__, T pref__, T const* vx__,
//        T const* vy__, gpu_complex_type<T>* out__)
//{
//    int i = blockDim.x * blockIdx.x + threadIdx.x;
//    if (i < nr__) {
//        out__[i] = mul_accNumbers(in__[i], make_accComplex(vx__[i], pref__ * vy__[i]));
//    }
//}
//
//extern "C" {
//
//void
//mul_by_veff_real_real_gpu_float(int nr__, float const* in__, float const* veff__, float* out__)
//{
//    dim3 grid_t(64);
//    dim3 grid_b(num_blocks(nr__, grid_t.x));
//
//    accLaunchKernel((mul_by_veff_real_real_gpu_kernel<float>), dim3(grid_b), dim3(grid_t), 0, 0,
//            nr__, in__, veff__, out__);
//}
//
//void
//mul_by_veff_real_real_gpu_double(int nr__, double const* in__, double const* veff__, double* out__)
//{
//    dim3 grid_t(64);
//    dim3 grid_b(num_blocks(nr__, grid_t.x));
//
//    accLaunchKernel((mul_by_veff_real_real_gpu_kernel<double>), dim3(grid_b), dim3(grid_t), 0, 0,
//            nr__, in__, veff__, out__);
//}
//
//void
//mul_by_veff_complex_real_gpu_float(int nr__, gpu_complex_type<float> const* in__, float const* veff__,
//        gpu_complex_type<float>* out__)
//{
//    dim3 grid_t(64);
//    dim3 grid_b(num_blocks(nr__, grid_t.x));
//
//    accLaunchKernel((mul_by_veff_complex_real_gpu_kernel<float>), dim3(grid_b), dim3(grid_t), 0, 0,
//            nr__, in__, veff__, out__);
//}
//
//void
//mul_by_veff_complex_real_gpu_double(int nr__, gpu_complex_type<double> const* in__, double const* veff__,
//        gpu_complex_type<double>* out__)
//{
//    dim3 grid_t(64);
//    dim3 grid_b(num_blocks(nr__, grid_t.x));
//
//    accLaunchKernel((mul_by_veff_complex_real_gpu_kernel<double>), dim3(grid_b), dim3(grid_t), 0, 0,
//            nr__, in__, veff__, out__);
//}
//
//void
//mul_by_veff_complex_complex_gpu_float(int nr__, gpu_complex_type<float> const* in__, float pref__,
//    float const* vx__, float const* vy__, gpu_complex_type<float>* out__)
//{
//    dim3 grid_t(64);
//    dim3 grid_b(num_blocks(nr__, grid_t.x));
//
//    accLaunchKernel((mul_by_veff_complex_complex_gpu_kernel<float>), dim3(grid_b), dim3(grid_t), 0, 0,
//            nr__, in__, pref__, vx__, vy__, out__);
//}
//
//void
//mul_by_veff_complex_complex_gpu_double(int nr__, gpu_complex_type<double> const* in__, double pref__,
//    double const* vx__, double const* vy__, gpu_complex_type<double>* out__)
//{
//    dim3 grid_t(64);
//    dim3 grid_b(num_blocks(nr__, grid_t.x));
//
//    accLaunchKernel((mul_by_veff_complex_complex_gpu_kernel<double>), dim3(grid_b), dim3(grid_t), 0, 0,
//            nr__, in__, pref__, vx__, vy__, out__);
//}

//}


