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

/** \file random.cu
 *
 *  \brief CUDA kernel to compute simple random noise on GPU.
 */

//#include "../SDDK/GPU/acc_runtime.hpp"
//== inline __device__ uint32_t random(size_t seed)
//== {
//==     uint32_t h = 5381;
//== 
//==     return (h << (seed % 15)) + h;
//== }
//== 
//== __global__ void randomize_on_gpu_kernel
//== (
//==     double* ptr__,
//==     size_t size__
//== )
//== {
//==     int i = blockIdx.x * blockDim.x + threadIdx.x;
//==     if (i < size__) ptr__[i] = double(random(i)) / (1 << 31);
//== }
//== 
//== extern "C" void randomize_on_gpu(double* ptr, size_t size)
//== {
//==     dim3 grid_t(64);
//==     dim3 grid_b(num_blocks(size, grid_t.x));
//== 
//==     randomize_on_gpu_kernel <<<grid_b, grid_t>>>
//==     (
//==         ptr,
//==         size
//==     );
//== }
