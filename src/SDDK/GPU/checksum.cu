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

/** \file checksum.cu
 *
 *  \brief CUDA kernel for the calculation of checksum.
 */

#include "cuda_common.hpp"

__global__ void double_complex_checksum_gpu_kernel
(
    cuDoubleComplex const* ptr__,
    size_t size__,
    cuDoubleComplex *result__
)
{
    int N = num_blocks(size__, blockDim.x);

    extern __shared__ char sdata_ptr[];
    double* sdata_x = (double*)&sdata_ptr[0];
    double* sdata_y = (double*)&sdata_ptr[blockDim.x * sizeof(double)];

    sdata_x[threadIdx.x] = 0.0;
    sdata_y[threadIdx.x] = 0.0;

    for (int n = 0; n < N; n++) {
        int j = n * blockDim.x + threadIdx.x;
        if (j < size__) {
            sdata_x[threadIdx.x] += ptr__[j].x;
            sdata_y[threadIdx.x] += ptr__[j].y;
        }
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            sdata_x[threadIdx.x] = sdata_x[threadIdx.x] + sdata_x[threadIdx.x + s];
            sdata_y[threadIdx.x] = sdata_y[threadIdx.x] + sdata_y[threadIdx.x + s];
        }
        __syncthreads();
    }

    *result__ = make_cuDoubleComplex(sdata_x[0], sdata_y[0]);
}

extern "C" void double_complex_checksum_gpu(cuDoubleComplex const* ptr__,
                                            size_t size__,
                                            cuDoubleComplex* result__)
{
    dim3 grid_t(64);
    dim3 grid_b(1);

    cuDoubleComplex* res;
    cudaMalloc(&res, sizeof(cuDoubleComplex));

    double_complex_checksum_gpu_kernel <<<grid_b, grid_t, 2 * grid_t.x * sizeof(double)>>>
    (
        ptr__,
        size__,
        res
    );

    cudaMemcpy(result__, res, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cudaFree(res);
}
