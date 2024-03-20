/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file checksum.cu
 *
 *  \brief Compute checkum on GPU.
 */

#include "core/acc/acc_common.hpp"
#include "core/acc/acc_runtime.hpp"
#include "core/acc/acc.hpp"

extern sirius::acc_stream_t* streams;

using namespace sirius;
using namespace sirius::acc;

template <typename T>
__global__ void add_checksum_gpu_kernel(gpu_complex_type<T> const* ptr__, int ld__, int n__,
        gpu_complex_type<T>* result__)
{
    int N = num_blocks(n__, blockDim.x);

    ACC_DYNAMIC_SHARED(char, sdata_ptr)
    T* sdata_x = (T*)&sdata_ptr[0];
    T* sdata_y = (T*)&sdata_ptr[blockDim.x * sizeof(T)];

    sdata_x[threadIdx.x] = 0.0;
    sdata_y[threadIdx.x] = 0.0;

    for (int i = 0; i < N; i++) {
        int j = i * blockDim.x + threadIdx.x;
        if (j < n__) {
            int k = array2D_offset(j, blockIdx.x, ld__);
            sdata_x[threadIdx.x] += ptr__[k].x;
            sdata_y[threadIdx.x] += ptr__[k].y;
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

    result__[blockIdx.x] = add_accNumbers(result__[blockIdx.x], make_accComplex(sdata_x[0], sdata_y[0]));
}

extern "C" {

void add_checksum_gpu_double(acc_complex_double_t* ptr__, int ld__, int nrow__, int ncol__, acc_complex_double_t* result__)
{
    dim3 grid_t(64);
    dim3 grid_b(ncol__);

    accLaunchKernel((add_checksum_gpu_kernel<double>), dim3(grid_b), dim3(grid_t), 2 * grid_t.x * sizeof(double), 0,
            ptr__, ld__, nrow__, result__);
}

void add_checksum_gpu_float(acc_complex_float_t* ptr__, int ld__, int nrow__, int ncol__, acc_complex_float_t* result__)
{
    dim3 grid_t(64);
    dim3 grid_b(ncol__);

    accLaunchKernel((add_checksum_gpu_kernel<float>), dim3(grid_b), dim3(grid_t), 2 * grid_t.x * sizeof(float), 0,
            ptr__, ld__, nrow__, result__);
}

}
