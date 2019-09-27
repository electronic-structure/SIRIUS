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

/** \file acc_runtime.hpp
 *
 *  \brief Uniform interface to the runtime API of CUDA and ROCm.
 *
 */
#ifndef __ACC_RUNTIME_HPP__
#define __ACC_RUNTIME_HPP__

#include "acc.hpp"

#if defined(__CUDA)
#include <cuda_runtime.h>
#endif

#if defined(__ROCM)
#include <hip/hip_runtime.h>
#endif

/*
 * CUDA runtime calls and definitions
 */
#ifdef __CUDA
#define accLaunchKernel(kernelName, numblocks, numthreads, memperblock, streamId, ...)                                 \
    do {                                                                                                               \
        kernelName<<<numblocks, numthreads, memperblock, streamId>>>(__VA_ARGS__);                                     \
    } while (0)

#define hipThreadIdx_x threadIdx.x
#define hipThreadIdx_y threadIdx.y
#define hipThreadIdx_z threadIdx.z

#define hipBlockIdx_x blockIdx.x
#define hipBlockIdx_y blockIdx.y
#define hipBlockIdx_z blockIdx.z

#define hipBlockDim_x blockDim.x
#define hipBlockDim_y blockDim.y
#define hipBlockDim_z blockDim.z

#define hipGridDim_x gridDim.x
#define hipGridDim_y gridDim.y
#define hipGridDim_z gridDim.z
#endif

/*
 * ROCM runtime calls and definitions
 */
#ifdef __ROCM
#define accLaunchKernel(...)                                                                                           \
    do {                                                                                                               \
        hipLaunchKernelGGL(__VA_ARGS__);                                                                               \
    } while (0)

#endif

#endif
