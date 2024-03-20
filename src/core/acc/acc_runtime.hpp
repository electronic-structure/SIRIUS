/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file acc_runtime.hpp
 *
 *  \brief Uniform interface to the runtime API of CUDA and ROCm.
 *
 */
#ifndef __ACC_RUNTIME_HPP__
#define __ACC_RUNTIME_HPP__

#include "acc.hpp"

#if defined(SIRIUS_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(SIRIUS_ROCM)
#include <hip/hip_runtime.h>
#endif

/*
 * CUDA runtime calls and definitions
 */
#ifdef SIRIUS_CUDA
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
#ifdef SIRIUS_ROCM
#define accLaunchKernel(...)                                                                                           \
    do {                                                                                                               \
        hipLaunchKernelGGL(__VA_ARGS__);                                                                               \
    } while (0)

#endif

#endif
