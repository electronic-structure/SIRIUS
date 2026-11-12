/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file cusolver_api.hpp
 *
 *  \brief Internal helpers for the cuSolver library (error handling, data types).
 *
 *  Only included by the .cpp files which call cuSolver directly; the rest of the code uses cusolver.hpp.
 */

#ifndef __CUSOLVER_API_HPP__
#define __CUSOLVER_API_HPP__

#include <complex>
#include <cstdio>
#include <unistd.h>
#include <cusolverDn.h>
#include "acc.hpp"
#include "cusolver.hpp"

namespace sirius {

namespace acc {

namespace cusolver {

inline void
error_message(cusolverStatus_t status)
{
    switch (status) {
        case CUSOLVER_STATUS_NOT_INITIALIZED: {
            std::printf("the CUDA Runtime initialization failed\n");
            break;
        }
        case CUSOLVER_STATUS_ALLOC_FAILED: {
            std::printf("the resources could not be allocated\n");
            break;
        }
        case CUSOLVER_STATUS_ARCH_MISMATCH: {
            std::printf("the device only supports compute capability 2.0 and above\n");
            break;
        }
        case CUSOLVER_STATUS_INVALID_VALUE: {
            std::printf("An unsupported value or parameter was passed to the function\n");
            break;
        }
        case CUSOLVER_STATUS_EXECUTION_FAILED: {
            std::printf("The GPU program failed to execute. This is often caused by a launch failure of the kernel on "
                        "the GPU, which can be caused by multiple reasons.\n");
            break;
        }
        case CUSOLVER_STATUS_INTERNAL_ERROR: {
            std::printf("An internal cuSolver operation failed. This error is usually caused by a cudaMemcpyAsync() "
                        "failure.\n");
            break;
        }
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: {
            std::printf("The matrix type is not supported by this function. This is usually caused by passing an "
                        "invalid matrix descriptor to the function.\n");
            break;
        }
        default: {
            std::printf("cusolver status unknown\n");
        }
    }
}

#define CALL_CUSOLVER(func__, args__)                                                                                  \
    {                                                                                                                  \
        cusolverStatus_t status;                                                                                       \
        if ((status = func__ args__) != CUSOLVER_STATUS_SUCCESS) {                                                     \
            acc::cusolver::error_message(status);                                                                      \
            char nm[1024];                                                                                             \
            gethostname(nm, 1024);                                                                                     \
            std::printf("hostname: %s\n", nm);                                                                         \
            std::printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__);                           \
            acc::stack_backtrace();                                                                                    \
        }                                                                                                              \
    }

template <typename>
struct type_wrapper;

template <>
struct type_wrapper<float>
{
    static constexpr cudaDataType type = CUDA_R_32F;
};

template <>
struct type_wrapper<double>
{
    static constexpr cudaDataType type = CUDA_R_64F;
};

template <>
struct type_wrapper<std::complex<float>>
{
    static constexpr cudaDataType type = CUDA_C_32F;
};

template <>
struct type_wrapper<std::complex<double>>
{
    static constexpr cudaDataType type = CUDA_C_64F;
};

} // namespace cusolver

} // namespace acc

} // namespace sirius

#endif
