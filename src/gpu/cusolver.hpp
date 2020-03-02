/** \file cusolver.hpp
 *
 *  \brief Interface to CUDA eigen-solver library.
 *
 */

#ifndef __CUSOLVER_HPP__
#define __CUSOLVER_HPP__

#include "acc.hpp"
#include <cusolverDn.h>

namespace cusolver {

inline void error_message(cusolverStatus_t status)
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
            std::printf("The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.\n");
            break;
        }
        case CUSOLVER_STATUS_INTERNAL_ERROR: {
            std::printf("An internal cuSolver operation failed. This error is usually caused by a cudaMemcpyAsync() failure.\n");
            break;
        }
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: {
            std::printf("The matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function.\n");
            break;
        }
        default: {
            std::printf("cusolver status unknown\n");
        }
    }
}

#define CALL_CUSOLVER(func__, args__)                                               \
{                                                                                   \
    cusolverStatus_t status;                                                        \
    if ((status = func__ args__) != CUSOLVER_STATUS_SUCCESS) {                      \
        cusolver::error_message(status);                                                      \
        char nm[1024];                                                              \
        gethostname(nm, 1024);                                                      \
        std::printf("hostname: %s\n", nm);                                               \
        std::printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
        stack_backtrace();                                                          \
    }                                                                               \
}

inline cusolverDnHandle_t& cusolver_handle()
{
    static cusolverDnHandle_t handle;
    return handle;
}

inline void create_handle()
{
    CALL_CUSOLVER(cusolverDnCreate, (&cusolver_handle()));
}

inline void destroy_handle()
{
    CALL_CUSOLVER(cusolverDnDestroy, (cusolver_handle()));
}

} // namespace cusolver

#endif
