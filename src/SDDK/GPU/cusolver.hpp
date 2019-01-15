#ifndef __CUSOLVER_HPP__
#define __CUSOLVER_HPP__

#include "acc.hpp"
#include <cusolverDn.h>

namespace cusolver {

inline void error_message(cusolverStatus_t status)
{
    switch (status) {
        case CUSOLVER_STATUS_NOT_INITIALIZED: {
            printf("the CUDA Runtime initialization failed\n");
            break;
        }
        case CUSOLVER_STATUS_ALLOC_FAILED: {
            printf("the resources could not be allocated\n");
            break;
        }
        case CUSOLVER_STATUS_ARCH_MISMATCH: {
            printf("the device only supports compute capability 2.0 and above\n");
            break;
        }
        default: {
            printf("cusolver status unknown");
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
        printf("hostname: %s\n", nm);                                               \
        printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
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

inline void zheevd()
{

}

} // namespace cusolver

#endif
