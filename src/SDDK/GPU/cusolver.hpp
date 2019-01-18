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

inline int zheevd(int32_t matrix_size, int nv, void* A, int32_t lda, void* B, int32_t ldb, double* eval)

{
    cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    auto w = acc::allocate<double>(matrix_size);

    int lwork;
    CALL_CUSOLVER(cusolverDnZhegvd_bufferSize, (cusolver_handle(), itype, jobz, uplo, matrix_size,
                                                static_cast<cuDoubleComplex*>(A), lda, 
                                                static_cast<cuDoubleComplex*>(B), ldb, w, &lwork));

    auto work = acc::allocate<cuDoubleComplex>(lwork);

    int info;
    CALL_CUSOLVER(cusolverDnZhegvd, (cusolver_handle(), itype, jobz, uplo, matrix_size,
                                     static_cast<cuDoubleComplex*>(A), lda,
                                     static_cast<cuDoubleComplex*>(B), ldb, w, work, lwork, &info));

    acc::copyout(eval, w, nv);

    acc::deallocate(work);
    acc::deallocate(w);

    return info;
}

inline void dsyevd()
{

}


inline void zhegvd()
{

}

inline void dsygvd()
{

}

} // namespace cusolver

#endif
