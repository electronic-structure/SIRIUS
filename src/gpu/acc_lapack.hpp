#ifndef ACC_LAPACK_H
#define ACC_LAPACK_H


#include "acc_blas.hpp"
#include "utils/rte.hpp"

#if defined(SIRIUS_CUDA)
#include "cusolver.hpp"
#elif defined(SIRIUS_ROCM)
#include "gpu/rocsolver.hpp"
#endif

namespace acclapack {

inline int getrf(int m, int n, acc_complex_double_t* A, int* devIpiv, int lda)
{
#if defined (SIRIUS_CUDA)
    auto& handle = cusolver::cusolver_handle();
    int* devInfo = acc::allocate<int>(1);

    int lwork;
    CALL_CUSOLVER(cusolverDnZgetrf_bufferSize, (handle,  m, n, A, lda, &lwork));
    auto workspace = acc::allocate<cuDoubleComplex>(lwork);
    CALL_CUSOLVER(cusolverDnZgetrf, (handle, m, n, reinterpret_cast<cuDoubleComplex *>(A), lda, workspace, devIpiv, devInfo));
    acc::deallocate(workspace);

    int cpuInfo;
    acc::copyout(&cpuInfo, devInfo, 1);
    acc::deallocate(devInfo);
    return cpuInfo;
#elif defined(SIRIUS_ROCM)
    auto& handle = rocsolver::rocsolver_handle();
    int cpuInfo;
    int* devInfo = acc::allocate<int>(1);

    rocsolver::zgetrf(handle, m, n, A, devIpiv, lda, devInfo);

    acc::copyout(&cpuInfo, devInfo, 1);
    acc::deallocate(devInfo);
    return cpuInfo;
#endif
}


inline int getrs(char trans, int n, int nrhs, const acc_complex_double_t* A, int lda, const int* devIpiv, acc_complex_double_t* B, int ldb)
{
#if defined(SIRIUS_CUDA)
    auto& handle = cusolver::cusolver_handle();
    int* devInfo = acc::allocate<int>(1);

    cublasOperation_t op = accblas::get_gpublasOperation_t(trans);

    CALL_CUSOLVER(cusolverDnZgetrs, (handle, op, n, nrhs, A, lda, devIpiv, B, ldb, devInfo));

    int cpuInfo;
    acc::copyout(&cpuInfo, devInfo, 1);
    acc::deallocate(devInfo);
    if (cpuInfo != 0) {
        RTE_THROW("Error: cusolver LU solve (Zgetrs) failed. " + std::to_string(cpuInfo));
    }
    return cpuInfo;
#elif defined(SIRIUS_ROCM)
    auto& handle = rocsolver::rocsolver_handle();
    rocsolver::zgetrs(handle, trans, n, nrhs, const_cast<acc_complex_double_t*>(A), lda, devIpiv, B, ldb);
    return 0;
#endif
}



} // namespace acclapack


#endif /* ACC_LAPACK_H */
