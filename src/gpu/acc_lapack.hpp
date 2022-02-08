#ifndef ACC_LAPACK_H
#define ACC_LAPACK_H

#if defined (SIRIUS_CUDA)

#include "cusolver.hpp"
#include "acc_blas.hpp"

namespace acclapack {

inline int getrf(int m, int n, acc_complex_double_t* A, int* devIpiv, int lda)
{
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
}


inline int getrs(char trans, int n, int nrhs, const acc_complex_double_t* A, int lda, const int* devIpiv, acc_complex_double_t* B, int ldb)
{
    auto& handle = cusolver::cusolver_handle();
    int* devInfo = acc::allocate<int>(1);

    cublasOperation_t op = accblas::get_gpublasOperation_t(trans);

    CALL_CUSOLVER(cusolverDnZgetrs, (handle, op, n, nrhs, A, lda, devIpiv, B, ldb, devInfo));

    int cpuInfo;
    acc::copyout(&cpuInfo, devInfo, 1);
    acc::deallocate(devInfo);
    if (cpuInfo != 0) {
        throw std::runtime_error("Error: cusolver LU solve (Zgetrs) failed. " + std::to_string(cpuInfo));
    }
    return cpuInfo;
}


} // namespace acclapack

#endif

#endif /* ACC_LAPACK_H */
