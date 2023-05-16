// Copyright (c) 2023 Simon Pintarelli, Anton Kozhevnikov, Thomas Schulthess
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

#ifndef __ACC_LAPACK_HPP__
#define __ACC_LAPACK_HPP__

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


#endif /* __ACC_LAPACK_HPP__ */
