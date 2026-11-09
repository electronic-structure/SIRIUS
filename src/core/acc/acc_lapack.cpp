/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file acc_lapack.cpp
 *
 *  \brief Interface to accelerated lapack functions.
 */

#if defined(SIRIUS_CUDA) || defined(SIRIUS_ROCM)
#include <string>
#include "acc_lapack.hpp"
#include "acc_blas_api.hpp"
#include "core/rte/rte.hpp"
#if defined(SIRIUS_CUDA)
#include "cusolver_api.hpp"
#elif defined(SIRIUS_ROCM)
#include "rocsolver.hpp"
#endif

namespace sirius {

namespace acc {

namespace lapack {

int
getrf(int m, int n, acc_complex_double_t* A, int* devIpiv, int lda)
{
#if defined(SIRIUS_CUDA)
    auto& handle = cusolver::cusolver_handle();
    int* devInfo = acc::allocate<int>(1);

    int lwork;
    CALL_CUSOLVER(cusolverDnZgetrf_bufferSize, (handle, m, n, A, lda, &lwork));
    auto workspace = acc::allocate<cuDoubleComplex>(lwork);
    CALL_CUSOLVER(cusolverDnZgetrf,
                  (handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, workspace, devIpiv, devInfo));
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

int
getrs(char trans, int n, int nrhs, const acc_complex_double_t* A, int lda, const int* devIpiv, acc_complex_double_t* B,
      int ldb)
{
#if defined(SIRIUS_CUDA)
    auto& handle = cusolver::cusolver_handle();
    int* devInfo = acc::allocate<int>(1);

    cublasOperation_t op = blas::get_gpublasOperation_t(trans);

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

} // namespace lapack

} // namespace acc

} // namespace sirius

#endif
