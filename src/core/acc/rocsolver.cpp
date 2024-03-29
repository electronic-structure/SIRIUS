/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifdef SIRIUS_ROCM

#include "rocsolver.hpp"
#include "acc_blas.hpp"

namespace sirius {

namespace acc {

namespace rocsolver {

acc::blas_api::handle_t&
rocsolver_handle()
{
    return acc::blas::null_stream_handle();
}

/// Linear Solvers
void
zgetrs(rocblas_handle handle, char trans, int n, int nrhs, acc_complex_double_t* A, int lda, const int* devIpiv,
       acc_complex_double_t* B, int ldb)
{
    rocblas_operation trans_op = get_rocblas_operation(trans);

    CALL_ROCSOLVER(rocsolver_zgetrs, (handle, trans_op, n, nrhs, reinterpret_cast<rocblas_double_complex*>(A), lda,
                                      devIpiv, reinterpret_cast<rocblas_double_complex*>(B), ldb));
}

void
zgetrf(rocblas_handle handle, int m, int n, acc_complex_double_t* A, int* devIpiv, int lda, int* devInfo)
{
    CALL_ROCSOLVER(rocsolver_zgetrf,
                   (handle, m, n, reinterpret_cast<rocblas_double_complex*>(A), lda, devIpiv, devInfo));
}

} // namespace rocsolver

} // namespace acc

} // namespace sirius

#endif
