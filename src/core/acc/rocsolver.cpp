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

#ifdef SIRIUS_ROCM

#include "rocsolver.hpp"
#include "acc_blas.hpp"

namespace sirius {

namespace acc {

namespace rocsolver {

acc::blas::handle_t&
rocsolver_handle()
{
    return blas::null_stream_handle();
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
