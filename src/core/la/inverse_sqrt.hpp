// Copyright (c) 2013-2023 Anton Kozhevnikov, Thomas Schulthess
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

/** \file inverse_sqrt.hpp
 *
 *  \brief Compute inverse square root of the matrix.
 */

#ifndef __INVERSE_SQRT_HPP__
#define __INVERSE_SQRT_HPP__

#include "core/la/dmatrix.hpp"
#include "core/la/eigensolver.hpp"
#include "core/la/linalg.hpp"
#include "core/rte/rte.hpp"

namespace sirius {

namespace la {

/// Compute inverse square root of the matrix.
/** As by-product, return the eigen-vectors and the eigen-values of the matrix. */
template <typename T>
inline auto
inverse_sqrt(la::dmatrix<T>& A__, int N__)
{
    auto solver = (A__.comm().size() == 1) ? la::Eigensolver_factory("lapack") : la::Eigensolver_factory("scalapack");

    std::unique_ptr<la::dmatrix<T>> Z;
    std::unique_ptr<la::dmatrix<T>> B;
    if (A__.comm().size() == 1) {
        Z = std::make_unique<la::dmatrix<T>>(A__.num_rows(), A__.num_cols());
        B = std::make_unique<la::dmatrix<T>>(A__.num_rows(), A__.num_cols());
    } else {
        Z = std::make_unique<la::dmatrix<T>>(A__.num_rows(), A__.num_cols(), A__.blacs_grid(), A__.bs_row(),
                                             A__.bs_col());
        B = std::make_unique<la::dmatrix<T>>(A__.num_rows(), A__.num_cols(), A__.blacs_grid(), A__.bs_row(),
                                             A__.bs_col());
    }
    std::vector<real_type<T>> eval(N__);

    if (solver->solve(N__, N__, A__, &eval[0], *Z)) {
        RTE_THROW("error in diagonalization");
    }
    for (int i = 0; i < Z->num_cols_local(); i++) {
        int icol = Z->icol(i);
        RTE_ASSERT(eval[icol] > 0);
        auto f = 1.0 / std::sqrt(eval[icol]);
        for (int j = 0; j < Z->num_rows_local(); j++) {
            A__(j, i) = (*Z)(j, i) * static_cast<T>(f);
        }
    }

    if (A__.comm().size() == 1) {
        la::wrap(la::lib_t::blas)
                .gemm('N', 'C', N__, N__, N__, &la::constant<T>::one(), &A__(0, 0), A__.ld(), Z->at(memory_t::host),
                      Z->ld(), &la::constant<T>::zero(), B->at(memory_t::host), B->ld());
    } else {
        la::wrap(la::lib_t::scalapack)
                .gemm('N', 'C', N__, N__, N__, &la::constant<T>::one(), A__, 0, 0, *Z, 0, 0, &la::constant<T>::zero(),
                      *B, 0, 0);
    }

    return std::make_tuple(std::move(B), std::move(Z), eval);
}

} // namespace la

} // namespace sirius
#endif
