/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file dlaf.hpp
 *
 *  \brief Interface to some of the DLA-Future functionsh.
 */

#ifndef __DLAF_HPP__
#define __DLAF_HPP__

#ifdef SIRIUS_DLAF

#include "core/la/dmatrix.hpp"

#ifdef SIRIUS_DLAF
#include <dlaf_c/grid.h>
#include <dlaf_c/eigensolver/eigensolver.h>
#include <dlaf_c/eigensolver/gen_eigensolver.h>
#endif

namespace sirius::la::dlaf {

void
init();

void
finalize();

template <typename T>
int
blacs_context(la::dmatrix<T>& M__)
{
    int blacs_context = M__.blacs_grid().context();
    if (blacs_context == -1) {
        blacs_context = dlaf_create_grid(M__.blacs_grid().comm().native(), M__.blacs_grid().num_ranks_row(),
                                         M__.blacs_grid().num_ranks_col(), 'R');
    } else {
#ifdef SIRIUS_SCALAPACK
        // Create DLAF grid from the BLACS context
        dlaf_create_grid_from_blacs(blacs_context);
#else
        RTE_THROW("not compiled with ScaLAPACK");
#endif
    }

    return blacs_context;
}

template <typename T>
int
hermitian_eigensolver(ftn_int matrix_size__, la::dmatrix<T>& A__, real_type<T>* eval__, la::dmatrix<T>& Z__)
{
    dlaf::init();
    auto blacs_context = dlaf::blacs_context(A__);

    DLAF_descriptor desca{
            matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0, 0, 0, static_cast<int>(A__.ld())};
    DLAF_descriptor descz{
            matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0, 0, 0, static_cast<int>(Z__.ld())};

    if (std::is_same_v<T, std::complex<double>>) {
        return dlaf_hermitian_eigensolver_z(blacs_context, 'L',
                                            reinterpret_cast<std::complex<double>*>(A__.at(memory_t::host)), desca,
                                            reinterpret_cast<double*>(eval__),
                                            reinterpret_cast<std::complex<double>*>(Z__.at(memory_t::host)), descz);
    } else if (std::is_same_v<T, std::complex<float>>) {
        return dlaf_hermitian_eigensolver_c(blacs_context, 'L',
                                            reinterpret_cast<std::complex<float>*>(A__.at(memory_t::host)), desca,
                                            reinterpret_cast<float*>(eval__),
                                            reinterpret_cast<std::complex<float>*>(Z__.at(memory_t::host)), descz);
    } else if (std::is_same_v<T, double>) {
        return dlaf_symmetric_eigensolver_d(blacs_context, 'L', reinterpret_cast<double*>(A__.at(memory_t::host)),
                                            desca, reinterpret_cast<double*>(eval__),
                                            reinterpret_cast<double*>(Z__.at(memory_t::host)), descz);
    } else if (std::is_same_v<T, float>) {
        return dlaf_symmetric_eigensolver_s(blacs_context, 'L', reinterpret_cast<float*>(A__.at(memory_t::host)), desca,
                                            reinterpret_cast<float*>(eval__),
                                            reinterpret_cast<float*>(Z__.at(memory_t::host)), descz);
    }
}

template <typename T>
int
hermitian_generalized_eigensolver(ftn_int matrix_size__, la::dmatrix<T>& A__, la::dmatrix<T>& B__, real_type<T>* eval__,
                                  la::dmatrix<T>& Z__)
{
    dlaf::init();
    auto blacs_context = dlaf::blacs_context(A__);

    DLAF_descriptor desca{
            matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0, 0, 0, static_cast<int>(A__.ld())};
    DLAF_descriptor descb{
            matrix_size__, matrix_size__, B__.bs_row(), B__.bs_col(), 0, 0, 0, 0, static_cast<int>(B__.ld())};
    DLAF_descriptor descz{
            matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0, 0, 0, static_cast<int>(Z__.ld())};

    if (std::is_same_v<T, std::complex<double>>) {
        return dlaf_hermitian_generalized_eigensolver_z(
                blacs_context, 'L', reinterpret_cast<std::complex<double>*>(A__.at(memory_t::host)), desca,
                reinterpret_cast<std::complex<double>*>(B__.at(memory_t::host)), descb,
                reinterpret_cast<double*>(eval__), reinterpret_cast<std::complex<double>*>(Z__.at(memory_t::host)),
                descz);
    } else if (std::is_same_v<T, std::complex<float>>) {
        return dlaf_hermitian_generalized_eigensolver_c(
                blacs_context, 'L', reinterpret_cast<std::complex<float>*>(A__.at(memory_t::host)), desca,
                reinterpret_cast<std::complex<float>*>(B__.at(memory_t::host)), descb, reinterpret_cast<float*>(eval__),
                reinterpret_cast<std::complex<float>*>(Z__.at(memory_t::host)), descz);
    } else if (std::is_same_v<T, double>) {
        return dlaf_symmetric_generalized_eigensolver_d(
                blacs_context, 'L', reinterpret_cast<double*>(A__.at(memory_t::host)), desca,
                reinterpret_cast<double*>(B__.at(memory_t::host)), descb, reinterpret_cast<double*>(eval__),
                reinterpret_cast<double*>(Z__.at(memory_t::host)), descz);
    } else if (std::is_same_v<T, float>) {
        return dlaf_symmetric_generalized_eigensolver_s(
                blacs_context, 'L', reinterpret_cast<float*>(A__.at(memory_t::host)), desca,
                reinterpret_cast<float*>(A__.at(memory_t::host)), descb, reinterpret_cast<float*>(eval__),
                reinterpret_cast<float*>(Z__.at(memory_t::host)), descz);
    }
}

} // namespace sirius::la::dlaf

#endif  // SIRIUS_DLAF                                                                                                 \
        //
#endif  // __DLAF_HPP__
