/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file linalg_spla.hpp
 *
 *  \brief Interface to SPLA library
 */

#ifndef __LINALG_SPLA_HPP__
#define __LINALG_SPLA_HPP__

#include <memory>
#include <spla/spla.hpp>
#include "blas_lapack.h"

namespace sirius {

namespace splablas {

inline SplaOperation
get_spla_operation(char c)
{
    switch (c) {
        case 'n':
        case 'N': {
            return SPLA_OP_NONE;
        }
        case 't':
        case 'T': {
            return SPLA_OP_TRANSPOSE;
        }
        case 'c':
        case 'C': {
            return SPLA_OP_CONJ_TRANSPOSE;
        }
        default: {
            throw std::runtime_error("get_spla_operation(): wrong operation");
        }
    }
    return SPLA_OP_NONE; // make compiler happy
}

std::shared_ptr<::spla::Context>&
get_handle_ptr();

inline void
set_handle_ptr(std::shared_ptr<::spla::Context> ptr)
{
    get_handle_ptr() = std::move(ptr);
}

inline void
reset_handle(SplaProcessingUnit op = SPLA_PU_HOST)
{
    get_handle_ptr().reset(new ::spla::Context{op});
}

inline void
sgemm(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_single const* alpha, ftn_single const* A,
      ftn_int lda, ftn_single const* B, ftn_int ldb, ftn_single const* beta, ftn_single* C, ftn_int ldc)
{
    ::spla::gemm(get_spla_operation(transa), get_spla_operation(transb), m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc,
                 *get_handle_ptr());
}

inline void
dgemm(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_double const* alpha, ftn_double const* A,
      ftn_int lda, ftn_double const* B, ftn_int ldb, ftn_double const* beta, ftn_double* C, ftn_int ldc)
{
    ::spla::gemm(get_spla_operation(transa), get_spla_operation(transb), m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc,
                 *get_handle_ptr());
}

inline void
cgemm(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_complex const* alpha, ftn_complex const* A,
      ftn_int lda, ftn_complex const* B, ftn_int ldb, ftn_complex const* beta, ftn_complex* C, ftn_int ldc)
{
    ::spla::gemm(get_spla_operation(transa), get_spla_operation(transb), m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc,
                 *get_handle_ptr());
}

inline void
zgemm(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_double_complex const* alpha,
      ftn_double_complex const* A, ftn_int lda, ftn_double_complex const* B, ftn_int ldb,
      ftn_double_complex const* beta, ftn_double_complex* C, ftn_int ldc)
{
    ::spla::gemm(get_spla_operation(transa), get_spla_operation(transb), m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc,
                 *get_handle_ptr());
}
} // namespace splablas
} // namespace sirius

#endif // __LINALG_SPLA_HPP__
