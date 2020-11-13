// Copyright (c) 2013-2020 Anton Kozhevnikov, Thomas Schulthess
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

/** \file linalg_spla.hpp
 *
 *  \brief Interface to SPLA library
 */

#ifndef __LINALG_SPLA_HPP__
#define __LINALG_SPLA_HPP__

#include <memory>
#include <spla/spla.hpp>
#include "blas_lapack.h"

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

std::shared_ptr<::spla::Context>& get_handle_ptr();

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
dgemm(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_double const* alpha, ftn_double const* A,
      ftn_int lda, ftn_double const* B, ftn_int ldb, ftn_double const* beta, ftn_double* C, ftn_int ldc)
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

#endif // __LINALG_SPLA_HPP__
