/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file cusolver.hpp
 *
 *  \brief Interface to CUDA eigen-solver library.
 *
 */

#ifndef __CUSOLVER_HPP__
#define __CUSOLVER_HPP__

/* opaque handle, same definition as in cusolverDn.h */
struct cusolverDnContext;
typedef struct cusolverDnContext* cusolverDnHandle_t;

namespace sirius {

namespace acc {

/// Interface to cuSolver library.
namespace cusolver {

cusolverDnHandle_t&
cusolver_handle();
void
create_handle();
void
destroy_handle();

/// Cholesky factorization of a matrix in device memory; return the info parameter of cuSolver.
/** Instantiated for float, double, std::complex<float> and std::complex<double>. */
template <typename T>
int
potrf(int n__, T* A__, int lda__);

/// Inversion of an upper triangular matrix in device memory; return the info parameter of cuSolver.
/** Instantiated for float, double, std::complex<float> and std::complex<double>. */
template <typename T>
int
trtri(int n__, T* A__, int lda__);

} // namespace cusolver

} // namespace acc

} // namespace sirius

#endif
