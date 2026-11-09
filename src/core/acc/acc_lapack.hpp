/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __ACC_LAPACK_HPP__
#define __ACC_LAPACK_HPP__

#include "acc.hpp"

namespace sirius {

namespace acc {

/// Interface to accelerated lapack functions.
namespace lapack {

int
getrf(int m, int n, acc_complex_double_t* A, int* devIpiv, int lda);

int
getrs(char trans, int n, int nrhs, const acc_complex_double_t* A, int lda, const int* devIpiv, acc_complex_double_t* B,
      int ldb);

} // namespace lapack

} // namespace acc

} // namespace sirius

#endif /* __ACC_LAPACK_HPP__ */
