/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef DIAG_MM_H
#define DIAG_MM_H

#include <complex>

namespace sirius {

extern "C" void
ddiagmm(const double* diag, int n, const double* X, int lda_x, int ncols, double* Y, int lda_y, double alpha);
extern "C" void
sdiagmm(const float* diag, int n, const float* X, int lda_x, int ncols, float* Y, int lda_y, float alpha);
extern "C" void
zdiagmm(const std::complex<double>* diag, int n, const std::complex<double>* X, int lda_x, int ncols,
        std::complex<double>* Y, int lda_y, std::complex<double> alpha);

} // namespace sirius

#endif /* DIAG_MM_H */
