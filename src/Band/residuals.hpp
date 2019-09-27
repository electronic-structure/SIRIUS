// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
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

/** \file residuals.hpp
 *
 *  \brief Compute residuals from the eigen-vectors and basis functions.
 */

#include "typedefs.hpp"
#include "memory.hpp"
#include "linalg.hpp"

namespace sddk {
template <typename T>
class dmatrix;
class Wave_functions;
};


#if defined(__GPU)
extern "C" void residuals_aux_gpu(int num_gvec_loc__,
                                  int num_res_local__,
                                  int* res_idx__,
                                  double* eval__,
                                  double_complex const* hpsi__,
                                  double_complex const* opsi__,
                                  double const* h_diag__,
                                  double const* o_diag__,
                                  double_complex* res__,
                                  double* res_norm__,
                                  double* p_norm__,
                                  int gkvec_reduced__,
                                  int mpi_rank__);

extern "C" void compute_residuals_gpu(double_complex* hpsi__,
                                      double_complex* opsi__,
                                      double_complex* res__,
                                      int num_gvec_loc__,
                                      int num_bands__,
                                      double* eval__);

extern "C" void apply_preconditioner_gpu(double_complex* res__,
                                         int num_rows_loc__,
                                         int num_bands__,
                                         double* eval__,
                                         const double* h_diag__,
                                         const double* o_diag__);

extern "C" void make_real_g0_gpu(double_complex* res__,
                                 int ld__,
                                 int n__);
#endif

namespace sirius {

/// Compute preconditionined residuals.
/** The residuals of wave-functions are difined as:
    \f[
      R_{i} = \hat H \psi_{i} - \epsilon_{i} \hat S \psi_{i}
    \f]
 */
template <typename T>
int
residuals(sddk::memory_t mem_type__, sddk::linalg_t la_type__, int ispn__, int N__, int num_bands__,
          sddk::mdarray<double, 1>& eval__, sddk::dmatrix<T>& evec__, sddk::Wave_functions& hphi__,
          sddk::Wave_functions& ophi__, sddk::Wave_functions& hpsi__,
          sddk::Wave_functions& opsi__, sddk::Wave_functions& res__, sddk::mdarray<double, 2> const& h_diag__,
          sddk::mdarray<double, 2> const& o_diag__, bool estimate_eval__, double norm_tolerance__,
          std::function<bool(int, int)> is_converged__);

}
