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

#ifndef __RESIDUALS_HPP__
#define __RESIDUALS_HPP__

#include "typedefs.hpp"
#include "memory.hpp"
#include "linalg/linalg.hpp"
#include "context/simulation_context.hpp"
#include "SDDK/type_definition.hpp"

namespace sddk {
template <typename T>
class Wave_functions;
class spin_range;
};

struct residual_result {
  int num_consecutive_smallest_converged;
  int unconverged_residuals;
  double frobenius_norm;
};

#if defined(SIRIUS_GPU)
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

extern "C" void compute_residuals_gpu_double(double_complex* hpsi__,
                                              double_complex* opsi__,
                                              double_complex* res__,
                                              int num_gvec_loc__,
                                              int num_bands__,
                                              double* eval__);

extern "C" void compute_residuals_gpu_float(std::complex<float>* hpsi__,
                                             std::complex<float>* opsi__,
                                             std::complex<float>* res__,
                                             int num_gvec_loc__,
                                             int num_bands__,
                                             float* eval__);

extern "C" void apply_preconditioner_gpu_double(double_complex* res__,
                                                 int num_rows_loc__,
                                                 int num_bands__,
                                                 double* eval__,
                                                 const double* h_diag__,
                                                 const double* o_diag__);

extern "C" void apply_preconditioner_gpu_float(std::complex<float>* res__,
                                                int num_rows_loc__,
                                                int num_bands__,
                                                float* eval__,
                                                const float* h_diag__,
                                                const float* o_diag__);

extern "C" void make_real_g0_gpu_double(double_complex* res__,
                                         int ld__,
                                         int n__);

extern "C" void make_real_g0_gpu_float(std::complex<float>* res__,
                                         int ld__,
                                         int n__);
#endif

namespace sirius {

/// Compute preconditionined residuals.
/** The residuals of wave-functions are defined as:
    \f[
      R_{i} = \hat H \psi_{i} - \epsilon_{i} \hat S \psi_{i}
    \f]
 */
template <typename T, typename F>
residual_result
residuals(Simulation_context& ctx__, sddk::memory_t mem_type__, sddk::linalg_t la_type__, sddk::spin_range ispn__,
          int N__, int num_bands__, int num_locked, sddk::mdarray<real_type<F>, 1>& eval__,
          sddk::dmatrix<F>& evec__, sddk::Wave_functions<real_type<T>>& hphi__, sddk::Wave_functions<real_type<T>>& ophi__,
          sddk::Wave_functions<real_type<T>>& hpsi__, sddk::Wave_functions<real_type<T>>& opsi__,
          sddk::Wave_functions<real_type<T>>& res__, sddk::mdarray<real_type<T>, 2> const& h_diag__,
          sddk::mdarray<real_type<T>, 2> const& o_diag__, bool estimate_eval__, real_type<T> norm_tolerance__,
          std::function<bool(int, int)> is_converged__);

template <typename T>
void
apply_preconditioner(sddk::memory_t mem_type__, sddk::spin_range spins__, int num_bands__, sddk::Wave_functions<T>& res__,
                     sddk::mdarray<T, 2> const& h_diag__, sddk::mdarray<T, 2> const& o_diag__,
                     sddk::mdarray<T, 1>& eval__);
}

#endif
