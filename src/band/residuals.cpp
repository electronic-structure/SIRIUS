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

/** \file residuals.cpp
 *
 *  \brief Compute residuals from the eigen-vectors and basis functions.
 */

#include "residuals.hpp"
#include "wave_functions.hpp"
#include "wf_inner.hpp"
#include "wf_ortho.hpp"
#include "wf_trans.hpp"

#include <iomanip>

namespace sirius {

#if defined(SIRIUS_GPU)
void compute_residuals_gpu(double_complex* hpsi__, double_complex* opsi__, double_complex* res__,
                            int num_gvec_loc__, int num_bands__, double* eval__)
{
    compute_residuals_gpu_double(hpsi__, opsi__, res__, num_gvec_loc__, num_bands__, eval__);
}

void compute_residuals_gpu(std::complex<float>* hpsi__, std::complex<float>* opsi__, std::complex<float>* res__,
                           int num_gvec_loc__, int num_bands__, float* eval__)
{
    compute_residuals_gpu_float(hpsi__, opsi__, res__, num_gvec_loc__, num_bands__, eval__);
}

void apply_preconditioner_gpu(double_complex* res__, int num_rows_loc__, int num_bands__,
                              double* eval__, const double* h_diag__, const double* o_diag__)
{
    apply_preconditioner_gpu_double(res__, num_rows_loc__, num_bands__, eval__, h_diag__, o_diag__);
}

void apply_preconditioner_gpu(std::complex<float>* res__, int num_rows_loc__, int num_bands__,
                              float* eval__, const float* h_diag__, const float* o_diag__)
{
    apply_preconditioner_gpu_float(res__, num_rows_loc__, num_bands__, eval__, h_diag__, o_diag__);
}

void make_real_g0_gpu(double_complex* res__,  int ld__,  int n__)
{
    make_real_g0_gpu_double(res__, ld__, n__);
}

void make_real_g0_gpu(std::complex<float>* res__,  int ld__,  int n__)
{
    make_real_g0_gpu_float(res__, ld__, n__);
}
#endif

template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
static void
compute_residuals(sddk::memory_t mem_type__, sddk::spin_range spins__, int num_bands__, sddk::mdarray<T, 1>& eval__,
                  sddk::Wave_functions<T>& hpsi__, sddk::Wave_functions<T>& opsi__, sddk::Wave_functions<T>& res__)
{
    for (int ispn: spins__) {
        if (is_host_memory(mem_type__)) {
            /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
            #pragma omp parallel for
            for (int i = 0; i < num_bands__; i++) {
                for (int ig = 0; ig < res__.pw_coeffs(ispn).num_rows_loc(); ig++) {
                    res__.pw_coeffs(ispn).prime(ig, i) = hpsi__.pw_coeffs(ispn).prime(ig, i) -
                        eval__[i] * opsi__.pw_coeffs(ispn).prime(ig, i);
                }
                if (res__.has_mt()) {
                    for (int j = 0; j < res__.mt_coeffs(ispn).num_rows_loc(); j++) {
                        res__.mt_coeffs(ispn).prime(j, i) = hpsi__.mt_coeffs(ispn).prime(j, i) -
                            eval__[i] * opsi__.mt_coeffs(ispn).prime(j, i);
                    }
                }
            }
        } else {
#if defined(SIRIUS_GPU)
            compute_residuals_gpu(hpsi__.pw_coeffs(ispn).prime().at(sddk::memory_t::device),
                                  opsi__.pw_coeffs(ispn).prime().at(sddk::memory_t::device),
                                  res__.pw_coeffs(ispn).prime().at(sddk::memory_t::device),
                                  res__.pw_coeffs(ispn).num_rows_loc(),
                                  num_bands__,
                                  eval__.at(sddk::memory_t::device));
            if (res__.has_mt()) {
                compute_residuals_gpu(hpsi__.mt_coeffs(ispn).prime().at(sddk::memory_t::device),
                                      opsi__.mt_coeffs(ispn).prime().at(sddk::memory_t::device),
                                      res__.mt_coeffs(ispn).prime().at(sddk::memory_t::device),
                                      res__.mt_coeffs(ispn).num_rows_loc(),
                                      num_bands__,
                                      eval__.at(sddk::memory_t::device));
            }
#endif
        }
    }
}

/// Apply preconditioner to the residuals.
template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
static void
apply_preconditioner(sddk::memory_t mem_type__, sddk::spin_range spins__, int num_bands__, sddk::Wave_functions<T>& res__,
                     sddk::mdarray<T, 2> const& h_diag__, sddk::mdarray<T, 2> const& o_diag__,
                     sddk::mdarray<T, 1>& eval__)
{
    for (int ispn: spins__) {
        if (is_host_memory(mem_type__)) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_bands__; i++) {
                for (int ig = 0; ig < res__.pw_coeffs(ispn).num_rows_loc(); ig++) {
                    T p = h_diag__(ig, ispn) - o_diag__(ig, ispn) * eval__[i];
                    p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                    res__.pw_coeffs(ispn).prime(ig, i) /= p;
                }
                if (res__.has_mt()) {
                    for (int j = 0; j < res__.mt_coeffs(ispn).num_rows_loc(); j++) {
                        T p = h_diag__(res__.pw_coeffs(ispn).num_rows_loc() + j, ispn) -
                                   o_diag__(res__.pw_coeffs(ispn).num_rows_loc() + j, ispn) * eval__[i];
                        p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                        res__.mt_coeffs(ispn).prime(j, i) /= p;
                    }
                }
            }
        } else {
#if defined(SIRIUS_GPU)
            apply_preconditioner_gpu(res__.pw_coeffs(ispn).prime().at(sddk::memory_t::device),
                                     res__.pw_coeffs(ispn).num_rows_loc(),
                                     num_bands__,
                                     eval__.at(sddk::memory_t::device),
                                     h_diag__.at(sddk::memory_t::device, 0, ispn),
                                     o_diag__.at(sddk::memory_t::device, 0, ispn));
            if (res__.has_mt()) {
                apply_preconditioner_gpu(res__.mt_coeffs(ispn).prime().at(sddk::memory_t::device),
                                         res__.mt_coeffs(ispn).num_rows_loc(),
                                         num_bands__,
                                         eval__.at(sddk::memory_t::device),
                                         h_diag__.at(sddk::memory_t::device, res__.pw_coeffs(ispn).num_rows_loc(), ispn),
                                         o_diag__.at(sddk::memory_t::device, res__.pw_coeffs(ispn).num_rows_loc(), ispn));
            }
#endif
        }
    }
}

template <typename T>
static int
normalized_preconditioned_residuals(sddk::memory_t mem_type__, sddk::spin_range spins__, int num_bands__,
                                    sddk::mdarray<real_type<T>,1>& eval__, sddk::Wave_functions<real_type<T>>& hpsi__,
                                    sddk::Wave_functions<real_type<T>>& opsi__, sddk::Wave_functions<real_type<T>>& res__,
                                    sddk::mdarray<real_type<T>, 2> const& h_diag__, sddk::mdarray<real_type<T>, 2> const& o_diag__,
                                    real_type<T> norm_tolerance__, sddk::mdarray<real_type<T>, 1> &residual_norms__)
{
    PROFILE("sirius::normalized_preconditioned_residuals");

    assert(num_bands__ != 0);

    auto pu = get_device_t(mem_type__);

    /* compute "raw" residuals */
    compute_residuals<real_type<T>>(mem_type__, spins__, num_bands__, eval__, hpsi__, opsi__, res__);

    /* compute norm of the "raw" residuals */
    residual_norms__ = res__.l2norm(pu, spins__, num_bands__);

    /* apply preconditioner */
    apply_preconditioner<real_type<T>>(mem_type__, spins__, num_bands__, res__, h_diag__, o_diag__, eval__);

    /* this not strictly necessary as the wave-function orthoronormalization can take care of this;
       however, normalization of residuals is harmless and gives a better numerical stability */
    res__.normalize(pu, spins__, num_bands__);

    int num_unconverged{0};

    for (int i = 0; i < num_bands__; i++) {
        /* take the residual if it's norm is above the threshold */
        if (residual_norms__[i] > norm_tolerance__) {
            /* shift unconverged residuals to the beginning of array */
            /* note: we can just keep them where they were  */
            if (num_unconverged != i) {
                for (int ispn: spins__) {
                    res__.copy_from(res__, 1, ispn, i, ispn, num_unconverged);
                }
            }
            num_unconverged++;
        }
    }

    /* prevent numerical noise */
    /* this only happens for real wave-functions (Gamma-point case), non-magnetic or collinear magnetic */
    if (std::is_same<T, real_type<T>>::value && res__.comm().rank() == 0 && num_unconverged != 0 && spins__() != 2) {
        if (is_device_memory(res__.preferred_memory_t())) {
#if defined(SIRIUS_GPU)
            make_real_g0_gpu(res__.pw_coeffs(spins__()).prime().at(sddk::memory_t::device), res__.pw_coeffs(spins__()).prime().ld(), num_unconverged);
#endif
        } else {
            for (int i = 0; i < num_unconverged; i++) {
                res__.pw_coeffs(spins__()).prime(0, i) = res__.pw_coeffs(spins__()).prime(0, i).real();
            }
        }
    }

    return num_unconverged;
}

/// Compute residuals from eigen-vectors.
template <typename T, typename F>
residual_result
residuals(Simulation_context& ctx__, sddk::memory_t mem_type__, sddk::linalg_t la_type__, sddk::spin_range ispn__, int N__,
          int num_bands__, int num_locked, sddk::mdarray<real_type<F>, 1>& eval__, sddk::dmatrix<F>& evec__,
          sddk::Wave_functions<real_type<T>>& hphi__, sddk::Wave_functions<real_type<T>>& ophi__, sddk::Wave_functions<real_type<T>>& hpsi__,
          sddk::Wave_functions<real_type<T>>& opsi__, sddk::Wave_functions<real_type<T>>& res__, sddk::mdarray<real_type<T>, 2> const& h_diag__,
          sddk::mdarray<real_type<T>, 2> const& o_diag__, bool estimate_eval__, real_type<T> norm_tolerance__,
          std::function<bool(int, int)> is_converged__)
{
    PROFILE("sirius::residuals");

    assert(N__ != 0);

    sddk::mdarray<real_type<T>, 1> res_norm;
    sddk::dmatrix<F> evec_tmp;

    //sddk::mdarray<real_type<T>, 1> eval_tmp;

    sddk::mdarray<real_type<T>, 1> eval(num_bands__);
    eval = [&](size_t j) -> real_type<T> { return eval__[j]; };

    sddk::dmatrix<F>* evec_ptr{nullptr};
    //sddk::mdarray<real_type<T>, 1>* eval_ptr{nullptr};

    /* total number of residuals to be computed */
    int num_residuals{0};

    /* number of lockable eigenvectors */
    int num_consecutive_converged{0};

    /* number of residuals that do not meet any convergence criterion */
    int num_unconverged{0};

    int ispn = (ispn__() == 2) ? 0 : ispn__();

    // When estimate_eval is set we only compute true residuals of unconverged eigenpairs
    // where convergence is determined just on the change in the eigenvalues.
    if (estimate_eval__) {
        // Locking is only based on the is_converged__ criterion, not on the actual
        // residual norms. We could lock more by considering the residual norm criterion
        // later, but since we're reordering eigenvectors too, this becomes messy.
        while (num_consecutive_converged < num_bands__ && is_converged__(num_consecutive_converged, ispn)) {
            ++num_consecutive_converged;
        }

        // Collect indices of unconverged eigenpairs.
        std::vector<int> ev_idx;
        for (int j = 0; j < num_bands__; j++) {
            if (!is_converged__(j, ispn)) {
                ev_idx.push_back(j);
            }
        }

        // If everything is converged, return early.
        if (ev_idx.empty()) {
            return residual_result{num_bands__, 0, 0};
        }

        // Otherwise copy / reorder the unconverged eigenpairs
        num_residuals = static_cast<int>(ev_idx.size());

        //eval_tmp = sddk::mdarray<real_type<T>, 1>(num_residuals);
        //eval_ptr = &eval_tmp;
        evec_tmp = sddk::dmatrix<F>(N__, num_residuals, evec__.blacs_grid(), evec__.bs_row(), evec__.bs_col());
        evec_ptr = &evec_tmp;

        int num_rows_local = evec_tmp.num_rows_local();
        for (int j = 0; j < num_residuals; j++) {
            //eval_tmp[j] = eval[ev_idx[j]];
            eval[j] = eval[ev_idx[j]];
            if (evec__.blacs_grid().comm().size() == 1) {
                /* do a local copy */
                std::copy(&evec__(0, ev_idx[j]), &evec__(0, ev_idx[j]) + num_rows_local, &evec_tmp(0, j));
            } else {
                auto pos_src  = evec__.spl_col().location(ev_idx[j]);
                auto pos_dest = evec_tmp.spl_col().location(j);
                /* do MPI send / receive */
                if (pos_src.rank == evec__.blacs_grid().comm_col().rank() && num_rows_local) {
                    evec__.blacs_grid().comm_col().isend(&evec__(0, pos_src.local_index), num_rows_local, pos_dest.rank, ev_idx[j]);
                }
                if (pos_dest.rank == evec__.blacs_grid().comm_col().rank() && num_rows_local) {
                    evec__.blacs_grid().comm_col().recv(&evec_tmp(0, pos_dest.local_index), num_rows_local, pos_src.rank, ev_idx[j]);
                }
            }
        }
        if (is_device_memory(mem_type__) && evec_tmp.blacs_grid().comm().size() == 1) {
            evec_tmp.allocate(sddk::memory_t::device);
        }
        //if (is_device_memory(mem_type__)) {
        //    eval_tmp.allocate(sddk::memory_t::device).copy_to(sddk::memory_t::device);
        //}
    } else {
        //if (is_device_memory(mem_type__)) {
        //    eval__.allocate(sddk::memory_t::device).copy_to(sddk::memory_t::device);
        //}
        //if (is_device_memory(mem_type__)) {
        //    eval.allocate(sddk::memory_t::device).copy_to(sddk::memory_t::device);
        //}
        evec_ptr = &evec__;
        //eval_ptr = &eval;
        num_residuals = num_bands__;
    }
    if (is_device_memory(mem_type__)) {
        eval.allocate(sddk::memory_t::device).copy_to(sddk::memory_t::device);
    }

    /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} and O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
    sddk::transform<T, F>(ctx__.spla_context(), ispn__(), {&hphi__, &ophi__}, num_locked, N__ - num_locked,
                          *evec_ptr, 0, 0, {&hpsi__, &opsi__}, 0, num_residuals);

    num_unconverged = normalized_preconditioned_residuals<T>(mem_type__, ispn__, num_residuals, eval, hpsi__, opsi__,
                                                             res__, h_diag__, o_diag__, norm_tolerance__, res_norm);

    // In case we're not using the delta in eigenvalues as a convergence criterion,
    // we lock eigenpairs using residual norms.
    if (!estimate_eval__) {
        while (num_consecutive_converged < num_residuals && res_norm[num_consecutive_converged] <= norm_tolerance__) {
            ++num_consecutive_converged;
        }
    }

    auto frobenius_norm = 0.0;
    for (int i = 0; i < num_residuals; i++) {
        frobenius_norm += res_norm[i] * res_norm[i];
    }
    frobenius_norm = std::sqrt(frobenius_norm);
    return {
        num_consecutive_converged,
        num_unconverged,
        frobenius_norm
    };
}

template residual_result
residuals<double, double>(Simulation_context& ctx__, sddk::memory_t mem_type__, sddk::linalg_t la_type__,
                  sddk::spin_range ispn__, int N__,
                  int num_bands__, int num_locked, sddk::mdarray<double, 1>& eval__, sddk::dmatrix<double>& evec__,
                  sddk::Wave_functions<double>& hphi__, sddk::Wave_functions<double>& ophi__, sddk::Wave_functions<double>& hpsi__,
                  sddk::Wave_functions<double>& opsi__, sddk::Wave_functions<double>& res__, sddk::mdarray<double, 2> const& h_diag__,
                  sddk::mdarray<double, 2> const& o_diag__, bool estimate_eval__, double norm_tolerance__,
                  std::function<bool(int, int)> is_converged__);

template residual_result
residuals<double_complex, double_complex>(Simulation_context& ctx__, sddk::memory_t mem_type__, sddk::linalg_t la_type__,
                          sddk::spin_range ispn__,
                          int N__, int num_bands__, int num_locked, sddk::mdarray<double, 1>& eval__,
                          sddk::dmatrix<double_complex>& evec__, sddk::Wave_functions<double>& hphi__,
                          sddk::Wave_functions<double>& ophi__, sddk::Wave_functions<double>& hpsi__, sddk::Wave_functions<double>& opsi__,
                          sddk::Wave_functions<double>& res__, sddk::mdarray<double, 2> const& h_diag__,
                          sddk::mdarray<double, 2> const& o_diag__, bool estimate_eval__, double norm_tolerance__,
                          std::function<bool(int, int)> is_converged__);

#if defined(USE_FP32)
template residual_result
residuals<float, float>(Simulation_context& ctx__, sddk::memory_t mem_type__, sddk::linalg_t la_type__,
                 sddk::spin_range ispn__, int N__,
                 int num_bands__, int num_locked, sddk::mdarray<float, 1>& eval__, sddk::dmatrix<float>& evec__,
                 sddk::Wave_functions<float>& hphi__, sddk::Wave_functions<float>& ophi__, sddk::Wave_functions<float>& hpsi__,
                 sddk::Wave_functions<float>& opsi__, sddk::Wave_functions<float>& res__, sddk::mdarray<float, 2> const& h_diag__,
                 sddk::mdarray<float, 2> const& o_diag__, bool estimate_eval__, float norm_tolerance__,
                 std::function<bool(int, int)> is_converged__);

template residual_result
residuals<float, double>(Simulation_context& ctx__, sddk::memory_t mem_type__, sddk::linalg_t la_type__,
                 sddk::spin_range ispn__, int N__,
                 int num_bands__, int num_locked, sddk::mdarray<double, 1>& eval__, sddk::dmatrix<double>& evec__,
                 sddk::Wave_functions<float>& hphi__, sddk::Wave_functions<float>& ophi__, sddk::Wave_functions<float>& hpsi__,
                 sddk::Wave_functions<float>& opsi__, sddk::Wave_functions<float>& res__, sddk::mdarray<float, 2> const& h_diag__,
                 sddk::mdarray<float, 2> const& o_diag__, bool estimate_eval__, float norm_tolerance__,
                 std::function<bool(int, int)> is_converged__);

template residual_result
residuals<std::complex<float>, std::complex<float>>(Simulation_context& ctx__, sddk::memory_t mem_type__, sddk::linalg_t la_type__,
                               sddk::spin_range ispn__,
                               int N__, int num_bands__, int num_locked, sddk::mdarray<float, 1>& eval__,
                               sddk::dmatrix<std::complex<float>>& evec__, sddk::Wave_functions<float>& hphi__,
                               sddk::Wave_functions<float>& ophi__, sddk::Wave_functions<float>& hpsi__, sddk::Wave_functions<float>& opsi__,
                               sddk::Wave_functions<float>& res__, sddk::mdarray<float, 2> const& h_diag__,
                               sddk::mdarray<float, 2> const& o_diag__, bool estimate_eval__, float norm_tolerance__,
                               std::function<bool(int, int)> is_converged__);

template residual_result
residuals<std::complex<float>, std::complex<double>>(Simulation_context& ctx__, sddk::memory_t mem_type__, sddk::linalg_t la_type__,
                               sddk::spin_range ispn__,
                               int N__, int num_bands__, int num_locked, sddk::mdarray<double, 1>& eval__,
                               sddk::dmatrix<std::complex<double>>& evec__, sddk::Wave_functions<float>& hphi__,
                               sddk::Wave_functions<float>& ophi__, sddk::Wave_functions<float>& hpsi__, sddk::Wave_functions<float>& opsi__,
                               sddk::Wave_functions<float>& res__, sddk::mdarray<float, 2> const& h_diag__,
                               sddk::mdarray<float, 2> const& o_diag__, bool estimate_eval__, float norm_tolerance__,
                               std::function<bool(int, int)> is_converged__);
#endif
} // namespace
