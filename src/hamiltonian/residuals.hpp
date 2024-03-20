/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file residuals.hpp
 *
 *  \brief Compute residuals from the eigen-vectors and basis functions.
 */

#ifndef __RESIDUALS_HPP__
#define __RESIDUALS_HPP__

#include "core/typedefs.hpp"
#include "core/la/linalg.hpp"
#include "core/wf/wave_functions.hpp"
#include "context/simulation_context.hpp"

struct residual_result_t
{
    int num_consecutive_smallest_converged;
    int unconverged_residuals;
    double frobenius_norm;
};

template <typename T>
struct normalized_preconditioned_residuals_result
{
    int num_unconverged;
    std::vector<T> norm;
};

#if defined(SIRIUS_GPU)
// extern "C" void residuals_aux_gpu(int num_gvec_loc__,
//                                   int num_res_local__,
//                                   int* res_idx__,
//                                   double* eval__,
//                                   std::complex<double> const* hpsi__,
//                                   std::complex<double> const* opsi__,
//                                   double const* h_diag__,
//                                   double const* o_diag__,
//                                   std::complex<double>* res__,
//                                   double* res_norm__,
//                                   double* p_norm__,
//                                   int gkvec_reduced__,
//                                   int mpi_rank__);
//
extern "C" {

void
compute_residuals_gpu_double(std::complex<double> const* hpsi__, std::complex<double> const* opsi__,
                             std::complex<double>* res__, int num_gvec_loc__, int num_bands__, double const* eval__);

void
compute_residuals_gpu_float(std::complex<float> const* hpsi__, std::complex<float> const* opsi__,
                            std::complex<float>* res__, int num_gvec_loc__, int num_bands__, float const* eval__);

void
apply_preconditioner_gpu_double(std::complex<double>* res__, int num_rows_loc__, int num_bands__, double const* eval__,
                                const double* h_diag__, const double* o_diag__);

void
apply_preconditioner_gpu_float(std::complex<float>* res__, int num_rows_loc__, int num_bands__, float const* eval__,
                               const float* h_diag__, const float* o_diag__);

void
make_real_g0_gpu_double(std::complex<double>* res__, int ld__, int n__);

void
make_real_g0_gpu_float(std::complex<float>* res__, int ld__, int n__);
}

inline void
compute_residuals_gpu(std::complex<double> const* hpsi__, std::complex<double> const* opsi__,
                      std::complex<double>* res__, int num_gvec_loc__, int num_bands__, double const* eval__)
{
    compute_residuals_gpu_double(hpsi__, opsi__, res__, num_gvec_loc__, num_bands__, eval__);
}

inline void
compute_residuals_gpu(std::complex<float> const* hpsi__, std::complex<float> const* opsi__, std::complex<float>* res__,
                      int num_gvec_loc__, int num_bands__, float const* eval__)
{
    compute_residuals_gpu_float(hpsi__, opsi__, res__, num_gvec_loc__, num_bands__, eval__);
}

inline void
apply_preconditioner_gpu(std::complex<double>* res__, int num_rows_loc__, int num_bands__, double const* eval__,
                         double const* h_diag__, double const* o_diag__)
{
    apply_preconditioner_gpu_double(res__, num_rows_loc__, num_bands__, eval__, h_diag__, o_diag__);
}

inline void
apply_preconditioner_gpu(std::complex<float>* res__, int num_rows_loc__, int num_bands__, float const* eval__,
                         const float* h_diag__, const float* o_diag__)
{
    apply_preconditioner_gpu_float(res__, num_rows_loc__, num_bands__, eval__, h_diag__, o_diag__);
}

inline void
make_real_g0_gpu(std::complex<double>* res__, int ld__, int n__)
{
    make_real_g0_gpu_double(res__, ld__, n__);
}

inline void
make_real_g0_gpu(std::complex<float>* res__, int ld__, int n__)
{
    make_real_g0_gpu_float(res__, ld__, n__);
}
#endif

namespace sirius {

/// Compute band residuals.
/**
 *
 * \tparam T Precision type of the wave-functions (float or double).
 *
 *
 */
template <typename T>
static void
compute_residuals(memory_t mem__, wf::spin_range spins__, wf::num_bands num_bands__, mdarray<T, 1> const& eval__,
                  wf::Wave_functions<T> const& hpsi__, wf::Wave_functions<T> const& opsi__,
                  wf::Wave_functions<T>& res__)
{
    RTE_ASSERT(hpsi__.ld() == opsi__.ld());
    RTE_ASSERT(hpsi__.ld() == res__.ld());
    RTE_ASSERT(hpsi__.num_md() == opsi__.num_md());
    RTE_ASSERT(hpsi__.num_md() == res__.num_md());

    for (auto s = spins__.begin(); s != spins__.end(); s++) {
        auto sp = hpsi__.actual_spin_index(s);
        if (is_host_memory(mem__)) {
            /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
            #pragma omp parallel for
            for (int i = 0; i < num_bands__.get(); i++) {
                auto hpsi_ptr = hpsi__.at(mem__, 0, sp, wf::band_index(i));
                auto opsi_ptr = opsi__.at(mem__, 0, sp, wf::band_index(i));
                auto res_ptr  = res__.at(mem__, 0, sp, wf::band_index(i));

                for (int j = 0; j < hpsi__.ld(); j++) {
                    res_ptr[j] = hpsi_ptr[j] - eval__[i] * opsi_ptr[j];
                }
            }
        } else {
#if defined(SIRIUS_GPU)
            auto hpsi_ptr = hpsi__.at(mem__, 0, sp, wf::band_index(0));
            auto opsi_ptr = opsi__.at(mem__, 0, sp, wf::band_index(0));
            auto res_ptr  = res__.at(mem__, 0, sp, wf::band_index(0));
            compute_residuals_gpu(hpsi_ptr, opsi_ptr, res_ptr, res__.ld(), num_bands__.get(), eval__.at(mem__));
#endif
        }
    }
}

/// Apply preconditioner to the residuals.
template <typename T>
void
apply_preconditioner(memory_t mem__, wf::spin_range spins__, wf::num_bands num_bands__, wf::Wave_functions<T>& res__,
                     mdarray<T, 2> const& h_diag__, mdarray<T, 2> const& o_diag__, mdarray<T, 1> const& eval__)
{
    PROFILE("sirius::apply_preconditioner");
    for (auto s = spins__.begin(); s != spins__.end(); s++) {
        auto sp = res__.actual_spin_index(s);
        if (is_host_memory(mem__)) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_bands__.get(); i++) {
                auto res_ptr = res__.at(mem__, 0, sp, wf::band_index(i));
                for (int j = 0; j < res__.ld(); j++) {
                    T p = h_diag__(j, s.get()) - o_diag__(j, s.get()) * eval__[i];
                    p   = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                    res_ptr[j] /= p;
                }
            }
        } else {
#if defined(SIRIUS_GPU)
            apply_preconditioner_gpu(res__.at(mem__, 0, sp, wf::band_index(0)), res__.ld(), num_bands__.get(),
                                     eval__.at(mem__), h_diag__.at(mem__, 0, s.get()), o_diag__.at(mem__, 0, s.get()));
#endif
        }
    }
}

template <typename T, typename F>
static auto
normalized_preconditioned_residuals(memory_t mem__, wf::spin_range spins__, wf::num_bands num_bands__,
                                    mdarray<T, 1> const& eval__, wf::Wave_functions<T> const& hpsi__,
                                    wf::Wave_functions<T> const& opsi__, wf::Wave_functions<T>& res__,
                                    mdarray<T, 2> const& h_diag__, mdarray<T, 2> const& o_diag__, T norm_tolerance__,
                                    bool gamma__)
{
    PROFILE("sirius::normalized_preconditioned_residuals");

    RTE_ASSERT(num_bands__.get() != 0);

    normalized_preconditioned_residuals_result<real_type<F>> result;
    result.norm = std::vector<real_type<F>>(num_bands__.get());

    /* compute "raw" residuals */
    compute_residuals<T>(mem__, spins__, num_bands__, eval__, hpsi__, opsi__, res__);

    /* compute norm of the "raw" residuals; if norm is small, residuals are close to convergence */
    auto res_norm = wf::inner_diag<T, F>(mem__, res__, res__, spins__, num_bands__);
    for (int i = 0; i < num_bands__.get(); i++) {
        result.norm[i] = std::sqrt(std::real(res_norm[i]));
    }

    /* apply preconditioner */
    apply_preconditioner<T>(mem__, spins__, num_bands__, res__, h_diag__, o_diag__, eval__);

    /* this not strictly necessary as the wave-function orthoronormalization can take care of this;
       however, normalization of residuals is harmless and gives a better numerical stability */
    res_norm = wf::inner_diag<T, F>(mem__, res__, res__, spins__, num_bands__);
    std::vector<real_type<F>> norm1;
    for (auto e : res_norm) {
        norm1.push_back(1.0 / std::sqrt(std::real(e)));
    }
    wf::axpby<T, real_type<F>>(mem__, spins__, wf::band_range(0, num_bands__.get()), nullptr, nullptr, norm1.data(),
                               &res__);

    int n{0};
    for (int i = 0; i < num_bands__.get(); i++) {
        /* take the residual if it's norm is above the threshold */
        if (result.norm[i] > norm_tolerance__) {
            /* shift unconverged residuals to the beginning of array */
            /* note: we can just keep them where they were  */
            if (n != i) {
                for (auto s = spins__.begin(); s != spins__.end(); s++) {
                    auto sp = res__.actual_spin_index(s);
                    wf::copy(mem__, res__, sp, wf::band_range(i, i + 1), res__, sp, wf::band_range(n, n + 1));
                }
            }
            n++;
        }
    }
    result.num_unconverged = n;

    /* prevent numerical noise */
    /* this only happens for real wave-functions (Gamma-point case), non-magnetic or collinear magnetic */
    if (gamma__ && res__.comm().rank() == 0 && n != 0) {
        RTE_ASSERT(spins__.begin().get() + 1 == spins__.end().get());
        if (is_device_memory(mem__)) {
#if defined(SIRIUS_GPU)
            make_real_g0_gpu(res__.at(mem__, 0, spins__.begin(), wf::band_index(0)), res__.ld(), n);
#endif
        } else {
            for (int i = 0; i < n; i++) {
                res__.pw_coeffs(0, spins__.begin(), wf::band_index(i)) =
                        res__.pw_coeffs(0, spins__.begin(), wf::band_index(i)).real();
            }
        }
    }

    return result;
}

/// Compute residuals from eigen-vectors.
/** \tparam T Precision type of the wave-functions (float or double).
    \tparam F Type of the subspace (float or double for Gamma-point calculation,
              complex<float> or complex<double> otherwise.

    The residuals of wave-functions are defined as:
    \f[
      R_{i} = \hat H \psi_{i} - \epsilon_{i} \hat S \psi_{i}
    \f]
 */
template <typename T, typename F>
auto
residuals(Simulation_context& ctx__, memory_t mem__, wf::spin_range sr__, int N__, int num_bands__, int num_locked__,
          mdarray<real_type<F>, 1>& eval__, la::dmatrix<F>& evec__, wf::Wave_functions<T>& hphi__,
          wf::Wave_functions<T>& ophi__, wf::Wave_functions<T>& hpsi__, wf::Wave_functions<T>& opsi__,
          wf::Wave_functions<T>& res__, mdarray<T, 2> const& h_diag__, mdarray<T, 2> const& o_diag__,
          bool estimate_eval__, T norm_tolerance__, std::function<bool(int, int)> is_converged__)
{
    PROFILE("sirius::residuals");

    RTE_ASSERT(N__ != 0);
    RTE_ASSERT(hphi__.num_sc() == hpsi__.num_sc());
    RTE_ASSERT(ophi__.num_sc() == opsi__.num_sc());

    la::dmatrix<F> evec_tmp;

    mdarray<T, 1> eval({num_bands__});
    eval = [&](size_t j) -> T { return eval__[j]; };

    la::dmatrix<F>* evec_ptr{nullptr};

    /* total number of residuals to be computed */
    int num_residuals{0};

    /* number of lockable eigenvectors */
    int num_consecutive_converged{0};

    /* when estimate_eval is set we only compute true residuals of unconverged eigenpairs
       where convergence is determined just on the change in the eigenvalues. */
    if (estimate_eval__) {
        /* Locking is only based on the "is_converged" criterion, not on the actual
           residual norms. We could lock more by considering the residual norm criterion
           later, but since we're reordering eigenvectors too, this becomes messy. */
        while (num_consecutive_converged < num_bands__ &&
               is_converged__(num_consecutive_converged, sr__.spinor_index())) {
            ++num_consecutive_converged;
        }

        /* collect indices of unconverged eigenpairs */
        std::vector<int> ev_idx;
        for (int j = 0; j < num_bands__; j++) {
            if (!is_converged__(j, sr__.spinor_index())) {
                ev_idx.push_back(j);
            }
        }

        /* if everything is converged, return early */
        if (ev_idx.empty()) {
            return residual_result_t{num_bands__, 0, 0};
        }

        // Otherwise copy / reorder the unconverged eigenpairs
        num_residuals = static_cast<int>(ev_idx.size());

        evec_tmp = la::dmatrix<F>(N__, num_residuals, evec__.blacs_grid(), evec__.bs_row(), evec__.bs_col());
        evec_ptr = &evec_tmp;

        int num_rows_local = evec_tmp.num_rows_local();
        for (int j = 0; j < num_residuals; j++) {
            eval[j] = eval[ev_idx[j]];
            if (evec__.blacs_grid().comm().size() == 1) {
                /* do a local copy */
                std::copy(&evec__(0, ev_idx[j]), &evec__(0, ev_idx[j]) + num_rows_local, &evec_tmp(0, j));
            } else {
                auto pos_src  = evec__.spl_col().location(ev_idx[j]);
                auto pos_dest = evec_tmp.spl_col().location(j);
                /* do MPI send / receive */
                if (pos_src.ib == evec__.blacs_grid().comm_col().rank() && num_rows_local) {
                    evec__.blacs_grid().comm_col().isend(&evec__(0, pos_src.index_local), num_rows_local, pos_dest.ib,
                                                         ev_idx[j]);
                }
                if (pos_dest.ib == evec__.blacs_grid().comm_col().rank() && num_rows_local) {
                    evec__.blacs_grid().comm_col().recv(&evec_tmp(0, pos_dest.index_local), num_rows_local, pos_src.ib,
                                                        ev_idx[j]);
                }
            }
        }
        if (is_device_memory(mem__) && evec_tmp.blacs_grid().comm().size() == 1) {
            evec_tmp.allocate(memory_t::device);
        }
    } else {
        evec_ptr      = &evec__;
        num_residuals = num_bands__;
    }
    if (is_device_memory(mem__)) {
        eval.allocate(memory_t::device).copy_to(memory_t::device);
    }

    for (auto s = sr__.begin(); s != sr__.end(); s++) {
        auto sp = hphi__.actual_spin_index(s);

        /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} */
        wf::transform<T, F>(ctx__.spla_context(), mem__, *evec_ptr, 0, 0, 1.0, hphi__, sp,
                            wf::band_range(num_locked__, N__), 0.0, hpsi__, sp, wf::band_range(0, num_residuals));

        sp = ophi__.actual_spin_index(s);
        /* compute O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
        wf::transform<T, F>(ctx__.spla_context(), mem__, *evec_ptr, 0, 0, 1.0, ophi__, sp,
                            wf::band_range(num_locked__, N__), 0.0, opsi__, sp, wf::band_range(0, num_residuals));
    }

    auto result = normalized_preconditioned_residuals<T, F>(mem__, sr__, wf::num_bands(num_residuals), eval, hpsi__,
                                                            opsi__, res__, h_diag__, o_diag__, norm_tolerance__,
                                                            std::is_same<F, real_type<F>>::value);

    // In case we're not using the delta in eigenvalues as a convergence criterion,
    // we lock eigenpairs using residual norms.
    if (!estimate_eval__) {
        while (num_consecutive_converged < num_residuals &&
               result.norm[num_consecutive_converged] <= norm_tolerance__) {
            num_consecutive_converged++;
        }
    }

    auto frobenius_norm = 0.0;
    for (int i = 0; i < num_residuals; i++) {
        frobenius_norm += result.norm[i] * result.norm[i];
    }
    frobenius_norm = std::sqrt(frobenius_norm);
    return residual_result_t{num_consecutive_converged, result.num_unconverged, frobenius_norm};
}

} // namespace sirius

#endif
