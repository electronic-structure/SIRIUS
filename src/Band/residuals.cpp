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

namespace sirius {

static void
compute_residuals(sddk::memory_t mem_type__, sddk::spin_range spins__, int num_bands__, sddk::mdarray<double, 1>& eval__,
                  sddk::Wave_functions& hpsi__, sddk::Wave_functions& opsi__, sddk::Wave_functions& res__)
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
#if defined(__GPU)
            compute_residuals_gpu(hpsi__.pw_coeffs(ispn).prime().at(memory_t::device),
                                  opsi__.pw_coeffs(ispn).prime().at(memory_t::device),
                                  res__.pw_coeffs(ispn).prime().at(memory_t::device),
                                  res__.pw_coeffs(ispn).num_rows_loc(),
                                  num_bands__,
                                  eval__.at(memory_t::device));
            if (res__.has_mt()) {
                compute_residuals_gpu(hpsi__.mt_coeffs(ispn).prime().at(memory_t::device),
                                      opsi__.mt_coeffs(ispn).prime().at(memory_t::device),
                                      res__.mt_coeffs(ispn).prime().at(memory_t::device),
                                      res__.mt_coeffs(ispn).num_rows_loc(),
                                      num_bands__,
                                      eval__.at(memory_t::device));
            }
#endif
        }
    }
}

/// Apply preconditioner to the residuals.
static void
apply_preconditioner(memory_t mem_type__, spin_range spins__, int num_bands__, Wave_functions& res__,
                     mdarray<double, 2> const& h_diag__, mdarray<double, 2> const& o_diag__, mdarray<double, 1>& eval__)
{
    for (int ispn: spins__) {
        if (is_host_memory(mem_type__)) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_bands__; i++) {
                for (int ig = 0; ig < res__.pw_coeffs(ispn).num_rows_loc(); ig++) {
                    double p = h_diag__(ig, ispn) - o_diag__(ig, ispn) * eval__[i];
                    p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                    res__.pw_coeffs(ispn).prime(ig, i) /= p;
                }
                if (res__.has_mt()) {
                    for (int j = 0; j < res__.mt_coeffs(ispn).num_rows_loc(); j++) {
                        double p = h_diag__(res__.pw_coeffs(ispn).num_rows_loc() + j, ispn) - 
                                   o_diag__(res__.pw_coeffs(ispn).num_rows_loc() + j, ispn) * eval__[i];
                        p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                        res__.mt_coeffs(ispn).prime(j, i) /= p;
                    }
                }
            }
        } else {
#if defined(__GPU)
            apply_preconditioner_gpu(res__.pw_coeffs(ispn).prime().at(memory_t::device),
                                     res__.pw_coeffs(ispn).num_rows_loc(),
                                     num_bands__,
                                     eval__.at(memory_t::device),
                                     h_diag__.at(memory_t::device, 0, ispn),
                                     o_diag__.at(memory_t::device, 0, ispn));
            if (res__.has_mt()) {
                apply_preconditioner_gpu(res__.mt_coeffs(ispn).prime().at(memory_t::device),
                                         res__.mt_coeffs(ispn).num_rows_loc(),
                                         num_bands__,
                                         eval__.at(memory_t::device),
                                         h_diag__.at(memory_t::device, res__.pw_coeffs(ispn).num_rows_loc(), ispn),
                                         o_diag__.at(memory_t::device, res__.pw_coeffs(ispn).num_rows_loc(), ispn));
            }
#endif
        }
    }
}

template <typename T>
static inline int
normalized_preconditioned_residuals(memory_t mem_type__, spin_range spins__, int num_bands__, mdarray<double,1>& eval__,
                                    Wave_functions& hpsi__, Wave_functions& opsi__, Wave_functions& res__,
                                    mdarray<double, 2> const& h_diag__, mdarray<double, 2> const& o_diag__,
                                    double norm_tolerance__)
{
    PROFILE("sirius::normalized_preconditioned_residuals");

    assert(num_bands__ != 0);

    auto pu = get_device_t(mem_type__);

    /* compute "raw" residuals */
    compute_residuals(mem_type__, spins__, num_bands__, eval__, hpsi__, opsi__, res__);

    /* compute norm of the "raw" residuals */
    auto res_norm = res__.l2norm(pu, spins__, num_bands__);

    /* apply preconditioner */
    apply_preconditioner(mem_type__, spins__, num_bands__, res__, h_diag__, o_diag__, eval__);

    /* this not strictly necessary as the wave-function orthoronormalization can take care of this;
       however, normalization of residuals is harmless and gives a better numerical stability */
    res__.normalize(pu, spins__, num_bands__);

    int n{0};
    for (int i = 0; i < num_bands__; i++) {
        /* take the residual if it's norm is above the threshold */
        if (res_norm[i] > norm_tolerance__) {
            /* shift unconverged residuals to the beginning of array */
            if (n != i) {
                for (int ispn: spins__) {
                    res__.copy_from(res__, 1, ispn, i, ispn, n);
                }
            }
            n++;
        }
    }

    /* prevent numerical noise */
    /* this only happens for real wave-functions (Gamma-point case), non-magnetic or collinear magnetic */
    if (std::is_same<T, double>::value && res__.comm().rank() == 0 && n != 0 && spins__() != 2) {
        if (is_device_memory(res__.preferred_memory_t())) {
#if defined(__GPU)
            make_real_g0_gpu(res__.pw_coeffs(spins__()).prime().at(memory_t::device), res__.pw_coeffs(spins__()).prime().ld(), n);
#endif
        } else {
            for (int i = 0; i < n; i++) {
                res__.pw_coeffs(spins__()).prime(0, i) = res__.pw_coeffs(spins__()).prime(0, i).real();
            }
        }
    }

    return n;
}

/// Compute residuals from eigen-vectors.
template <typename T>
int
residuals(memory_t mem_type__, linalg_t la_type__, int ispn__, int N__, int num_bands__, mdarray<double, 1>& eval__,
          dmatrix<T>& evec__, Wave_functions& hphi__, Wave_functions& ophi__, Wave_functions& hpsi__,
          Wave_functions& opsi__, Wave_functions& res__, mdarray<double, 2> const& h_diag__,
          mdarray<double, 2> const& o_diag__, bool estimate_eval__, double norm_tolerance__,
          std::function<bool(int, int)> is_converged__)
{
    PROFILE("sirius::residuals");

    assert(N__ != 0);

    mdarray<double, 1> res_norm;
    dmatrix<T> evec_tmp;
    mdarray<double, 1> eval_tmp;

    dmatrix<T>* evec_ptr{nullptr};
    mdarray<double, 1>* eval_ptr{nullptr};

    int n{0};
    if (estimate_eval__) {
        std::vector<int> ev_idx;
        for (int j = 0; j < num_bands__; j++) {
            if (!is_converged__(j, ispn__)) {
                ev_idx.push_back(j);
            }
        }

        n = static_cast<int>(ev_idx.size());

        if (n) {
            eval_tmp = mdarray<double, 1>(n);
            eval_ptr = &eval_tmp;
            evec_tmp = dmatrix<T>(N__, n, evec__.blacs_grid(), evec__.bs_row(), evec__.bs_col());
            evec_ptr = &evec_tmp;

            int num_rows_local = evec_tmp.num_rows_local();
            for (int j = 0; j < n; j++) {
                eval_tmp[j] = eval__[ev_idx[j]];
                if (evec__.blacs_grid().comm().size() == 1) {
                    /* do a local copy */
                    std::copy(&evec__(0, ev_idx[j]), &evec__(0, ev_idx[j]) + num_rows_local, &evec_tmp(0, j));
                } else {
                    auto pos_src  = evec__.spl_col().location(ev_idx[j]);
                    auto pos_dest = evec_tmp.spl_col().location(j);
                    /* do MPI send / recieve */
                    if (pos_src.rank == evec__.blacs_grid().comm_col().rank()) {
                        evec__.blacs_grid().comm_col().isend(&evec__(0, pos_src.local_index), num_rows_local, pos_dest.rank, ev_idx[j]);
                    }
                    if (pos_dest.rank == evec__.blacs_grid().comm_col().rank()) {
                       evec__.blacs_grid().comm_col().recv(&evec_tmp(0, pos_dest.local_index), num_rows_local, pos_src.rank, ev_idx[j]);
                    }
                }
            }
            if (is_device_memory(mem_type__) && evec_tmp.blacs_grid().comm().size() == 1) {
                evec_tmp.allocate(memory_t::device);
            }
            if (is_device_memory(mem_type__)) {
                eval_tmp.allocate(memory_t::device).copy_to(memory_t::device);
            }
        }
    } else { /* compute all residuals first */
        if (is_device_memory(mem_type__)) {
            eval__.allocate(memory_t::device).copy_to(memory_t::device);
        }
        evec_ptr = &evec__;
        eval_ptr = &eval__;
        n = num_bands__;
    }
    if (!n) {
        return 0;
    }

    /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} and O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
    transform<T>(mem_type__, la_type__, ispn__, {&hphi__, &ophi__}, 0, N__, *evec_ptr, 0, 0, {&hpsi__, &opsi__}, 0, n);

    n = normalized_preconditioned_residuals<T>(mem_type__, spin_range(ispn__), n, *eval_ptr, hpsi__, opsi__, res__,
                                               h_diag__, o_diag__, norm_tolerance__);

    return n;
}

template
int
residuals<double>(memory_t mem_type__, linalg_t la_type__, int ispn__, int N__, int num_bands__, mdarray<double, 1>& eval__,
                  dmatrix<double>& evec__, Wave_functions& hphi__, Wave_functions& ophi__, Wave_functions& hpsi__,
                  Wave_functions& opsi__, Wave_functions& res__, mdarray<double, 2> const& h_diag__,
                  mdarray<double, 2> const& o_diag__, bool estimate_eval__, double norm_tolerance__,
                  std::function<bool(int, int)> is_converged__);

template
int
residuals<double_complex>(memory_t mem_type__, linalg_t la_type__, int ispn__, int N__, int num_bands__, mdarray<double, 1>& eval__,
                          dmatrix<double_complex>& evec__, Wave_functions& hphi__, Wave_functions& ophi__, Wave_functions& hpsi__,
                          Wave_functions& opsi__, Wave_functions& res__, mdarray<double, 2> const& h_diag__,
                          mdarray<double, 2> const& o_diag__, bool estimate_eval__, double norm_tolerance__,
                          std::function<bool(int, int)> is_converged__);

} // namespace
