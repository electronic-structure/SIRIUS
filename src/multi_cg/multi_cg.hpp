/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file multi_cg.hpp
 *
 *  \brief Linear response functionality.
 */

#ifndef __MULTI_CG_HPP__
#define __MULTI_CG_HPP__

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <complex>
#include "core/wf/wave_functions.hpp"
#include "hamiltonian/residuals.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "hamiltonian/non_local_operator.hpp"
#include "k_point/k_point.hpp"

namespace sirius {
/// Conjugate-gradient solver.
namespace cg {

template <class T>
void
repack(std::vector<T>& data, std::vector<int> const& ids)
{
    for (size_t i = 0; i < ids.size(); ++i) {
        data[i] = data[ids[i]];
    }
}

template <typename Matrix, typename Prec, typename StateVec>
auto
multi_cg(Matrix& A, Prec& P, StateVec& X, StateVec& B, StateVec& U, StateVec& C, int maxiters = 10, double tol = 1e-3,
         bool initial_guess_is_zero = false)
{

    PROFILE("sirius::multi_cg");

    auto const n = X.cols();

    U.zero();

    // Use R for residual, we modify the right-hand side B in-place.
    auto& R = B;

    // Use B effectively as the residual block-vector
    // R = B - A * X -- don't multiply when initial guess is zero.
    if (!initial_guess_is_zero) {
        A.multiply(-1.0, X, 1.0, R, n);
    }

    auto rhos     = std::vector<typename StateVec::value_type>(n);
    auto rhos_old = rhos;
    auto sigmas   = rhos;
    auto alphas   = rhos;

    // When vectors converge we move them to the front, but we can't really do
    // that with X, so we have to keep track of where is what.
    auto ids = std::vector<int>(n);
    std::iota(ids.begin(), ids.end(), 0);

    size_t num_unconverged = n;

    double niter_eff{0};

    auto residual_history = std::vector<std::vector<typename StateVec::value_type>>(n);
    int niter{0};
    for (int iter = 0; iter < maxiters; ++iter) {
        niter = iter;
        // Check the residual norms in the P-norm
        // that means whenever P is approximately inv(A)
        // since (r, Pr) = (Ae, PAe) ~= (e, Ae)
        // we check the errors roughly in the A-norm.
        // When P = I, we just check the residual norm.

        // C = P * R.
        P.apply(C, R);

        rhos_old = rhos;

        // rhos = dot(C, R)
        C.block_dot(R, rhos, num_unconverged);

        for (size_t i = 0; i < num_unconverged; ++i) {
            residual_history[ids[i]].push_back(std::sqrt(std::abs(rhos[i])));
        }

        auto not_converged = std::vector<int>{};
        for (size_t i = 0; i < num_unconverged; ++i) {
            if (std::abs(rhos[i]) > tol * tol) {
                not_converged.push_back(i);
            }
        }

        num_unconverged = not_converged.size();

        niter_eff += static_cast<double>(num_unconverged) / n;

        if (not_converged.empty()) {
            break;
        }

        // Move everything contiguously to the front,
        // except for X, since that's updated in-place.
        repack(ids, not_converged); // use repack on 1-D vector
        repack(rhos, not_converged);
        repack(rhos_old, not_converged);

        U.repack(not_converged); // use repack from the Wave_functions_wrap
        C.repack(not_converged);
        R.repack(not_converged);

        A.repack(not_converged); // use repack from the Linear_response_operator
        P.repack(not_converged); // use repack from the preconditioner

        /* The repack on A and P changes the eigenvalue vectors of A and P respectively */
        /* The eigenvalues of the Linear_response_operator A are sent to device when needed */
        /* Update P.eigvals on device here */
        if (is_device_memory(P.mem)) {
            P.eigvals.copy_to(memory_t::device);
        }

        // In the first iteration we have U == 0, so no need for an axpy.
        if (iter == 0) {
            U.copy(C, num_unconverged);
        } else {
            for (size_t i = 0; i < num_unconverged; ++i) {
                alphas[i] = rhos[i] / rhos_old[i];
            }

            // U[:, i] = C[:, i] + alpha[i] * U[:, i] for i < num_unconverged
            U.block_xpby(C, alphas, num_unconverged);
        }

        // C = A * U.
        A.multiply(1.0, U, 0.0, C, num_unconverged);

        // compute the optimal distance for the search direction
        // sigmas = dot(U, C)
        U.block_dot(C, sigmas, num_unconverged);

        // Update the solution and the residual
        for (size_t i = 0; i < num_unconverged; ++i) {
            alphas[i] = rhos[i] / sigmas[i];
        }

        // X[:, ids[i]] += alpha[i] * U[:, i]
        X.block_axpy_scatter(alphas, U, ids, num_unconverged);

        for (size_t i = 0; i < num_unconverged; ++i) {
            alphas[i] *= -1;
        }

        // R[:, i] += alpha[i] * C[:, i] for i < num_unconverged
        R.block_axpy(alphas, C, num_unconverged);
    }
    struct
    {
        std::vector<std::vector<typename StateVec::value_type>> residual_history;
        int niter;
        int niter_eff;
    } result{residual_history, niter, static_cast<int>(niter_eff)};
    return result;
}
} // namespace cg

/// Linear respone functions and objects.
namespace lr {

struct Wave_functions_wrap
{
    wf::Wave_functions<double>* x;
    memory_t mem;

    typedef std::complex<double> value_type;

    void
    zero()
    {
        x->zero(mem);
    }

    int
    cols() const
    {
        return x->num_wf().get();
    }

    void
    block_dot(Wave_functions_wrap const& y__, std::vector<value_type>& rhos__, size_t N__)
    {
        rhos__ = wf::inner_diag<double, value_type>(mem, *x, *y__.x, wf::spin_range(0), wf::num_bands(N__));
    }

    void
    repack(std::vector<int> const& ids__)
    {
        PROFILE("sirius::Wave_functions_wrap::repack");
        int j{0};
        for (auto i : ids__) {
            if (j != i) {
                wf::copy(mem, *x, wf::spin_index(0), wf::band_range(i, i + 1), *x, wf::spin_index(0),
                         wf::band_range(j, j + 1));
            }
            ++j;
        }
    }

    void
    copy(Wave_functions_wrap const& y__, size_t N__)
    {
        wf::copy(mem, *y__.x, wf::spin_index(0), wf::band_range(0, N__), *x, wf::spin_index(0), wf::band_range(0, N__));
    }

    void
    block_xpby(Wave_functions_wrap const& y__, std::vector<value_type> const& alphas, int N__)
    {
        std::vector<value_type> ones(N__, 1.0);
        wf::axpby(mem, wf::spin_range(0), wf::band_range(0, N__), ones.data(), y__.x, alphas.data(), x);
    }

    void
    block_axpy_scatter(std::vector<value_type> const& alphas__, Wave_functions_wrap const& y__,
                       std::vector<int> const& idx__, int n__)
    {
        wf::axpy_scatter<double, value_type>(mem, wf::spin_range(0), alphas__.data(), y__.x, idx__.data(), x, n__);
    }

    void
    block_axpy(std::vector<value_type> const& alphas__, Wave_functions_wrap const& y__, int N__)
    {
        std::vector<value_type> ones(N__, 1.0);
        wf::axpby(mem, wf::spin_range(0), wf::band_range(0, N__), alphas__.data(), y__.x, ones.data(), x);
    }
};

struct Identity_preconditioner
{
    size_t num_active;

    void
    apply(Wave_functions_wrap& x, Wave_functions_wrap const& y)
    {
        x.copy(y, num_active);
    }

    void
    repack(std::vector<int> const& ids)
    {
        num_active = ids.size();
    }
};

struct Smoothed_diagonal_preconditioner
{
    mdarray<double, 2> H_diag;
    mdarray<double, 2> S_diag;
    mdarray<double, 1> eigvals;
    int num_active;
    memory_t mem;
    wf::spin_range sr;

    void
    apply(Wave_functions_wrap& x, Wave_functions_wrap const& y)
    {
        // Could avoid a copy here, but apply_precondition is in-place.
        x.copy(y, num_active);
        sirius::apply_preconditioner(mem, sr, wf::num_bands(num_active), *x.x, H_diag, S_diag, eigvals);
    }

    void
    repack(std::vector<int> const& ids)
    {
        num_active = ids.size();
        for (size_t i = 0; i < ids.size(); ++i) {
            eigvals[i] = eigvals[ids[i]];
        }
    }
};

struct Linear_response_operator
{
    sirius::Simulation_context& ctx;
    sirius::Hamiltonian_k<double>& Hk;
    std::vector<double> min_eigenvals;
    wf::Wave_functions<double>* Hphi;
    wf::Wave_functions<double>* Sphi;
    wf::Wave_functions<double>* evq;
    wf::Wave_functions<double>* tmp;
    double alpha_pv;
    wf::band_range br;
    wf::spin_range sr;
    memory_t mem;
    la::dmatrix<std::complex<double>> overlap;

    Linear_response_operator(sirius::Simulation_context& ctx, sirius::Hamiltonian_k<double>& Hk,
                             std::vector<double> const& eigvals, wf::Wave_functions<double>* Hphi,
                             wf::Wave_functions<double>* Sphi, wf::Wave_functions<double>* evq,
                             wf::Wave_functions<double>* tmp, double alpha_pv, wf::band_range br, wf::spin_range sr,
                             memory_t mem)
        : ctx(ctx)
        , Hk(Hk)
        , min_eigenvals(eigvals)
        , Hphi(Hphi)
        , Sphi(Sphi)
        , evq(evq)
        , tmp(tmp)
        , alpha_pv(alpha_pv)
        , br(br)
        , sr(sr)
        , mem(mem)
        , overlap(br.size(), br.size())
    {
        // I think we could just compute alpha_pv here by just making it big enough
        // s.t. the operator H - e * S + alpha_pv * Q is positive, e.g:
        // alpha_pv = 2 * min_eigenvals.back();
        // but QE has a very specific way to compute it, so we just forward it from
        // there.;

        // flip the sign of the eigenvals so that the axpby works
        for (auto& e : min_eigenvals) {
            e *= -1;
        }
    }

    void
    repack(std::vector<int> const& ids)
    {
        for (size_t i = 0; i < ids.size(); ++i) {
            min_eigenvals[i] = min_eigenvals[ids[i]];
        }
    }

    // y[:, i] <- alpha * A * x[:, i] + beta * y[:, i] where A = (H - e_j S + constant   * SQ * SQ')
    // where SQ is S * eigenvectors.
    void
    multiply(double alpha, Wave_functions_wrap x, double beta, Wave_functions_wrap y, int num_active)
    {
        PROFILE("sirius::Linear_response_operator::multiply");
        // Hphi = H * x, Sphi = S * x
        Hk.apply_h_s<std::complex<double>>(sr, wf::band_range(0, num_active), *x.x, Hphi, Sphi);

        std::vector<double> ones(num_active, 1.0);

        // effectively tmp := (H - e * S) * x, as an axpy, modifying Hphi.
        wf::axpby(mem, wf::spin_range(0), wf::band_range(0, num_active), min_eigenvals.data(), Sphi, ones.data(), Hphi);
        wf::copy(mem, *Hphi, wf::spin_index(0), wf::band_range(0, num_active), *tmp, wf::spin_index(0),
                 wf::band_range(0, num_active));

        // Projector, add alpha_pv * (S * (evq * (evq' * (S * x))))

        // overlap := evq' * (S * x)
        wf::inner(ctx.spla_context(), mem, wf::spin_range(0), *evq, br, *Sphi, wf::band_range(0, num_active), overlap,
                  0, 0);

        // Hphi := evq * overlap
        wf::transform(ctx.spla_context(), mem, overlap, 0, 0, 1.0, *evq, wf::spin_index(0), br, 0.0, *Hphi,
                      wf::spin_index(0), wf::band_range(0, num_active));

        Hk.apply_s<std::complex<double>>(wf::spin_range(0), wf::band_range(0, num_active), *Hphi, *Sphi);

        // tmp := alpha_pv * Sphi + tmp = (H - e * S) * x + alpha_pv * (S * (evq * (evq' * (S * x))))
        std::vector<double> alpha_pvs(num_active, alpha_pv);
        wf::axpby(mem, wf::spin_range(0), wf::band_range(0, num_active), alpha_pvs.data(), Sphi, ones.data(), tmp);
        // y[:, i] <- alpha * tmp + beta * y[:, i]
        std::vector<double> alphas(num_active, alpha);
        std::vector<double> betas(num_active, beta);
        wf::axpby(mem, wf::spin_range(0), wf::band_range(0, num_active), alphas.data(), tmp, betas.data(), y.x);
    }
};

} // namespace lr

} // namespace sirius
#endif
