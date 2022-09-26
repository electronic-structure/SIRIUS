// Copyright (c) 2018-2022 Harmen Stoppels, Anton Kozhevnikov
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
#include "SDDK/wave_functions.hpp"
#include "SDDK/wf_inner.hpp"
#include "SDDK/wf_trans.hpp"
#include "band/residuals.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "hamiltonian/non_local_operator.hpp"
#include "k_point/k_point.hpp"

namespace sirius {
namespace cg {

template <class T>
void repack(std::vector<T> &data, std::vector<size_t> const&ids) {
    for (size_t i = 0; i < ids.size(); ++i) {
        data[i] = data[ids[i]];
    }
}

template<class Matrix, class Prec, class StateVec>
std::vector<std::vector<typename StateVec::value_type>> multi_cg(
    Matrix &A, Prec &P, StateVec &X, StateVec &B, StateVec &U, StateVec &C,
    size_t maxiters = 10, double tol = 1e-3, bool initial_guess_is_zero = false
) {
    auto n = X.cols();

    U.fill(0);

    // Use R for residual, we modify the right-hand side B in-place.
    auto &R = B;

    // Use B effectively as the residual block-vector
    // R = B - A * X -- don't multiply when initial guess is zero.
    if (!initial_guess_is_zero)
        A.multiply(-1.0, X, 1.0, R, n);

    auto rhos = std::vector<typename StateVec::value_type>(n);
    auto rhos_old = rhos;
    auto sigmas = rhos;
    auto alphas = rhos;

    // When vectors converge we move them to the front, but we can't really do
    // that with X, so we have to keep track of where is what.
    auto ids = std::vector<size_t>(n);
    std::iota(ids.begin(), ids.end(), 0);

    size_t num_unconverged = n;

    auto residual_history = std::vector<std::vector<typename StateVec::value_type>>(n);

    for (size_t iter = 0; iter < maxiters; ++iter) {
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

        auto not_converged = std::vector<size_t>{};
        for (size_t i = 0; i < num_unconverged; ++i) {
            if (std::abs(rhos[i]) > tol * tol) {
                not_converged.push_back(i);
            }
        }

        num_unconverged = not_converged.size();

        if (not_converged.empty()) {
            break;
        }

        // Move everything contiguously to the front,
        // except for X, since that's updated in-place.
        repack(ids, not_converged);
        repack(rhos, not_converged);
        repack(rhos_old, not_converged);

        U.repack(not_converged);
        C.repack(not_converged);
        R.repack(not_converged);

        A.repack(not_converged);
        P.repack(not_converged);

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

        R.block_axpy(alphas, C, num_unconverged);
    }

    return residual_history;
}
}

namespace lr {

struct Wave_functions_wrap {
    sddk::Wave_functions<double> *x;

    typedef double_complex value_type;

    void fill(double_complex val) {
        x->pw_coeffs(0).prime() = [=](){
            return val;
        };
    }

    int cols() const {
        return x->num_wf();
    }

    void block_dot(Wave_functions_wrap const &y, std::vector<double_complex> &rhos, size_t num_unconverged) {
        auto result = x->dot(sddk::device_t::CPU, sddk::spin_range(0), *y.x, static_cast<int>(num_unconverged));
        for (int i = 0; i < static_cast<int>(num_unconverged); ++i)
            rhos[i] = result(i);
    }

    void repack(std::vector<size_t> const &ids) {
        size_t j = 0;
        for (auto i : ids) {
            if (j != i) {
                x->copy_from(*x, 1, 0, i, 0, j);
            }

            ++j;
        }
    }

    void copy(Wave_functions_wrap const &y, size_t num) {
        x->copy_from(*y.x, static_cast<int>(num), 0, 0, 0, 0);
    }

    void block_xpby(Wave_functions_wrap const &y, std::vector<double_complex> const &alphas, size_t num) {
        x->xpby(sddk::device_t::CPU, sddk::spin_range(0), *y.x, alphas, static_cast<int>(num));
    }

    void block_axpy_scatter(std::vector<double_complex> const &alphas, Wave_functions_wrap const &y, std::vector<size_t> const &ids, size_t num) {
        x->axpy_scatter(sddk::device_t::CPU, sddk::spin_range(0), alphas, *y.x, ids, static_cast<int>(num));
    }

    void block_axpy(std::vector<double_complex> const &alphas, Wave_functions_wrap const &y, size_t num) {
        x->axpy(sddk::device_t::CPU, sddk::spin_range(0), alphas, *y.x, static_cast<int>(num));
    }
};

struct Identity_preconditioner {
    size_t num_active;

    void apply(Wave_functions_wrap &x, Wave_functions_wrap const &y) {
        x.copy(y, num_active);
    }

    void repack(std::vector<size_t> const &ids) {
        num_active = ids.size();
    }
};

struct Smoothed_diagonal_preconditioner {
    sddk::mdarray<double, 2> H_diag;
    sddk::mdarray<double, 2> S_diag;
    sddk::mdarray<double, 1> eigvals;
    int num_active;

    void apply(Wave_functions_wrap &x, Wave_functions_wrap const &y) {
        // Could avoid a copy here, but apply_precondition is in-place.
        x.copy(y, num_active);
        sirius::apply_preconditioner(
            sddk::memory_t::host,
            sddk::spin_range(0),
            static_cast<int>(num_active),
            *x.x,
            H_diag,
            S_diag,
            eigvals);
    }

    void repack(std::vector<size_t> const &ids) {
        num_active = ids.size();
        for (size_t i = 0; i < ids.size(); ++i) {
            eigvals[i] = eigvals[ids[i]];
        }
    }
};


struct Linear_response_operator {
    sirius::Simulation_context &ctx;
    sirius::Hamiltonian_k<double> &Hk;
    std::vector<double> min_eigenvals;
    sddk::Wave_functions<double> * Hphi;
    sddk::Wave_functions<double> * Sphi;
    sddk::Wave_functions<double> * evq;
    sddk::Wave_functions<double> * tmp;
    double alpha_pv;
    sddk::dmatrix<double_complex> overlap;


    Linear_response_operator(
        sirius::Simulation_context &ctx,
        sirius::Hamiltonian_k<double> & Hk,
        std::vector<double> const &eigvals,
        sddk::Wave_functions<double> * Hphi,
        sddk::Wave_functions<double> * Sphi,
        sddk::Wave_functions<double> * evq,
        sddk::Wave_functions<double> * tmp,
        double alpha_pv)
    : ctx(ctx), Hk(Hk), min_eigenvals(eigvals), Hphi(Hphi), Sphi(Sphi), evq(evq), tmp(tmp),
      alpha_pv(alpha_pv), overlap(ctx.num_bands(), ctx.num_bands())
    {
        // I think we could just compute alpha_pv here by just making it big enough
        // s.t. the operator H - e * S + alpha_pv * Q is positive, e.g:
        // alpha_pv = 2 * min_eigenvals.back();
        // but QE has a very specific way to compute it, so we just forward it from
        // there.;

        // flip the sign of the eigenvals so that the axpby works
        for (auto &e : min_eigenvals) {
            e *= -1;
        }
    }

    void repack(std::vector<size_t> const &ids) {
        for (size_t i = 0; i < ids.size(); ++i) {
            min_eigenvals[i] = min_eigenvals[ids[i]];
        }
    }

    // y[:, i] <- alpha * A * x[:, i] + beta * y[:, i] where A = (H - e_j S + constant   * SQ * SQ')
    // where SQ is S * eigenvectors.
    void multiply(double alpha, Wave_functions_wrap x, double beta, Wave_functions_wrap y, int num_active) {
        // Hphi = H * x, Sphi = S * x
        Hk.apply_h_s<double_complex>(
            sddk::spin_range(0),
            0,
            num_active,
            *x.x,
            Hphi,
            Sphi
        );

        // effectively tmp := (H - e * S) * x, as an axpy, modifying Hphi.
        Hphi->axpy(sddk::device_t::CPU, sddk::spin_range(0), min_eigenvals, *Sphi, num_active);
        tmp->copy_from(*Hphi, num_active, 0, 0, 0, 0);

        // Projector, add alpha_pv * (S * (evq * (evq' * (S * x))))

        // overlap := evq' * (S * x)
        sddk::inner(
            ctx.spla_context(),
            sddk::spin_range(0),
            *evq, 0, ctx.num_bands(),
            *Sphi, 0, num_active,
            overlap, 0, 0);

        // Hphi := evq * overlap
        sddk::transform<double_complex>(
            ctx.spla_context(),
            0, 1.0,
            {evq}, 0, ctx.num_bands(),
            overlap, 0, 0,
            0.0, {Hphi}, 0, num_active);

        // Sphi := S * Hphi = S * (evq * (evq' * (S * x)))
        // sirius::apply_S_operator<double_complex>(
        //     ctx.processing_unit(),
        //     sddk::spin_range(0),
        //     0,
        //     num_active,
        //     Hk.kp().beta_projectors(),
        //     *Hphi,
        //     &Hk.H0().Q(),
        //     *Sphi);

        // tmp := alpha_pv * Sphi + tmp = (H - e * S) * x + alpha_pv * (S * (evq * (evq' * (S * x))))
        {
            // okay, the block-scalar api is a bit awkward...
            std::vector<double_complex> alpha_pvs(num_active, alpha_pv);
            tmp->axpy(sddk::device_t::CPU, sddk::spin_range(0), alpha_pvs, *Sphi, num_active);
        }
        // y[:, i] <- alpha * tmp + beta * y[:, i]
        y.x->axpby(sddk::device_t::CPU, sddk::spin_range(0), alpha, *tmp, beta, num_active);
    }
};

}

}
#endif
