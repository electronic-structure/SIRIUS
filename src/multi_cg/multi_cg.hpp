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

#if defined(SIRIUS_GPU)
extern "C" {

void
apply_preconditioner_gpu_complex_double(std::complex<double>* res__, int num_rows_loc__, int num_bands__,
                                        const double* eval__, const double* h_diag__, const double* o_diag__,
                                        double omega_real, double omega_imag);

void
conjugate_gpu_complex_double(std::complex<double>* ptr__, int ld__, int num_wf__);
}
#endif

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

    bool is_herm = A.is_hermitian();

    auto const n = X.cols();

    U.zero();

    auto U1 = is_herm ? U : U.deep_copy();

    // Use R for residual, we modify the right-hand side B in-place.
    auto& R = B;

    // Use B effectively as the residual block-vector
    // R = B - A * X -- don't multiply when initial guess is zero.
    if (!initial_guess_is_zero) {
        A.multiply(-1.0, X, 1.0, R, n);
    }

    auto R1 = is_herm ? R : R.deep_copy_conj();

    auto C1 = is_herm ? C : C.deep_copy();

    auto rhos     = std::vector<typename StateVec::value_type>(n);
    auto rhos_old = rhos;
    auto sigmas   = rhos;
    auto alphas   = rhos;
    auto alphas1  = rhos;

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
        // BiCG C1 = P^+ * R1
        if (!is_herm) {
            P.apply(C1, R1, true);
        }

	//std::cout << "Preconditioner applied" << std::endl;

        rhos_old = rhos;

        // CG rhos = dot(C, R) -> <R | P | R>
        // BiCG rhos = dot(C1, R) -> <R1 | P | R>
        if (is_herm) {
            C.block_dot(R, rhos, num_unconverged);
        } else {
            C1.block_dot(R, rhos, num_unconverged);
        }

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
        if (!is_herm) {
            U1.repack(not_converged);
            C1.repack(not_converged);
            R1.repack(not_converged);
        }

        A.repack(not_converged); // use repack from the Linear_response_operator
        P.repack(not_converged); // use repack from the preconditioner

        /* The repack on A and P changes the eigenvalue vectors of A and P respectively */
        /* The eigenvalues of the Linear_response_operator A are sent to device when needed */
        /* Update P.eigvals on device here */
        if (is_device_memory(P.mem)) {
            P.eigvals.copy_to(memory_t::device);
        }

	//std::cout << "Finished check convergence" << std::endl;


        // In the first iteration we have U == 0, so no need for an axpy.
        if (iter == 0) {
            U.copy(C, num_unconverged);
            if (!is_herm) {
                U1.copy(C1, num_unconverged);
            }
        } else {
            for (size_t i = 0; i < num_unconverged; ++i) {
                alphas[i] = rhos[i] / rhos_old[i];
                if (!is_herm) {
                    alphas1[i] = sirius::conj(alphas[i]);
                }
            }

            // U[:, i] = C[:, i] + alpha[i] * U[:, i] for i < num_unconverged
            U.block_xpby(C, alphas, num_unconverged);
            // BiCG U1[:, i] = C1[:, i] + alpha1[i] * U1[:, i] for i < num_unconverged
            if (!is_herm) {
                U1.block_xpby(C1, alphas1, num_unconverged);
            }
        }


	//std::cout << "Defined U" << std::endl;


        // C = A * U.
        A.multiply(1.0, U, 0.0, C, num_unconverged);
        // BiCG C1 = A^+ * U1
        if (!is_herm) {
            A.multiply(1.0, U1, 0.0, C1, num_unconverged, true);
	}

	//std::cout << "Applied A" << std::endl;


        // compute the optimal distance for the search direction
        // sigmas = dot(U, C)
        // C = A * U, then sigma = U * A * U
        // U is a search direction
        if (is_herm) {
            U.block_dot(C, sigmas, num_unconverged);
        } else {
            // BiCG sigma = dot(U1,C) = U1 * A * U
            U1.block_dot(C, sigmas, num_unconverged);
        }


	//std::cout << "Block dot done" << std::endl;


        // Update the solution and the residual
        // alpha is the step length
        for (size_t i = 0; i < num_unconverged; ++i) {
            alphas[i] = rhos[i] / sigmas[i];
            if (!is_herm) {
                alphas1[i] = sirius::conj(alphas[i]);
            }
        }

        // X[:, ids[i]] += alpha[i] * U[:, i]
        X.block_axpy_scatter(alphas, U, ids, num_unconverged);

        for (size_t i = 0; i < num_unconverged; ++i) {
            alphas[i] *= -1;
            if (!is_herm) {
                alphas1[i] *= -1;
            }
        }

        // R[:, i] += alpha[i] * C[:, i] for i < num_unconverged
        R.block_axpy(alphas, C, num_unconverged);
        // BiCG R1.block_axpy(alphas1, C1, num_unconverged);
        if (!is_herm) {
            R1.block_axpy(alphas1, C1, num_unconverged);
        }

//	std::cout << "End of iteration" << std::endl;

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

/// Wave-function wrapper for linear reponse solver.
struct Wave_functions_wrap
{
    std::shared_ptr<wf::Wave_functions<double>> x;
    /// Location of the data.
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
        wf::axpby(mem, wf::spin_range(0), wf::band_range(0, N__), ones.data(), y__.x.get(), alphas.data(), x.get());
    }

    void
    block_axpy_scatter(std::vector<value_type> const& alphas__, Wave_functions_wrap const& y__,
                       std::vector<int> const& idx__, int n__)
    {
        wf::axpy_scatter<double, value_type>(mem, wf::spin_range(0), alphas__.data(), y__.x.get(), idx__.data(),
                                             x.get(), n__);
    }

    void
    block_axpy(std::vector<value_type> const& alphas__, Wave_functions_wrap const& y__, int N__)
    {
        std::vector<value_type> ones(N__, 1.0);
        wf::axpby(mem, wf::spin_range(0), wf::band_range(0, N__), alphas__.data(), y__.x.get(), ones.data(), x.get());
    }

    /// Make deep copy of the wave-functions wrapper and underlying wave-functions object.
    inline auto
    deep_copy() const
    {
        /* allocate new wave-functions */
        auto wf_out = std::make_shared<wf::Wave_functions<double>>(x->gkvec_sptr(), x->num_md(), x->num_wf(), mem);

        //if (sirius::is_device_memory(mem)) {
        //    wf_out->allocate(sirius::memory_t::device);
        //}

        /* band range to copy: all */
        auto br = wf::band_range(0, x->num_wf().get());
        /* copy from existing to new */
        wf::copy(mem, *x, wf::spin_index(0), br, *wf_out, wf::spin_index(0), br);
        /* return new wrapper */
        return Wave_functions_wrap({wf_out, mem});
    }

    inline auto
    deep_copy_conj() const
    {
	//std::cout << "Entering deep copy conj" << std::endl;
        auto out = this->deep_copy();
        if (sirius::is_host_memory(mem)) {
            #pragma omp parallel for
            for (int i = 0; i < x->num_wf().get(); i++) {
                auto ptr = out.x->at(mem, 0, wf::spin_index(0), wf::band_index(i));
                for (int j = 0; j < x->ld(); j++) {
                    ptr[j] = std::conj(ptr[j]);
                }
            }
        } else {
#if defined(SIRIUS_GPU)
	  //  std::cout << "Entering deep copy conj GPU" << std::endl;
            auto base_ptr = out.x->at(mem, 0, wf::spin_index(0), wf::band_index(0));
            conjugate_gpu_complex_double(base_ptr, out.x->ld(), x->num_wf().get());
#endif
        }
        return out;
    }
};

struct Identity_preconditioner
{
    size_t num_active;

    void
    apply(Wave_functions_wrap& x, Wave_functions_wrap const& y, bool adjoint = false)
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
    std::complex<double> omega;

    template <typename T>
    void
    apply_preconditioner_unified(wf::Wave_functions<T>& res__, bool adjoint__)
    {
        PROFILE("sirius::apply_preconditioner_unified");
        for (auto s = sr.begin(); s != sr.end(); s++) {
            auto sp = res__.actual_spin_index(s);
            if (is_host_memory(mem)) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < num_active; i++) {
                    auto res_ptr = res__.at(mem, 0, sp, wf::band_index(i));
                    for (int j = 0; j < res__.ld(); j++) {
                        auto p = H_diag(j, s.get()) -
                                 S_diag(j, s.get()) * (eigvals[i] + (adjoint__ ? std::conj(omega) : omega));
                        // Step preconditioner
                        if (std::abs(p) > 1.0) {
                            p = 1.0 / p;
                            res_ptr[j] *= p;
                        }
                    }
                }
            } else {
#if defined(SIRIUS_GPU)
                double om_r = std::real(omega);
                double om_i = adjoint__ ? -std::imag(omega) : std::imag(omega);
                apply_preconditioner_gpu_complex_double(res__.at(mem, 0, sp, wf::band_index(0)), res__.ld(), num_active,
                                                        eigvals.at(mem), H_diag.at(mem, 0, s.get()),
                                                        S_diag.at(mem, 0, s.get()), om_r, om_i);
#endif
            }
        }
    }

    void
    apply(Wave_functions_wrap& x, Wave_functions_wrap const& y, bool adjoint = false)
    {
        // Could avoid a copy here, but apply_precondition is in-place.
        x.copy(y, num_active);
        apply_preconditioner_unified(*x.x, adjoint);
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
    std::vector<double> eigenvals;
    /// Work array, stores H|x> and intermediate results
    std::shared_ptr<wf::Wave_functions<double>> Hphi;
    /// Work array, stores S|x> and intermediate results
    std::shared_ptr<wf::Wave_functions<double>> Sphi;
    /// |Psi_k> of the P projector
    std::shared_ptr<wf::Wave_functions<double>> evq;
    /// Work array
    std::shared_ptr<wf::Wave_functions<double>> tmp;
    double alpha_pv;
    std::complex<double> omega;
    /// Band range of the projectors |Psi_k><Psi_k|
    wf::band_range br;
    /// Spin range: currently single up or dn spin is implemented
    wf::spin_range sr;
    memory_t mem;
    la::dmatrix<std::complex<double>> overlap;
    std::vector<wf::device_memory_guard> mg;

    Linear_response_operator(sirius::Hamiltonian_k<double>& Hk, std::vector<double> const& eigvals,
                             std::shared_ptr<wf::Wave_functions<double>> evq, double alpha_pv,
                             std::complex<double> omega, wf::band_range br, wf::spin_range sr, memory_t mem)
        : ctx(Hk.H0().ctx())
        , Hk(Hk)
        , eigenvals(eigvals)
        , Hphi(wave_function_factory<double>(ctx, Hk.kp(), wf::num_bands(eigvals.size()), wf::num_mag_dims(0), false))
        , Sphi(wave_function_factory<double>(ctx, Hk.kp(), wf::num_bands(eigvals.size()), wf::num_mag_dims(0), false))
        , evq(evq)
        , tmp(wave_function_factory<double>(ctx, Hk.kp(), wf::num_bands(eigvals.size()), wf::num_mag_dims(0), false))
        , alpha_pv(alpha_pv)
        , omega(omega)
        , br(br)
        , sr(sr)
        , mem(mem)
        , overlap(br.size(), Hphi->num_wf())
    {
        // I think we could just compute alpha_pv here by just making it big enough
        // s.t. the operator H - e * S + alpha_pv * Q is positive, e.g:
        // alpha_pv = 2 * min_eigenvals.back();
        // but QE has a very specific way to compute it, so we just forward it from
        // there.;

        mg.emplace_back(Hphi->memory_guard(mem));
        mg.emplace_back(Sphi->memory_guard(mem));
        mg.emplace_back(tmp->memory_guard(mem));
    }

    inline bool
    is_hermitian() const
    {
        return std::abs(std::imag(omega)) < 1e-12;
    }

    void
    repack(std::vector<int> const& ids)
    {
        for (size_t i = 0; i < ids.size(); ++i) {
            eigenvals[i] = eigenvals[ids[i]];
        }
    }

    // y[:, i] <- alpha * A * x[:, i] + beta * y[:, i] where A = (H - (e_j + w) S + constant   * SQ * SQ')
    // where SQ is S * eigenvectors.
    void
    multiply(double alpha, Wave_functions_wrap& x, double beta, Wave_functions_wrap& y, int num_active,
             bool adjoint = false)
    {
        PROFILE("sirius::Linear_response_operator::multiply");

        // Hphi = H * x, Sphi = S * x
        Hk.apply_h_s<std::complex<double>>(sr, wf::band_range(0, num_active), *x.x, Hphi.get(), Sphi.get());

        // overlap := evq' * (S * x) = <Psi_k | S | X>
        wf::inner(ctx.spla_context(), mem, wf::spin_range(0), *evq, br, *Sphi, wf::band_range(0, num_active), overlap,
                  0, 0);

        std::vector<std::complex<double>> ones(num_active, 1.0);
        std::vector<std::complex<double>> ev(num_active);

        for (int i = 0; i < num_active; i++) {
            if (!adjoint) {
                // Multiply A * x
                ev[i] = -(eigenvals[i] + omega);
            } else {
                // Multiply A^+ * |x>
                ev[i] = -(eigenvals[i] + std::conj(omega));
            }
        }

        // Hphi contains H|x> - e S|x>
        wf::axpby(mem, wf::spin_range(0), wf::band_range(0, num_active), ev.data(), Sphi.get(), ones.data(),
                  Hphi.get());

        // tmp := evq * overlap
        wf::transform(ctx.spla_context(), mem, overlap, 0, 0, 1.0, *evq, wf::spin_index(0), br, 0.0, *tmp,
                      wf::spin_index(0), wf::band_range(0, num_active));

        // Sphi contains S|Psi_k><Psi_k| S |X>
        Hk.apply_s<std::complex<double>>(wf::spin_range(0), wf::band_range(0, num_active), *tmp, *Sphi);

        // Projector, add alpha_pv * (S * (evq * (evq' * (S * x))))

        // Hphi := alpha_pv * Sphi + Hphi = (H - e * S) * x + alpha_pv * (S * (evq * (evq' * (S * x))))
        std::vector<std::complex<double>> alpha_pvs(num_active, alpha_pv);
        wf::axpby(mem, wf::spin_range(0), wf::band_range(0, num_active), alpha_pvs.data(), Sphi.get(), ones.data(),
                  Hphi.get());

        // y[:, i] <- alpha * Hphi + beta * y[:, i]
        std::vector<std::complex<double>> alphas(num_active, alpha);
        std::vector<std::complex<double>> betas(num_active, beta);
        wf::axpby(mem, wf::spin_range(0), wf::band_range(0, num_active), alphas.data(), Hphi.get(), betas.data(),
                  y.x.get());
    }
};

} // namespace lr

} // namespace sirius
#endif
