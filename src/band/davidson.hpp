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

/** \file davidson.hpp
 *
 *  \brief Davidson iterative solver
 */

#ifndef __DAVIDSON_HPP__
#define __DAVIDSON_HPP__

#include "utils/profiler.hpp"
#include "SDDK/wf_ortho.hpp"
#include "residuals.hpp"

struct davidson_result {
    int niter;
    sddk::mdarray<double, 2> eval;
};

namespace sirius {

/// Solve the eigen-problem using Davidson iterative method.
/**
\tparam T                  Type of the wave-functions in real space
\param [in]     Hk         Hamiltonian for a given k-point.
\param [in,out] psi        Wave-functions. On input they are used for the starting guess of the subspace basis.
                           On output they are the solutions of Hk|psi> = e|psi> eigen-problem. 
\param [in]     occupancy  Lambda-function for the band occupancy numbers.
\return                    List of eigen-values.
*/
template <typename T>
inline davidson_result
davidson(Hamiltonian_k<real_type<T>>& Hk__, Wave_functions<real_type<T>>& psi__,
         std::function<double(int, int)> occupancy__, std::function<double(int, int)> tolerance__)
{
    PROFILE("sirius::davidson");

    auto& ctx = Hk__.H0().ctx();
    ctx.print_memory_usage(__FILE__, __LINE__);


    auto& kp                = Hk__.kp();
    auto& itso              = ctx.cfg().iterative_solver();
    bool converge_by_energy = (itso.converge_by_energy() == 1);

    //kp.message(2, __function_name__, "eigen-value tolerance  : %18.12f\n", ctx.iterative_solver_tolerance());
    //kp.message(2, __function_name__, "empty states tolerance : %18.12f\n",
    //        std::max(ctx.iterative_solver_tolerance() * ctx.cfg().settings().itsol_tol_ratio(), itso.empty_states_tolerance()));


    // bool const estimate_eval{true};
    // bool const skip_initial_diag{false};

    // ctx_.message(2, __function_name__, "iterative solver tolerance: %18.12f\n", ctx_.iterative_solver_tolerance());

    /* true if this is a non-collinear case */
    const bool nc_mag = (ctx.num_mag_dims() == 3);

    /* number of spin components, treated simultaneously
     *   1 - in case of non-magnetic or collinear calculation
     *   2 - in case of non-collinear calculation
     */
    const int num_sc = nc_mag ? 2 : 1;

    /* short notation for number of KS wave-functions to compute */
    // int const num_bands = psi__.num_wf();
    const int num_bands = ctx.num_bands();

    /* maximum subspace size */
    const int num_phi = itso.subspace_size() * num_bands;

    if (num_phi > kp.num_gkvec()) {
        std::stringstream s;
        s << "subspace size is too large!";
        RTE_THROW(s);
    }

    /* alias for memory pool */
    auto& mp = ctx.mem_pool(ctx.host_memory_t());

    auto& gkvecp = kp.gkvec_partition();

    /* allocate wave-functions */

    /* auxiliary wave-functions */
    Wave_functions<real_type<T>> phi(mp, gkvecp, num_phi, ctx.aux_preferred_memory_t(), num_sc);

    /* Hamiltonian, applied to auxiliary wave-functions */
    Wave_functions<real_type<T>> hphi(mp, gkvecp, num_phi, ctx.preferred_memory_t(), num_sc);

    /* S operator, applied to auxiliary wave-functions */
    Wave_functions<real_type<T>> sphi(mp, gkvecp, num_phi, ctx.preferred_memory_t(), num_sc);

    /* Hamiltonain, applied to new Psi wave-functions */
    Wave_functions<real_type<T>> hpsi(mp, gkvecp, num_bands, ctx.preferred_memory_t(), num_sc);

    /* S operator, applied to new Psi wave-functions */
    Wave_functions<real_type<T>> spsi(mp, gkvecp, num_bands, ctx.preferred_memory_t(), num_sc);

    /* residuals */
    Wave_functions<real_type<T>> res(mp, gkvecp, num_bands, ctx.preferred_memory_t(), num_sc);

    const int bs = ctx.cyclic_block_size();

    dmatrix<T> hmlt(num_phi, num_phi, ctx.blacs_grid(), bs, bs, mp);
    dmatrix<T> ovlp(num_phi, num_phi, ctx.blacs_grid(), bs, bs, mp);
    dmatrix<T> evec(num_phi, num_phi, ctx.blacs_grid(), bs, bs, mp);
    dmatrix<T> hmlt_old(num_phi, num_phi, ctx.blacs_grid(), bs, bs, mp);
    dmatrix<T> ovlp_old(num_phi, num_phi, ctx.blacs_grid(), bs, bs, mp);

    if (is_device_memory(ctx.aux_preferred_memory_t())) {
        auto& mpd = ctx.mem_pool(memory_t::device);

        for (int i = 0; i < num_sc; i++) {
            phi.pw_coeffs(i).allocate(mpd);
        }
    }

    if (is_device_memory(ctx.preferred_memory_t())) {
        auto& mpd = ctx.mem_pool(memory_t::device);

        for (int ispn = 0; ispn < psi__.num_sc(); ispn++) {
            psi__.pw_coeffs(ispn).allocate(mpd);
            psi__.pw_coeffs(ispn).copy_to(memory_t::device, 0, num_bands);
        }

        for (int i = 0; i < num_sc; i++) {
            res.pw_coeffs(i).allocate(mpd);

            hphi.pw_coeffs(i).allocate(mpd);
            sphi.pw_coeffs(i).allocate(mpd);

            hpsi.pw_coeffs(i).allocate(mpd);
            spsi.pw_coeffs(i).allocate(mpd);
        }

        if (ctx.blacs_grid().comm().size() == 1) {
            evec.allocate(mpd);
            ovlp.allocate(mpd);
            hmlt.allocate(mpd);
        }
    }

    kp.copy_hubbard_orbitals_on_device();

    ctx.print_memory_usage(__FILE__, __LINE__);

    /* get diagonal elements for preconditioning */
    auto h_o_diag = Hk__.template get_h_o_diag_pw<T, 3>();

    if (ctx.print_checksum()) {
        auto cs1 = h_o_diag.first.checksum();
        auto cs2 = h_o_diag.second.checksum();
        kp.comm().allreduce(&cs1, 1);
        kp.comm().allreduce(&cs2, 1);
        if (kp.comm().rank() == 0) {
            utils::print_checksum("h_diag", cs1);
            utils::print_checksum("o_diag", cs2);
        }
    }

    auto& std_solver = ctx.std_evp_solver();

    if (ctx.print_checksum()) {
        for (int ispn = 0; ispn < ctx.num_spins(); ispn++) {
            auto cs = psi__.checksum_pw(get_device_t(psi__.preferred_memory_t()), ispn, 0, num_bands);
            std::stringstream s;
            s << "input spinor_wave_functions_" << ispn;
            if (kp.comm().rank() == 0) {
                utils::print_checksum(s.str(), cs);
            }
        }
    }

    davidson_result result{0, sddk::mdarray<double, 2>(num_bands, ctx.num_spinors())};

    PROFILE_START("sirius::davidson|iter");
    for (int ispin_step = 0; ispin_step < ctx.num_spinors(); ispin_step++) {

        /* converged vectors */
        int num_locked = 0;

        sddk::mdarray<real_type<T>, 1> eval(num_bands);
        sddk::mdarray<real_type<T>, 1> eval_old(num_bands);
        eval_old = []() { return 1e10; };

        /* check if band energy is converged */
        auto is_converged = [&](int j__, int ispn__) -> bool {
            return std::abs(eval[j__] - eval_old[j__]) <= tolerance__(j__ + num_locked, ispn__);
        };

        if (itso.init_eval_old()) {
            eval_old = [&](int64_t j) { return kp.band_energy(j, ispin_step); };
        }

        /* trial basis functions */
        for (int ispn = 0; ispn < num_sc; ispn++) {
            phi.copy_from(psi__, num_bands, nc_mag ? ispn : ispin_step, 0, ispn, 0);
        }

        if (ctx.print_checksum()) {
            for (int ispn = 0; ispn < num_sc; ispn++) {
                auto cs = phi.checksum_pw(get_device_t(phi.preferred_memory_t()), ispn, 0, num_bands);
                std::stringstream s;
                s << "input phi" << ispn;
                if (kp.comm().rank() == 0) {
                    utils::print_checksum(s.str(), cs);
                }
            }
        }

        /* first phase: setup and diagonalize reduced Hamiltonian and get eigen-values;
         * this is done before the main itertive loop */

        /* apply Hamiltonian and S operators to the basis functions */
        Hk__.template apply_h_s<T>(spin_range(nc_mag ? 2 : ispin_step), 0, num_bands, phi, &hphi, &sphi);

        orthogonalize<T>(ctx.spla_context(), ctx.preferred_memory_t(), ctx.blas_linalg_t(), nc_mag ? 2 : 0, phi, hphi,
                         sphi, 0, num_bands, ovlp, res);

        /* setup eigen-value problem */
        Band(ctx).set_subspace_mtrx<T>(0, num_bands, 0, phi, hphi, hmlt, &hmlt_old);

        if (ctx.cfg().control().verification() >= 1) {
            auto max_diff = check_hermitian(hmlt, num_bands);
            if (max_diff > (std::is_same<real_type<T>, double>::value ? 1e-12 : 1e-6)) {
                std::stringstream s;
                s << "H matrix is not Hermitian, max_err = " << max_diff << std::endl
                  << "  happened before entering the iterative loop" << std::endl;
                WARNING(s);
                if (num_bands <= 20) {
                    hmlt.serialize("davidson:H", num_bands);
                }
            }
        }

        /* current subspace size */
        int N = num_bands;

        // Seems like a smaller block size is not always improving time to solution much,
        // so keep it num_bands.
        int block_size = num_bands;

        /* solve generalized eigen-value problem with the size N and get lowest num_bands eigen-vectors */
        if (std_solver.solve(N, num_bands, hmlt, &eval[0], evec)) {
            std::stringstream s;
            s << "error in diagonalziation";
            RTE_THROW(s);
        }

        ctx.evp_work_count(1);

        for (int i = 0; i < num_bands; i++) {
            kp.message(4, __function_name__, "eval[%i]=%20.16f\n", i, eval[i]);
        }

        /* number of newly added basis functions */
        int num_unconverged{0};

        /* tolerance for the norm of L2-norms of the residuals, used for
         * relative convergence criterion. We can only compute this after
         * we have the first residual norms available */
        real_type<T> relative_frobenius_tolerance{0};
        real_type<T> current_frobenius_norm{0};

        /* second phase: start iterative diagonalization */
        for (int k = 0; k < itso.num_steps(); k++) {
            int num_lockable = 0;

            bool last_iteration = k == (itso.num_steps() - 1);

            int num_ritz = num_bands - num_locked;

            /* don't compute residuals on last iteration */
            if (!last_iteration) {
                /* get new preconditionined residuals, and also hpsi and opsi as a by-product */
                auto result = residuals<T>(ctx, ctx.preferred_memory_t(), ctx.blas_linalg_t(),
                                           spin_range(nc_mag ? 2 : ispin_step), N, num_ritz, num_locked, eval, evec,
                                           hphi, sphi, hpsi, spsi, res, h_o_diag.first, h_o_diag.second,
                                           itso.converge_by_energy(), itso.residual_tolerance(), is_converged);

                num_unconverged        = result.unconverged_residuals;
                num_lockable           = result.num_consecutive_smallest_converged;
                current_frobenius_norm = result.frobenius_norm;

                /* set the relative tolerance convergence criterion */
                if (k == 0) {
                    relative_frobenius_tolerance = current_frobenius_norm * itso.relative_tolerance();
                }
            }

            /* verify convergence criteria */
            int num_converged              = num_ritz - num_unconverged;
            bool converged_by_relative_tol = k > 0 && current_frobenius_norm < relative_frobenius_tolerance;
            bool converged_by_absolute_tol = num_locked + num_converged + itso.min_num_res() >= num_bands;

            bool converged = converged_by_relative_tol || converged_by_absolute_tol;

            /* Todo: num_unconverged might be very small at some point slowing down convergence
                     can we add more? */
            int expand_with     = std::min(num_unconverged, block_size);
            bool should_restart = ((N + expand_with) > num_phi) ||
                                  (num_lockable > 5 && num_unconverged < itso.early_restart() * num_lockable);

            kp.message(3, __function_name__,
                       "Restart = %s. Locked = %d. Converged = %d. Wanted = %d. Lockable = %d.i "
                       "Num ritz = %d. Expansion size = %d\n",
                       should_restart ? "yes" : "no", num_locked, num_converged, num_bands, num_lockable, num_ritz,
                       expand_with);

            /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
            if (should_restart || converged || last_iteration) {
                PROFILE("sirius::davidson|update_phi");
                /* recompute wave-functions */
                /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */

                /* No need to recompute the wave functions when converged in the first iteration */
                if (k != 0 || num_unconverged != 0 || ctx.cfg().settings().always_update_wf()) {
                    /* in case of non-collinear magnetism transform two components */
                    transform<T>(ctx.spla_context(), nc_mag ? 2 : ispin_step, {&phi}, num_locked, N - num_locked, evec,
                                 0, 0, {&psi__}, num_locked, num_ritz);

                    /* update eigen-values */
                    for (int j = num_locked; j < num_bands; j++) {
                        result.eval(j, ispin_step) = eval[j - num_locked];
                    }

                } else {
                    kp.message(3, __function_name__, "%s", "wave-functions are not recomputed\n");
                }

                if (last_iteration && !converged) {
                    kp.message(3, __function_name__,
                               "Warning: maximum number of iterations reached, but %i "
                               "residual(s) did not converge for k-point %f %f %f, eigen-solver tolerance: %18.12f\n",
                               num_unconverged, kp.vk()[0], kp.vk()[1], kp.vk()[2], ctx.iterative_solver_tolerance());
                }

                /* exit the loop if the eigen-vectors are converged or this is a last iteration */
                if (converged || last_iteration) {
                    kp.message(3, __function_name__, "end of iterative diagonalization; n=%i, k=%i\n", num_unconverged, k);
                    break;
                } else { /* otherwise, set Psi as a new trial basis */
                    kp.message(3, __function_name__, "%s", "subspace size limit reached\n");

                    // TODO: consider keeping more than num_bands when nearly all Ritz vectors have converged.
                    int keep = num_bands;

                    /* need to compute all hpsi and opsi states (not only unconverged) */
                    if (converge_by_energy) {
                        transform<T>(ctx.spla_context(), nc_mag ? 2 : ispin_step, 1.0,
                                     std::vector<Wave_functions<real_type<T>>*>({&hphi, &sphi}), num_locked,
                                     N - num_locked, evec, 0, 0, 0.0, {&hpsi, &spsi}, 0, num_ritz);
                    }

                    /* update basis functions, hphi and ophi */
                    for (int ispn = 0; ispn < num_sc; ispn++) {
                        phi.copy_from(psi__, keep - num_locked, nc_mag ? ispn : ispin_step, num_locked,
                                      nc_mag ? ispn : 0, num_locked);
                        hphi.copy_from(hpsi, keep - num_locked, ispn, 0, ispn, num_locked);
                        sphi.copy_from(spsi, keep - num_locked, ispn, 0, ispn, num_locked);
                    }

                    // Remove locked Ritz values so indexing starts at unconverged eigenpairs
                    if (itso.locking() && num_lockable > 0) {
                        for (int i = num_lockable; i < num_ritz; ++i) {
                            eval[i - num_lockable] = eval[i];
                        }
                    }

                    // Remove the locked block from the projected matrix too.
                    hmlt_old.zero();
                    for (int i = 0; i < keep - num_locked; i++) {
                        hmlt_old.set(i, i, eval[i]);
                    }

                    /* number of basis functions that we already have */
                    N = keep;

                    // Only when we do orthogonalization we can lock vecs
                    if (itso.locking()) {
                        num_locked += num_lockable;
                    }
                }
            }

            /* expand variational subspace with new basis vectors obtatined from residuals */
            for (int ispn = 0; ispn < num_sc; ispn++) {
                phi.copy_from(res, expand_with, ispn, 0, ispn, N);
            }

            /* apply Hamiltonian and S operators to the new basis functions */
            Hk__.template apply_h_s<T>(spin_range(nc_mag ? 2 : ispin_step), N, expand_with, phi, &hphi, &sphi);

            kp.message(3, __function_name__, "Orthogonalize %d to %d\n", expand_with, N);

            orthogonalize<T>(ctx.spla_context(), ctx.preferred_memory_t(), ctx.blas_linalg_t(), nc_mag ? 2 : 0, phi,
                             hphi, sphi, N, expand_with, ovlp, res);

            /* setup eigen-value problem
             * N is the number of previous basis functions
             * expand_with is the number of new basis functions */
            Band(ctx).set_subspace_mtrx(N, expand_with, num_locked, phi, hphi, hmlt, &hmlt_old);

            if (ctx.cfg().control().verification() >= 1) {
                real_type<T> max_diff = check_hermitian(hmlt, N + expand_with - num_locked);
                if (max_diff > (std::is_same<real_type<T>, double>::value ? 1e-12 : 1e-6)) {
                    std::stringstream s;
                    kp.message(1, __function_name__, "H matrix of size %i is not Hermitian, maximum error: %18.12e\n",
                               N + expand_with - num_locked, max_diff);
                }
            }

            /* increase size of the variation space */
            N += expand_with;

            // Copy the Ritz values
            eval >> eval_old;

            kp.message(3, __function_name__, "Computing %d pre-Ritz pairs\n", num_bands - num_locked);
            /* solve standard eigen-value problem with the size N */
            if (std_solver.solve(N - num_locked, num_bands - num_locked, hmlt, &eval[0], evec)) {
                std::stringstream s;
                s << "error in diagonalziation";
                RTE_THROW(s);
            }

            ctx.evp_work_count(std::pow(static_cast<double>(N - num_locked) / num_bands, 3));

            kp.message(3, __function_name__, "step: %i, current subspace size: %i, maximum subspace size: %i\n", k, N,
                       num_phi);
            for (int i = 0; i < num_bands - num_locked; i++) {
                kp.message(4, __function_name__, "eval[%i]=%20.16f, diff=%20.16f, occ=%20.16f\n", i, eval[i],
                           std::abs(eval[i] - eval_old[i]), occupancy__(i, ispin_step));
            }
            result.niter++;

        } /* loop over iterative steps k */
    } /* loop over ispin_step */
    PROFILE_STOP("sirius::davidson|iter");

    if (is_device_memory(ctx.preferred_memory_t())) {
        for (int ispn = 0; ispn < ctx.num_spins(); ispn++) {
            psi__.pw_coeffs(ispn).copy_to(memory_t::host, 0, num_bands);
            psi__.pw_coeffs(ispn).deallocate(memory_t::device);
        }
    }

    kp.release_hubbard_orbitals_on_device();
    return result;
}

}

#endif
