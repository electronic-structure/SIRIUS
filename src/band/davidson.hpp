// Copyright (c) 2013-2021 Anton Kozhevnikov, Thomas Schulthess
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
 *  \brief Davidson iterative solver implementation.
 */

#ifndef __DAVIDSON_HPP__
#define __DAVIDSON_HPP__

#include "utils/profiler.hpp"
#include "SDDK/wf_ortho.hpp"
#include "SDDK/wf_trans.hpp"
#include "residuals.hpp"

/// Result of Davidson solver.
struct davidson_result_t {
    int niter;
    sddk::mdarray<double, 2> eval;
};

namespace sirius {

template <typename T>
inline void
project_out_subspace(::spla::Context& spla_ctx__, spin_range spins__, Wave_functions<real_type<T>>& phi__,
                     Wave_functions<real_type<T>>& sphi__, Wave_functions<real_type<T>>& phi_new__, int N__, int n__,
                     sddk::dmatrix<T>& o__)
{
    PROFILE("sirius::project_out_subspace");

    /* project out the old subspace:
     * |\tilda phi_new> = |phi_new> - |phi_old><phi_old|S|phi_new> */
    inner(spla_ctx__, spins__, sphi__, 0, N__, phi_new__, 0, n__, o__, 0, 0);
    sddk::transform<T>(spla_ctx__, spins__(), -1.0, {&phi__}, 0, N__, o__, 0, 0, 1.0, {&phi_new__}, 0, n__);

    //inner(spla_ctx__, spins__, sphi__, 0, N__, phi_new__, 0, n__, o__, 0, 0);
    //for (int i = 0; i < N__; i++) {
    //    for (int j = 0; j < n__; j++) {
    //        std::cout << i << " " << j << " " << o__(i, j) << std::endl;
    //    }
    //}
}

template <typename T>
inline int
remove_linearly_dependent(::spla::Context& spla_ctx__, spin_range spins__, Wave_functions<real_type<T>>& phi__,
                          int n__, sddk::dmatrix<T>& o__)

{
    PROFILE("sirius::remove_linearly_dependent");

    /* compute <phi | phi> */
    inner(spla_ctx__, spins__, phi__, 0, n__, phi__, 0, n__, o__, 0, 0);

    auto la = (o__.comm().size() == 1) ? linalg_t::lapack : linalg_t::scalapack;
    linalg(la).geqrf(n__, n__, o__, 0, 0);
    auto diag = o__.get_diag(n__);

    auto eps = std::numeric_limits<real_type<T>>::epsilon();

    int n{0};
    for (int i = 0; i < n__; i++) {
        if (std::abs(diag[i]) >= eps * 10) {
            /* shift linearly independent basis functions to the beginning of phi */
            if (n != i) {
                for (int ispn: spins__) {
                    phi__.copy_from(phi__, 1, ispn, i, ispn, n);
                }
            }
            n++;
        }
    }
    return n;
}

/// Solve the eigen-problem using Davidson iterative method.
/**
\tparam T                     Type of the wave-functions in real space (one of float, double, complex<float>, complex<double>).
\param [in]     Hk            Hamiltonian for a given k-point.
\param [in]     num_bands     Number of eigen-states (bands) to compute.
\param [in]     num_mag_dims  Number of magnetic dimensions (0, 1 or 3).
\param [in,out] psi           Wave-functions. On input they are used for the starting guess of the subspace basis.
                              On output they are the solutions of Hk|psi> = e S|psi> eigen-problem. 
\param [in]     occupancy     Lambda-function for the band occupancy numbers.
\param [in]     tolerance     Lambda-function for the band energy tolerance.
\param [in]     res_tol       Residual tolerance.
\param [in]     num_stpes     Number of iterative steps.
\param [in]     locking       Lock and do not update of the converged wave-functions.
\param [in]     subspace_size Size of the diagonalziation subspace.
\param [in]     estimate_eval Estimate eigen-values to get the converrged rersiduals.
\param [out]    out           Output stream.
\param [in]     verbosity     Verbosity level.
\return                       List of eigen-values.
*/
template <typename T>
inline davidson_result_t
davidson(Hamiltonian_k<real_type<T>>& Hk__, int num_bands__, int num_mag_dims__, Wave_functions<real_type<T>>& psi__,
         std::function<double(int, int)> occupancy__, std::function<double(int, int)> tolerance__, double res_tol__,
         int num_steps__, bool locking__, int subspace_size__, bool estimate_eval__, std::ostream& out__,
         int verbosity__ = 0)
{
    PROFILE("sirius::davidson");

    auto& ctx = Hk__.H0().ctx();
    ctx.print_memory_usage(__FILE__, __LINE__);


    auto& kp      = Hk__.kp();
    auto& gkvecp  = kp.gkvec_partition();

    auto& itso = ctx.cfg().iterative_solver();

    /* true if this is a non-collinear case */
    const bool nc_mag = (num_mag_dims__ == 3);

    /* number of spin components, treated simultaneously
     *   1 - in case of non-magnetic or collinear calculation
     *   2 - in case of non-collinear calculation
     */
    const int num_sc = nc_mag ? 2 : 1;

    /* number of spinor components stored under the same band index */
    const int num_spinors = (num_mag_dims__ == 1) ? 2 : 1; 

    /* number of spins */
    const int num_spins = (num_mag_dims__ == 0) ? 1 : 2;

    /* maximum subspace size */
    const int num_phi = subspace_size__ * num_bands__;

    if (num_phi > kp.num_gkvec()) {
        std::stringstream s;
        s << "subspace size is too large!";
        RTE_THROW(s);
    }

    /* alias for memory pool */
    auto& mp = ctx.mem_pool(ctx.host_memory_t());

    bool project_out_here{false};

    /* allocate wave-functions */

    /* auxiliary wave-functions */
    Wave_functions<real_type<T>> phi(mp, gkvecp, num_phi, ctx.aux_preferred_memory_t(), num_sc);

    /* Hamiltonian, applied to auxiliary wave-functions */
    Wave_functions<real_type<T>> hphi(mp, gkvecp, num_phi, ctx.preferred_memory_t(), num_sc);

    /* S operator, applied to auxiliary wave-functions */
    Wave_functions<real_type<T>> sphi(mp, gkvecp, num_phi, ctx.preferred_memory_t(), num_sc);

    /* Hamiltonain, applied to new Psi wave-functions */
    Wave_functions<real_type<T>> hpsi(mp, gkvecp, num_bands__, ctx.preferred_memory_t(), num_sc);

    /* S operator, applied to new Psi wave-functions */
    Wave_functions<real_type<T>> spsi(mp, gkvecp, num_bands__, ctx.preferred_memory_t(), num_sc);

    /* residuals */
    Wave_functions<real_type<T>> res(mp, gkvecp, num_bands__, ctx.preferred_memory_t(), num_sc);

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
            psi__.pw_coeffs(ispn).copy_to(memory_t::device, 0, num_bands__);
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
        for (int ispn = 0; ispn < num_spins; ispn++) {
            auto cs = psi__.checksum_pw(get_device_t(psi__.preferred_memory_t()), ispn, 0, num_bands__);
            std::stringstream s;
            s << "input spinor_wave_functions_" << ispn;
            if (kp.comm().rank() == 0) {
                utils::print_checksum(s.str(), cs);
            }
        }
    }

    davidson_result_t result{0, sddk::mdarray<double, 2>(num_bands__, num_spinors)};

    if (verbosity__ >= 1) {
        out__ << "starting Davidson iterative solver" << std::endl
              << "  number of bands : " << num_bands__ << std::endl
              << "  subspace size   : " << num_phi << std::endl
              << "  locking         : " << locking__ << std::endl
              << "  number of spins : " << num_spins << std::endl
              << "  non-collinear   : " << nc_mag << std::endl;
    }

    PROFILE_START("sirius::davidson|iter");
    for (int ispin_step = 0; ispin_step < num_spinors; ispin_step++) {

        if (verbosity__ >= 1) {
            out__ << "ispin_step " << ispin_step << " out of " << num_spinors << std::endl;
        }

        /* converged vectors */
        int num_locked{0};

        sddk::mdarray<real_type<T>, 1> eval(num_bands__);
        sddk::mdarray<real_type<T>, 1> eval_old(num_bands__);

        /* check if band energy is converged */
        auto is_converged = [&](int j__, int ispn__) -> bool {
            return std::abs(eval[j__] - eval_old[j__]) <= tolerance__(j__, ispn__);
        };

        if (itso.init_eval_old()) {
            eval_old = [&](int64_t j) { return kp.band_energy(j, ispin_step); };
        } else {
            eval_old = []() { return std::numeric_limits<real_type<T>>::max(); };
        }

        /* trial basis functions */
        for (int ispn = 0; ispn < num_sc; ispn++) {
            phi.copy_from(psi__, num_bands__, nc_mag ? ispn : ispin_step, 0, ispn, 0);
        }

        if (ctx.print_checksum()) {
            for (int ispn = 0; ispn < num_sc; ispn++) {
                auto cs = phi.checksum_pw(get_device_t(phi.preferred_memory_t()), ispn, 0, num_bands__);
                std::stringstream s;
                s << "input phi" << ispn;
                if (kp.comm().rank() == 0) {
                    utils::print_checksum(s.str(), cs);
                }
            }
        }

        /* first phase: setup and diagonalize reduced Hamiltonian and get eigen-values;
         * this is done before the main itertive loop */

        if (verbosity__ >= 1) {
            out__ << "apply Hamiltonian to phi(0:" << num_bands__ - 1 << ")" << std::endl;
        }
        /* apply Hamiltonian and S operators to the basis functions */
        Hk__.template apply_h_s<T>(spin_range(nc_mag ? 2 : ispin_step), 0, num_bands__, phi, &hphi, &sphi);

        /* DEBUG */
        if (ctx.cfg().control().verification() >= 1) {
            /* setup eigen-value problem */
            Band(ctx).set_subspace_mtrx<T>(0, num_bands__, 0, phi, hphi, hmlt, &hmlt_old);
            Band(ctx).set_subspace_mtrx<T>(0, num_bands__, 0, phi, sphi, ovlp, &ovlp_old);

            auto max_diff = check_hermitian(hmlt, num_bands__);
            if (max_diff > (std::is_same<real_type<T>, double>::value ? 1e-12 : 1e-6)) {
                std::stringstream s;
                s << "H matrix is not Hermitian, max_err = " << max_diff << std::endl
                  << "  happened before entering the iterative loop" << std::endl;
                WARNING(s);
                if (num_bands__ <= 20) {
                    hmlt.serialize("davidson:H_first", num_bands__);
                }
            }
            max_diff = check_hermitian(ovlp, num_bands__);
            if (max_diff > (std::is_same<real_type<T>, double>::value ? 1e-12 : 1e-6)) {
                std::stringstream s;
                s << "O matrix is not Hermitian, max_err = " << max_diff << std::endl
                  << "  happened before entering the iterative loop" << std::endl;
                WARNING(s);
                if (num_bands__ <= 20) {
                    hmlt.serialize("davidson:O_first", num_bands__);
                }
            }
        }
        /* END DEBUG */

        if (verbosity__ >= 1) {
            out__ << "orthogonalize " << num_bands__ << " states" << std::endl;
        }

        orthogonalize<T>(ctx.spla_context(), ctx.preferred_memory_t(), ctx.blas_linalg_t(),
                         spin_range(nc_mag ? 2 : 0), phi, hphi, sphi, 0, num_bands__, ovlp, res, false, false);

        if (verbosity__ >= 1) {
            out__ << "set " << num_bands__ << " x " << num_bands__ << " subspace matrix" << std::endl;
        }
        /* setup eigen-value problem */
        Band(ctx).set_subspace_mtrx<T>(0, num_bands__, 0, phi, hphi, hmlt, &hmlt_old);

        if (ctx.cfg().control().verification() >= 1) {
            auto max_diff = check_hermitian(hmlt, num_bands__);
            if (max_diff > (std::is_same<real_type<T>, double>::value ? 1e-12 : 1e-6)) {
                std::stringstream s;
                s << "H matrix is not Hermitian, max_err = " << max_diff << std::endl
                  << "  happened before entering the iterative loop" << std::endl;
                WARNING(s);
                if (num_bands__ <= 20) {
                    hmlt.serialize("davidson:H", num_bands__);
                }
            }
        }

        /* current subspace size */
        int N = num_bands__;

        /* upper limit fot the subspace expansion;
           seems like a smaller block size is not always improving time to solution much, so keep it num_bands */
        int block_size = num_bands__;

        if (verbosity__ >= 1) {
            out__ << "diagonalize " << N << " x " << N << " Hamiltonian" << std::endl;
        }
        /* solve eigen-value problem with the size N and get lowest num_bands eigen-vectors */
        if (std_solver.solve(N, num_bands__, hmlt, &eval[0], evec)) {
            std::stringstream s;
            s << "error in diagonalziation";
            RTE_THROW(s);
        } else {
            ctx.evp_work_count(1);
        }

        /* number of newly added basis functions */
        int num_unconverged{0};

        /* tolerance for the norm of L2-norms of the residuals, used for
         * relative convergence criterion. We can only compute this after
         * we have the first residual norms available */
        real_type<T> relative_frobenius_tolerance{0};
        real_type<T> current_frobenius_norm{0};

        /* second phase: start iterative diagonalization */
        for (int iter_step = 0; iter_step < num_steps__; iter_step++) {
            if (verbosity__ >= 1) {
                out__ << "iter_step " << iter_step << " out of " << num_steps__ << std::endl;
            }

            bool last_iteration = iter_step == (num_steps__ - 1);

            bool converged{true};
            for (int i = 0; i < num_bands__; i++) {
                converged = converged & is_converged(i, ispin_step);
            }

            /* don't compute residuals on last iteration */
            if (!last_iteration && !converged) {
                //int num_ritz = num_bands__ - num_locked;
                if (verbosity__ >= 1) {
                    out__ << "compute " << num_bands__ - num_locked
                          << " residuals from phi(" << num_locked << ":" << N - 1 << ")" << std::endl;
                }
                /* get new preconditionined residuals, and also hpsi and opsi as a by-product */
                auto result = residuals<T>(ctx, ctx.preferred_memory_t(), ctx.blas_linalg_t(),
                                           spin_range(nc_mag ? 2 : ispin_step), N, num_bands__, num_locked, eval, evec,
                                           hphi, sphi, hpsi, spsi, res, h_o_diag.first, h_o_diag.second,
                                           estimate_eval__, res_tol__, is_converged);

                num_unconverged        = result.unconverged_residuals;
                //num_lockable           = result.num_consecutive_smallest_converged;
                current_frobenius_norm = result.frobenius_norm;

                if (verbosity__) {
                    out__ << "number of unconverged residuals : " << num_unconverged << std::endl;
                }

                /* set the relative tolerance convergence criterion */
                if (iter_step == 0) {
                    relative_frobenius_tolerance = current_frobenius_norm * itso.relative_tolerance();
                }

                if (num_unconverged && project_out_here) {
                    if (verbosity__) {
                        out__ << "project out " << N << " basis states" << std::endl;
                    }
                    project_out_subspace(ctx.spla_context(), spin_range(nc_mag ? 2 : 0), phi, sphi, res, N,
                                         num_unconverged, ovlp);
                    int nli = remove_linearly_dependent(ctx.spla_context(), spin_range(nc_mag ? 2 : 0), res,
                                                        num_unconverged, ovlp);
                    if (verbosity__) {
                        out__ << "number of linearly independent residuals : " << nli << std::endl;
                    }
                    num_unconverged = nli;
                }
            }

            /* verify convergence criteria */
            bool converged_by_relative_tol = iter_step > 0 && current_frobenius_norm < relative_frobenius_tolerance;
            bool converged_by_absolute_tol = num_unconverged <= itso.min_num_res();

            converged = converged || converged_by_relative_tol || converged_by_absolute_tol;

            /* Todo: num_unconverged might be very small at some point slowing down convergence
                     can we add more? */
            int expand_with     = std::min(num_unconverged, block_size);
            bool should_restart = (N + expand_with) > num_phi;
            //bool should_restart = ((N + expand_with) > num_phi) ||
            //                      (num_lockable > 5 && num_unconverged < itso.early_restart() * num_lockable);

            //kp.message(3, __function_name__,
            //           "Restart = %s. Locked = %d. Converged = %d. Wanted = %d. Lockable = %d. "
            //           "Num ritz = %d. Expansion size = %d\n",
            //           should_restart ? "yes" : "no", num_locked, num_converged, num_bands__, num_lockable, num_ritz,
            //           expand_with);

            if (verbosity__ >= 2) {
                out__ << "converged_by_relative_tol : " << converged_by_relative_tol << std::endl
                      << "converged_by_absolute_tol : " << converged_by_absolute_tol << std::endl;
                if (num_unconverged == 0) {
                    for (int i = 0; i < num_bands__; i++) {
                        out__ << "eval[" << i << "]=" << eval[i]
                              << ", eval_old[" << i << "]=" << eval_old[i]
                              << ", diff=" << std::abs(eval[i] - eval_old[i])
                              << ", tol=" << tolerance__(i, ispin_step)
                              << ", converged=" << is_converged(i, ispin_step)
                              << ", occ=" << occupancy__(i, ispin_step) << std::endl;
                    }
                }
            }

            /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
            if (should_restart || converged || last_iteration) {
                PROFILE("sirius::davidson|update_phi");
                /* recompute wave-functions */
                /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */

                /* No need to recompute the wave functions when converged in the first iteration */
                if (iter_step != 0 || num_unconverged != 0 || ctx.cfg().settings().always_update_wf()) {
                    /* in case of non-collinear magnetism transform two components */
                    if (verbosity__ >= 2) {
                        out__ << "recomputing psi(" << num_locked << ":" << num_bands__ - 1 << ") "
                              << "from phi(" << num_locked << ":" << N - 1 << ")" << std::endl;
                    }
                    transform<T>(ctx.spla_context(), nc_mag ? 2 : ispin_step, {&phi}, num_locked, N - num_locked, evec,
                                 0, 0, {&psi__}, num_locked, num_bands__ - num_locked);

                    /* update eigen-values */
                    for (int j = 0; j < num_bands__; j++) {
                        result.eval(j, ispin_step) = eval[j];
                    }

                } else {
                    if (verbosity__ >= 2) {
                        out__ << "psi is not recomputed" << std::endl;
                    }
                }

                if (last_iteration && !converged) {
                    //kp.message(3, __function_name__,
                    //           "Warning: maximum number of iterations reached, but %i "
                    //           "residual(s) did not converge for k-point %f %f %f, eigen-solver tolerance: %18.12f\n",
                    //           num_unconverged, kp.vk()[0], kp.vk()[1], kp.vk()[2], ctx.iterative_solver_tolerance());
                }

                /* exit the loop if the eigen-vectors are converged or this is a last iteration */
                if (converged || last_iteration) {
                    if (verbosity__) {
                        out__ << "end of iterative diagonalization; num_unconverged = " << num_unconverged
                              << ", iter_step = " << iter_step << std::endl;
                    }
                    break;
                } else { /* otherwise, set Psi as a new trial basis */
                    //kp.message(3, __function_name__, "%s", "subspace size limit reached\n");

                    //// TODO: consider keeping more than num_bands when nearly all Ritz vectors have converged.
                    //int keep = num_bands__;

                    /* need to compute all hpsi and opsi states (not only unconverged) */
                    if (estimate_eval__) {
                        transform<T>(ctx.spla_context(), nc_mag ? 2 : ispin_step, 1.0,
                                     std::vector<Wave_functions<real_type<T>>*>({&hphi, &sphi}), num_locked,
                                     N - num_locked, evec, 0, 0, 0.0, {&hpsi, &spsi}, 0, num_bands__ - num_locked);
                    }

                    /* update basis functions, hphi and ophi */
                    for (int ispn = 0; ispn < num_sc; ispn++) {
                        phi.copy_from(psi__, num_bands__ - num_locked, nc_mag ? ispn : ispin_step, num_locked,
                                      nc_mag ? ispn : 0, num_locked);
                        hphi.copy_from(hpsi, num_bands__ - num_locked, ispn, 0, ispn, num_locked);
                        sphi.copy_from(spsi, num_bands__ - num_locked, ispn, 0, ispn, num_locked);
                    }

                    /* only when we do orthogonalization we can lock vecs */
                    if (locking__) {
                        int nlock{0};
                        while (nlock < num_bands__ && is_converged(nlock, ispin_step)) {
                            nlock++;
                        }
                        if (verbosity__) {
                            out__ << "old num_locked = " << num_locked << ", new num_locked = " << nlock << std::endl;
                        }
                        if (nlock < num_locked) {
                            std::stringstream s;
                            s << "new number of locked eigen-pairs is smaller than it was" << std::endl
                              << "  old num_locked = " << num_locked << ", new num_locked = " << nlock << std::endl;
                            RTE_THROW(s);
                        }
                        num_locked = nlock;
                    }
                    if (num_locked == num_bands__) {
                        RTE_THROW("should stop here");
                    }

                    /* remove the locked block from the projected matrix too. */
                    hmlt_old.zero();
                    for (int i = 0; i < num_bands__ - num_locked; i++) {
                        hmlt_old.set(i, i, eval[i + num_locked]);
                    }

                    /* number of basis functions that we already have */
                    N = num_bands__;

                }
            }

            if (verbosity__) {
                out__ << "expanding subspace of size " << N << " with " << expand_with 
                      << " new basis functions" << std::endl;
            }

            /* expand variational subspace with new basis vectors obtatined from residuals */
            for (int ispn = 0; ispn < num_sc; ispn++) {
                phi.copy_from(res, expand_with, ispn, 0, ispn, N);
            }
            //inner(ctx.spla_context(), spin_range(nc_mag ? 2 : 0), sphi, 0, N, phi, N, expand_with, ovlp, 0, 0);
            //std::cout << "<phi|S | res>" << std::endl;
            //for (int i = 0; i < N; i++) {
            //    for (int j = 0; j < expand_with; j++) {
            //        std::cout << i << " " << j << " " << ovlp(i, j) << std::endl;
            //    }
            //}

            if (verbosity__ >= 1) {
                out__ << "apply Hamiltonian to phi(" << N << ":" << N + expand_with - 1 << ")" << std::endl;
            }
            /* apply Hamiltonian and S operators to the new basis functions */
            Hk__.template apply_h_s<T>(spin_range(nc_mag ? 2 : ispin_step), N, expand_with, phi, &hphi, &sphi);

            //inner(ctx.spla_context(), spin_range(nc_mag ? 2 : 0), phi, 0, N + expand_with, sphi, 0, N + expand_with, ovlp, 0, 0);
            //ovlp.serialize("davidson:ovlp1", N + expand_with);

            if (verbosity__ >= 1) {
                out__ << "orthogonalize " << expand_with << " states to " << N << " previous states" << std::endl;
            }
            orthogonalize<T>(ctx.spla_context(), ctx.preferred_memory_t(), ctx.blas_linalg_t(),
                             spin_range(nc_mag ? 2 : 0), phi, hphi, sphi, N, expand_with, ovlp, res, !project_out_here, false);

            if (verbosity__ >= 1) {
                out__ << "set " << N + expand_with - num_locked << " x " << N + expand_with - num_locked
                      << " subspace matrix" << std::endl;
            }
            /* setup eigen-value problem
             * N is the number of previous basis functions
             * expand_with is the number of new basis functions */
            Band(ctx).set_subspace_mtrx(N, expand_with, num_locked, phi, hphi, hmlt, &hmlt_old);

            //hmlt.serialize("davidson:hmlt", N + expand_with);
            //
            //Band(ctx).set_subspace_mtrx(0, N + expand_with, 0, sphi, phi, ovlp);
            //auto err = check_identity(ovlp, N + expand_with);
            //std::cout << "overlap matrix error : " << err << std::endl;
            //if (err > 1e-10) {
            //    Band(ctx).set_subspace_mtrx(0, N + expand_with, 0, sphi, phi, ovlp);
            //    ovlp.serialize("davidson:ovlp", N + expand_with);
            //    //RTE_THROW("phi are not orrthogonal");
            //}

            if (ctx.cfg().control().verification() >= 1) {
                real_type<T> max_diff = check_hermitian(hmlt, N + expand_with - num_locked);
                if (max_diff > (std::is_same<real_type<T>, double>::value ? 1e-12 : 1e-6)) {
                    std::stringstream s;
                    //kp.message(1, __function_name__, "H matrix of size %i is not Hermitian, maximum error: %18.12e\n",
                    //           N + expand_with - num_locked, max_diff);
                }
            }

            /* increase size of the variation space */
            N += expand_with;

            /* copy the Ritz values */
            eval >> eval_old;

            if (verbosity__ >= 1) {
                out__ << "diagonalize " << N - num_locked << " x " << N - num_locked << " Hamiltonian" << std::endl;
            }

            //kp.message(3, __function_name__, "Computing %d pre-Ritz pairs\n", num_bands__ - num_locked);
            /* solve standard eigen-value problem with the size N */
            if (std_solver.solve(N - num_locked, num_bands__ - num_locked, hmlt, &eval[num_locked], evec)) {
                std::stringstream s;
                s << "error in diagonalziation";
                RTE_THROW(s);
            } else {
                ctx.evp_work_count(std::pow(static_cast<double>(N - num_locked) / num_bands__, 3));
            }

            //kp.message(3, __function_name__, "step: %i, current subspace size: %i, maximum subspace size: %i\n",
            //    iter_step, N, num_phi);
            if (verbosity__ >= 2) {
                for (int i = 0; i < num_bands__; i++) {
                    out__ << "eval[" << i << "]=" << eval[i]
                          << ", eval_old[" << i << "]=" << eval_old[i]
                          << ", diff=" << std::abs(eval[i] - eval_old[i])
                          << ", tol=" << tolerance__(i, ispin_step)
                          << ", converged=" << is_converged(i, ispin_step)
                          << ", occ=" << occupancy__(i, ispin_step) << std::endl;
                }
            }
            result.niter++;

        } /* loop over iterative steps k */
    } /* loop over ispin_step */
    PROFILE_STOP("sirius::davidson|iter");

    if (is_device_memory(ctx.preferred_memory_t())) {
        for (int ispn = 0; ispn < num_spins; ispn++) {
            psi__.pw_coeffs(ispn).copy_to(memory_t::host, 0, num_bands__);
            psi__.pw_coeffs(ispn).deallocate(memory_t::device);
        }
    }

    kp.release_hubbard_orbitals_on_device();
    return result;
}

}

#endif
