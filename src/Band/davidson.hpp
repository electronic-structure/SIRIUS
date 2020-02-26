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
#include "residuals.hpp"

/// Solve the eigen-problem using Davidson iterative method.
/**
\tparam T  type of the wave-functions in real space
\param [in] Hk  Hamiltonian for a given k-point
\return list of eigen-values
*/
template <typename T>
inline mdarray<double, 1>
davidson(Hamiltonian_k& Hk__, Wave_functions& psi__, int num_mag_dims__, int subspace_size__, int num_steps__,
         double eval_tolerance__, double eval_tolerance_empty__, double norm_tolerance__,
         std::function<double(int, int)> occupancy__, bool keep_phi_orthogonal__ = true)
{
    PROFILE("sirius::davidson");

    /* number of KS wave-functions to compute */
    int const num_bands = psi__.num_wf();
    /* number of spins */
    int const num_spins = psi__.num_sc();

    bool const estimate_eval{true};

    bool const skip_initial_diag{false};

    //ctx_.print_memory_usage(__FILE__, __LINE__);

    //bool converge_by_energy = (itso.converge_by_energy_ == 1);

    //ctx_.message(2, __function_name__, "iterative solver tolerance: %18.12f\n", ctx_.iterative_solver_tolerance());

    /* true if this is a non-collinear case */
    bool const nc_mag = (num_mag_dims__ == 3);

    /* number of spin components, treated simultaneously
     *   1 - in case of non-magnetic or collinear calculation
     *   2 - in case of non-collinear calculation */
    int const num_sc = nc_mag ? 2 : 1;

    int const num_spin_steps = nc_mag ? 1 : num_spins;

    /* maximum subspace size */
    int const num_phi = subspace_size__ * num_bands;

    auto& ctx = Hk__.H0().ctx();
    auto& gkvecp = Hk__.kp().gkvec_partition();

    //if (num_phi > kp__->num_gkvec()) {
    //    std::stringstream s;
    //    s << "subspace size is too large!";
    //    TERMINATE(s);
    //}

    /* alias for memory pool */
    auto& mp = ctx.mem_pool(ctx.host_memory_t());

    /* allocate wave-functions */

    /* auxiliary wave-functions */
    Wave_functions phi(mp, gkvecp, num_phi, ctx.aux_preferred_memory_t(), num_sc);

    /* Hamiltonian, applied to auxiliary wave-functions */
    Wave_functions hphi(mp, gkvecp, num_phi, ctx.preferred_memory_t(), num_sc);

    /* S operator, applied to auxiliary wave-functions */
    Wave_functions sphi(mp, gkvecp, num_phi, ctx.preferred_memory_t(), num_sc);

    /* Hamiltonain, applied to new Psi wave-functions */
    Wave_functions hpsi(mp, gkvecp, num_bands, ctx.preferred_memory_t(), num_sc);

    /* S operator, applied to new Psi wave-functions */
    Wave_functions spsi(mp, gkvecp, num_bands, ctx.preferred_memory_t(), num_sc);

    /* residuals */
    Wave_functions res(mp, gkvecp, num_bands, ctx.preferred_memory_t(), num_sc);

    const int bs = ctx.cyclic_block_size();

    dmatrix<T> hmlt(mp, num_phi, num_phi, ctx.blacs_grid(), bs, bs);
    dmatrix<T> ovlp(mp, num_phi, num_phi, ctx.blacs_grid(), bs, bs);
    dmatrix<T> evec(mp, num_phi, num_phi, ctx.blacs_grid(), bs, bs);
    dmatrix<T> hmlt_old(mp, num_phi, num_phi, ctx.blacs_grid(), bs, bs);
    dmatrix<T> ovlp_old(mp, num_phi, num_phi, ctx.blacs_grid(), bs, bs);

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

    Hk__.kp().copy_hubbard_orbitals_on_device();

    //ctx_.print_memory_usage(__FILE__, __LINE__);
    //t2.stop();

    /* get diagonal elements for preconditioning */
    auto h_o_diag = Hk__.get_h_o_diag_pw<T, 3>();

    auto& std_solver = ctx.std_evp_solver();
    auto& gen_solver = ctx.gen_evp_solver();

    int niter{0};

    mdarray<double, 1> eval_out(num_bands);

    for (int ispin_step = 0; ispin_step < num_spin_steps; ispin_step++) {

        mdarray<double, 1> eval(num_bands);
        mdarray<double, 1> eval_old(num_bands);
        eval_old = [](){return 1e10;};

        /* check if band energy is converged */
        auto is_converged = [&](int j__, int ispn__) -> bool
        {
            double o1 = std::abs(occupancy__(j__, ispn__));
            double o2 = std::abs(1 - o1);

            double tol = o1 * eval_tolerance__ + o2 * (eval_tolerance__ + eval_tolerance_empty__);
            if (std::abs(eval[j__] - eval_old[j__]) > tol) {
                return false;
            } else {
                return true;
            }
        };

        /* trial basis functions */
        for (int ispn = 0; ispn < num_sc; ispn++) {
            phi.copy_from(psi__, num_bands, nc_mag ? ispn : ispin_step, 0, ispn, 0);
        }

        /* fisrt phase: setup and diagonalize reduced Hamiltonian and get eigen-values;
         * this is done before the main itertive loop */

        /* apply Hamiltonian and S operators to the basis functions */
        Hk__.apply_h_s<T>(nc_mag ? 2 : ispin_step, 0, num_bands, phi, &hphi, &sphi);

        if (keep_phi_orthogonal__) {
            orthogonalize<T>(ctx.preferred_memory_t(), ctx.blas_linalg_t(), nc_mag ? 2 : 0, phi, hphi, sphi, 0, num_bands, ovlp, res);
        }

        /* setup eigen-value problem */
        Band(ctx).set_subspace_mtrx<T>(0, num_bands, phi, hphi, hmlt, &hmlt_old);
        if (!keep_phi_orthogonal__) {
            /* setup overlap matrix */
            Band(ctx).set_subspace_mtrx<T>(0, num_bands, phi, sphi, ovlp, &ovlp_old);
        }

        /* current subspace size */
        int N = num_bands;

        if (skip_initial_diag) {
            mdarray<double, 2> hsdiag(num_bands, 2);
            hsdiag.zero();
            #pragma omp parallel for
            for (int i = 0; i < num_bands; i++) {
                for (int ig = 0; ig < gkvecp.gvec().count(); ig++) {
                    hsdiag(i, 0) += std::real(std::conj(phi.pw_coeffs(0).prime(ig, i)) * hphi.pw_coeffs(0).prime(ig, i));
                    hsdiag(i, 1) += std::real(std::conj(phi.pw_coeffs(0).prime(ig, i)) * sphi.pw_coeffs(0).prime(ig, i));
                }
            }
            for (int i = 0; i < num_bands; i++) {
                eval[i] = hsdiag(i, 0) / hsdiag(i, 1);
            }
        } else {
            PROFILE("sirius::davidson|evp");
            /* solve generalized eigen-value problem with the size N and get lowest num_bands eigen-vectors */
            if (keep_phi_orthogonal__) {
                if (std_solver.solve(N, num_bands, hmlt, eval.at(memory_t::host), evec)) {
                    std::stringstream s;
                    s << "error in diagonalziation";
                    TERMINATE(s);
                }
            } else  {
                if (gen_solver.solve(N, num_bands, hmlt, ovlp, eval.at(memory_t::host), evec)) {
                    std::stringstream s;
                    s << "error in diagonalziation";
                    TERMINATE(s);
                }
            }
        }

        ////if (ctx_.control().verbosity_ >= 4 && kp__->comm().rank() == 0) {
        //    for (int i = 0; i < num_bands; i++) {
        //        std::printf("eval[%i]=%20.16f\n", i, eval[i]);
        //    }
        ////}

        /* number of newly added basis functions */
        int n{0};

        /* second phase: start iterative diagonalization */
        for (int k = 1; k <= num_steps__; k++) {

            /* don't compute residuals on last iteration */
            if (k != num_steps__) {
                if (k == 1 && skip_initial_diag) {
                    /* get residuals */
                    n = normalized_preconditioned_residuals<T>(ctx.preferred_memory_t(), nc_mag ? spin_range(2) : spin_range(ispin_step),
                                                           num_bands, eval, hphi, sphi, res, h_o_diag.first,
                                                           h_o_diag.second, norm_tolerance__);
                } else {
                    /* get new preconditionined residuals, and also hpsi and opsi as a by-product */
                    n = residuals<T>(ctx.preferred_memory_t(), ctx.blas_linalg_t(), nc_mag ? 2 : ispin_step,
                                     N, num_bands, eval, evec, hphi, sphi, hpsi, spsi, res, h_o_diag.first, h_o_diag.second,
                                     estimate_eval, norm_tolerance__, is_converged);
                }
            }

            /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
            if (N + n > num_phi || n == 0 || k == num_steps__) {
                PROFILE("sirius::davidson|update_phi");
                /* recompute wave-functions */
                /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
                /* in case of non-collinear magnetism transform two components */
                transform<T>(ctx.preferred_memory_t(), ctx.blas_linalg_t(), nc_mag ? 2 : ispin_step, {&phi}, 0, N,
                             evec, 0, 0, {&psi__}, 0, num_bands);

                /* exit the loop if the eigen-vectors are converged or this is a last iteration */
                if (n == 0 || k == num_steps__) {
                    break;
                } else { /* otherwise, set Psi as a new trial basis */
                    hmlt_old.zero();
                    for (int i = 0; i < num_bands; i++) {
                        hmlt_old.set(i, i, eval[i]);
                    }
                    if (!keep_phi_orthogonal__) {
                        ovlp_old.zero();
                        for (int i = 0; i < num_bands; i++) {
                            ovlp_old.set(i, i, 1);
                        }
                    }

                    /* need to compute all hpsi and opsi states (not only unconverged) */
                    if (estimate_eval) {
                        transform<T>(ctx.preferred_memory_t(), ctx.blas_linalg_t(), nc_mag ? 2 : ispin_step, 1.0,
                                     std::vector<Wave_functions*>({&hphi, &sphi}), 0, N, evec, 0, 0, 0.0,
                                     {&hpsi, &spsi}, 0, num_bands);
                    }

                    /* update basis functions, hphi and ophi */
                    for (int ispn = 0; ispn < num_sc; ispn++) {
                        phi.copy_from(psi__, num_bands, nc_mag ? ispn : ispin_step, 0, ispn, 0);
                        hphi.copy_from(hpsi, num_bands, ispn, 0, ispn, 0);
                        sphi.copy_from(spsi, num_bands, ispn, 0, ispn, 0);
                    }
                    /* number of basis functions that we already have */
                    N = num_bands;
                }
            }

            /* expand variational subspace with new basis vectors obtatined from residuals */
            for (int ispn = 0; ispn < num_sc; ispn++) {
                phi.copy_from(res, n, ispn, 0, ispn, N);
            }

            /* apply Hamiltonian and S operators to the new basis functions */
            Hk__.apply_h_s<T>(nc_mag ? 2 : ispin_step, N, n, phi, &hphi, &sphi);

            if (keep_phi_orthogonal__) {
                orthogonalize<T>(ctx.preferred_memory_t(), ctx.blas_linalg_t(), nc_mag ? 2 : 0, phi, hphi, sphi, N, n, ovlp, res);
            }

            /* setup eigen-value problem
             * N is the number of previous basis functions
             * n is the number of new basis functions */
            Band(ctx).set_subspace_mtrx<T>(N, n, phi, hphi, hmlt, &hmlt_old);

            //if (ctx_.control().verification_ >= 1) {
            //    double max_diff = check_hermitian(hmlt, N + n);
            //    if (max_diff > 1e-12) {
            //        std::stringstream s;
            //        s << "H matrix is not hermitian, max_err = " << max_diff;
            //        WARNING(s);
            //    }
            //}

            if (!keep_phi_orthogonal__) {
                /* setup overlap matrix */
                Band(ctx).set_subspace_mtrx<T>(N, n, phi, sphi, ovlp, &ovlp_old);

                //if (ctx_.control().verification_ >= 1) {
                //    double max_diff = check_hermitian(ovlp, N + n);
                //    if (max_diff > 1e-12) {
                //        std::stringstream s;
                //        s << "S matrix is not hermitian, max_err = " << max_diff;
                //        WARNING(s);
                //    }
                //}
            }

            /* increase size of the variation space */
            N += n;

            eval >> eval_old;

            PROFILE_START("sirius::davidson|evp");
            if (keep_phi_orthogonal__) {
                /* solve standard eigen-value problem with the size N */
                if (std_solver.solve(N, num_bands, hmlt, eval.at(memory_t::host), evec)) {
                    std::stringstream s;
                    s << "error in diagonalziation";
                    TERMINATE(s);
                }
            } else {
                /* solve generalized eigen-value problem with the size N */
                if (gen_solver.solve(N, num_bands, hmlt, ovlp, eval.at(memory_t::host), evec)) {
                    std::stringstream s;
                    s << "error in diagonalziation";
                    TERMINATE(s);
                }
            }
            PROFILE_STOP("sirius::davidson|evp");

            //if (ctx_.control().verbosity_ >= 2 && kp__->comm().rank() == 0) {
                std::printf("step: %i, current subspace size: %i, maximum subspace size: %i\n", k, N, num_phi);
                //if (ctx_.control().verbosity_ >= 4) {
                    for (int i = 0; i < num_bands; i++) {
                        std::printf("eval[%i]=%20.16f, diff=%20.16f, occ=%20.16f\n", i, eval[i], std::abs(eval[i] - eval_old[i]),
                             occupancy__(i, ispin_step));
                    }
                //}
            //}
            //niter++;
        } /* loop over iterative steps k */
        for (int i = 0; i < num_bands; i++) {
            eval_out[i] = eval[i];
        }
    } /* loop over ispin_step */
    //t3.stop();

    if (is_device_memory(ctx.preferred_memory_t())) {
        psi__.copy_to(spin_range(psi__.num_sc()), memory_t::host, 0, num_bands);
        psi__.deallocate(spin_range(psi__.num_sc()), memory_t::device);
    }

    Hk__.kp().release_hubbard_orbitals_on_device();
    //return niter;
    return eval_out;
}

#endif
