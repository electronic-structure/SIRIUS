// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

// #include ...

// namespace ...

template <typename T>
inline void davidson(Hamiltonian_k& Hk__, Wave_functions& psi__, int subspace_size__, int num_mag_dims__)
{
    PROFILE("sirius::davidson");

    /* number of KS wave-functions to compute */
    int const num_bands = psi__.num_wf();
    /* number of spins */
    int const num_spins = psi__.num_sc();

    //ctx_.print_memory_usage(__FILE__, __LINE__);

    //auto& itso = ctx_.iterative_solver_input();

    //bool converge_by_energy = (itso.converge_by_energy_ == 1);

    //ctx_.message(2, __func__, "iterative solver tolerance: %18.12f\n", ctx_.iterative_solver_tolerance());

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
    //utils::timer t2("sirius::Band::diag_pseudo_potential_davidson|alloc");

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

    //ctx_.print_memory_usage(__FILE__, __LINE__);
    //t2.stop();

    /* get diagonal elements for preconditioning */
    auto h_o_diag = Hk__.get_h_o_diag_pw<T, 3>();

    //if (ctx_.control().print_checksum_) {
    //    auto cs1 = h_diag.checksum();
    //    auto cs2 = o_diag.checksum();
    //    kp__->comm().allreduce(&cs1, 1);
    //    kp__->comm().allreduce(&cs2, 1);
    //    if (kp__->comm().rank() == 0) {
    //        utils::print_checksum("h_diag", cs1);
    //        utils::print_checksum("o_diag", cs2);
    //    }
    //}

    auto& std_solver = ctx.std_evp_solver();
    auto& gen_solver = ctx.gen_evp_solver();

    //if (ctx_.control().print_checksum_) {
    //    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
    //        auto cs = psi.checksum_pw(get_device_t(psi.preferred_memory_t()), ispn, 0, num_bands);
    //        std::stringstream s;
    //        s << "input spinor_wave_functions_" << ispn;
    //        if (kp__->comm().rank() == 0) {
    //            utils::print_checksum(s.str(), cs);
    //        }
    //    }
    //}

    int niter{0};

    //utils::timer t3("sirius::Band::diag_pseudo_potential_davidson|iter");
    for (int ispin_step = 0; ispin_step < num_spin_steps; ispin_step++) {

        std::vector<double> eval(num_bands);
        std::vector<double> eval_old(num_bands, 1e100);

    //    if (itso.init_eval_old_) {
    //        for (int j = 0; j < num_bands; j++) {
    //            eval_old[j] = kp__->band_energy(j, ispin_step);
    //        }
    //    }

        /* trial basis functions */
        for (int ispn = 0; ispn < num_sc; ispn++) {
            phi.copy_from(psi__, num_bands, nc_mag ? ispn : ispin_step, 0, ispn, 0);
        }
        //if (ctx_.control().print_checksum_) {
        //    for (int ispn = 0; ispn < num_sc; ispn++) {
        //        auto cs = phi.checksum_pw(get_device_t(phi.preferred_memory_t()), ispn, 0, num_bands);
        //        std::stringstream s;
        //        s << "input phi" << ispn;
        //        if (kp__->comm().rank() == 0) {
        //            utils::print_checksum(s.str(), cs);
        //        }
        //    }
        //}

        /* fisrt phase: setup and diagonalize reduced Hamiltonian and get eigen-values;
         * this is done before the main itertive loop */

        /* apply Hamiltonian and S operators to the basis functions */
        Hk__.apply_h_s<T>(nc_mag ? 2 : ispin_step, 0, num_bands, phi, &hphi, &sphi);

    //    /* setup eigen-value problem
    //     * N is the number of previous basis functions
    //     * n is the number of new basis functions */
    //    set_subspace_mtrx(0, num_bands, phi, hphi, hmlt, &hmlt_old);
    //    /* setup overlap matrix */
    //    set_subspace_mtrx(0, num_bands, phi, sphi, ovlp, &ovlp_old);

    //    if (ctx_.control().verification_ >= 1) {
    //        double max_diff = check_hermitian(hmlt, num_bands);
    //        if (max_diff > 1e-12) {
    //            std::stringstream s;
    //            s << "H matrix is not hermitian, max_err = " << max_diff;
    //            WARNING(s);
    //        }
    //        max_diff = check_hermitian(ovlp, num_bands);
    //        if (max_diff > 1e-12) {
    //            std::stringstream s;
    //            s << "S matrix is not hermitian, max_err = " << max_diff;
    //            WARNING(s);
    //        }

    //    }
    //    if (ctx_.control().verification_ >= 2 && ctx_.control().verbosity_ >= 2) {
    //        hmlt.serialize("H matrix", num_bands);
    //        ovlp.serialize("S matrix", num_bands);
    //    }

    //    /* current subspace size */
    //    int N = num_bands;

    //    utils::timer t1("sirius::Band::diag_pseudo_potential_davidson|evp");
    //    /* solve generalized eigen-value problem with the size N and get lowest num_bands eigen-vectors */
    //    if (gen_solver.solve(N, num_bands, hmlt, ovlp, eval.data(), evec)) {
    //        std::stringstream s;
    //        s << "error in diagonalziation";
    //        TERMINATE(s);
    //    }
    //    t1.stop();

    //    evp_work_count() += 1;

    //    if (ctx_.control().verbosity_ >= 4 && kp__->comm().rank() == 0) {
    //        for (int i = 0; i < num_bands; i++) {
    //            printf("eval[%i]=%20.16f\n", i, eval[i]);
    //        }
    //    }

    //    /* number of newly added basis functions */
    //    int n{0};

    //    /* second phase: start iterative diagonalization */
    //    for (int k = 0; k < itso.num_steps_; k++) {

    //        /* don't compute residuals on last iteration */
    //        if (k != itso.num_steps_ - 1) {
    //            /* get new preconditionined residuals, and also hpsi and opsi as a by-product */
    //            n = residuals<T>(kp__, nc_mag ? 2 : ispin_step, N, num_bands, eval, eval_old, evec, hphi,
    //                             sphi, hpsi, spsi, res, h_diag, o_diag, itso.energy_tolerance_,
    //                             itso.residual_tolerance_);
    //        }

    //        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
    //        if (N + n > num_phi || n <= itso.min_num_res_ || k == (itso.num_steps_ - 1)) {
    //            utils::timer t1("sirius::Band::diag_pseudo_potential_davidson|update_phi");
    //            /* recompute wave-functions */
    //            /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
    //            if (ctx_.settings().always_update_wf_ || k + n > 0) {
    //                /* in case of non-collinear magnetism transform two components */
    //                transform<T>(ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), nc_mag ? 2 : ispin_step, {&phi}, 0, N, evec, 0, 0,
    //                             {&psi}, 0, num_bands);
    //                /* update eigen-values */
    //                for (int j = 0; j < num_bands; j++) {
    //                    kp__->band_energy(j, ispin_step, eval[j]);
    //                }
    //            } else {
    //                if (ctx_.control().verbosity_ >= 2 && kp__->comm().rank() == 0) {
    //                    printf("wave-functions are not recomputed\n");
    //                }
    //            }

    //            /* exit the loop if the eigen-vectors are converged or this is a last iteration */
    //            if (n <= itso.min_num_res_ || k == (itso.num_steps_ - 1)) {
    //                break;
    //            } else { /* otherwise, set Psi as a new trial basis */
    //                kp__->message(3, __func__, "subspace size limit reached\n");
    //                hmlt_old.zero();
    //                for (int i = 0; i < num_bands; i++) {
    //                    hmlt_old.set(i, i, eval[i]);
    //                }
    //                if (!itso.orthogonalize_) {
    //                    ovlp_old.zero();
    //                    for (int i = 0; i < num_bands; i++) {
    //                        ovlp_old.set(i, i, 1);
    //                    }
    //                }

    //                /* need to compute all hpsi and opsi states (not only unconverged) */
    //                if (converge_by_energy) {
    //                    transform<T>(ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), nc_mag ? 2 : ispin_step, 1.0,
    //                                 std::vector<Wave_functions*>({&hphi, &sphi}), 0, N, evec, 0, 0, 0.0,
    //                                 {&hpsi, &spsi}, 0, num_bands);
    //                }

    //                /* update basis functions, hphi and ophi */
    //                for (int ispn = 0; ispn < num_sc; ispn++) {
    //                    phi.copy_from(psi, num_bands, nc_mag ? ispn : ispin_step, 0, nc_mag ? ispn : 0, 0);
    //                    hphi.copy_from(hpsi, num_bands, ispn, 0, ispn, 0);
    //                    sphi.copy_from(spsi, num_bands, ispn, 0, ispn, 0);
    //                }
    //                /* number of basis functions that we already have */
    //                N = num_bands;
    //            }
    //        }

    //        /* expand variational subspace with new basis vectors obtatined from residuals */
    //        for (int ispn = 0; ispn < num_sc; ispn++) {
    //            phi.copy_from(res, n, ispn, 0, ispn, N);
    //        }

    //        /* apply Hamiltonian and S operators to the new basis functions */
    //        H__.apply_h_s<T>(kp__, nc_mag ? 2 : ispin_step, N, n, phi, &hphi, &sphi);

    //        if (itso.orthogonalize_) {
    //            orthogonalize<T>(ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), nc_mag ? 2 : 0, phi, hphi, sphi, N, n, ovlp, res);
    //        }

    //        /* setup eigen-value problem
    //         * N is the number of previous basis functions
    //         * n is the number of new basis functions */
    //        set_subspace_mtrx(N, n, phi, hphi, hmlt, &hmlt_old);

    //        if (ctx_.control().verification_ >= 1) {
    //            double max_diff = check_hermitian(hmlt, N + n);
    //            if (max_diff > 1e-12) {
    //                std::stringstream s;
    //                s << "H matrix is not hermitian, max_err = " << max_diff;
    //                WARNING(s);
    //            }
    //        }

    //        if (!itso.orthogonalize_) {
    //            /* setup overlap matrix */
    //            set_subspace_mtrx(N, n, phi, sphi, ovlp, &ovlp_old);

    //            if (ctx_.control().verification_ >= 1) {
    //                double max_diff = check_hermitian(ovlp, N + n);
    //                if (max_diff > 1e-12) {
    //                    std::stringstream s;
    //                    s << "S matrix is not hermitian, max_err = " << max_diff;
    //                    WARNING(s);
    //                }
    //            }
    //        }

    //        /* increase size of the variation space */
    //        N += n;

    //        eval_old = eval;

    //        utils::timer t1("sirius::Band::diag_pseudo_potential_davidson|evp");
    //        if (itso.orthogonalize_) {
    //            /* solve standard eigen-value problem with the size N */
    //            if (std_solver.solve(N, num_bands, hmlt, eval.data(), evec)) {
    //                std::stringstream s;
    //                s << "error in diagonalziation";
    //                TERMINATE(s);
    //            }
    //        } else {
    //            /* solve generalized eigen-value problem with the size N */
    //            if (gen_solver.solve(N, num_bands, hmlt, ovlp, eval.data(), evec)) {
    //                std::stringstream s;
    //                s << "error in diagonalziation";
    //                TERMINATE(s);
    //            }
    //        }
    //        t1.stop();

    //        evp_work_count() += std::pow(static_cast<double>(N) / num_bands, 3);

    //        if (ctx_.control().verbosity_ >= 2 && kp__->comm().rank() == 0) {
    //            printf("step: %i, current subspace size: %i, maximum subspace size: %i\n", k, N, num_phi);
    //            if (ctx_.control().verbosity_ >= 4) {
    //                for (int i = 0; i < num_bands; i++) {
    //                    printf("eval[%i]=%20.16f, diff=%20.16f, occ=%20.16f\n", i, eval[i], std::abs(eval[i] - eval_old[i]),
    //                         kp__->band_occupancy(i, ispin_step));
    //                }
    //            }
    //        }
    //        niter++;
    //    }
    } /* loop over ispin_step */
    //t3.stop();

    ////if (ctx_.control().print_checksum_) {
    ////    auto cs = psi.checksum(0, ctx_.num_fv_states());
    ////    if (kp__->comm().rank() == 0) {
    ////        DUMP("checksum(psi): %18.10f %18.10f", cs.real(), cs.imag());
    ////    }
    ////}
    //if (is_device_memory(ctx_.preferred_memory_t())) {
    //    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
    //        psi.pw_coeffs(ispn).copy_to(memory_t::host, 0, num_bands);
    //        psi.pw_coeffs(ispn).deallocate(memory_t::device);
    //    }
    //}

    ////== std::cout << "checking psi" << std::endl;
    ////== for (int i = 0; i < ctx_.num_bands(); i++) {
    ////==     for (int j = 0; j < ctx_.num_bands(); j++) {
    ////==         double_complex z(0, 0);
    ////==         for (int ig = 0; ig < kp__->num_gkvec(); ig++) {
    ////==             z += std::conj(psi.pw_coeffs(0).prime(ig, i)) * psi.pw_coeffs(0).prime(ig, j);
    ////==         }
    ////==         if (i == j) {
    ////==             z -= 1.0;
    ////==         }
    ////==         if (std::abs(z) > 1e-10) {
    ////==             std::cout << "non-orthogonal  wave-functions " << i << " " << j << ", diff: " << z << std::endl;
    ////==         }
    ////==     }
    ////== }

    //return niter;
}

#endif
