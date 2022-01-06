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

/** \file diag_full_potential.cpp
 *
 *  \brief Diagonalization of full-potential Hamiltonian.
 */

#include "band.hpp"
#include "residuals.hpp"
#include "wf_inner.hpp"
#include "wf_ortho.hpp"
#include "wf_trans.hpp"
#include "context/simulation_context.hpp"
#include "k_point/k_point.hpp"
#include "utils/profiler.hpp"
#include "davidson.hpp"

namespace sirius {

void
Band::diag_full_potential_first_variation_exact(Hamiltonian_k<double>& Hk__) const
{
    PROFILE("sirius::Band::diag_fv_exact");

    auto& kp = Hk__.kp();

    auto& solver = ctx_.gen_evp_solver();

    /* total eigen-value problem size */
    int ngklo = kp.gklo_basis_size();

    /* block size of scalapack 2d block-cyclic distribution */
    int bs = ctx_.cyclic_block_size();

    sddk::dmatrix<double_complex> h(ngklo, ngklo, ctx_.blacs_grid(), bs, bs, ctx_.mem_pool(solver.host_memory_t()));
    sddk::dmatrix<double_complex> o(ngklo, ngklo, ctx_.blacs_grid(), bs, bs, ctx_.mem_pool(solver.host_memory_t()));

    /* setup Hamiltonian and overlap */
    Hk__.set_fv_h_o(h, o);

    if (ctx_.gen_evp_solver().type() == ev_solver_t::cusolver) {
        auto& mpd = ctx_.mem_pool(memory_t::device);
        h.allocate(mpd);
        o.allocate(mpd);
        kp.fv_eigen_vectors().allocate(mpd);
    }

    if (ctx_.cfg().control().verification() >= 1) {
        double max_diff = check_hermitian(h, ngklo);
        if (max_diff > 1e-12) {
            std::stringstream s;
            s << "H matrix is not hermitian" << std::endl
              << "max error: " << max_diff;
            TERMINATE(s);
        }
        max_diff = check_hermitian(o, ngklo);
        if (max_diff > 1e-12) {
            std::stringstream s;
            s << "O matrix is not hermitian" << std::endl
              << "max error: " << max_diff;
            TERMINATE(s);
        }
    }

    if (ctx_.print_checksum()) {
        auto z1 = h.checksum();
        auto z2 = o.checksum();
        kp.comm().allreduce(&z1, 1);
        kp.comm().allreduce(&z2, 1);
        if (kp.comm().rank() == 0) {
            utils::print_checksum("h_lapw", z1);
            utils::print_checksum("o_lapw", z2);
        }
    }

    assert(kp.gklo_basis_size() > ctx_.num_fv_states());

    std::vector<double> eval(ctx_.num_fv_states());

    ctx_.print_memory_usage(__FILE__, __LINE__);
    if (solver.solve(kp.gklo_basis_size(), ctx_.num_fv_states(), h, o, eval.data(), kp.fv_eigen_vectors())) {
        RTE_THROW("error in generalized eigen-value problem");
    }
    ctx_.print_memory_usage(__FILE__, __LINE__);

    if (ctx_.gen_evp_solver().type() == ev_solver_t::cusolver) {
        h.deallocate(memory_t::device);
        o.deallocate(memory_t::device);
        kp.fv_eigen_vectors().deallocate(memory_t::device);
    }
    kp.set_fv_eigen_values(&eval[0]);

    for (int i = 0; i < ctx_.num_fv_states(); i++) {
        kp.message(4, __function_name__, "eval[%i]=%20.16f\n", i, eval[i]);
    }

    if (ctx_.print_checksum()) {
        auto z1 = kp.fv_eigen_vectors().checksum();
        kp.comm().allreduce(&z1, 1);
        if (kp.comm().rank() == 0) {
            utils::print_checksum("fv_eigen_vectors", z1);
        }
    }

    /* remap to slab */
    kp.fv_eigen_vectors_slab().pw_coeffs(0).remap_from(kp.fv_eigen_vectors(), 0);
    kp.fv_eigen_vectors_slab().mt_coeffs(0).remap_from(kp.fv_eigen_vectors(), kp.num_gkvec());

    /* renormalize wave-functions */
    if (ctx_.valence_relativity() == relativity_t::iora) {
        Wave_functions<double> ofv(kp.gkvec_partition(), unit_cell_.num_atoms(),
                           [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states(),
                           ctx_.preferred_memory_t(), 1);
        if (ctx_.processing_unit() == device_t::GPU) {
            kp.fv_eigen_vectors_slab().allocate(spin_range(0), memory_t::device);
            kp.fv_eigen_vectors_slab().copy_to(spin_range(0), memory_t::device, 0, ctx_.num_fv_states());
            ofv.allocate(spin_range(0), memory_t::device);
        }

        Hk__.apply_fv_h_o(false, false, 0, ctx_.num_fv_states(), kp.fv_eigen_vectors_slab(), nullptr, &ofv);

        if (ctx_.processing_unit() == device_t::GPU) {
            kp.fv_eigen_vectors_slab().deallocate(spin_range(0), memory_t::device);
        }

        //if (true) {
        //    Wave_functions phi(kp.gkvec_partition(), unit_cell_.num_atoms(),
        //                       [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states(),
        //                       ctx_.preferred_memory_t(), 1);
        //    Wave_functions ofv(kp.gkvec_partition(), unit_cell_.num_atoms(),
        //                       [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states(),
        //                       ctx_.preferred_memory_t(), 1);
        //    phi.allocate(spin_range(0), memory_t::device);
        //    ofv.allocate(spin_range(0), memory_t::device);

        //    for (int i = 0; i < kp.num_gkvec(); i++) {
        //        phi.zero(device_t::CPU, 0, 0, ctx_.num_fv_states());
        //        for (int j = 0; j < ctx_.num_fv_states(); j++) {
        //            phi.pw_coeffs(0).prime(i, j) = 1.0;
        //        }
        //        phi.copy_to(spin_range(0), memory_t::device, 0, ctx_.num_fv_states());
        //        Hk__.apply_fv_h_o(false, false, 0, ctx_.num_fv_states(), phi, nullptr, &ofv);
        //    }

        //    for (int i = 0; i < unit_cell_.mt_lo_basis_size(); i++) {
        //        phi.zero(device_t::CPU, 0, 0, ctx_.num_fv_states());
        //        for (int j = 0; j < ctx_.num_fv_states(); j++) {
        //            phi.mt_coeffs(0).prime(i, j) = 1.0;
        //        }
        //        phi.copy_to(spin_range(0), memory_t::device, 0, ctx_.num_fv_states());
        //        Hk__.apply_fv_h_o(false, false, 0, ctx_.num_fv_states(), phi, nullptr, &ofv);
        //    }
        //}

        std::vector<double> norm(ctx_.num_fv_states(), 0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < ctx_.num_fv_states(); i++) {
            for (int j = 0; j < ofv.pw_coeffs(0).num_rows_loc(); j++) {
                norm[i] += std::real(std::conj(kp.fv_eigen_vectors_slab().pw_coeffs(0).prime(j, i)) * ofv.pw_coeffs(0).prime(j, i));
            }
            for (int j = 0; j < ofv.mt_coeffs(0).num_rows_loc(); j++) {
                norm[i] += std::real(std::conj(kp.fv_eigen_vectors_slab().mt_coeffs(0).prime(j, i)) * ofv.mt_coeffs(0).prime(j, i));
            }
        }
        kp.comm().allreduce(norm);
        if (ctx_.verbosity() >= 2) {
            for (int i = 0; i < ctx_.num_fv_states(); i++) {
                kp.message(2, __function_name__, "norm(%i)=%18.12f\n", i, norm[i]);
            }
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < ctx_.num_fv_states(); i++) {
            norm[i] = 1 / std::sqrt(norm[i]);
            for (int j = 0; j < ofv.pw_coeffs(0).num_rows_loc(); j++) {
                kp.fv_eigen_vectors_slab().pw_coeffs(0).prime(j, i) *= norm[i];
            }
            for (int j = 0; j < ofv.mt_coeffs(0).num_rows_loc(); j++) {
                kp.fv_eigen_vectors_slab().mt_coeffs(0).prime(j, i) *= norm[i];
            }
        }
    }

    if (ctx_.cfg().control().verification() >= 2) {
        kp.message(1, __function_name__, "%s", "checking application of H and O\n");
        /* check application of H and O */
        Wave_functions<double> hphi(kp.gkvec_partition(), unit_cell_.num_atoms(),
                            [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states(),
                            ctx_.preferred_memory_t());
        Wave_functions<double> ophi(kp.gkvec_partition(), unit_cell_.num_atoms(),
                            [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states(),
                            ctx_.preferred_memory_t());

        if (ctx_.processing_unit() == device_t::GPU) {
            kp.fv_eigen_vectors_slab().allocate(spin_range(0), memory_t::device);
            kp.fv_eigen_vectors_slab().copy_to(spin_range(0), memory_t::device, 0, ctx_.num_fv_states());
            hphi.allocate(spin_range(0), memory_t::device);
            ophi.allocate(spin_range(0), memory_t::device);
        }

        Hk__.apply_fv_h_o(false, false, 0, ctx_.num_fv_states(), kp.fv_eigen_vectors_slab(), &hphi, &ophi);

        dmatrix<double_complex> hmlt(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(),
                                     ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
        dmatrix<double_complex> ovlp(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(),
                                     ctx_.cyclic_block_size(), ctx_.cyclic_block_size());

        inner(ctx_.spla_context(), spin_range(0), kp.fv_eigen_vectors_slab(), 0, ctx_.num_fv_states(),
              hphi, 0, ctx_.num_fv_states(), hmlt, 0, 0);
        inner(ctx_.spla_context(), spin_range(0), kp.fv_eigen_vectors_slab(), 0, ctx_.num_fv_states(),
              ophi, 0, ctx_.num_fv_states(), ovlp, 0, 0);

        double max_diff{0};
        for (int i = 0; i < hmlt.num_cols_local(); i++) {
            int icol = hmlt.icol(i);
            for (int j = 0; j < hmlt.num_rows_local(); j++) {
                int jrow = hmlt.irow(j);
                if (icol == jrow) {
                    max_diff = std::max(max_diff, std::abs(hmlt(j, i) - eval[icol]));
                } else {
                    max_diff = std::max(max_diff, std::abs(hmlt(j, i)));
                }
            }
        }
        if (max_diff > 1e-9) {
            std::stringstream s;
            s << "application of Hamiltonian failed, maximum error: " << max_diff;
            WARNING(s);
        }

        max_diff = 0;
        for (int i = 0; i < ovlp.num_cols_local(); i++) {
            int icol = ovlp.icol(i);
            for (int j = 0; j < ovlp.num_rows_local(); j++) {
                int jrow = ovlp.irow(j);
                if (icol == jrow) {
                    max_diff = std::max(max_diff, std::abs(ovlp(j, i) - 1.0));
                } else {
                    max_diff = std::max(max_diff, std::abs(ovlp(j, i)));
                }
            }
        }
        if (max_diff > 1e-9) {
            std::stringstream s;
            s << "application of overlap failed, maximum error: " << max_diff;
            WARNING(s);
        }
    }
}

void Band::get_singular_components(Hamiltonian_k<double>& Hk__) const
{
    PROFILE("sirius::Band::get_singular_components");

    auto& kp = Hk__.kp();

    auto& psi = kp.singular_components();

    int ncomp = psi.num_wf();

    ctx_.message(3, __function_name__, "number of singular components: %i\n", ncomp);

    auto& itso = ctx_.cfg().iterative_solver();

    auto result = davidson<double_complex, double_complex, davidson_evp_t::overlap>(Hk__, ncomp, 0, psi,
            [](int i, int ispn){ return 1e-10; }, itso.residual_tolerance(), itso.num_steps(), itso.locking(),
            itso.subspace_size(), itso.converge_by_energy(), itso.extra_ortho(), std::cout, 0);

    kp.message(2, __function_name__, "smallest eigen-value of the singular components: %20.16f\n", result.eval[0]);
}

void Band::diag_full_potential_first_variation_davidson(Hamiltonian_k<double>& Hk__) const
{
    PROFILE("sirius::Band::diag_fv_davidson");

    auto& kp = Hk__.kp();

    auto h_o_diag = Hk__.get_h_o_diag_lapw<3>();

    if (ctx_.print_checksum()) {
        auto cs1 = h_o_diag.first.checksum();
        auto cs2 = h_o_diag.second.checksum();
        if (kp.comm().rank() == 0) {
            utils::print_checksum("h_dial_lapw", cs1);
            utils::print_checksum("o_diag_lapw", cs2);
         }
    }

    get_singular_components(Hk__);

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_fv_states();

    auto& itso = ctx_.cfg().iterative_solver();

    /* short notation for target wave-functions */
    auto& psi = kp.fv_eigen_vectors_slab();

    /* total number of local orbitals */
    int nlo = ctx_.unit_cell().mt_lo_basis_size();

    /* number of singular components */
    int ncomp = kp.singular_components().num_wf();

    /* number of auxiliary basis functions */
    int num_phi = nlo + ncomp + itso.subspace_size() * num_bands;
    /* sanity check */
    if (num_phi >= kp.num_gkvec()) {
        TERMINATE("subspace is too big");
    }

    /* allocate wave-functions */
    Wave_functions<double> phi(kp.gkvec_partition(), unit_cell_.num_atoms(),
                       [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, num_phi,
                       ctx_.preferred_memory_t());
    Wave_functions<double> hphi(kp.gkvec_partition(), unit_cell_.num_atoms(),
                        [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, num_phi,
                        ctx_.preferred_memory_t());
    Wave_functions<double> ophi(kp.gkvec_partition(), unit_cell_.num_atoms(),
                        [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, num_phi,
                        ctx_.preferred_memory_t());
    Wave_functions<double> hpsi(kp.gkvec_partition(), unit_cell_.num_atoms(),
                        [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, num_bands,
                        ctx_.preferred_memory_t());
    Wave_functions<double> opsi(kp.gkvec_partition(), unit_cell_.num_atoms(),
                        [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, num_bands,
                        ctx_.preferred_memory_t());

    /* residuals */
    /* res is also used as a temporary array in orthogonalize() and the first time nlo + ncomp + num_bands
     * states will be orthogonalized */
    Wave_functions<double> res(kp.gkvec_partition(), unit_cell_.num_atoms(),
                       [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, nlo + ncomp + num_bands,
                       ctx_.preferred_memory_t());

    //auto mem_type = (gen_evp_solver_->type() == ev_magma) ? memory_t::host_pinned : memory_t::host;

    int bs = ctx_.cyclic_block_size();

    dmatrix<double_complex> hmlt(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> ovlp(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> evec(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> hmlt_old(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);

    /* add pure local orbitals to the basis */
    if (nlo) {
        phi.pw_coeffs(0).zero(memory_t::host, 0, nlo);
        phi.mt_coeffs(0).zero(memory_t::host, 0, nlo);
        for (int ialoc = 0; ialoc < phi.spl_num_atoms().local_size(); ialoc++) {
            int ia = phi.spl_num_atoms()[ialoc];
            for (int xi = 0; xi < unit_cell_.atom(ia).mt_lo_basis_size(); xi++) {
                phi.mt_coeffs(0).prime(phi.offset_mt_coeffs(ialoc) + xi, unit_cell_.atom(ia).offset_lo() + xi) = 1.0;
            }
        }
    }

    /* add singular components to the basis */
    if (ncomp != 0) {
        phi.mt_coeffs(0).zero(memory_t::host, nlo, ncomp);
        for (int j = 0; j < ncomp; j++) {
            std::memcpy(phi.pw_coeffs(0).prime().at(memory_t::host, 0, nlo + j),
                        kp.singular_components().pw_coeffs(0).prime().at(memory_t::host, 0, j),
                        phi.pw_coeffs(0).num_rows_loc() * sizeof(double_complex));
        }
    }

    if (ctx_.processing_unit() == device_t::GPU) {
        psi.allocate(spin_range(0), memory_t::device);
        psi.copy_to(spin_range(0), memory_t::device, 0, num_bands);

        phi.allocate(spin_range(0), memory_t::device);
        phi.copy_to(spin_range(0), memory_t::device, 0, nlo + ncomp);

        res.allocate(spin_range(0), memory_t::device);

        hphi.allocate(spin_range(0), memory_t::device);
        ophi.allocate(spin_range(0), memory_t::device);

        hpsi.allocate(spin_range(0), memory_t::device);
        opsi.allocate(spin_range(0), memory_t::device);

        if (ctx_.blacs_grid().comm().size() == 1) {
            evec.allocate(memory_t::device);
            ovlp.allocate(memory_t::device);
            hmlt.allocate(memory_t::device);
        }
    }

    mdarray<double, 1> eval(num_bands);
    mdarray<double, 1> eval_old(num_bands);
    eval_old = [](){return -1.0;};

    for (int i = 0; i < num_bands; i++) {
        eval[i] = kp.fv_eigen_value(i);
    }

    /* trial basis functions */
    phi.copy_from(ctx_.processing_unit(), num_bands, psi, 0, 0, 0, nlo + ncomp);

    if (ctx_.print_checksum()) {
        kp.message(1, __function_name__, "%s", "checksum of initial wave-functions\n");
        psi.print_checksum(ctx_.processing_unit(), "psi", 0, num_bands);
        phi.print_checksum(ctx_.processing_unit(), "phi", 0,  nlo + ncomp + num_bands);
    }

    /* current subspace size */
    int N = 0;

    /* number of newly added basis functions */
    int n = nlo + ncomp + num_bands;

    ctx_.print_memory_usage(__FILE__, __LINE__);

    auto& std_solver = ctx_.std_evp_solver();

    /* tolerance for the norm of L2-norms of the residuals, used for
     * relative convergence criterion. We can only compute this after
     * we have the first residual norms available */
    double relative_frobenius_tolerance{0};
    double current_frobenius_norm{0};

    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps(); k++) {

        bool last_iteration = k == (itso.num_steps() - 1);

        /* apply Hamiltonian and overlap operators to the new basis functions */
        if (k == 0) {
            Hk__.apply_fv_h_o(false, true, 0, nlo, phi, &hphi, &ophi);
            Hk__.apply_fv_h_o(false, false, nlo, ncomp + num_bands, phi, &hphi, &ophi);
        } else {
            Hk__.apply_fv_h_o(false, false, N, n, phi, &hphi, &ophi);
        }

        orthogonalize<std::complex<double>>(ctx_.spla_context(), ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), spin_range(0), phi,
                      hphi, ophi, N, n, ovlp, res);

        /* setup eigen-value problem
         * N is the number of previous basis functions
         * n is the number of new basis functions */
        set_subspace_mtrx<double_complex, double_complex>(N, n, 0, phi, hphi, hmlt, &hmlt_old);

        /* increase size of the variation space */
        N += n;

        eval >> eval_old;

        /* solve standard eigen-value problem with the size N */
        if (std_solver.solve(N, num_bands, hmlt, &eval[0], evec)) {
            std::stringstream s;
            s << "[sirius::Band::diag_full_potential_first_variation_davidson] error in diagonalziation";
            TERMINATE(s);
        }
        kp.message(2, __function_name__, "step: %i, current subspace size: %i, maximum subspace size: %i\n", k, N, num_phi);
        for (int i = 0; i < num_bands; i++) {
            kp.message(4, __function_name__, "eval[%i]=%20.16f, diff=%20.16f\n", i, eval[i], std::abs(eval[i] - eval_old[i]));
        }

        /* don't compute residuals on last iteration */
        if (!last_iteration) {
            /* get new preconditionined residuals, and also hpsi and opsi as a by-product */
            auto result = sirius::residuals<double_complex, double_complex>(
                ctx_, ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), spin_range(0), N, num_bands, 0, eval, evec, hphi, ophi, hpsi,
                opsi, res, h_o_diag.first, h_o_diag.second, itso.converge_by_energy(), itso.residual_tolerance(),
                [&](int i, int ispn) { return std::abs(eval[i] - eval_old[i]) < itso.energy_tolerance(); });
            n = result.unconverged_residuals;
            current_frobenius_norm = result.frobenius_norm;

            /* set the relative tolerance convergence criterion */
            if (k == 0) {
                relative_frobenius_tolerance = current_frobenius_norm * itso.relative_tolerance();
            }
        }

        /* verify convergence criteria */
        bool converged_by_relative_tol = k > 0 && current_frobenius_norm < relative_frobenius_tolerance ;
        bool converged_by_absolute_tol = n <= itso.min_num_res();
        bool converged = converged_by_absolute_tol || converged_by_relative_tol;

        /* check if running out of space */
        bool should_restart = N + n > num_phi;

        if (converged) {
            kp.message(3, __function_name__, "converged by %s tolerance\n", converged_by_relative_tol ? "relative" : "absolute");
        }

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (should_restart || converged || last_iteration) {
            PROFILE("sirius::Band::diag_fv_davidson|update_phi");
            /* recompute wave-functions */
            /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
            transform<double_complex, double_complex>(ctx_.spla_context(), 0, phi, 0, N, evec, 0, 0, psi, 0, num_bands);

            /* exit the loop if the eigen-vectors are converged or this is a last iteration */
            if (converged || last_iteration) {
                break;
            } else { /* otherwise, set Psi as a new trial basis */
                kp.message(3, __function_name__, "%s", "subspace size limit reached\n");
                /* update basis functions */
                /* first nlo + ncomp functions are fixed, don't update them */
                phi.copy_from(ctx_.processing_unit(), num_bands, psi, 0, 0, 0, nlo + ncomp);
                phi.copy_from(ctx_.processing_unit(), n, res, 0, 0, 0, nlo + ncomp + num_bands);
                /* number of basis functions that we already have */
                N = nlo + ncomp;
                n += num_bands;
            }
        } else {
            /* expand variational subspace with new basis vectors obtatined from residuals */
            phi.copy_from(ctx_.processing_unit(), n, res, 0, 0, 0, N);
        }
    }

    if (ctx_.processing_unit() == device_t::GPU) {
        psi.pw_coeffs(0).copy_to(memory_t::host, 0, num_bands);
        psi.mt_coeffs(0).copy_to(memory_t::host, 0, num_bands);
        psi.deallocate(spin_range(0), memory_t::device);
    }
    kp.set_fv_eigen_values(&eval[0]);
}

void Band::diag_full_potential_second_variation(Hamiltonian_k<double>& Hk__) const
{
    PROFILE("sirius::Band::diag_sv");

    auto& kp = Hk__.kp();

    if (!ctx_.need_sv()) {
        kp.bypass_sv();
        return;
    }

    sddk::mdarray<double, 2> band_energies(ctx_.num_bands(), ctx_.num_spinors());

    /* product of the second-variational Hamiltonian and a first-variational wave-function */
    std::vector<Wave_functions<double>> hpsi;
    for (int i = 0; i < ctx_.num_mag_comp(); i++) {
        hpsi.push_back(Wave_functions<double>(kp.gkvec_partition(), unit_cell_.num_atoms(),
                                      [this](int ia) { return unit_cell_.atom(ia).mt_basis_size(); },
                                      ctx_.num_fv_states(), ctx_.preferred_memory_t()));
    }

    /* compute product of magnetic field and wave-function */
    if (ctx_.num_spins() == 2) {
        Hk__.apply_b(kp.fv_states(), hpsi);
    } else {
        hpsi[0].pw_coeffs(0).prime().zero();
        hpsi[0].mt_coeffs(0).prime().zero();
    }

    ctx_.print_memory_usage(__FILE__, __LINE__);

    //== if (ctx_.uj_correction())
    //== {
    //==     apply_uj_correction<uu>(kp->fv_states_col(), hpsi);
    //==     if (ctx_.num_mag_dims() != 0) apply_uj_correction<dd>(kp->fv_states_col(), hpsi);
    //==     if (ctx_.num_mag_dims() == 3)
    //==     {
    //==         apply_uj_correction<ud>(kp->fv_states_col(), hpsi);
    //==         if (ctx_.std_evp_solver()->parallel()) apply_uj_correction<du>(kp->fv_states_col(), hpsi);
    //==     }
    //== }

    if (ctx_.so_correction()) {
        Hk__.H0().apply_so_correction(kp.fv_states(), hpsi);
    }

    int nfv = ctx_.num_fv_states();
    int bs  = ctx_.cyclic_block_size();

    if (ctx_.processing_unit() == device_t::GPU) {
        kp.fv_states().allocate(spin_range(0), ctx_.mem_pool(memory_t::device));
        kp.fv_states().copy_to(spin_range(0), memory_t::device, 0, nfv);
        for (int i = 0; i < ctx_.num_mag_comp(); i++) {
            hpsi[i].allocate(spin_range(0), ctx_.mem_pool(memory_t::device));
            hpsi[i].copy_to(spin_range(0), memory_t::device, 0, nfv);
        }
    }

    ctx_.print_memory_usage(__FILE__, __LINE__);

    //#ifdef __PRINT_OBJECT_CHECKSUM
    //auto z1 = kp->fv_states().checksum(0, nfv);
    //DUMP("checksum(fv_states): %18.10f %18.10f", std::real(z1), std::imag(z1));
    //for (int i = 0; i < ctx_.num_mag_comp(); i++) {
    //    z1 = hpsi[i].checksum(0, nfv);
    //    DUMP("checksum(hpsi[i]): %18.10f %18.10f", std::real(z1), std::imag(z1));
    //}
    //#endif

    auto& std_solver = ctx_.std_evp_solver();

    if (ctx_.num_mag_dims() != 3) {
        dmatrix<double_complex> h(nfv, nfv, ctx_.blacs_grid(), bs, bs);
        if (ctx_.blacs_grid().comm().size() == 1 && ctx_.processing_unit() == device_t::GPU) {
            h.allocate(ctx_.mem_pool(memory_t::device));
        }
        /* perform one or two consecutive diagonalizations */
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {

            /* compute <wf_i | h * wf_j> */
            inner(ctx_.spla_context(), spin_range(0), kp.fv_states(), 0, nfv, hpsi[ispn], 0, nfv, h, 0, 0);

            for (int i = 0; i < nfv; i++) {
                h.add(i, i, kp.fv_eigen_value(i));
            }
            //#ifdef __PRINT_OBJECT_CHECKSUM
            //auto z1 = h.checksum();
            //DUMP("checksum(h): %18.10f %18.10f", std::real(z1), std::imag(z1));
            //#endif
            PROFILE("sirius::Band::diag_sv|stdevp");
            std_solver.solve(nfv, nfv, h, &band_energies(0, ispn), kp.sv_eigen_vectors(ispn));
        }
    } else {
        int nb = ctx_.num_bands();
        dmatrix<double_complex> h(nb, nb, ctx_.blacs_grid(), bs, bs);
        if (ctx_.blacs_grid().comm().size() == 1 && ctx_.processing_unit() == device_t::GPU) {
            h.allocate(ctx_.mem_pool(memory_t::device));
        }
        /* compute <wf_i | h * wf_j> for up-up block */
        inner(ctx_.spla_context(), spin_range(0), kp.fv_states(), 0, nfv, hpsi[0], 0, nfv, h, 0, 0);
        /* compute <wf_i | h * wf_j> for dn-dn block */
        inner(ctx_.spla_context(), spin_range(0), kp.fv_states(), 0, nfv, hpsi[1], 0, nfv, h, nfv, nfv);
        /* compute <wf_i | h * wf_j> for up-dn block */
        inner(ctx_.spla_context(), spin_range(0), kp.fv_states(), 0, nfv, hpsi[2], 0, nfv, h, 0, nfv);

        if (kp.comm().size() == 1) {
            for (int i = 0; i < nfv; i++) {
                for (int j = 0; j < nfv; j++) {
                    h(nfv + j, i) = std::conj(h(i, nfv + j));
                }
            }
        } else {
            linalg(linalg_t::scalapack).tranc(nfv, nfv, h, 0, nfv, h, nfv, 0);
        }

        for (int i = 0; i < nfv; i++) {
            h.add(i, i, kp.fv_eigen_value(i));
            h.add(i + nfv, i + nfv, kp.fv_eigen_value(i));
        }
        //#ifdef __PRINT_OBJECT_CHECKSUM
        //auto z1 = h.checksum();
        //DUMP("checksum(h): %18.10f %18.10f", std::real(z1), std::imag(z1));
        //#endif
        PROFILE("sirius::Band::diag_sv|stdevp");
        std_solver.solve(nb, nb, h, &band_energies(0, 0), kp.sv_eigen_vectors(0));
    }

    if (ctx_.processing_unit() == device_t::GPU) {
        kp.fv_states().deallocate(spin_range(0), memory_t::device);
        for (int i = 0; i < ctx_.num_mag_comp(); i++) {
            hpsi[i].deallocate(spin_range(0), memory_t::device);
        }
    }
    for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
        for (int j = 0; j < ctx_.num_bands(); j++) {
            kp.band_energy(j, ispn, band_energies(j, ispn));
        }
    }
}

//inline int Band::diag_full_potential_single_variation(K_point& kp__, Hamiltonian& hamiltonian__) const
//{
//     if (kp->num_ranks() > 1 && !parameters_.gen_evp_solver()->parallel())
//         error_local(__FILE__, __LINE__, "eigen-value solver is not parallel");
//
//     mdarray<double_complex, 2> h(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
//     mdarray<double_complex, 2> o(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
//
//     set_o(kp, o);
//
//     std::vector<double> eval(parameters_.num_bands());
//     mdarray<double_complex, 2>& fd_evec = kp->fd_eigen_vectors();
//
//     if (parameters_.num_mag_dims() == 0)
//     {
//         assert(kp->gklo_basis_size() >= parameters_.num_fv_states());
//         set_h<nm>(kp, effective_potential, effective_magnetic_field, h);
//
//         Timer t2("sirius::Band::solve_fd|diag");
//         parameters_.gen_evp_solver()->solve(kp->gklo_basis_size(), kp->gklo_basis_size_row(),
//     kp->gklo_basis_size_col(),
//                                             parameters_.num_fv_states(), h.ptr(), h.ld(), o.ptr(), o.ld(),
//                                             &eval[0], fd_evec.ptr(), fd_evec.ld());
//     }
//
//     if (parameters_.num_mag_dims() == 1)
//     {
//         assert(kp->gklo_basis_size() >= parameters_.num_fv_states());
//
//         mdarray<double_complex, 2> o1(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
//         memcpy(&o1(0, 0), &o(0, 0), o.size() * sizeof(double_complex));
//
//         set_h<uu>(kp, effective_potential, effective_magnetic_field, h);
//
//         Timer t2("sirius::Band::solve_fd|diag");
//         parameters_.gen_evp_solver()->solve(kp->gklo_basis_size(), kp->gklo_basis_size_row(),
//     kp->gklo_basis_size_col(),
//                                             parameters_.num_fv_states(), h.ptr(), h.ld(), o.ptr(), o.ld(),
//                                             &eval[0], &fd_evec(0, 0), fd_evec.ld());
//         t2.stop();
//
//         set_h<dd>(kp, effective_potential, effective_magnetic_field, h);
//
//         t2.start();
//         parameters_.gen_evp_solver()->solve(kp->gklo_basis_size(), kp->gklo_basis_size_row(),
//     kp->gklo_basis_size_col(),
//                                            parameters_.num_fv_states(), h.ptr(), h.ld(), o1.ptr(), o1.ld(),
//                                            &eval[parameters_.num_fv_states()],
//                                            &fd_evec(0, parameters_.spl_fv_states().local_size()), fd_evec.ld());
//        t2.stop();
//    }
//
//    kp->set_band_energies(&eval[0]);
//    return niter;
//}

} // namespace
