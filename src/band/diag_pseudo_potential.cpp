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

/** \file diag_pseudo_potential.cpp
 *
 *  \brief Diagonalization of pseudopotential Hamiltonian.
 */

#include "band.hpp"
#include "residuals.hpp"
#include "davidson.hpp"
#include "potential/potential.hpp"
#include "utils/profiler.hpp"

#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
#include "gpu/acc.hpp"
extern "C" void compute_chebyshev_polynomial_gpu(int num_gkvec,
                                                 int n,
                                                 double c,
                                                 double r,
                                                 acc_complex_double_t* phi0,
                                                 acc_complex_double_t* phi1,
                                                 acc_complex_double_t* phi2);
#endif

namespace sirius {

template <typename T>
void
Band::diag_pseudo_potential_exact(int ispn__, Hamiltonian_k<real_type<T>>& Hk__) const
{
    PROFILE("sirius::Band::diag_pseudo_potential_exact");

    auto& kp = Hk__.kp();

    if (ctx_.gamma_point()) {
        TERMINATE("exact diagonalization for Gamma-point case is not implemented");
    }

    const int bs = ctx_.cyclic_block_size();
    dmatrix<T> hmlt(kp.num_gkvec(), kp.num_gkvec(), ctx_.blacs_grid(), bs, bs);
    dmatrix<T> ovlp(kp.num_gkvec(), kp.num_gkvec(), ctx_.blacs_grid(), bs, bs);
    dmatrix<T> evec(kp.num_gkvec(), kp.num_gkvec(), ctx_.blacs_grid(), bs, bs);
    std::vector<real_type<T>> eval(kp.num_gkvec());

    hmlt.zero();
    ovlp.zero();

    auto& gen_solver = ctx_.gen_evp_solver();

    for (int ig = 0; ig < kp.num_gkvec(); ig++) {
        hmlt.set(ig, ig, 0.5 * std::pow(kp.gkvec().template gkvec_cart<index_domain_t::global>(ig).length(), 2));
        ovlp.set(ig, ig, 1);
    }

    auto veff = Hk__.H0().potential().effective_potential().gather_f_pw();
    std::vector<double_complex> beff;
    if (ctx_.num_mag_dims() == 1) {
        beff = Hk__.H0().potential().effective_magnetic_field(0).gather_f_pw();
        for (int ig = 0; ig < ctx_.gvec().num_gvec(); ig++) {
            auto z1 = veff[ig];
            auto z2 = beff[ig];
            veff[ig] = z1 + z2;
            beff[ig] = z1 - z2;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int igk_col = 0; igk_col < kp.num_gkvec_col(); igk_col++) {
        int ig_col    = kp.igk_col(igk_col);
        auto gvec_col = kp.gkvec().gvec(ig_col);
        for (int igk_row = 0; igk_row < kp.num_gkvec_row(); igk_row++) {
            int ig_row    = kp.igk_row(igk_row);
            auto gvec_row = kp.gkvec().gvec(ig_row);
            auto ig12 = ctx_.gvec().index_g12_safe(gvec_row, gvec_col);

            if (ispn__ == 0) {
                if (ig12.second) {
                    hmlt(igk_row, igk_col) += std::conj(veff[ig12.first]);
                } else {
                    hmlt(igk_row, igk_col) += veff[ig12.first];
                }
            } else {
                if (ig12.second) {
                    hmlt(igk_row, igk_col) += std::conj(beff[ig12.first]);
                } else {
                    hmlt(igk_row, igk_col) += beff[ig12.first];
                }
            }
        }
    }

    auto& Dop = Hk__.H0().D();
    auto& Qop = Hk__.H0().Q();

    sddk::mdarray<T, 2> dop(ctx_.unit_cell().max_mt_basis_size(), ctx_.unit_cell().max_mt_basis_size());
    sddk::mdarray<T, 2> qop(ctx_.unit_cell().max_mt_basis_size(), ctx_.unit_cell().max_mt_basis_size());

    sddk::mdarray<T, 2> btmp(kp.num_gkvec_row(), ctx_.unit_cell().max_mt_basis_size());

    kp.beta_projectors_row().prepare();
    kp.beta_projectors_col().prepare();
    for (int ichunk = 0; ichunk <  kp.beta_projectors_row().num_chunks(); ichunk++) {
        /* generate beta-projectors for a block of atoms */
        kp.beta_projectors_row().generate(ichunk);
        kp.beta_projectors_col().generate(ichunk);

        auto& beta_row = kp.beta_projectors_row().pw_coeffs_a();
        auto& beta_col = kp.beta_projectors_col().pw_coeffs_a();

        for (int i = 0; i <  kp.beta_projectors_row().chunk(ichunk).num_atoms_; i++) {
            /* number of beta functions for a given atom */
            int nbf  = kp.beta_projectors_row().chunk(ichunk).desc_(static_cast<int>(beta_desc_idx::nbf), i);
            int offs = kp.beta_projectors_row().chunk(ichunk).desc_(static_cast<int>(beta_desc_idx::offset), i);
            int ia   = kp.beta_projectors_row().chunk(ichunk).desc_(static_cast<int>(beta_desc_idx::ia), i);

            for (int xi1 = 0; xi1 < nbf; xi1++) {
                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    dop(xi1, xi2) = Dop.template value<T>(xi1, xi2, ispn__, ia);
                    qop(xi1, xi2) = Qop.template value<T>(xi1, xi2, ispn__, ia);
                }
            }
            /* compute <G+k|beta> D */
            linalg(linalg_t::blas).gemm('N', 'N', kp.num_gkvec_row(), nbf, nbf,
                &linalg_const<T>::one(), &beta_row(0, offs), beta_row.ld(), &dop(0, 0), dop.ld(),
                &linalg_const<T>::zero(), &btmp(0, 0), btmp.ld());
            /* compute (<G+k|beta> D ) <beta|G+k> */
            linalg(linalg_t::blas).gemm('N', 'C', kp.num_gkvec_row(), kp.num_gkvec_col(), nbf,
                &linalg_const<T>::one(), &btmp(0, 0), btmp.ld(), &beta_col(0, offs), beta_col.ld(),
                &linalg_const<T>::one(), &hmlt(0, 0), hmlt.ld());
            /* update the overlap matrix */
            if (ctx_.unit_cell().atom(ia).type().augment()) {
                linalg(linalg_t::blas).gemm('N', 'N', kp.num_gkvec_row(), nbf, nbf,
                    &linalg_const<T>::one(), &beta_row(0, offs), beta_row.ld(), &qop(0, 0), qop.ld(),
                    &linalg_const<T>::zero(), &btmp(0, 0), btmp.ld());
                linalg(linalg_t::blas).gemm('N', 'C', kp.num_gkvec_row(), kp.num_gkvec_col(), nbf,
                    &linalg_const<T>::one(), &btmp(0, 0), btmp.ld(), &beta_col(0, offs), beta_col.ld(),
                    &linalg_const<T>::one(), &ovlp(0, 0), ovlp.ld());
            }
        } // i (atoms in chunk)
    }
    kp.beta_projectors_row().dismiss();
    kp.beta_projectors_col().dismiss();

    if (ctx_.cfg().control().verification() >= 1) {
        double max_diff = check_hermitian(ovlp, kp.num_gkvec());
        if (max_diff > 1e-12) {
            std::stringstream s;
            s << "overlap matrix is not hermitian, max_err = " << max_diff;
            TERMINATE(s);
        }
        max_diff = check_hermitian(hmlt, kp.num_gkvec());
        if (max_diff > 1e-12) {
            std::stringstream s;
            s << "Hamiltonian matrix is not hermitian, max_err = " << max_diff;
            TERMINATE(s);
        }
    }
    if (ctx_.cfg().control().verification() >= 2) {
        ctx_.message(1, __function_name__, "%s", "checking eigen-values of S-matrix\n");

        dmatrix<T> ovlp1(kp.num_gkvec(), kp.num_gkvec(), ctx_.blacs_grid(), bs, bs);
        dmatrix<T> evec(kp.num_gkvec(), kp.num_gkvec(), ctx_.blacs_grid(), bs, bs);

        ovlp >> ovlp1;

        std::vector<real_type<T>> eo(kp.num_gkvec());

        auto solver = Eigensolver_factory("scalapack", nullptr);
        solver->solve(kp.num_gkvec(), ovlp1, eo.data(), evec);

        for (int i = 0; i < kp.num_gkvec(); i++) {
            if (eo[i] < 1e-6) {
                ctx_.message(1, __function_name__, "small eigen-value: %18.10f\n", eo[i]);
            }
        }
    }

    if (gen_solver.solve(kp.num_gkvec(), ctx_.num_bands(), hmlt, ovlp, eval.data(), evec)) {
        std::stringstream s;
        s << "error in full diagonalziation";
        TERMINATE(s);
    }

    for (int j = 0; j < ctx_.num_bands(); j++) {
        kp.band_energy(j, ispn__, eval[j]);
    }

    kp.spinor_wave_functions().pw_coeffs(ispn__).remap_from(evec, 0);
}

template <typename T>
sddk::mdarray<real_type<T>, 1>
Band::diag_S_davidson(Hamiltonian_k<real_type<T>>& Hk__) const
{
    PROFILE("sirius::Band::diag_S_davidson");

    auto& kp = Hk__.kp();

    auto& itso = ctx_.cfg().iterative_solver();

    double iterative_solver_tolerance = 1e-12;

    /* true if this is a non-collinear case */
    const bool nc_mag = (ctx_.num_mag_dims() == 3);

    /* number of spin components, treated simultaneously
     *   1 - in case of non-magnetic or collinear calculation
     *   2 - in case of non-collinear calculation
     */
    const int num_sc = nc_mag ? 2 : 1;

    /* number of eigen-vectors to find */
    const int nevec = 1;

    /* maximum subspace size */
    int num_phi = itso.subspace_size() * nevec;

    if (num_phi > kp.num_gkvec()) {
        std::stringstream s;
        s << "subspace size is too large!";
        TERMINATE(s);
    }
    /* alias for memory pool */
    auto& mp = ctx_.mem_pool(ctx_.host_memory_t());

    /* eigen-vectors */
    Wave_functions<real_type<T>> psi(mp, kp.gkvec_partition(), nevec, ctx_.aux_preferred_memory_t(), num_sc);
    for (int i = 0; i < nevec; i++) {
        for (int ispn = 0; ispn < num_sc; ispn++) {
            for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
                /* global index of G+k vector */
                int igk = kp.idxgk(igk_loc);
                if (igk == i + 1) {
                    psi.pw_coeffs(ispn).prime(igk_loc, i) = 1.0;
                }
                if (igk == i + 2) {
                    psi.pw_coeffs(ispn).prime(igk_loc, i) = 0.5;
                }
                if (igk == i + 3) {
                    psi.pw_coeffs(ispn).prime(igk_loc, i) = 0.25;
                }
                if (igk == i + 4) {
                    psi.pw_coeffs(ispn).prime(igk_loc, i) = 0.125;
                }
            }
        }
    }
    std::vector<double> tmp(4096);
    for (int i = 0; i < 4096; i++) {
        tmp[i] = utils::random<double>();
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nevec; i++) {
        for (int ispn = 0; ispn < num_sc; ispn++) {
            for (int igk_loc = kp.gkvec().skip_g0(); igk_loc < kp.num_gkvec_loc(); igk_loc++) {
                /* global index of G+k vector */
                int igk = kp.idxgk(igk_loc);
                psi.pw_coeffs(ispn).prime(igk_loc, i) += tmp[igk & 0xFFF] * 1e-5;
            }
        }
    }

    /* auxiliary wave-functions */
    Wave_functions<real_type<T>> phi(mp, kp.gkvec_partition(), num_phi, ctx_.aux_preferred_memory_t(), num_sc);

    /* S operator, applied to auxiliary wave-functions */
    Wave_functions<real_type<T>> sphi(mp, kp.gkvec_partition(), num_phi, ctx_.preferred_memory_t(), num_sc);

    /* S operator, applied to new Psi wave-functions */
    Wave_functions<real_type<T>> spsi(mp, kp.gkvec_partition(), nevec, ctx_.preferred_memory_t(), num_sc);

    /* residuals */
    Wave_functions<real_type<T>> res(mp, kp.gkvec_partition(), nevec, ctx_.preferred_memory_t(), num_sc);

    const int bs = ctx_.cyclic_block_size();

    dmatrix<T> ovlp(num_phi, num_phi, ctx_.blacs_grid(), bs, bs, mp);
    dmatrix<T> evec(num_phi, num_phi, ctx_.blacs_grid(), bs, bs, mp);
    dmatrix<T> ovlp_old(num_phi, num_phi, ctx_.blacs_grid(), bs, bs, mp);

    if (is_device_memory(ctx_.aux_preferred_memory_t())) {
        auto& mpd = ctx_.mem_pool(memory_t::device);
        for (int ispn = 0; ispn < num_sc; ispn++) {
            phi.pw_coeffs(ispn).allocate(mpd);
        }
    }

    if (is_device_memory(ctx_.preferred_memory_t())) {
        auto& mpd = ctx_.mem_pool(memory_t::device);
        for (int ispn = 0; ispn < num_sc; ispn++) {
            psi.pw_coeffs(ispn).allocate(mpd);
            psi.pw_coeffs(ispn).copy_to(memory_t::device, 0, nevec);
        }

        for (int i = 0; i < num_sc; i++) {
            res.pw_coeffs(i).allocate(mpd);
            sphi.pw_coeffs(i).allocate(mpd);
            spsi.pw_coeffs(i).allocate(mpd);
        }

        if (ctx_.blacs_grid().comm().size() == 1) {
            evec.allocate(mpd);
            ovlp.allocate(mpd);
        }
    }

    /* allocate memory for the hubbard orbitals on device */
    Hk__.kp().copy_hubbard_orbitals_on_device();

    auto o_diag = Hk__.template get_h_o_diag_pw<T, 2>().second;

    mdarray<real_type<T>, 2> o_diag1(kp.num_gkvec_loc(), num_sc);
    for (int ispn = 0; ispn < num_sc; ispn++) {
        for (int ig = 0; ig < kp.num_gkvec_loc(); ig++) {
            o_diag1(ig, ispn) = 1.0;
        }
    }
    if (ctx_.processing_unit() == device_t::GPU) {
        o_diag1.allocate(memory_t::device).copy_to(memory_t::device);
    }

    auto& std_solver = ctx_.std_evp_solver();

    for (int ispn = 0; ispn < num_sc; ispn++) {
        /* trial basis functions */
        phi.copy_from(psi, nevec, ispn, 0, ispn, 0);
    }

    /* current subspace size */
    int N{0};

    /* number of newly added basis functions */
    int n = nevec;

    mdarray<real_type<T>, 1> eval(nevec);
    mdarray<real_type<T>, 1> eval_old(nevec);
    eval_old = [](){return 1e10;};

    /* tolerance for the norm of L2-norms of the residuals, used for
     * relative convergence criterion. We can only compute this after
     * we have the first residual norms available */
    double relative_frobenius_tolerance{0};
    double current_frobenius_norm{0};

    for (int k = 0; k < itso.num_steps(); k++) {

        /* apply Hamiltonian and S operators to the basis functions */
        Hk__.template apply_h_s<T>(spin_range(nc_mag ? 2 : 0), N, n, phi, nullptr, &sphi);

        orthogonalize<T>(ctx_.spla_context(), ctx_.preferred_memory_t(), ctx_.blas_linalg_t(),
                         spin_range(nc_mag ? 2 : 0), phi, sphi, N, n, ovlp, res);

        /* setup eigen-value problem
         * N is the number of previous basis functions
         * n is the number of new basis functions */
        set_subspace_mtrx<T, T>(N, n, 0, phi, sphi, ovlp, &ovlp_old);
        if (ctx_.cfg().control().verification() >= 2 && ctx_.verbosity() >= 2) {
            ovlp.serialize("<i|S|j> subspace matrix", N + n);
        }

        /* increase size of the variation space */
        N += n;

        eval >> eval_old;

        /* solve standard eigen-value problem with the size N */
        if (std_solver.solve(N, nevec, ovlp, &eval[0], evec)) {
            std::stringstream s;
            s << "error in diagonalziation";
            TERMINATE(s);
        }

        kp.message(3, __function_name__, "step: %i, current subspace size: %i, maximum subspace size: %i\n", k, N, num_phi);
        for (int i = 0; i < nevec; i++) {
            kp.message(4, __function_name__, "eval[%i]=%20.16f, diff=%20.16f\n", i, eval[i], std::abs(eval[i] - eval_old[i]));
        }

        /* don't compute residuals on last iteration */
        if (k != itso.num_steps() - 1) {
            if (ctx_.processing_unit() == device_t::GPU) {
                o_diag.allocate(memory_t::device).copy_to(memory_t::device);
                o_diag1.allocate(memory_t::device).copy_to(memory_t::device);
            }

            /* get new preconditionined residuals, and also opsi and psi as a by-product */
            auto result = sirius::residuals<T>(ctx_, ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), spin_range(nc_mag ? 2 : 0),
                                     N, nevec, 0, eval, evec, sphi, phi, spsi, psi, res, o_diag, o_diag1,
                                     itso.converge_by_energy(), itso.residual_tolerance(),
                                     [&](int i, int ispn){return std::abs(eval[i] - eval_old[i]) < iterative_solver_tolerance;});
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

        bool last_iteration = k == (itso.num_steps() - 1);

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (should_restart || converged || last_iteration) {
            /* recompute wave-functions */
            /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
            transform<T, T>(ctx_.spla_context(), nc_mag ? 2 : 0, phi, 0, N, evec, 0, 0, psi, 0, nevec);

            /* exit the loop if the eigen-vectors are converged or this is a last iteration */
            if (converged || last_iteration) {
                break;
            } else { /* otherwise, set Psi as a new trial basis */
                kp.message(3, __function_name__, "%s", "subspace size limit reached\n");

                if (itso.converge_by_energy()) {
                    transform<T, T>(ctx_.spla_context(), nc_mag ? 2 : 0, sphi, 0, N, evec, 0, 0, spsi, 0, nevec);
                }

                ovlp_old.zero();
                for (int i = 0; i < nevec; i++) {
                    ovlp_old.set(i, i, eval[i]);
                }
                /* update basis functions */
                for (int ispn = 0; ispn < num_sc; ispn++) {
                    phi.copy_from(ctx_.processing_unit(), nevec, psi, ispn, 0, ispn, 0);
                    sphi.copy_from(ctx_.processing_unit(), nevec, spsi, ispn, 0, ispn, 0);
                }
                /* number of basis functions that we already have */
                N = nevec;
            }
        }
        for (int ispn = 0; ispn < num_sc; ispn++) {
            /* expand variational subspace with new basis vectors obtatined from residuals */
            phi.copy_from(ctx_.processing_unit(), n, res, ispn, 0, ispn, N);
        }
    }

    Hk__.kp().release_hubbard_orbitals_on_device();

    return eval;
}

template
mdarray<double, 1>
Band::diag_S_davidson<double>(Hamiltonian_k<double>& Hk__) const;

template
mdarray<double, 1>
Band::diag_S_davidson<std::complex<double>>(Hamiltonian_k<double>& Hk__) const;

template
void
Band::diag_pseudo_potential_exact<std::complex<double>>(int ispn__, Hamiltonian_k<double>& Hk__) const;

template<>
void
Band::diag_pseudo_potential_exact<double>(int ispn__, Hamiltonian_k<double>& Hk__) const
{
    RTE_THROW("not implemented");
}

#if defined(USE_FP32)
template
mdarray<float, 1>
Band::diag_S_davidson<float>(Hamiltonian_k<float>& Hk__) const;

template
mdarray<float, 1>
Band::diag_S_davidson<std::complex<float>>(Hamiltonian_k<float>& Hk__) const;

template
void
Band::diag_pseudo_potential_exact<std::complex<float>>(int ispn__, Hamiltonian_k<float>& Hk__) const;

template<>
void
Band::diag_pseudo_potential_exact<float>(int ispn__, Hamiltonian_k<float>& Hk__) const
{
    RTE_THROW("not implemented");
}
#endif

}
