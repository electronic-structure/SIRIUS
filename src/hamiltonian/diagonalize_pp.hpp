// Copyright (c) 2013-2023 Anton Kozhevnikov, Thomas Schulthess
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

/** \file diagonalize_pp.hpp
 *
 *  \brief Diagonalize pseudo-potential Hamiltonian.
 */
#ifndef __DIAGONALIZE_PP_HPP__
#define __DIAGONALIZE_PP_HPP__

#include "davidson.hpp"
#include "check_wave_functions.hpp"
#include "k_point/k_point.hpp"

namespace sirius {

template <typename T, typename F>
inline std::enable_if_t<!std::is_same<T, real_type<F>>::value, void>
diagonalize_pp_exact(int ispn__, Hamiltonian_k<T> const& Hk__, K_point<T>& kp)
{
    RTE_THROW("not implemented");
}

template <typename T, typename F>
inline std::enable_if_t<std::is_same<T, real_type<F>>::value, void>
diagonalize_pp_exact(int ispn__, Hamiltonian_k<T> const& Hk__, K_point<T>& kp__)
{
    PROFILE("sirius::diagonalize_pp_exact");

    auto& ctx = Hk__.H0().ctx();

    if (ctx.gamma_point()) {
        RTE_THROW("exact diagonalization for Gamma-point case is not implemented");
    }

    const int bs = ctx.cyclic_block_size();
    la::dmatrix<F> hmlt(kp__.num_gkvec(), kp__.num_gkvec(), ctx.blacs_grid(), bs, bs);
    la::dmatrix<F> ovlp(kp__.num_gkvec(), kp__.num_gkvec(), ctx.blacs_grid(), bs, bs);
    la::dmatrix<F> evec(kp__.num_gkvec(), kp__.num_gkvec(), ctx.blacs_grid(), bs, bs);
    std::vector<real_type<F>> eval(kp__.num_gkvec());

    hmlt.zero();
    ovlp.zero();

    auto& gen_solver = ctx.gen_evp_solver();

    for (int ig = 0; ig < kp__.num_gkvec(); ig++) {
        hmlt.set(ig, ig, 0.5 * std::pow(kp__.gkvec().gkvec_cart(gvec_index_t::global(ig)).length(), 2));
        ovlp.set(ig, ig, 1);
    }

    auto veff = Hk__.H0().potential().effective_potential().rg().gather_f_pw();
    std::vector<std::complex<double>> beff;
    if (ctx.num_mag_dims() == 1) {
        beff = Hk__.H0().potential().effective_magnetic_field(0).rg().gather_f_pw();
        for (int ig = 0; ig < ctx.gvec().num_gvec(); ig++) {
            auto z1  = veff[ig];
            auto z2  = beff[ig];
            veff[ig] = z1 + z2;
            beff[ig] = z1 - z2;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int igk_col = 0; igk_col < kp__.num_gkvec_col(); igk_col++) {
        auto gvec_col = kp__.gkvec_col().gvec(gvec_index_t::local(igk_col));
        for (int igk_row = 0; igk_row < kp__.num_gkvec_row(); igk_row++) {
            auto gvec_row = kp__.gkvec_row().gvec(gvec_index_t::local(igk_row));
            auto ig12     = ctx.gvec().index_g12_safe(gvec_row, gvec_col);

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

    int const nmt = ctx.unit_cell().max_mt_basis_size();
    mdarray<F, 2> dop({nmt, nmt});
    mdarray<F, 2> qop({nmt, ctx.unit_cell().max_mt_basis_size()});

    mdarray<F, 2> btmp({kp__.num_gkvec_row(), ctx.unit_cell().max_mt_basis_size()});

    auto bp_gen_row    = kp__.beta_projectors_row().make_generator();
    auto bp_coeffs_row = bp_gen_row.prepare();

    auto bp_gen_col    = kp__.beta_projectors_col().make_generator();
    auto bp_coeffs_col = bp_gen_col.prepare();

    for (int ichunk = 0; ichunk < kp__.beta_projectors_row().num_chunks(); ichunk++) {
        /* generate beta-projectors for a block of atoms */

        bp_gen_row.generate(bp_coeffs_row, ichunk);
        bp_gen_col.generate(bp_coeffs_col, ichunk);

        auto& beta_row = bp_coeffs_row.pw_coeffs_a_;
        auto& beta_col = bp_coeffs_col.pw_coeffs_a_;

        for (int i = 0; i < bp_coeffs_row.beta_chunk_->num_atoms_; i++) {
            /* number of beta functions for a given atom */
            int nbf  = bp_coeffs_row.beta_chunk_->desc_(beta_desc_idx::nbf, i);
            int offs = bp_coeffs_row.beta_chunk_->desc_(beta_desc_idx::offset, i);
            int ia   = bp_coeffs_row.beta_chunk_->desc_(beta_desc_idx::ia, i);

            for (int xi1 = 0; xi1 < nbf; xi1++) {
                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    dop(xi1, xi2) = Dop.template value<F>(xi1, xi2, ispn__, ia);
                    qop(xi1, xi2) = Qop.template value<F>(xi1, xi2, ispn__, ia);
                }
            }
            /* compute <G+k|beta> D */
            la::wrap(la::lib_t::blas)
                    .gemm('N', 'N', kp__.num_gkvec_row(), nbf, nbf, &la::constant<F>::one(), &beta_row(0, offs),
                          beta_row.ld(), &dop(0, 0), dop.ld(), &la::constant<F>::zero(), &btmp(0, 0), btmp.ld());
            /* compute (<G+k|beta> D ) <beta|G+k> */
            la::wrap(la::lib_t::blas)
                    .gemm('N', 'C', kp__.num_gkvec_row(), kp__.num_gkvec_col(), nbf, &la::constant<F>::one(),
                          &btmp(0, 0), btmp.ld(), &beta_col(0, offs), beta_col.ld(), &la::constant<F>::one(),
                          &hmlt(0, 0), hmlt.ld());
            /* update the overlap matrix */
            if (ctx.unit_cell().atom(ia).type().augment()) {
                la::wrap(la::lib_t::blas)
                        .gemm('N', 'N', kp__.num_gkvec_row(), nbf, nbf, &la::constant<F>::one(), &beta_row(0, offs),
                              beta_row.ld(), &qop(0, 0), qop.ld(), &la::constant<F>::zero(), &btmp(0, 0), btmp.ld());
                la::wrap(la::lib_t::blas)
                        .gemm('N', 'C', kp__.num_gkvec_row(), kp__.num_gkvec_col(), nbf, &la::constant<F>::one(),
                              &btmp(0, 0), btmp.ld(), &beta_col(0, offs), beta_col.ld(), &la::constant<F>::one(),
                              &ovlp(0, 0), ovlp.ld());
            }
        } // i (atoms in chunk)
    }
    // kp.beta_projectors_row().dismiss();
    // kp.beta_projectors_col().dismiss();

    if (ctx.cfg().control().verification() >= 1) {
        double max_diff = check_hermitian(ovlp, kp__.num_gkvec());
        if (max_diff > 1e-12) {
            std::stringstream s;
            s << "overlap matrix is not hermitian, max_err = " << max_diff;
            RTE_THROW(s);
        }
        max_diff = check_hermitian(hmlt, kp__.num_gkvec());
        if (max_diff > 1e-12) {
            std::stringstream s;
            s << "Hamiltonian matrix is not hermitian, max_err = " << max_diff;
            RTE_THROW(s);
        }
    }
    if (ctx.cfg().control().verification() >= 2) {
        RTE_OUT(ctx.out()) << "checking eigen-values of S-matrix\n";

        la::dmatrix<F> ovlp1(kp__.num_gkvec(), kp__.num_gkvec(), ctx.blacs_grid(), bs, bs);
        la::dmatrix<F> evec(kp__.num_gkvec(), kp__.num_gkvec(), ctx.blacs_grid(), bs, bs);

        copy(ovlp, ovlp1);

        std::vector<real_type<F>> eo(kp__.num_gkvec());

        auto solver = la::Eigensolver_factory("scalapack");
        solver->solve(kp__.num_gkvec(), ovlp1, eo.data(), evec);

        for (int i = 0; i < kp__.num_gkvec(); i++) {
            if (eo[i] < 1e-6) {
                RTE_OUT(ctx.out()) << "small eigen-value: " << eo[i] << std::endl;
            }
        }
    }

    if (gen_solver.solve(kp__.num_gkvec(), ctx.num_bands(), hmlt, ovlp, eval.data(), evec)) {
        std::stringstream s;
        s << "error in full diagonalization";
        RTE_THROW(s);
    }

    for (int j = 0; j < ctx.num_bands(); j++) {
        kp__.band_energy(j, ispn__, eval[j]);
    }

    auto layout_in = evec.grid_layout(0, 0, kp__.num_gkvec(), ctx.num_bands());
    auto layout_out =
            kp__.spinor_wave_functions().grid_layout_pw(wf::spin_index(ispn__), wf::band_range(0, ctx.num_bands()));

    costa::transform(layout_in, layout_out, 'N', la::constant<std::complex<T>>::one(),
                     la::constant<std::complex<T>>::zero(), kp__.gkvec().comm().native());
}

/// Diagonalize S-operator of the ultrasoft or PAW methods.
/** Sometimes this is needed to check the quality of pseudopotential. */
template <typename T, typename F>
inline mdarray<real_type<F>, 1>
diag_S_davidson(Hamiltonian_k<T> const& Hk__, K_point<T>& kp__)
{
    PROFILE("sirius::diag_S_davidson");

    RTE_THROW("implement this");

    auto& ctx = Hk__.H0().ctx();

    auto& itso = ctx.cfg().iterative_solver();

    /* for overlap matrix we do non-magnetic or non-collinear diagonalization */
    auto num_mag_dims = wf::num_mag_dims((ctx.num_mag_dims() == 3) ? 3 : 0);

    /* number of spin components, treated simultaneously
     *   1 - in case of non-magnetic
     *   2 - in case of non-collinear calculation
     */
    const int num_sc = (num_mag_dims == 3) ? 2 : 1;

    /* number of eigen-vectors to find */
    const int nevec{1};

    /* eigen-vectors */
    auto psi = wave_function_factory(ctx, kp__, wf::num_bands(nevec), num_mag_dims, false);
    for (int i = 0; i < nevec; i++) {
        auto ib = wf::band_index(i);
        for (int ispn = 0; ispn < num_sc; ispn++) {
            auto s = wf::spin_index(ispn);
            for (int igk_loc = 0; igk_loc < kp__.num_gkvec_loc(); igk_loc++) {
                /* global index of G+k vector */
                int igk = kp__.gkvec().offset() + igk_loc;
                if (igk == i + 1) {
                    psi->pw_coeffs(igk_loc, s, ib) = 1.0;
                }
                if (igk == i + 2) {
                    psi->pw_coeffs(igk_loc, s, ib) = 0.5;
                }
                if (igk == i + 3) {
                    psi->pw_coeffs(igk_loc, s, ib) = 0.25;
                }
                if (igk == i + 4) {
                    psi->pw_coeffs(igk_loc, s, ib) = 0.125;
                }
            }
        }
    }
    std::vector<double> tmp(4096);
    for (int i = 0; i < 4096; i++) {
        tmp[i] = random<double>();
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nevec; i++) {
        for (int ispn = 0; ispn < num_sc; ispn++) {
            for (int igk_loc = kp__.gkvec().skip_g0(); igk_loc < kp__.num_gkvec_loc(); igk_loc++) {
                /* global index of G+k vector */
                int igk = kp__.gkvec().offset() + igk_loc;
                psi->pw_coeffs(igk_loc, wf::spin_index(ispn), wf::band_index(i)) += tmp[igk & 0xFFF] * 1e-5;
            }
        }
    }

    auto result = davidson<T, F, davidson_evp_t::overlap>(
            Hk__, kp__, wf::num_bands(nevec), num_mag_dims, *psi, [](int i, int ispn) { return 1e-10; },
            itso.residual_tolerance(), itso.num_steps(), itso.locking(), 10, itso.converge_by_energy(),
            itso.extra_ortho(), std::cout, 0);

    mdarray<real_type<F>, 1> eval({nevec});
    for (int i = 0; i < nevec; i++) {
        eval(i) = result.eval(i, 0);
    }

    return eval;
}

template <typename T, typename F>
inline auto
diagonalize_pp(Hamiltonian_k<T> const& Hk__, K_point<T>& kp__, double itsol_tol__, double empy_tol__,
               int itsol_num_steps__)
{
    auto& ctx = Hk__.H0().ctx();
    print_memory_usage(ctx.out(), FILE_LINE);

    davidson_result_t result{0, mdarray<double, 2>(), true, {0, 0}};

    auto& itso = ctx.cfg().iterative_solver();
    if (itso.type() == "davidson") {
        auto tolerance = [&](int j__, int ispn__) -> double {
            /* tolerance for occupied states */
            double tol = itsol_tol__;
            /* if band is empty, make tolerance larger (in most cases we don't need high precision on
             * unoccupied states) */
            if (std::abs(kp__.band_occupancy(j__, ispn__)) < ctx.min_occupancy() * ctx.max_occupancy()) {
                tol += empy_tol__;
            }

            return tol;
        };

        std::stringstream s;
        std::ostream* out = (kp__.comm().rank() == 0) ? &std::cout : &s;
        result            = davidson<T, F, davidson_evp_t::hamiltonian>(
                Hk__, kp__, wf::num_bands(ctx.num_bands()), wf::num_mag_dims(ctx.num_mag_dims()),
                kp__.spinor_wave_functions(), tolerance, itso.residual_tolerance(), itsol_num_steps__, itso.locking(),
                itso.subspace_size(), itso.converge_by_energy(), itso.extra_ortho(), *out, 0);
        for (int ispn = 0; ispn < ctx.num_spinors(); ispn++) {
            for (int j = 0; j < ctx.num_bands(); j++) {
                kp__.band_energy(j, ispn, result.eval(j, ispn));
            }
        }
    } else {
        RTE_THROW("unknown iterative solver type");
    }

    /* check wave-functions */
    if (ctx.cfg().control().verification() >= 2) {
        if (ctx.num_mag_dims() == 3) {
            auto eval = kp__.band_energies(0);
            check_wave_functions<T, F>(Hk__, kp__.spinor_wave_functions(), wf::spin_range(0, 2),
                                       wf::band_range(0, ctx.num_bands()), eval.data());
        } else {
            for (int ispn = 0; ispn < ctx.num_spins(); ispn++) {
                auto eval = kp__.band_energies(ispn);
                check_wave_functions<T, F>(Hk__, kp__.spinor_wave_functions(), wf::spin_range(ispn),
                                           wf::band_range(0, ctx.num_bands()), eval.data());
            }
        }
    }

    print_memory_usage(ctx.out(), FILE_LINE);

    return result;
}

} // namespace sirius

#endif
