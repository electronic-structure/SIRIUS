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

template <typename T, typename F>
void
Band::diag_pseudo_potential_exact(int ispn__, Hamiltonian_k<T>& Hk__) const
{
    PROFILE("sirius::Band::diag_pseudo_potential_exact");

    auto& kp = Hk__.kp();

    if (ctx_.gamma_point()) {
        TERMINATE("exact diagonalization for Gamma-point case is not implemented");
    }

    const int bs = ctx_.cyclic_block_size();
    la::dmatrix<F> hmlt(kp.num_gkvec(), kp.num_gkvec(), ctx_.blacs_grid(), bs, bs);
    la::dmatrix<F> ovlp(kp.num_gkvec(), kp.num_gkvec(), ctx_.blacs_grid(), bs, bs);
    la::dmatrix<F> evec(kp.num_gkvec(), kp.num_gkvec(), ctx_.blacs_grid(), bs, bs);
    std::vector<real_type<F>> eval(kp.num_gkvec());

    hmlt.zero();
    ovlp.zero();

    auto& gen_solver = ctx_.gen_evp_solver();

    for (int ig = 0; ig < kp.num_gkvec(); ig++) {
        hmlt.set(ig, ig, 0.5 * std::pow(kp.gkvec().template gkvec_cart<sddk::index_domain_t::global>(ig).length(), 2));
        ovlp.set(ig, ig, 1);
    }

    auto veff = Hk__.H0().potential().effective_potential().rg().gather_f_pw();
    std::vector<std::complex<double>> beff;
    if (ctx_.num_mag_dims() == 1) {
        beff = Hk__.H0().potential().effective_magnetic_field(0).rg().gather_f_pw();
        for (int ig = 0; ig < ctx_.gvec().num_gvec(); ig++) {
            auto z1 = veff[ig];
            auto z2 = beff[ig];
            veff[ig] = z1 + z2;
            beff[ig] = z1 - z2;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int igk_col = 0; igk_col < kp.num_gkvec_col(); igk_col++) {
        auto gvec_col = kp.gkvec_col().template gvec<sddk::index_domain_t::local>(igk_col);
        for (int igk_row = 0; igk_row < kp.num_gkvec_row(); igk_row++) {
            auto gvec_row = kp.gkvec_row().template gvec<sddk::index_domain_t::local>(igk_row);
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

    sddk::mdarray<F, 2> dop(ctx_.unit_cell().max_mt_basis_size(), ctx_.unit_cell().max_mt_basis_size());
    sddk::mdarray<F, 2> qop(ctx_.unit_cell().max_mt_basis_size(), ctx_.unit_cell().max_mt_basis_size());

    sddk::mdarray<F, 2> btmp(kp.num_gkvec_row(), ctx_.unit_cell().max_mt_basis_size());

    kp.beta_projectors_row().prepare();
    kp.beta_projectors_col().prepare();
    for (int ichunk = 0; ichunk <  kp.beta_projectors_row().num_chunks(); ichunk++) {
        /* generate beta-projectors for a block of atoms */
        kp.beta_projectors_row().generate(sddk::memory_t::host, ichunk);
        kp.beta_projectors_col().generate(sddk::memory_t::host, ichunk);

        auto& beta_row = kp.beta_projectors_row().pw_coeffs_a();
        auto& beta_col = kp.beta_projectors_col().pw_coeffs_a();

        for (int i = 0; i <  kp.beta_projectors_row().chunk(ichunk).num_atoms_; i++) {
            /* number of beta functions for a given atom */
            int nbf  = kp.beta_projectors_row().chunk(ichunk).desc_(beta_desc_idx::nbf, i);
            int offs = kp.beta_projectors_row().chunk(ichunk).desc_(beta_desc_idx::offset, i);
            int ia   = kp.beta_projectors_row().chunk(ichunk).desc_(beta_desc_idx::ia, i);

            for (int xi1 = 0; xi1 < nbf; xi1++) {
                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    dop(xi1, xi2) = Dop.template value<F>(xi1, xi2, ispn__, ia);
                    qop(xi1, xi2) = Qop.template value<F>(xi1, xi2, ispn__, ia);
                }
            }
            /* compute <G+k|beta> D */
            la::wrap(la::lib_t::blas).gemm('N', 'N', kp.num_gkvec_row(), nbf, nbf,
                &la::constant<F>::one(), &beta_row(0, offs), beta_row.ld(), &dop(0, 0), dop.ld(),
                &la::constant<F>::zero(), &btmp(0, 0), btmp.ld());
            /* compute (<G+k|beta> D ) <beta|G+k> */
            la::wrap(la::lib_t::blas).gemm('N', 'C', kp.num_gkvec_row(), kp.num_gkvec_col(), nbf,
                &la::constant<F>::one(), &btmp(0, 0), btmp.ld(), &beta_col(0, offs), beta_col.ld(),
                &la::constant<F>::one(), &hmlt(0, 0), hmlt.ld());
            /* update the overlap matrix */
            if (ctx_.unit_cell().atom(ia).type().augment()) {
                la::wrap(la::lib_t::blas).gemm('N', 'N', kp.num_gkvec_row(), nbf, nbf,
                    &la::constant<F>::one(), &beta_row(0, offs), beta_row.ld(), &qop(0, 0), qop.ld(),
                    &la::constant<F>::zero(), &btmp(0, 0), btmp.ld());
                la::wrap(la::lib_t::blas).gemm('N', 'C', kp.num_gkvec_row(), kp.num_gkvec_col(), nbf,
                    &la::constant<F>::one(), &btmp(0, 0), btmp.ld(), &beta_col(0, offs), beta_col.ld(),
                    &la::constant<F>::one(), &ovlp(0, 0), ovlp.ld());
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
        RTE_OUT(ctx_.out()) << "checking eigen-values of S-matrix\n";

        la::dmatrix<F> ovlp1(kp.num_gkvec(), kp.num_gkvec(), ctx_.blacs_grid(), bs, bs);
        la::dmatrix<F> evec(kp.num_gkvec(), kp.num_gkvec(), ctx_.blacs_grid(), bs, bs);

        sddk::copy(ovlp, ovlp1);

        std::vector<real_type<F>> eo(kp.num_gkvec());

        auto solver = la::Eigensolver_factory("scalapack");
        solver->solve(kp.num_gkvec(), ovlp1, eo.data(), evec);

        for (int i = 0; i < kp.num_gkvec(); i++) {
            if (eo[i] < 1e-6) {
                RTE_OUT(ctx_.out()) << "small eigen-value: " << eo[i] << std::endl;
            }
        }
    }

    if (gen_solver.solve(kp.num_gkvec(), ctx_.num_bands(), hmlt, ovlp, eval.data(), evec)) {
        std::stringstream s;
        s << "error in full diagonalization";
        RTE_THROW(s);
    }

    for (int j = 0; j < ctx_.num_bands(); j++) {
        kp.band_energy(j, ispn__, eval[j]);
    }

    auto layout_in  = evec.grid_layout(0, 0, kp.num_gkvec(), ctx_.num_bands());
    auto layout_out = kp.spinor_wave_functions().grid_layout_pw(wf::spin_index(ispn__), wf::band_range(0, ctx_.num_bands()));

    costa::transform(layout_in, layout_out, 'N', la::constant<std::complex<T>>::one(),
            la::constant<std::complex<T>>::zero(), kp.gkvec().comm().native());
}

template <typename T>
sddk::mdarray<real_type<T>, 1>
Band::diag_S_davidson(Hamiltonian_k<real_type<T>>& Hk__) const
{
    PROFILE("sirius::Band::diag_S_davidson");

    RTE_THROW("implement this");

//    auto& kp = Hk__.kp();
//
//    auto& itso = ctx_.cfg().iterative_solver();
//
//    /* for overlap matrix we do non-magnetic or non-collinear diagonalization */
//    const int num_mag_dims = (ctx_.num_mag_dims() == 3) ? 3 : 0;
//
//    /* number of spin components, treated simultaneously
//     *   1 - in case of non-magnetic
//     *   2 - in case of non-collinear calculation
//     */
//    const int num_sc = (num_mag_dims == 3) ? 2 : 1;
//
    /* number of eigen-vectors to find */
    const int nevec{1};
//
//    /* alias for memory pool */
//    auto& mp = ctx_.mem_pool(ctx_.host_memory_t());
//
//    /* eigen-vectors */
//    sddk::Wave_functions<real_type<T>> psi(mp, kp.gkvec_partition(), nevec, ctx_.aux_preferred_memory_t(), num_sc);
//    for (int i = 0; i < nevec; i++) {
//        for (int ispn = 0; ispn < num_sc; ispn++) {
//            for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
//                /* global index of G+k vector */
//                int igk = kp.idxgk(igk_loc);
//                if (igk == i + 1) {
//                    psi.pw_coeffs(ispn).prime(igk_loc, i) = 1.0;
//                }
//                if (igk == i + 2) {
//                    psi.pw_coeffs(ispn).prime(igk_loc, i) = 0.5;
//                }
//                if (igk == i + 3) {
//                    psi.pw_coeffs(ispn).prime(igk_loc, i) = 0.25;
//                }
//                if (igk == i + 4) {
//                    psi.pw_coeffs(ispn).prime(igk_loc, i) = 0.125;
//                }
//            }
//        }
//    }
//    std::vector<double> tmp(4096);
//    for (int i = 0; i < 4096; i++) {
//        tmp[i] = utils::random<double>();
//    }
//    #pragma omp parallel for schedule(static)
//    for (int i = 0; i < nevec; i++) {
//        for (int ispn = 0; ispn < num_sc; ispn++) {
//            for (int igk_loc = kp.gkvec().skip_g0(); igk_loc < kp.num_gkvec_loc(); igk_loc++) {
//                /* global index of G+k vector */
//                int igk = kp.idxgk(igk_loc);
//                psi.pw_coeffs(ispn).prime(igk_loc, i) += tmp[igk & 0xFFF] * 1e-5;
//            }
//        }
//    }
//
//    auto result = davidson<T, T, davidson_evp_t::overlap>(Hk__, nevec, num_mag_dims, psi,
//            [](int i, int ispn){ return 1e-10; }, itso.residual_tolerance(), itso.num_steps(), itso.locking(),
//            10, itso.converge_by_energy(), itso.extra_ortho(), std::cout, 0);
//
    sddk::mdarray<real_type<T>, 1> eval(nevec);
//    for (int i = 0; i < nevec; i++) {
//        eval(i) = result.eval(i, 0);
//    }
//
    return eval;
}

template
sddk::mdarray<double, 1>
Band::diag_S_davidson<double>(Hamiltonian_k<double>& Hk__) const;

template
sddk::mdarray<double, 1>
Band::diag_S_davidson<std::complex<double>>(Hamiltonian_k<double>& Hk__) const;

template
void
Band::diag_pseudo_potential_exact<double, std::complex<double>>(int ispn__, Hamiltonian_k<double>& Hk__) const;

template<>
void
Band::diag_pseudo_potential_exact<double, double>(int ispn__, Hamiltonian_k<double>& Hk__) const
{
    RTE_THROW("not implemented");
}

#if defined(USE_FP32)
template
sddk::mdarray<float, 1>
Band::diag_S_davidson<float>(Hamiltonian_k<float>& Hk__) const;

template
sddk::mdarray<float, 1>
Band::diag_S_davidson<std::complex<float>>(Hamiltonian_k<float>& Hk__) const;

template
void
Band::diag_pseudo_potential_exact<float, std::complex<float>>(int ispn__, Hamiltonian_k<float>& Hk__) const;

template<>
void
Band::diag_pseudo_potential_exact<float, float>(int ispn__, Hamiltonian_k<float>& Hk__) const
{
    RTE_THROW("not implemented");
}
template<>
void
Band::diag_pseudo_potential_exact<float, double>(int ispn__, Hamiltonian_k<float>& Hk__) const
{
    RTE_THROW("not implemented");
}
template<>
void
Band::diag_pseudo_potential_exact<float, std::complex<double>>(int ispn__, Hamiltonian_k<float>& Hk__) const
{
    RTE_THROW("not implemented");
}
#endif

}
