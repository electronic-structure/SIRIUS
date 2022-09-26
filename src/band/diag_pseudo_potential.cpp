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

using namespace sddk;

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
        auto gvec_col = kp.gkvec().template gvec<index_domain_t::global>(ig_col);
        for (int igk_row = 0; igk_row < kp.num_gkvec_row(); igk_row++) {
            int ig_row    = kp.igk_row(igk_row);
            auto gvec_row = kp.gkvec().template gvec<index_domain_t::global>(ig_row);
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

    {
        auto bp_gen        = kp.beta_projectors_row().make_generator();
        auto bp_coeffs_row = bp_gen.prepare();
        auto bp_coeffs_col = bp_gen.prepare();

        for (int ichunk = 0; ichunk < kp.beta_projectors_row().num_chunks(); ichunk++) {
            /* generate beta-projectors for a block of atoms */
            bp_gen.generate(bp_coeffs_row, ichunk);
            bp_gen.generate(bp_coeffs_col, ichunk);

            auto& beta_row = bp_coeffs_row.pw_coeffs_a;
            auto& beta_col = bp_coeffs_col.pw_coeffs_a;

            for (int i = 0; i < bp_coeffs_row.beta_chunk.num_atoms_; i++) {
                /* number of beta functions for a given atom */
                int nbf  = bp_coeffs_row.beta_chunk.desc_(static_cast<int>(beta_desc_idx::nbf), i);
                int offs = bp_coeffs_row.beta_chunk.desc_(static_cast<int>(beta_desc_idx::offset), i);
                int ia   = bp_coeffs_row.beta_chunk.desc_(static_cast<int>(beta_desc_idx::ia), i);

                for (int xi1 = 0; xi1 < nbf; xi1++) {
                    for (int xi2 = 0; xi2 < nbf; xi2++) {
                        dop(xi1, xi2) = Dop.template value<T>(xi1, xi2, ispn__, ia);
                        qop(xi1, xi2) = Qop.template value<T>(xi1, xi2, ispn__, ia);
                    }
                }
                /* compute <G+k|beta> D */
                linalg(linalg_t::blas)
                    .gemm('N', 'N', kp.num_gkvec_row(), nbf, nbf, &linalg_const<T>::one(), &beta_row(0, offs),
                          beta_row.ld(), &dop(0, 0), dop.ld(), &linalg_const<T>::zero(), &btmp(0, 0), btmp.ld());
                /* compute (<G+k|beta> D ) <beta|G+k> */
                linalg(linalg_t::blas)
                    .gemm('N', 'C', kp.num_gkvec_row(), kp.num_gkvec_col(), nbf, &linalg_const<T>::one(), &btmp(0, 0),
                          btmp.ld(), &beta_col(0, offs), beta_col.ld(), &linalg_const<T>::one(), &hmlt(0, 0),
                          hmlt.ld());
                /* update the overlap matrix */
                if (ctx_.unit_cell().atom(ia).type().augment()) {
                    linalg(linalg_t::blas)
                        .gemm('N', 'N', kp.num_gkvec_row(), nbf, nbf, &linalg_const<T>::one(), &beta_row(0, offs),
                              beta_row.ld(), &qop(0, 0), qop.ld(), &linalg_const<T>::zero(), &btmp(0, 0), btmp.ld());
                    linalg(linalg_t::blas)
                        .gemm('N', 'C', kp.num_gkvec_row(), kp.num_gkvec_col(), nbf, &linalg_const<T>::one(),
                              &btmp(0, 0), btmp.ld(), &beta_col(0, offs), beta_col.ld(), &linalg_const<T>::one(),
                              &ovlp(0, 0), ovlp.ld());
                }
            } // i (atoms in chunk)
        }
    } // end of scope for bp_coeffs_row/col

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
        s << "error in full diagonalization";
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

    /* for overlap matrix we do non-magnetic or non-collinear diagonalization */
    const int num_mag_dims = (ctx_.num_mag_dims() == 3) ? 3 : 0;

    /* number of spin components, treated simultaneously
     *   1 - in case of non-magnetic
     *   2 - in case of non-collinear calculation
     */
    const int num_sc = (num_mag_dims == 3) ? 2 : 1;

    /* number of eigen-vectors to find */
    const int nevec{1};

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

    auto result = davidson<T, T, davidson_evp_t::overlap>(Hk__, nevec, num_mag_dims, psi,
            [](int i, int ispn){ return 1e-10; }, itso.residual_tolerance(), itso.num_steps(), itso.locking(),
            10, itso.converge_by_energy(), itso.extra_ortho(), std::cout, 0);

    sddk::mdarray<real_type<T>, 1> eval(nevec);
    for (int i = 0; i < nevec; i++) {
        eval(i) = result.eval(i, 0);
    }

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
