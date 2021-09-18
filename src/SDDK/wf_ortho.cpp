// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file wf_ortho.cpp
 *
 *  \brief Definitions.
 *
 */

#include "wf_ortho.hpp"
#include "wf_inner.hpp"
#include "wf_trans.hpp"
#include "utils/profiler.hpp"
#include "linalg/eigensolver.hpp"
#include "type_definition.hpp"

namespace sddk {

template <typename T>
int
orthogonalize(::spla::Context& spla_ctx__, memory_t mem__, linalg_t la__, spin_range spins__,
              int idx_bra__, int idx_ket__, std::vector<Wave_functions<real_type<T>>*> wfs__, int N__, int n__,
              sddk::dmatrix<T>& o__, Wave_functions<real_type<T>>& tmp__, bool project_out__, bool remove_lr__)
{
    PROFILE("sddk::orthogonalize");

    const char* sddk_pp_raw = std::getenv("SDDK_PRINT_PERFORMANCE");
    int sddk_pp             = (sddk_pp_raw == NULL) ? 0 : std::atoi(sddk_pp_raw);

    auto& comm = wfs__[0]->comm();

    int K{0};

    if (sddk_pp) {
        K = wfs__[0]->gkvec().num_gvec() + wfs__[0]->num_mt_coeffs();
        if (std::is_same<T, real_type<T>>::value) {
            K *= 2;
        }
    }

    auto sddk_debug_ptr = utils::get_env<int>("SDDK_DEBUG");
    int sddk_debug      = (sddk_debug_ptr) ? (*sddk_debug_ptr) : 0;

    /* prefactor for the number of floating point operatoins (for complex matrices) */
    double ngop{8e-9};  // default value for complex type
    if (std::is_same<T, real_type<T>>::value) { // change it if it is real type
        ngop = 2e-9;
    }

    auto eps = std::numeric_limits<real_type<T>>::epsilon();

    if (sddk_pp) {
        comm.barrier();
    }
    // double time = -omp_get_wtime();

    double gflops{0};

    if (project_out__) {
        /* project out the old subspace:
         * |\tilda phi_new> = |phi_new> - |phi_old><phi_old|phi_new> */
        if (N__ > 0) {
            inner(spla_ctx__, spins__, *wfs__[idx_bra__], 0, N__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);
            transform(spla_ctx__, spins__(), -1.0, wfs__, 0, N__, o__, 0, 0, 1.0, wfs__, N__, n__);

            if (sddk_pp) {
                /* inner and transform have the same number of flops */
                gflops += static_cast<int>(1 + wfs__.size()) * ngop * N__ * n__ * K;
            }
        }
    }

    /* actual number of newly added basis functions */
    int n = n__;

    /* if we remove linearly dependent basis functions */
    if (remove_lr__) {

        /* compute < \tilda phi_new | \tilda phi_new > */
        inner(spla_ctx__, spins__, *wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);

        linalg_t la1 = (o__.comm().size() == 1) ? linalg_t::lapack : linalg_t::scalapack;
        linalg(la1).geqrf(n__, n__, o__, 0, 0);
        auto diag = o__.get_diag(n__);

        n = 0;
        for (int i = 0; i < n__; i++) {
            if (std::abs(diag[i]) >= eps * 10) {
                /* shift linearly independent basis functions to the beginning of the block which starts
                   after N already existing basis functions */
                if (n != i) {
                    for (int j = 0; j < wfs__.size(); j++) {
                        for (int ispn: spins__) {
                            wfs__[j]->copy_from(*wfs__[j], 1, ispn, N__ + i, ispn, N__ + n);
                        }
                    }
                }
                n++;
            }
        }

    }

    if (sddk_debug >= 2) {
        if (o__.comm().rank() == 0) {
            std::printf("check QR decomposition, matrix size : %i\n", n);
        }
        inner(spla_ctx__, spins__, *wfs__[idx_bra__], N__, n, *wfs__[idx_ket__], N__, n, o__, 0, 0);

        linalg(linalg_t::scalapack).geqrf(n, n, o__, 0, 0);
        auto diag = o__.get_diag(n);
        if (o__.comm().rank() == 0) {
            for (int i = 0; i < n; i++) {
                if (std::abs(diag[i]) < eps * 10) {
                    std::cout << "small diagonal of QR : " << i << " " << diag[i] << std::endl;
                }
            }
        }

        if (o__.comm().rank() == 0) {
            std::printf("check eigen-values, matrix size : %i\n", n);
        }
        inner(spla_ctx__, spins__, *wfs__[idx_bra__], N__, n, *wfs__[idx_ket__], N__, n, o__, 0, 0);

        // if (sddk_debug >= 3) {
        //    save_to_hdf5("nxn_overlap.h5", o__, n__);
        //}

        std::vector<real_type<T>> eo(n);
        dmatrix<T> evec(o__.num_rows(), o__.num_cols(), o__.blacs_grid(), o__.bs_row(), o__.bs_col());

        auto solver = Eigensolver_factory("lapack", nullptr);
        solver->solve(n, o__, eo.data(), evec);

        if (o__.comm().rank() == 0) {
            for (int i = 0; i < n; i++) {
                if (eo[i] < eps * 10) {
                    std::cout << "small eigen-value " << i << " " << eo[i] << std::endl;
                }
            }
        }
    }

    /* orthogonalize new n x n block */
    inner(spla_ctx__, spins__, *wfs__[idx_bra__], N__, n, *wfs__[idx_ket__], N__, n, o__, 0, 0);

    if (sddk_debug >= 1) {
        if (o__.comm().rank() == 0) {
            std::printf("check diagonal\n");
        }
        auto diag = o__.get_diag(n);
        for (int i = 0; i < n; i++) {
            if (std::real(diag[i]) <= 0 || std::imag(diag[i]) > eps) {
                std::cout << "wrong diagonal: " << i << " " << diag[i] << std::endl;
            }
        }
        if (o__.comm().rank() == 0) {
            std::printf("check hermitian\n");
        }
        auto d = check_hermitian(o__, n);
        if (d > eps && o__.comm().rank() == 0) {
            std::stringstream s;
            s << "matrix is not hermitian, max diff = " << d;
            WARNING(s);
        }
    }

    if (sddk_pp) {
        gflops += ngop * n * n * K;
    }

    /* single MPI rank */
    if (o__.comm().size() == 1) {
        bool use_magma{false};

        // MAGMA performance for Cholesky and inversion is not good enough; use lapack for the moment
        //#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
        //        if (pu__ == GPU) {
        //            use_magma = true;
        //        }
        //#endif

        PROFILE_START("sddk::orthogonalize|tmtrx");
        if (use_magma) {
            /* Cholesky factorization */
            if (int info = linalg(linalg_t::magma).potrf(n, o__.at(memory_t::device), o__.ld())) {
                std::stringstream s;
                s << "error in GPU factorization, info = " << info;
                TERMINATE(s);
            }
            /* inversion of triangular matrix */
            if (linalg(linalg_t::magma).trtri(n, o__.at(memory_t::device), o__.ld())) {
                TERMINATE("error in inversion");
            }
        } else { /* CPU version */
            /* Cholesky factorization */
            if (int info = linalg(linalg_t::lapack).potrf(n, &o__(0, 0), o__.ld())) {
                std::stringstream s;
                s << "error in factorization, info = " << info << std::endl
                  << "number of existing states: " << N__ << std::endl
                  << "number of new states: " << n << std::endl
                  << "number of wave_functions: " << wfs__.size() << std::endl
                  << "idx_bra: " << idx_bra__ << " "
                  << "idx_ket:" << idx_ket__;
                TERMINATE(s);
            }
            /* inversion of triangular matrix */
            if (linalg(linalg_t::lapack).trtri(n, &o__(0, 0), o__.ld())) {
                TERMINATE("error in inversion");
            }
            if (is_device_memory(mem__)) {
                acc::copyin(o__.at(memory_t::device), o__.ld(), o__.at(memory_t::host), o__.ld(), n, n);
            }
        }
        PROFILE_STOP("sddk::orthogonalize|tmtrx");

        PROFILE_START("sddk::orthogonalize|transform");

        int sid{0};
        for (int s : spins__) {
            /* multiplication by triangular matrix */
            for (auto& e : wfs__) {
                /* wave functions are complex, transformation matrix is complex */
                if (!std::is_scalar<T>::value) {
                    linalg(la__).trmm('R', 'U', 'N', e->pw_coeffs(s).num_rows_loc(), n, &linalg_const<T>::one(),
                                      reinterpret_cast<T*>(o__.at(mem__)), o__.ld(),
                                      reinterpret_cast<T*>(e->pw_coeffs(s).prime().at(e->preferred_memory_t(), 0, N__)),
                                      e->pw_coeffs(s).prime().ld(), stream_id(sid++));

                    if (e->has_mt()) {
                        linalg(la__).trmm(
                            'R', 'U', 'N', e->mt_coeffs(s).num_rows_loc(), n, &linalg_const<T>::one(),
                            reinterpret_cast<T*>(o__.at(mem__)), o__.ld(),
                            reinterpret_cast<T*>(e->mt_coeffs(s).prime().at(e->preferred_memory_t(), 0, N__)),
                            e->mt_coeffs(s).prime().ld(), stream_id(sid++));
                    }
                }
                /* wave functions are real (psi(G) = psi^{*}(-G)), transformation matrix is real */
                if (std::is_scalar<T>::value) {
                    linalg(la__).trmm('R', 'U', 'N', 2 * e->pw_coeffs(s).num_rows_loc(), n, &linalg_const<T>::one(),
                                      reinterpret_cast<T*>(o__.at(mem__)), o__.ld(),
                                      reinterpret_cast<T*>(e->pw_coeffs(s).prime().at(e->preferred_memory_t(), 0, N__)),
                                      2 * e->pw_coeffs(s).prime().ld(), stream_id(sid++));

                    if (e->has_mt()) {
                        linalg(la__).trmm(
                            'R', 'U', 'N', 2 * e->mt_coeffs(s).num_rows_loc(), n, &linalg_const<T>::one(),
                            reinterpret_cast<T*>(o__.at(mem__)), o__.ld(),
                            reinterpret_cast<T*>(e->mt_coeffs(s).prime().at(e->preferred_memory_t(), 0, N__)),
                            2 * e->mt_coeffs(s).prime().ld(), stream_id(sid++));
                    }
                }
            }
        }
        if (la__ == linalg_t::gpublas || la__ == linalg_t::cublasxt || la__ == linalg_t::magma) {
            // sync stream only if processing unit is gpu
            for (int i = 0; i < sid; i++) {
                acc::sync_stream(stream_id(i));
            }
        }
        PROFILE_STOP("sddk::orthogonalize|transform");
    } else { /* parallel transformation */
        PROFILE_START("sddk::orthogonalize|potrf");
        mdarray<T, 1> diag;
        o__.make_real_diag(n);
        if (sddk_debug >= 1) {
            diag = o__.get_diag(n);
        }
        if (int info = linalg(linalg_t::scalapack).potrf(n, o__.at(memory_t::host), o__.ld(), o__.descriptor())) {
            std::stringstream s;
            s << "error in Cholesky factorization, info = " << info << ", matrix size = " << n;
            if (sddk_debug >= 1) {
                s << std::endl << "  diag = " << diag[info - 1];
            }
            TERMINATE(s);
        }
        PROFILE_STOP("sddk::orthogonalize|potrf");

        PROFILE_START("sddk::orthogonalize|trtri");
        if (linalg(linalg_t::scalapack).trtri(n, o__.at(memory_t::host), o__.ld(), o__.descriptor())) {
            TERMINATE("error in inversion");
        }
        PROFILE_STOP("sddk::orthogonalize|trtri");

        /* o is upper triangular matrix */
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                o__.set(j, i, 0);
            }
        }

        /* phi is transformed into phi, so we can't use it as the output buffer; use tmp instead and then overwrite phi
         */
        for (auto& e : wfs__) {
            transform(spla_ctx__, spins__(), *e, N__, n, o__, 0, 0, tmp__, 0, n);
            for (int s : spins__) {
                e->copy_from(tmp__, n, s, 0, s, N__);
            }
        }
    }
    if (sddk_debug >= 2) {
        inner(spla_ctx__, spins__, *wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);
        auto err = check_identity(o__, n__);
        std::cout << "wf_ortho: error in (n, n) overlap matrix : " << err << std::endl;
    }
    return n;
}

// instantiate for required types
template
int
orthogonalize<double>(::spla::Context& spla_ctx__, memory_t mem__, linalg_t la__, spin_range spins__,
                      int idx_bra__, int idx_ket__, std::vector<Wave_functions<double>*> wfs__, int N__, int n__,
                      sddk::dmatrix<double>& o__, Wave_functions<double>& tmp__, bool project_out__, bool remove_lr__);

template
int
orthogonalize<std::complex<double>>(::spla::Context& spla_ctx__, memory_t mem__, linalg_t la__, spin_range spins__,
                      int idx_bra__, int idx_ket__, std::vector<Wave_functions<double>*> wfs__, int N__, int n__,
                      sddk::dmatrix<std::complex<double>>& o__, Wave_functions<double>& tmp__, bool project_out__,
                      bool remove_lr__);

#if defined(USE_FP32)
template
int
orthogonalize<float>(::spla::Context& spla_ctx__, memory_t mem__, linalg_t la__, spin_range spins__,
                      int idx_bra__, int idx_ket__, std::vector<Wave_functions<float>*> wfs__, int N__, int n__,
                      sddk::dmatrix<float>& o__, Wave_functions<float>& tmp__, bool project_out__, bool remove_lr__);

template
int
orthogonalize<std::complex<float>>(::spla::Context& spla_ctx__, memory_t mem__, linalg_t la__, spin_range spins__,
                      int idx_bra__, int idx_ket__, std::vector<Wave_functions<float>*> wfs__, int N__, int n__,
                      sddk::dmatrix<std::complex<float>>& o__, Wave_functions<float>& tmp__, bool project_out__,
                      bool remove_lr__);
#endif

} // namespace sddk
