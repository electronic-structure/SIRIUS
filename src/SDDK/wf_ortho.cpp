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

namespace sddk {

template <typename T, int idx_bra__, int idx_ket__>
void orthogonalize(memory_t mem__, linalg_t la__, int ispn__, std::vector<Wave_functions*> wfs__, int N__, int n__,
                   dmatrix<T>& o__, Wave_functions& tmp__)
{
    PROFILE("sddk::orthogonalize");

    const char* sddk_pp_raw = std::getenv("SDDK_PRINT_PERFORMANCE");
    int sddk_pp             = (sddk_pp_raw == NULL) ? 0 : std::atoi(sddk_pp_raw);

    auto& comm = wfs__[0]->comm();

    auto spins = (ispn__ == 2) ? std::vector<int>({0, 1}) : std::vector<int>({ispn__});

    int K{0};
    if (sddk_pp) {
        K = wfs__[0]->gkvec().num_gvec() + wfs__[0]->num_mt_coeffs();
        if (std::is_same<T, double>::value) {
            K *= 2;
        }
    }

    auto sddk_debug_ptr = utils::get_env<int>("SDDK_DEBUG");
    int sddk_debug      = (sddk_debug_ptr) ? (*sddk_debug_ptr) : 0;

    double ngop{0};
    if (std::is_same<T, double>::value) {
        ngop = 2e-9;
    }
    if (std::is_same<T, double_complex>::value) {
        ngop = 8e-9;
    }

    if (sddk_pp) {
        comm.barrier();
    }
    // double time = -omp_get_wtime();

    double gflops{0};

    /* project out the old subspace:
     * |\tilda phi_new> = |phi_new> - |phi_old><phi_old|phi_new> */
    if (N__ > 0) {
        inner(mem__, la__, ispn__, *wfs__[idx_bra__], 0, N__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);
        transform(mem__, la__, ispn__, -1.0, wfs__, 0, N__, o__, 0, 0, 1.0, wfs__, N__, n__);

        if (sddk_pp) {
            gflops += static_cast<int>(1 + wfs__.size()) * ngop * N__ * n__ *
                      K; // inner and transfrom have the same number of flops
        }
    }

    if (sddk_debug >= 2) {
        //if (o__.comm().rank() == 0) {
        //    std::printf("check QR decomposition, matrix size : %i\n", n__);
        //}
        //inner(mem__, la__, ispn__, *wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);

        //linalg<device_t::CPU>::geqrf(n__, n__, o__, 0, 0);
        //auto diag = o__.get_diag(n__);
        //if (o__.comm().rank() == 0) {
        //    for (int i = 0; i < n__; i++) {
        //        if (std::abs(diag[i]) < 1e-6) {
        //            std::cout << "small norm: " << i << " " << diag[i] << std::endl;
        //        }
        //    }
        //}

        if (o__.comm().rank() == 0) {
            std::printf("check eigen-values, matrix size : %i\n", n__);
        }
        inner(mem__, la__, ispn__, *wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);

        // if (sddk_debug >= 3) {
        //    save_to_hdf5("nxn_overlap.h5", o__, n__);
        //}

        std::vector<double> eo(n__);
        dmatrix<T> evec(o__.num_rows(), o__.num_cols(), o__.blacs_grid(), o__.bs_row(), o__.bs_col());

        auto solver = Eigensolver_factory(ev_solver_t::scalapack);
        solver->solve(n__, o__, eo.data(), evec);

        if (o__.comm().rank() == 0) {
            for (int i = 0; i < n__; i++) {
                if (eo[i] < 1e-6) {
                    std::cout << "small eigen-value " << i << " " << eo[i] << std::endl;
                }
            }
        }
    }

    /* orthogonalize new n__ x n__ block */
    inner(mem__, la__, ispn__, *wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);

    if (sddk_debug >= 1) {
        if (o__.comm().rank() == 0) {
            std::printf("check diagonal\n");
        }
        auto diag = o__.get_diag(n__);
        for (int i = 0; i < n__; i++) {
            if (std::real(diag[i]) <= 0 || std::imag(diag[i]) > 1e-12) {
                std::cout << "wrong diagonal: " << i << " " << diag[i] << std::endl;
            }
        }
        if (o__.comm().rank() == 0) {
            std::printf("check hermitian\n");
        }
        double d = check_hermitian(o__, n__);
        if (d > 1e-12 && o__.comm().rank() == 0) {
            std::stringstream s;
            s << "matrix is not hermitian, max diff = " << d;
            WARNING(s);
        }
    }

    if (sddk_pp) {
        gflops += ngop * n__ * n__ * K;
    }

    /* single MPI rank */
    if (o__.comm().size() == 1) {
        bool use_magma{false};

        // MAGMA performance for Cholesky and inversion is not good enough; use lapack for the moment
        //#if defined(__GPU) && defined(__MAGMA)
        //        if (pu__ == GPU) {
        //            use_magma = true;
        //        }
        //#endif

        PROFILE_START("sddk::orthogonalize|tmtrx");
        if (use_magma) {
            /* Cholesky factorization */
            if (int info = linalg(linalg_t::magma).potrf(n__, o__.at(memory_t::device), o__.ld())) {
                std::stringstream s;
                s << "error in GPU factorization, info = " << info;
                TERMINATE(s);
            }
            /* inversion of triangular matrix */
            if (linalg(linalg_t::magma).trtri(n__, o__.at(memory_t::device), o__.ld())) {
                TERMINATE("error in inversion");
            }
        } else { /* CPU version */
            /* Cholesky factorization */
            if (int info = linalg(linalg_t::lapack).potrf(n__, &o__(0, 0), o__.ld())) {
                std::stringstream s;
                s << "error in factorization, info = " << info << std::endl
                  << "number of existing states: " << N__ << std::endl
                  << "number of new states: " << n__ << std::endl
                  << "number of wave_functions: " << wfs__.size() << std::endl
                  << "idx_bra: " << idx_bra__ << " "
                  << "idx_ket:" << idx_ket__;
                TERMINATE(s);
            }
            /* inversion of triangular matrix */
            if (linalg(linalg_t::lapack).trtri(n__, &o__(0, 0), o__.ld())) {
                TERMINATE("error in inversion");
            }
            if (is_device_memory(mem__)) {
                acc::copyin(o__.at(memory_t::device), o__.ld(), o__.at(memory_t::host), o__.ld(), n__, n__);
            }
        }
        PROFILE_STOP("sddk::orthogonalize|tmtrx");

        PROFILE_START("sddk::orthogonalize|transform");

        int sid{0};
        for (int s : spins) {
            /* multiplication by triangular matrix */
            for (auto& e : wfs__) {
                /* wave functions are complex, transformation matrix is complex */
                if (std::is_same<T, double_complex>::value) {
                    linalg(la__).trmm('R', 'U', 'N', e->pw_coeffs(s).num_rows_loc(), n__,
                                       &linalg_const<double_complex>::one(),
                                       reinterpret_cast<double_complex*>(o__.at(mem__)), o__.ld(),
                                       e->pw_coeffs(s).prime().at(e->preferred_memory_t(), 0, N__),
                                       e->pw_coeffs(s).prime().ld(), stream_id(sid++));

                    if (e->has_mt()) {
                        linalg(la__).trmm('R', 'U', 'N', e->mt_coeffs(s).num_rows_loc(), n__,
                                           &linalg_const<double_complex>::one(),
                                           reinterpret_cast<double_complex*>(o__.at(mem__)), o__.ld(),
                                           e->mt_coeffs(s).prime().at(e->preferred_memory_t(), 0, N__),
                                           e->mt_coeffs(s).prime().ld(), stream_id(sid++));
                    }
                }
                /* wave functions are real (psi(G) = psi^{*}(-G)), transformation matrix is real */
                if (std::is_same<T, double>::value) {
                    linalg(la__).trmm(
                        'R', 'U', 'N', 2 * e->pw_coeffs(s).num_rows_loc(), n__, &linalg_const<double>::one(),
                        reinterpret_cast<double*>(o__.at(mem__)), o__.ld(),
                        reinterpret_cast<double*>(e->pw_coeffs(s).prime().at(e->preferred_memory_t(), 0, N__)),
                        2 * e->pw_coeffs(s).prime().ld(), stream_id(sid++));

                    if (e->has_mt()) {
                        linalg(la__).trmm(
                            'R', 'U', 'N', 2 * e->mt_coeffs(s).num_rows_loc(), n__, &linalg_const<double>::one(),
                            reinterpret_cast<double*>(o__.at(mem__)), o__.ld(),
                            reinterpret_cast<double*>(e->mt_coeffs(s).prime().at(e->preferred_memory_t(), 0, N__)),
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
        o__.make_real_diag(n__);
        if (sddk_debug >= 1) {
            diag = o__.get_diag(n__);
        }
        if (int info = linalg(linalg_t::scalapack).potrf(n__, o__.at(memory_t::host), o__.ld(), o__.descriptor())) {
            std::stringstream s;
            s << "error in Cholesky factorization, info = " << info << ", matrix size = " << n__;
            if (sddk_debug >= 1) {
                s << std::endl << "  diag = " << diag[info - 1];
            }
            TERMINATE(s);
        }
        PROFILE_STOP("sddk::orthogonalize|potrf");

        PROFILE_START("sddk::orthogonalize|trtri");
        if (linalg(linalg_t::scalapack).trtri(n__, o__.at(memory_t::host), o__.ld(), o__.descriptor())) {
            TERMINATE("error in inversion");
        }
        PROFILE_STOP("sddk::orthogonalize|trtri");

        /* o is upper triangular matrix */
        for (int i = 0; i < n__; i++) {
            for (int j = i + 1; j < n__; j++) {
                o__.set(j, i, 0);
            }
        }

        /* phi is transformed into phi, so we can't use it as the output buffer; use tmp instead and then overwrite phi
         */
        for (auto& e : wfs__) {
            transform(mem__, la__, ispn__, *e, N__, n__, o__, 0, 0, tmp__, 0, n__);
            for (int s : spins) {
                e->copy_from(tmp__, n__, s, 0, s, N__);
            }
        }
    }
}

// instantiate for required types
template void orthogonalize<double, 0, 2>(memory_t mem__, linalg_t la__, int ispn__, std::vector<Wave_functions*> wfs__,
                                          int N__, int n__, dmatrix<double>& o__, Wave_functions& tmp__);

template void orthogonalize<double, 0, 0>(memory_t mem__, linalg_t la__, int ispn__, std::vector<Wave_functions*> wfs__,
                                          int N__, int n__, dmatrix<double>& o__, Wave_functions& tmp__);

template void orthogonalize<double_complex, 0, 2>(memory_t mem__, linalg_t la__, int ispn__,
                                                  std::vector<Wave_functions*> wfs__, int N__, int n__,
                                                  dmatrix<double_complex>& o__, Wave_functions& tmp__);

template void orthogonalize<double_complex, 0, 0>(memory_t mem__, linalg_t la__, int ispn__,
                                                  std::vector<Wave_functions*> wfs__, int N__, int n__,
                                                  dmatrix<double_complex>& o__, Wave_functions& tmp__);
} // namespace sddk
