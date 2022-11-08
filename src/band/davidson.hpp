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

#include "k_point/k_point.hpp"
#include "band/band.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "utils/profiler.hpp"
#include "residuals.hpp"

namespace sirius {

/// Result of Davidson solver.
struct davidson_result_t {
    int niter;
    sddk::mdarray<double, 2> eval;
};

enum class davidson_evp_t {
    hamiltonian,
    overlap
};

template <typename T, typename F>
inline void
project_out_subspace(::spla::Context& spla_ctx__, sddk::memory_t mem__, wf::spin_range spins__, wf::Wave_functions<T>& phi__,
                     wf::Wave_functions<T>& sphi__, int N__, int n__, sddk::dmatrix<F>& o__)
{
    PROFILE("sirius::project_out_subspace");

    /* project out the old subspace:
     * |\tilda phi_new> = |phi_new> - |phi_old><phi_old|S|phi_new> */
    wf::inner(spla_ctx__, mem__, spins__, sphi__, wf::band_range(0, N__), phi__, wf::band_range(N__, N__ + n__), o__, 0, 0);
    for (auto s = spins__.begin(); s != spins__.end(); s++) {
        auto sp = phi__.actual_spin_index(s);
        wf::transform<T, F>(spla_ctx__, mem__, o__, 0, 0, -1.0, phi__, sp, wf::band_range(0, N__), 1.0,
                phi__, sp, wf::band_range(N__, N__ + n__));
    }

    //auto norms = phi__.l2norm(device_t::CPU, spins__, N__ + n__);

    //for (int i = 0; i < N__ + n__; i++) {
    //    std::cout << "phi: " << i << ", l2norm: " << norms[i] << std::endl;
    //}
    //inner(spla_ctx__, spins__, sphi__, 0, N__, phi_new__, 0, n__, o__, 0, 0);
    //for (int i = 0; i < N__; i++) {
    //    for (int j = 0; j < n__; j++) {
    //        std::cout << i << " " << j << " " << o__(i, j) << std::endl;
    //    }
    //}
}

//template <typename T>
//inline int
//remove_linearly_dependent(::spla::Context& spla_ctx__, sddk::spin_range spins__, sddk::Wave_functions<real_type<T>>& phi__,
//                          int N__, int n__, sddk::dmatrix<T>& o__)
//
//{
//    PROFILE("sirius::remove_linearly_dependent");
//
//    /* compute <phi | phi> */
//    inner(spla_ctx__, spins__, phi__, N__, n__, phi__, N__, n__, o__, 0, 0);
//
//    auto la = (o__.comm().size() == 1) ? sddk::linalg_t::lapack : sddk::linalg_t::scalapack;
//    sddk::linalg(la).geqrf(n__, n__, o__, 0, 0);
//    auto diag = o__.get_diag(n__);
//
//    auto eps = std::numeric_limits<real_type<T>>::epsilon();
//
//    int n{0};
//    for (int i = 0; i < n__; i++) {
//        if (std::abs(diag[i]) >= eps * 10) {
//            /* shift linearly independent basis functions to the beginning of phi */
//            if (n != i) {
//                for (int ispn: spins__) {
//                    phi__.copy_from(phi__, 1, ispn, N__ + i, ispn, N__ + n);
//                }
//            }
//            n++;
//        }
//    }
//    return n;
//}


/// Solve the eigen-problem using Davidson iterative method.
/**
\tparam T                     Precision type of wave-functions (float or double).
\tparam F                     Type of the subspace matrices (fload or duble for Gamma case,
                              complex<float> or complex<doouble> for general k-point case.
\tparam what                  What to solve: H|psi> = e*S|psi> or S|psi> = o|psi>
\param [in]     Hk            Hamiltonian for a given k-point.
\param [in]     num_bands     Number of eigen-states (bands) to compute.
\param [in]     num_mag_dims  Number of magnetic dimensions (0, 1 or 3).
\param [in,out] psi           Wave-functions. On input they are used for the starting guess of the subspace basis.
                              On output they are the solutions of Hk|psi> = e S|psi> eigen-problem.
\param [in]     tolerance     Lambda-function for the band energy tolerance.
\param [in]     res_tol       Residual tolerance.
\param [in]     num_stpes     Number of iterative steps.
\param [in]     locking       Lock and do not update of the converged wave-functions.
\param [in]     subspace_size Size of the diagonalziation subspace.
\param [in]     estimate_eval Estimate eigen-values to get the converrged rersiduals.
\param [in]     extra_ortho   Orthogonalize new subspace basis one extra time.
\param [out]    out           Output stream.
\param [in]     verbosity     Verbosity level.
\param [in]     phi_extra     Pointer to the additional (fixed) auxiliary basis functions (used in LAPW).
\return                       List of eigen-values.
*/
template <typename T, typename F, davidson_evp_t what>
inline auto
davidson(Hamiltonian_k<T>& Hk__, wf::num_bands num_bands__, wf::num_mag_dims num_mag_dims__,
        wf::Wave_functions<T>& psi__, std::function<double(int, int)> tolerance__, double res_tol__,
        int num_steps__, bool locking__, int subspace_size__, bool estimate_eval__, bool extra_ortho__,
        std::ostream& out__, int verbosity__, wf::Wave_functions<T>* phi_extra__ = nullptr)
{
    PROFILE("sirius::davidson");

    auto& ctx = Hk__.H0().ctx();
    ctx.print_memory_usage(__FILE__, __LINE__);

    auto pcs = env::print_checksum();

    auto& kp = Hk__.kp();

    auto& itso = ctx.cfg().iterative_solver();

    /* true if this is a non-collinear case */
    const bool nc_mag = (num_mag_dims__.get() == 3);

    auto num_md = wf::num_mag_dims(nc_mag ? 3 : 0);

    /* number of spin components, treated simultaneously
     *   1 - in case of non-magnetic or collinear calculation
     *   2 - in case of non-collinear calculation
     */
    const int num_sc = nc_mag ? 2 : 1;

    /* number of spinor components stored under the same band index */
    const int num_spinors = (num_mag_dims__.get() == 1) ? 2 : 1;

    /* number of spins */
    const int num_spins = (num_mag_dims__.get() == 0) ? 1 : 2;

    /* maximum subspace size */
    int num_phi = subspace_size__ * num_bands__.get();
    int num_extra_phi{0};
    int nlo{0};
    if (phi_extra__) {
        num_extra_phi = phi_extra__->num_wf().get();
        num_phi += num_extra_phi;
        /* total number of local orbitals (needed for LAPW) */
        nlo = ctx.unit_cell().mt_lo_basis_size();
    }

    if (num_phi > kp.gklo_basis_size()) {
        std::stringstream s;
        s << "subspace size is too large!";
        RTE_THROW(s);
    }

    /* alias for memory pool */
    auto& mp = get_memory_pool(ctx.host_memory_t());

    sddk::memory_t mem = ctx.processing_unit_memory_t();

    /* allocate wave-functions */

    using wf_t = wf::Wave_functions<T>;

    bool mt_part{false};
    //if (ctx.full_potential() && what == davidson_evp_t::hamiltonian) {
    if (ctx.full_potential()) {
        mt_part = true;
    }

    std::vector<wf::device_memory_guard> mg;

    mg.emplace_back(psi__.memory_guard(mem, wf::copy_to::device | wf::copy_to::host));

    /* auxiliary wave-functions */
    auto phi = wave_function_factory(ctx, kp, wf::num_bands(num_phi), num_md, mt_part);
    mg.emplace_back(phi->memory_guard(mem));

    /* Hamiltonian, applied to auxiliary wave-functions */
    std::unique_ptr<wf_t> hphi{nullptr};
    if (what == davidson_evp_t::hamiltonian) {
        hphi = wave_function_factory(ctx, kp, wf::num_bands(num_phi), num_md, mt_part);
        mg.emplace_back(hphi->memory_guard(mem));
    }

    /* S operator, applied to auxiliary wave-functions */
    auto sphi = wave_function_factory(ctx, kp, wf::num_bands(num_phi), num_md, mt_part);
    mg.emplace_back(sphi->memory_guard(mem));

    /* Hamiltonain, applied to new Psi wave-functions */
    std::unique_ptr<wf_t> hpsi{nullptr};
    if (what == davidson_evp_t::hamiltonian) {
        hpsi = wave_function_factory(ctx, kp, num_bands__, num_md, mt_part);
        mg.emplace_back(hpsi->memory_guard(mem));
    }

    /* S operator, applied to new Psi wave-functions */
    auto spsi = wave_function_factory(ctx, kp, num_bands__, num_md, mt_part);
    mg.emplace_back(spsi->memory_guard(mem));

    /* residuals */
    /* res is also used as a temporary array in orthogonalize() and the first time num_extra_phi + num_bands
     * states will be orthogonalized */
    auto res = wave_function_factory(ctx, kp, wf::num_bands(num_bands__.get() + num_extra_phi), num_md, mt_part);
    mg.emplace_back(res->memory_guard(mem));

    std::unique_ptr<wf_t> hphi_extra{nullptr};
    std::unique_ptr<wf_t> sphi_extra{nullptr};

    if (phi_extra__) {
        hphi_extra = wave_function_factory(ctx, kp, wf::num_bands(num_extra_phi), num_md, mt_part);
        sphi_extra = wave_function_factory(ctx, kp, wf::num_bands(num_extra_phi), num_md, mt_part);
        mg.emplace_back(phi_extra__->memory_guard(mem, wf::copy_to::device));
        mg.emplace_back(hphi_extra->memory_guard(mem));
        mg.emplace_back(sphi_extra->memory_guard(mem));
        if (pcs) {
            auto cs = phi_extra__->checksum(mem, wf::band_range(0, num_extra_phi));
            if (kp.comm().rank() == 0) {
                utils::print_checksum("phi_extra", cs, RTE_OUT(std::cout));
            }
        }
    }

    int const bs = ctx.cyclic_block_size();

    sddk::dmatrix<F> H(num_phi, num_phi, ctx.blacs_grid(), bs, bs, mp);
    sddk::dmatrix<F> H_old(num_phi, num_phi, ctx.blacs_grid(), bs, bs, mp);
    sddk::dmatrix<F> evec(num_phi, num_phi, ctx.blacs_grid(), bs, bs, mp);

    int const num_ortho_steps = extra_ortho__ ? 2 : 1;

    if (is_device_memory(mem)) {
        auto& mpd = get_memory_pool(mem);
        if (ctx.blacs_grid().comm().size() == 1) {
            evec.allocate(mpd);
            H.allocate(mpd);
        }
    }

    ctx.print_memory_usage(__FILE__, __LINE__);

    /* get diagonal elements for preconditioning */
    auto h_o_diag = (ctx.full_potential()) ?
        Hk__.template get_h_o_diag_lapw<3>() : Hk__.template get_h_o_diag_pw<T, 3>();

    sddk::mdarray<T, 2>* h_diag{nullptr};;
    sddk::mdarray<T, 2>* o_diag{nullptr};

    switch (what) {
        case davidson_evp_t::hamiltonian: {
            h_diag = &h_o_diag.first;
            o_diag = &h_o_diag.second;
            break;
        }
        case davidson_evp_t::overlap: {
            h_diag = &h_o_diag.second;
            o_diag = &h_o_diag.first;
            *o_diag = [](){ return 1.0;};
            if (is_device_memory(mem)) {
                o_diag->copy_to(mem);
            }
            break;
        }
    }

    /* checksum info */
    if (pcs) {
        auto cs1 = h_o_diag.first.checksum();
        auto cs2 = h_o_diag.second.checksum();
        kp.comm().allreduce(&cs1, 1);
        kp.comm().allreduce(&cs2, 1);
        utils::print_checksum("h_diag", cs1, RTE_OUT(out__));
        utils::print_checksum("o_diag", cs2, RTE_OUT(out__));

        auto cs = psi__.checksum(mem, wf::band_range(0, num_bands__.get()));
        utils::print_checksum("input spinor_wave_functions", cs, RTE_OUT(out__));
    }

    auto& std_solver = ctx.std_evp_solver();

    davidson_result_t result{0, sddk::mdarray<double, 2>(num_bands__.get(), num_spinors)};

    if (verbosity__ >= 1) {
         RTE_OUT(out__) << "starting Davidson iterative solver" << std::endl
               << "  number of bands     : " << num_bands__.get() << std::endl
               << "  subspace size       : " << num_phi << std::endl
               << "  locking             : " << locking__ << std::endl
               << "  number of spins     : " << num_spins << std::endl
               << "  non-collinear       : " << nc_mag << std::endl
               << "  number of extra phi : " << num_extra_phi << std::endl;
    }

    PROFILE_START("sirius::davidson|iter");
    for (int ispin_step = 0; ispin_step < num_spinors; ispin_step++) {
        if (verbosity__ >= 1) {
            RTE_OUT(out__) << "ispin_step " << ispin_step << " out of " << num_spinors << std::endl;
        }
        auto sr = nc_mag ? wf::spin_range(0, 2) : wf::spin_range(ispin_step);

        ctx.print_memory_usage(__FILE__, __LINE__);

        /* converged vectors */
        int num_locked{0};

        sddk::mdarray<real_type<F>, 1> eval(num_bands__.get());
        sddk::mdarray<real_type<F>, 1> eval_old(num_bands__.get());

        /* lambda function thatcheck if band energy is converged */
        auto is_converged = [&](int j__, int ispn__) -> bool {
            return std::abs(eval[j__] - eval_old[j__]) <= tolerance__(j__ + num_locked, ispn__);
        };

        //if (itso.init_eval_old()) {
        //    eval_old = [&](int64_t j) { return kp.band_energy(j, ispin_step); };
        //} else {
            eval_old = []() { return 1e10; };
        //}

        /* trial basis functions */
        for (int ispn = 0; ispn < num_sc; ispn++) {
            wf::copy(mem, psi__, wf::spin_index(nc_mag ? ispn : ispin_step),
                    wf::band_range(0, num_bands__.get()), *phi, wf::spin_index(ispn),
                    wf::band_range(0, num_bands__.get()));
        }

        /* extra basis functions for LAPW go after phi */
        if (phi_extra__) {
            if (num_mag_dims__.get() != 0) {
                RTE_THROW("not supported");
            }
            wf::copy(mem, *phi_extra__, wf::spin_index(0), wf::band_range(0, num_extra_phi),
                    *phi, wf::spin_index(0), wf::band_range(num_bands__.get(), num_bands__.get() + num_extra_phi));
        }

        if (pcs) {
            if (phi_extra__) {
                auto cs = phi_extra__->checksum(mem, wf::band_range(0, num_extra_phi));
                utils::print_checksum("extra phi", cs, RTE_OUT(out__));
            }
            auto cs = phi->checksum(mem, wf::band_range(0, num_bands__.get()));
            utils::print_checksum("input phi", cs, RTE_OUT(out__));
        }

        /* current subspace size */
        int N = num_bands__.get() + num_extra_phi;

        /* first phase: setup and diagonalise reduced Hamiltonian and get eigen-values;
         * this is done before the main itertive loop */

        if (verbosity__ >= 1) {
            RTE_OUT(out__) << "apply Hamiltonian to phi(0:" << N - 1 << ")" << std::endl;
        }

        /* apply Hamiltonian and S operators to the basis functions */
        switch (what) {
            case davidson_evp_t::hamiltonian: {
                if (ctx.full_potential()) {
                    /* we save time by not applying APW part to pure local orbital basis */
                    /* aplpy full LAPW Hamiltonian to first N - nlo states */
                    Hk__.apply_fv_h_o(false, false, wf::band_range(0, N - nlo), *phi, hphi.get(), sphi.get());
                    /* aplpy local orbital part to remaining states */
                    Hk__.apply_fv_h_o(false, true, wf::band_range(N - nlo, N), *phi, hphi.get(), sphi.get());
                    /* phi_extra is constant, so the hphi_extra and sphi_extra */
                    if (phi_extra__) {
                        auto s = wf::spin_index(0);
                        wf::copy(mem, *hphi, s, wf::band_range(num_bands__.get(), num_bands__.get() + num_extra_phi),
                                *hphi_extra, s, wf::band_range(0, num_extra_phi));
                        wf::copy(mem, *sphi, s, wf::band_range(num_bands__.get(), num_bands__.get() + num_extra_phi),
                                *sphi_extra, s, wf::band_range(0, num_extra_phi));
                    }
                } else {
                    Hk__.template apply_h_s<F>(sr, wf::band_range(0, num_bands__.get()), *phi, hphi.get(), sphi.get());
                }
                break;
            }
            case davidson_evp_t::overlap: {
                if (ctx.full_potential()) {
                    Hk__.apply_fv_h_o(true, false, wf::band_range(0, num_bands__.get()), *phi, nullptr, sphi.get());
                } else {
                    Hk__.template apply_h_s<F>(sr, wf::band_range(0, num_bands__.get()), *phi, nullptr, sphi.get());
                }
                break;
            }
        }
//
//        /* DEBUG */
//        if (ctx.cfg().control().verification() >= 1) {
//            /* setup eigen-value problem */
//            if (what != davidson_evp_t::overlap) {
//                Band(ctx).set_subspace_mtrx<T, F>(0, N, 0, *phi, *hphi, H);
//                auto max_diff = check_hermitian(H, N);
//                if (max_diff > (std::is_same<real_type<T>, double>::value ? 1e-12 : 1e-6)) {
//                    std::stringstream s;
//                    s << "H matrix is not Hermitian, max_err = " << max_diff << std::endl
//                      << "  happened before entering the iterative loop" << std::endl;
//                    WARNING(s);
//                    if (N <= 20) {
//                        auto s1 = H.serialize("davidson:H_first", N, N);
//                        if (Hk__.kp().comm().rank() == 0) {
//                            RTE_OUT(out__) << s1.str() << std::endl;
//                        }
//                    }
//                }
//            }
//
//            Band(ctx).set_subspace_mtrx<T, F>(0, N, 0, *phi, *sphi, H);
//            auto max_diff = check_hermitian(H, N);
//            if (max_diff > (std::is_same<real_type<T>, double>::value ? 1e-12 : 1e-6)) {
//                std::stringstream s;
//                s << "O matrix is not Hermitian, max_err = " << max_diff << std::endl
//                  << "  happened before entering the iterative loop" << std::endl;
//                WARNING(s);
//                if (N <= 20) {
//                    auto s1 = H.serialize("davidson:O_first", N, N);
//                    if (Hk__.kp().comm().rank() == 0) {
//                        RTE_OUT(out__) << s1.str() << std::endl;
//                    }
//                }
//            }
//        }
//        /* END DEBUG */
//
        if (pcs) {
            auto cs = phi->checksum(mem, wf::band_range(0, N));
            utils::print_checksum("phi", cs, RTE_OUT(out__));
            if (hphi) {
                cs = hphi->checksum(mem, wf::band_range(0, N));
                utils::print_checksum("hphi", cs, RTE_OUT(out__));
            }
            cs = sphi->checksum(mem, wf::band_range(0, N));
            utils::print_checksum("sphi", cs, RTE_OUT(out__));
        }

        if (verbosity__ >= 1) {
            RTE_OUT(out__) << "orthogonalize " << N << " states" << std::endl;
        }

        /* orthogonalize subspace basis functions and setup eigen-value problem */
        switch (what) {
            case davidson_evp_t::hamiltonian: {
                wf::orthogonalize(ctx.spla_context(), mem,
                        nc_mag ? wf::spin_range(0, 2) : wf::spin_range(0), wf::band_range(0, 0), wf::band_range(0, N), *phi, *sphi,
                        {phi.get(), hphi.get(), sphi.get()}, H, *res, false);
                /* checksum info */
                if (pcs) {
                    auto cs = phi->checksum(mem, wf::band_range(0, N));
                    utils::print_checksum("phi", cs, RTE_OUT(out__));
                    if (hphi) {
                        cs = hphi->checksum(mem, wf::band_range(0, N));
                        utils::print_checksum("hphi", cs, RTE_OUT(out__));
                    }
                    cs = sphi->checksum(mem, wf::band_range(0, N));
                    utils::print_checksum("sphi", cs, RTE_OUT(out__));
                }
                /* setup eigen-value problem */
                Band(ctx).set_subspace_mtrx(0, N, 0, *phi, *hphi, H, &H_old);
                break;
            }
            case davidson_evp_t::overlap: {
                wf::orthogonalize(ctx.spla_context(), mem,
                        wf::spin_range(nc_mag ? 2 : 0), wf::band_range(0, 0), wf::band_range(0, N), *phi, *phi,
                        {phi.get(), sphi.get()}, H, *res, false);
                /* setup eigen-value problem */
                Band(ctx).set_subspace_mtrx(0, N, 0, *phi, *sphi, H, &H_old);
                break;
            }
        }

        if (verbosity__ >= 1) {
            RTE_OUT(out__) << "set " << N << " x " << N << " subspace matrix" << std::endl;
        }

        /* DEBUG */
        if (ctx.cfg().control().verification() >= 1) {
            auto max_diff = check_hermitian(H, N);
            if (max_diff > (std::is_same<real_type<F>, double>::value ? 1e-12 : 1e-6)) {
                std::stringstream s;
                s << "H matrix is not Hermitian, max_err = " << max_diff << std::endl
                  << "  happened before entering the iterative loop" << std::endl;
                WARNING(s);
                if (N <= 20) {
                    auto s1 = H.serialize("davidson:H", N, N);
                    if (Hk__.kp().comm().rank() == 0) {
                        RTE_OUT(out__) << s1.str() << std::endl;
                    }
                }
            }
        }

        /* Seems like a smaller block size is not always improving time to solution much,
           so keep it num_bands. */
        int block_size = num_bands__.get();

        if (verbosity__ >= 1) {
            RTE_OUT(out__) << "diagonalize " << N << " x " << N << " Hamiltonian" << std::endl;
        }

        /* solve eigen-value problem with the size N and get lowest num_bands eigen-vectors */
        if (std_solver.solve(N, num_bands__.get(), H, &eval[0], evec)) {
            std::stringstream s;
            s << "error in diagonalziation";
            RTE_THROW(s);
        }

        ctx.evp_work_count(1);

        for (int i = 0; i < num_bands__.get(); i++) {
            kp.message(4, __function_name__, "eval[%i]=%20.16f\n", i, eval[i]);
        }

        /* number of newly added basis functions */
        int num_unconverged{0};

        /* tolerance for the norm of L2-norms of the residuals, used for
         * relative convergence criterion. We can only compute this after
         * we have the first residual norms available */
        F relative_frobenius_tolerance{0};
        F current_frobenius_norm{0};

        /* second phase: start iterative diagonalization */
        for (int iter_step = 0; iter_step < num_steps__; iter_step++) {
            if (verbosity__ >= 1) {
                RTE_OUT(out__) << "iter_step " << iter_step << " out of " << num_steps__ << std::endl;
            }

            int num_lockable{0};

            bool last_iteration = iter_step == (num_steps__ - 1);

            int num_ritz = num_bands__.get() - num_locked;

            /* don't compute residuals on last iteration */
            if (!last_iteration) {
                if (verbosity__ >= 1) {
                    RTE_OUT(out__) << "compute " << num_bands__.get() - num_locked
                          << " residuals from phi(" << num_locked << ":" << N - 1 << ")" << std::endl;
                }
                residual_result result;
                /* get new preconditionined residuals, and also hpsi and spsi as a by-product */
                switch (what) {
                    case davidson_evp_t::hamiltonian: {
                        result = residuals<T, F>(ctx, mem, sr, N, num_ritz, num_locked, eval, evec, *hphi, *sphi,
                                *hpsi, *spsi, *res, *h_diag, *o_diag, estimate_eval__, res_tol__, is_converged);

                        break;
                    }
                    case davidson_evp_t::overlap: {
                        result = residuals<T, F>(ctx, mem, sr, N, num_ritz, num_locked, eval, evec, *sphi, *phi, *spsi,
                                psi__, *res, *h_diag, *o_diag, estimate_eval__, res_tol__, is_converged);
                        break;
                    }
                }
                num_unconverged        = result.unconverged_residuals;
                num_lockable           = result.num_consecutive_smallest_converged;
                current_frobenius_norm = result.frobenius_norm;

                /* set the relative tolerance convergence criterion */
                if (iter_step == 0) {
                    relative_frobenius_tolerance = std::abs(current_frobenius_norm) * itso.relative_tolerance();
                }

                if (verbosity__ >= 1) {
                    RTE_OUT(out__) << "number of unconverged residuals : " << num_unconverged << std::endl;
                    RTE_OUT(out__) << "current_frobenius_norm : " << current_frobenius_norm << std::endl;
                }
                /* checksum info */
                if (pcs) {
                    auto cs_pw = res->checksum_pw(mem, wf::spin_index(0), wf::band_range(0, num_unconverged));
                    auto cs_mt = res->checksum_mt(mem, wf::spin_index(0), wf::band_range(0, num_unconverged));
                    auto cs = res->checksum(mem, wf::band_range(0, num_unconverged));
                    utils::print_checksum("res_pw", cs_pw, RTE_OUT(out__));
                    utils::print_checksum("res_mt", cs_mt, RTE_OUT(out__));
                    utils::print_checksum("res", cs, RTE_OUT(out__));
                }
            }

            /* verify convergence criteria */
            int num_converged              = num_ritz - num_unconverged;
            bool converged_by_relative_tol = (iter_step > 0) && (std::abs(current_frobenius_norm) < std::abs(relative_frobenius_tolerance));
            bool converged_by_absolute_tol = (num_locked + num_converged + itso.min_num_res()) >= num_bands__.get();

            bool converged = converged_by_relative_tol || converged_by_absolute_tol;

            /* TODO: num_unconverged might be very small at some point slowing down convergence
                     can we add more? */
            int expand_with     = std::min(num_unconverged, block_size);
            bool should_restart = (N + expand_with > num_phi) ||
                                  (num_lockable > 5 && num_unconverged < itso.early_restart() * num_lockable);

            kp.message(3, __function_name__,
                       "Restart = %s. Locked = %d. Converged = %d. Wanted = %d. Lockable = %d. "
                       "Num ritz = %d. Expansion size = %d\n",
                       should_restart ? "yes" : "no", num_locked, num_converged, num_bands__, num_lockable, num_ritz,
                       expand_with);

            /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
            if (should_restart || converged || last_iteration) {
                PROFILE("sirius::davidson|update_phi");
                /* recompute wave-functions */
                /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
                for (auto s = sr.begin(); s != sr.end(); s++) {
                    auto sp = phi->actual_spin_index(s);
                    wf::transform(ctx.spla_context(), mem, evec, 0, 0, 1.0, *phi, sp, wf::band_range(num_locked, N),
                            0.0, psi__, s, wf::band_range(num_locked, num_locked + num_ritz));
                }

                /* update eigen-values */
                for (int j = num_locked; j < num_bands__.get(); j++) {
                    result.eval(j, ispin_step) = eval[j - num_locked];
                }

                if (last_iteration && !converged) {
                    kp.message(3, __function_name__,
                               "Warning: maximum number of iterations reached, but %i "
                               "residual(s) did not converge for k-point %f %f %f\n",
                               num_unconverged, kp.vk()[0], kp.vk()[1], kp.vk()[2]);
                }

                /* exit the loop if the eigen-vectors are converged or this is a last iteration */
                if (converged || last_iteration) {
                    kp.message(3, __function_name__, "end of iterative diagonalization; n=%i, k=%i\n",
                               num_unconverged, iter_step);
                    break;
                } else { /* otherwise, set Psi as a new trial basis */
                    kp.message(3, __function_name__, "%s", "subspace size limit reached\n");

                    /* need to compute all hpsi and spsi states (not only unconverged - that was done
                     * by the residuals() function before) */
                    if (estimate_eval__) {
                        for (auto s = sr.begin(); s != sr.end(); s++) {
                            auto sp = sphi->actual_spin_index(s);
                            switch (what) {
                                case davidson_evp_t::hamiltonian: {
                                    wf::transform(ctx.spla_context(), mem, evec, 0, 0, 1.0, *hphi, sp,
                                        wf::band_range(num_locked, N), 0.0, *hpsi, sp,
                                        wf::band_range(0, num_ritz));
                                }
                                /* attention! there is no break statement here for a reason:
                                 * we want to coontinue and compute the update to spsi */
                                case davidson_evp_t::overlap: {
                                    wf::transform(ctx.spla_context(), mem, evec, 0, 0, 1.0, *sphi, sp,
                                        wf::band_range(num_locked, N), 0.0, *spsi, sp,
                                        wf::band_range(0, num_ritz));
                                }
                            }
                        }
                    }

                    // TODO: consider keeping more than num_bands when nearly all Ritz vectors have converged.
                    // TODO: remove
                    int keep = num_bands__.get();

                    /* update basis functions, hphi and sphi */
                    for (auto s = sr.begin(); s != sr.end(); s++) {
                        auto sp = phi->actual_spin_index(s);
                        wf::copy(mem, psi__, s, wf::band_range(num_locked, keep), *phi, sp, wf::band_range(num_locked, keep));
                        wf::copy(mem, *spsi, sp, wf::band_range(0, num_ritz), *sphi, sp, wf::band_range(num_locked, keep));
                        if (what == davidson_evp_t::hamiltonian) {
                            wf::copy(mem, *hpsi, sp, wf::band_range(0, num_ritz), *hphi, sp, wf::band_range(num_locked, keep));
                        }
                    }

                    /* remove locked Ritz values so indexing starts at unconverged eigenpairs */
                    if (locking__ && num_lockable > 0) {
                        for (int i = num_lockable; i < num_ritz; ++i) {
                            eval[i - num_lockable] = eval[i];
                        }
                    }

                    /* remove the locked block from the projected matrix too */
                    H_old.zero();
                    for (int i = 0; i < keep - num_locked; i++) {
                        H_old.set(i, i, eval[i]);
                    }

                    /* number of basis functions that we already have */
                    N = keep;

                    /* only when we do orthogonalization we can lock vecs */
                    if (locking__) {
                        num_locked += num_lockable;
                    }
                }
            }

            /* expand variational subspace with new basis vectors obtatined from residuals */
            for (auto s = sr.begin(); s != sr.end(); s++) {
                auto sp = phi->actual_spin_index(s);
                wf::copy(mem, *res, sp, wf::band_range(0, expand_with), *phi, sp, wf::band_range(N, N + expand_with));
            }
            if (should_restart && phi_extra__) {
                wf::copy(mem, *phi_extra__, wf::spin_index(0), wf::band_range(0, num_extra_phi), *phi,
                        wf::spin_index(0), wf::band_range(N + expand_with, N + expand_with + num_extra_phi));
                expand_with += num_extra_phi;
            }
            if (verbosity__ >= 1) {
                RTE_OUT(out__) << "expanding subspace of size " << N << " with " << expand_with
                      << " new basis functions" << std::endl;
            }

            if (verbosity__ >= 1) {
                RTE_OUT(out__) << "project out " << N << " existing basis states" << std::endl;
            }
            /* now, when we added new basis functions to phi, we can project out the old subspace,
             * then apply Hamiltonian and overlap to the remaining part and then orthogonalise the
             * new part of phi and finally, setup the eigen-vaule problem
             *
             * N is the number of previous basis functions
             * expand_with is the number of new basis functions */
            switch (what) {
                case davidson_evp_t::hamiltonian: {
                    if (ctx.full_potential()) {
                        if (should_restart && phi_extra__) {
                            /* apply Hamiltonian to expand_with - num_extra_phi states; copy the rest */
                            Hk__.apply_fv_h_o(false, false, wf::band_range(N, N + expand_with - num_extra_phi),
                                    *phi, hphi.get(), sphi.get());
                            wf::copy(mem, *hphi_extra, wf::spin_index(0), wf::band_range(0, num_extra_phi), *hphi,
                                    wf::spin_index(0), wf::band_range(N + expand_with - num_extra_phi, N + expand_with));
                            wf::copy(mem, *sphi_extra, wf::spin_index(0), wf::band_range(0, num_extra_phi), *sphi,
                                    wf::spin_index(0), wf::band_range(N + expand_with - num_extra_phi, N + expand_with));
                        } else {
                            Hk__.apply_fv_h_o(false, false, wf::band_range(N, N + expand_with), *phi, hphi.get(), sphi.get());
                        }
                        wf::orthogonalize(ctx.spla_context(), mem, sr, wf::band_range(0, N), wf::band_range(N, N + expand_with),
                                *phi, *sphi, {phi.get(), hphi.get(), sphi.get()}, H, *res, true);
                    } else {
                        /* for pseudopotential case we first project out the old subspace; this takes little less
                         * operations and gives a slighly more stable procedure, especially for fp32 */
                        project_out_subspace<T, F>(ctx.spla_context(), mem, sr, *phi, *sphi, N, expand_with, H);
                        Hk__.template apply_h_s<F>(sr, wf::band_range(N, N + expand_with), *phi, hphi.get(), sphi.get());
                        for (int j = 0; j < num_ortho_steps; j++) {
                            wf::orthogonalize(ctx.spla_context(), mem, sr, wf::band_range(0, N), wf::band_range(N, N + expand_with),
                                    *phi, *sphi, {phi.get(), hphi.get(), sphi.get()}, H, *res, false);
                        }
                    }
                    Band(ctx).set_subspace_mtrx<T, F>(N, expand_with, num_locked, *phi, *hphi, H, &H_old);
                    break;
                }
                case davidson_evp_t::overlap: {
                    project_out_subspace(ctx.spla_context(), mem, sr, *phi, *phi, N, expand_with, H);
                    if (ctx.full_potential()) {
                        Hk__.apply_fv_h_o(true, false, wf::band_range(N, N + expand_with), *phi, nullptr, sphi.get());
                    } else {
                        Hk__.template apply_h_s<F>(sr, wf::band_range(N, N + expand_with), *phi, nullptr, sphi.get());
                    }
                    for (int j = 0; j < num_ortho_steps; j++) {
                        wf::orthogonalize(ctx.spla_context(), mem, sr, wf::band_range(0, N), wf::band_range(N, N + expand_with),
                                         *phi, *phi, {phi.get(), sphi.get()}, H, *res, false);
                    }
                    Band(ctx).set_subspace_mtrx<T, F>(N, expand_with, num_locked, *phi, *sphi, H, &H_old);
                    break;
                }
            }

            kp.message(3, __function_name__, "Orthogonalized %d to %d\n", expand_with, N);

            if (ctx.cfg().control().verification() >= 1) {
                auto max_diff = check_hermitian(H, N + expand_with - num_locked);
                if (max_diff > (std::is_same<T, double>::value ? 1e-12 : 1e-6)) {
                    std::stringstream s;
                    kp.message(1, __function_name__, "H matrix of size %i is not Hermitian, maximum error: %18.12e\n",
                               N + expand_with - num_locked, max_diff);
                }
            }

            /* increase size of the variation space */
            N += expand_with;

            /* copy the Ritz values */
            eval >> eval_old;

            kp.message(3, __function_name__, "Computing %d pre-Ritz pairs\n", num_bands__.get() - num_locked);
            /* solve standard eigen-value problem with the size N */
            if (std_solver.solve(N - num_locked, num_bands__.get() - num_locked, H, &eval[0], evec)) {
                std::stringstream s;
                s << "error in diagonalziation";
                RTE_THROW(s);
            }

            ctx.evp_work_count(std::pow(static_cast<double>(N - num_locked) / num_bands__.get(), 3));

            kp.message(3, __function_name__, "step: %i, current subspace size: %i, maximum subspace size: %i\n",
                iter_step, N, num_phi);
            for (int i = 0; i < num_bands__.get() - num_locked; i++) {
                kp.message(4, __function_name__, "eval[%i]=%20.16f, diff=%20.16f\n", i, eval[i],
                           std::abs(eval[i] - eval_old[i]));
            }
            result.niter++;

        } /* loop over iterative steps k */
        ctx.print_memory_usage(__FILE__, __LINE__);
    } /* loop over ispin_step */
    PROFILE_STOP("sirius::davidson|iter");

    mg.clear();
    ctx.print_memory_usage(__FILE__, __LINE__);
    return result;
}


} // namespace

#endif
