/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file diagonalize.hpp
 *
 *  \brief Entry point for Hamiltonain diagonalization.
 */

#ifndef __DIAGONALIZE_HPP__
#define __DIAGONALIZE_HPP__

#include "diagonalize_fp.hpp"
#include "diagonalize_pp.hpp"
#include "k_point/k_point_set.hpp"

namespace sirius {

struct diagonalize_result_t
{
    davidson_result_t davidson_result;
    double avg_num_iter{0};
    bool converged;
};

/// Diagonalize KS Hamiltonian for all k-points in the set.
/** \tparam T  Precision type of the wave-functions
 *  \tparam F  Precition type of the Hamiltonian matrix
 */
template <typename T, typename F>
inline auto
diagonalize(Hamiltonian0<T> const& H0__, K_point_set& kset__, double itsol_tol__, int itsol_num_steps__)
{
    PROFILE("sirius::diagonalize");

    auto& ctx = H0__.ctx();
    print_memory_usage(ctx.out(), FILE_LINE);

    diagonalize_result_t result;

    auto& itso = ctx.cfg().iterative_solver();

    double empy_tol{itsol_tol__};
    if (itso.type() == "davidson") {
        empy_tol =
                std::max(itsol_tol__ * ctx.cfg().iterative_solver().tolerance_ratio(), itso.empty_states_tolerance());
        RTE_OUT(ctx.out(2)) << "iterative solver tolerance (occupied, empty): " << itsol_tol__ << " "
                            << itsol_tol__ + empy_tol << std::endl;
    }

    int num_dav_iter{0};
    bool converged{true};
    /* solve secular equation and generate wave functions */
    for (auto it : kset__.spl_num_kpoints()) {
        auto kp = kset__.get<T>(it.i);

        auto Hk = H0__(*kp);
        if (ctx.full_potential()) {
            diagonalize_fp<T>(Hk, *kp, itsol_tol__);
        } else {
            if (itso.type() == "exact") {
                if (ctx.gamma_point() || ctx.num_mag_dims() == 3) {
                    RTE_THROW("not implemented");
                }
                for (int ispn = 0; ispn < ctx.num_spins(); ispn++) {
                    diagonalize_pp_exact<T, std::complex<F>>(ispn, Hk, *kp);
                }
            } else {
                if (ctx.gamma_point() && (ctx.so_correction() == false)) {
                    result.davidson_result = diagonalize_pp<T, F>(Hk, *kp, itsol_tol__, empy_tol, itsol_num_steps__);
                } else {
                    result.davidson_result =
                            diagonalize_pp<T, std::complex<F>>(Hk, *kp, itsol_tol__, empy_tol, itsol_num_steps__);
                }
                num_dav_iter += result.davidson_result.niter;
                converged = converged & result.davidson_result.converged;
            }
        }
    }
    kset__.comm().allreduce(&num_dav_iter, 1);
    kset__.comm().allreduce<bool, mpi::op_t::land>(&converged, 1);
    ctx.num_itsol_steps(num_dav_iter);
    result.avg_num_iter = static_cast<double>(num_dav_iter) / kset__.num_kpoints();
    result.converged    = converged;
    if (!ctx.full_potential()) {
        RTE_OUT(ctx.out(2)) << "average number of iterations: " << result.avg_num_iter << std::endl;
    }

    /* synchronize eigen-values */
    kset__.sync_band<T, sync_band_t::energy>();

    if (ctx.verbosity() >= 2) {
        std::stringstream s;
        s << "Lowest band energies" << std::endl;
        int nbnd = std::min(ctx.cfg().control().num_bands_to_print(), ctx.num_bands());
        for (int ik = 0; ik < kset__.num_kpoints(); ik++) {
            s << "ik:" << std::setw(5) << ik;
            for (int j = 0; j < nbnd; j++) {
                s << ffmt(12, 6) << kset__.get<T>(ik)->band_energy(j, 0);
            }
            if (ctx.num_mag_dims() == 1) {
                s << std::endl << "        ";
                for (int j = 0; j < nbnd; j++) {
                    s << ffmt(12, 6) << kset__.get<T>(ik)->band_energy(j, 1);
                }
            }
            s << std::endl;
        }
        RTE_OUT(ctx.out(2)) << s.str();
    }
    print_memory_usage(ctx.out(), FILE_LINE);

    return result;
}

} // namespace sirius

#endif
