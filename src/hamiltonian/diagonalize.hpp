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
