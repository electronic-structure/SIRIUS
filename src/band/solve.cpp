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

/** \file solve.cpp
 *
 *   \brief Contains interfaces to the sirius::Band solvers.
 */
#include "band.hpp"
#include "davidson.hpp"
#include "potential/potential.hpp"

namespace sirius {

template <typename T>
void
Band::solve_full_potential(Hamiltonian_k<T>& Hk__) const
{
    if (ctx_.cfg().control().use_second_variation()) {
        /* solve non-magnetic Hamiltonian (so-called first variation) */
        auto& itso = ctx_.cfg().iterative_solver();
        if (itso.type() == "exact") {
            diag_full_potential_first_variation_exact(Hk__);
        } else if (itso.type() == "davidson") {
            diag_full_potential_first_variation_davidson(Hk__);
        }
        /* generate first-variational states */
        Hk__.kp().generate_fv_states();
        /* solve magnetic Hamiltonian */
        diag_full_potential_second_variation(Hk__);
        /* generate spinor wave-functions */
        Hk__.kp().generate_spinor_wave_functions();
    } else {
        throw std::runtime_error("not implemented");
        //diag_full_potential_single_variation();
    }
}

template
void
Band::solve_full_potential<double>(Hamiltonian_k<double>& Hk__) const;

template<>
void
Band::solve_full_potential<float>(Hamiltonian_k<float>& Hk__) const
{
    RTE_THROW("FP32 is not implemented for FP-LAPW");
}

template <typename T>
int
Band::solve_pseudo_potential(Hamiltonian_k<real_type<T>>& Hk__) const
{
    ctx_.print_memory_usage(__FILE__, __LINE__);

    int niter{0};

    auto& itso = ctx_.cfg().iterative_solver();
    if (itso.type() == "exact") {
        if (ctx_.num_mag_dims() != 3) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                diag_pseudo_potential_exact<T>(ispn, Hk__);
            }
        } else {
            STOP();
        }
    } else if (itso.type() == "davidson") {
        auto& kp = Hk__.kp();

        auto tolerance = [&](int j__, int ispn__) -> double {

            double tol      = ctx_.iterative_solver_tolerance();
            double empy_tol = std::max(tol * ctx_.cfg().settings().itsol_tol_ratio(),
                                       ctx_.cfg().iterative_solver().empty_states_tolerance());

            /* if band is empty, make tolerance larger (in most cases we don't need high precision on
             * unoccupied  states) */
            if (std::abs(kp.band_occupancy(j__, ispn__)) < ctx_.min_occupancy() * ctx_.max_occupancy()) {
                tol += empy_tol;
            }

            return tol;
        };

        auto result = davidson<T>(Hk__, kp.spinor_wave_functions(),
                [&](int i, int ispn){return kp.band_occupancy(i, ispn);}, tolerance);
        niter = result.niter;
        for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
            for (int j = 0; j < ctx_.num_bands(); j++) {
                kp.band_energy(j, ispn, result.eval(j, ispn));
            }
        }
        //niter = diag_pseudo_potential_davidson<T>(Hk__);
    } else {
        RTE_THROW("unknown iterative solver type");
    }

    /* check residuals */
    if (ctx_.cfg().control().verification() >= 2) {
        check_residuals<T>(Hk__);
        check_wave_functions<T>(Hk__);
    }

    ctx_.print_memory_usage(__FILE__, __LINE__);

    return niter;
}

template <typename T>
void
Band::solve(K_point_set& kset__, Hamiltonian0<T>& H0__, bool precompute__) const
{
    PROFILE("sirius::Band::solve");

    if (precompute__ && ctx_.full_potential()) {
        H0__.potential().generate_pw_coefs();
        H0__.potential().update_atomic_potential();
        unit_cell_.generate_radial_functions();
        unit_cell_.generate_radial_integrals();
    }

    ctx_.print_memory_usage(__FILE__, __LINE__);

    if (!ctx_.full_potential()) {
        ctx_.message(2, __function_name__, "iterative solver tolerance: %1.4e\n", ctx_.iterative_solver_tolerance());
    }

    int num_dav_iter{0};
    /* solve secular equation and generate wave functions */
    for (int ikloc = 0; ikloc < kset__.spl_num_kpoints().local_size(); ikloc++) {
        int ik  = kset__.spl_num_kpoints(ikloc);
        auto kp = kset__.get<T>(ik);

        auto Hk = H0__(*kp);
        if (ctx_.full_potential()) {
            solve_full_potential<T>(Hk);
        } else {
            if (ctx_.gamma_point() && (ctx_.so_correction() == false)) {
                num_dav_iter += solve_pseudo_potential<T>(Hk);
            } else {
                num_dav_iter += solve_pseudo_potential<std::complex<T>>(Hk);
            }
        }
    }
    kset__.comm().allreduce(&num_dav_iter, 1);
    ctx_.num_itsol_steps(num_dav_iter);
    if (!ctx_.full_potential()) {
        ctx_.message(2, __function_name__, "average number of iterations: %12.6f\n",
                     static_cast<double>(num_dav_iter) / kset__.num_kpoints());
    }

    /* synchronize eigen-values */
    kset__.sync_band<T, sync_band_t::energy>();

    ctx_.message(2, __function_name__, "%s", "Lowest band energies\n");
    if (ctx_.verbosity() >= 2 && ctx_.comm().rank() == 0) {
        for (int ik = 0; ik < kset__.num_kpoints(); ik++) {
            std::printf("ik : %2i, ", ik);
            for (int j = 0; j < std::min(ctx_.cfg().control().num_bands_to_print(), ctx_.num_bands()); j++) {
                std::printf("%12.6f", kset__.get<T>(ik)->band_energy(j, 0));
            }
            if (ctx_.num_mag_dims() == 1) {
                std::printf("\n         ");
                for (int j = 0; j < std::min(ctx_.cfg().control().num_bands_to_print(), ctx_.num_bands()); j++) {
                    std::printf("%12.6f", kset__.get<T>(ik)->band_energy(j, 1));
                }
            }
            std::printf("\n");
        }
    }
}

template
void
Band::solve<double>(K_point_set& kset__, Hamiltonian0<double>& H0__, bool precompute__) const;

#if defined (USE_FP32)
template
void
Band::solve<float>(K_point_set& kset__, Hamiltonian0<float>& H0__, bool precompute__) const;
#endif

} // namespace
