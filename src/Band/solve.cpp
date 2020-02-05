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
#include "Potential/potential.hpp"

namespace sirius {

void
Band::solve_full_potential(Hamiltonian_k& Hk__) const
{
    if (ctx_.control().use_second_variation_) {
        /* solve non-magnetic Hamiltonian (so-called first variation) */
        auto& itso = ctx_.iterative_solver_input();
        if (itso.type_ == "exact") {
            diag_full_potential_first_variation_exact(Hk__);
        } else if (itso.type_ == "davidson") {
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

template <typename T>
int
Band::solve_pseudo_potential(Hamiltonian_k& Hk__) const
{
    ctx_.print_memory_usage(__FILE__, __LINE__);

    int niter{0};

    auto& itso = ctx_.iterative_solver_input();
    if (itso.type_ == "exact") {
        if (ctx_.num_mag_dims() != 3) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                diag_pseudo_potential_exact<double_complex>(ispn, Hk__);
            }
        } else {
            STOP();
        }
    } else if (itso.type_ == "davidson") {
        niter = diag_pseudo_potential_davidson<T>(Hk__);
    //} else if (itso.type_ == "rmm-diis") {
    //    if (ctx_.num_mag_dims() != 3) {
    //        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
    //            diag_pseudo_potential_rmm_diis<T>(&kp__, ispn, hamiltonian__);
    //        }
    //    } else {
    //        STOP();
    //    }
    //} else if (itso.type_ == "chebyshev") {
    //    P_operator<T> p_op(ctx_, kp__.p_mtrx());
    //    if (ctx_.num_mag_dims() != 3) {
    //        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
    //            diag_pseudo_potential_chebyshev<T>(&kp__, ispn, hamiltonian__, p_op);
    //        }
    //    } else {
    //        STOP();
    //    }
    } else {
        TERMINATE("unknown iterative solver type");
    }

    /* check residuals */
    if (ctx_.control().verification_ >= 1) {
        check_residuals<T>(Hk__);
        check_wave_functions<T>(Hk__);
    }

    ctx_.print_memory_usage(__FILE__, __LINE__);

    return niter;
}

void
Band::solve(K_point_set& kset__, Hamiltonian0& H0__, bool precompute__) const
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
        ctx_.message(1, __function_name__, "iterative solver tolerance: %18.12f\n", ctx_.iterative_solver_tolerance());
    }

    int num_dav_iter{0};
    /* solve secular equation and generate wave functions */
    for (int ikloc = 0; ikloc < kset__.spl_num_kpoints().local_size(); ikloc++) {
        int ik  = kset__.spl_num_kpoints(ikloc);
        auto kp = kset__[ik];

        auto Hk = H0__(*kp);
        if (ctx_.full_potential()) {
            solve_full_potential(Hk);
        } else {
            if (ctx_.gamma_point() && (ctx_.so_correction() == false)) {
                num_dav_iter += solve_pseudo_potential<double>(Hk);
            } else {
                num_dav_iter += solve_pseudo_potential<double_complex>(Hk);
            }
        }
    }
    kset__.comm().allreduce(&num_dav_iter, 1);
    if (!ctx_.full_potential()) {
        ctx_.message(1, __function_name__, "average number of iterations: %12.6f\n",
                     static_cast<double>(num_dav_iter) / kset__.num_kpoints());
    }

    /* synchronize eigen-values */
    kset__.sync_band_energies();

    ctx_.message(2, __function_name__, "%s", "Lowest band energies\n");
    if (ctx_.control().verbosity_ >= 2 && ctx_.comm().rank() == 0) {
        for (int ik = 0; ik < kset__.num_kpoints(); ik++) {
            std::printf("ik : %2i, ", ik);
            for (int j = 0; j < std::min(ctx_.control().num_bands_to_print_, ctx_.num_bands()); j++) {
                std::printf("%12.6f", kset__[ik]->band_energy(j, 0));
            }
            if (ctx_.num_mag_dims() == 1) {
                std::printf("\n         ");
                for (int j = 0; j < std::min(ctx_.control().num_bands_to_print_, ctx_.num_bands()); j++) {
                    std::printf("%12.6f", kset__[ik]->band_energy(j, 1));
                }
            }
            std::printf("\n");
        }
    }
}

} // namespace
