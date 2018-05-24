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

/** \file solve.hpp
 *
 *   \brief Contains interfaces to the sirius::Band solvers.
 */

inline void Band::solve_full_potential(K_point& kp__, Hamiltonian& hamiltonian__) const
{
    if (use_second_variation) {
        /* solve non-magnetic Hamiltonian (so-called first variation) */
        auto& itso = ctx_.iterative_solver_input();
        if (itso.type_ == "exact") {
            diag_full_potential_first_variation_exact(kp__, hamiltonian__);
        } else if (itso.type_ == "davidson") {
            diag_full_potential_first_variation_davidson(kp__, hamiltonian__);
        }
        /* generate first-variational states */
        kp__.generate_fv_states();
        /* solve magnetic Hamiltonian */
        diag_full_potential_second_variation(kp__, hamiltonian__);
        /* generate spinor wave-functions */
        kp__.generate_spinor_wave_functions();
    } else {
        TERMINATE_NOT_IMPLEMENTED
        //diag_full_potential_single_variation();
    }
}

template <typename T>
inline int Band::solve_pseudo_potential(K_point& kp__, Hamiltonian& hamiltonian__) const
{
    hamiltonian__.local_op().prepare(kp__.gkvec_partition());
    ctx_.fft_coarse().prepare(kp__.gkvec_partition());

    int niter{0};

    auto& itso = ctx_.iterative_solver_input();
    if (itso.type_ == "exact") {
        if (ctx_.num_mag_dims() != 3) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                diag_pseudo_potential_exact<double_complex>(&kp__, ispn, hamiltonian__);
            }
        } else {
            STOP();
        }
    } else if (itso.type_ == "davidson") {
        niter = diag_pseudo_potential_davidson<T>(&kp__, hamiltonian__);
    } else if (itso.type_ == "rmm-diis") {
        if (ctx_.num_mag_dims() != 3) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                diag_pseudo_potential_rmm_diis<T>(&kp__, ispn, hamiltonian__);
            }
        } else {
            STOP();
        }
    } else if (itso.type_ == "chebyshev") {
        P_operator<T> p_op(ctx_, kp__.p_mtrx());
        if (ctx_.num_mag_dims() != 3) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                diag_pseudo_potential_chebyshev<T>(&kp__, ispn, hamiltonian__, p_op);
            }
        } else {
            STOP();
        }
    } else {
        TERMINATE("unknown iterative solver type");
    }

    /* check residuals */
    if (ctx_.control().verification_ >= 1) {
        check_residuals<T>(&kp__, hamiltonian__); 
    }

    ctx_.fft_coarse().dismiss();
    return niter;
}

inline void Band::solve(K_point_set& kset__, Hamiltonian& hamiltonian__, bool precompute__) const
{
    PROFILE("sirius::Band::solve");

    if (precompute__ && ctx_.full_potential()) {
        hamiltonian__.potential().generate_pw_coefs();
        hamiltonian__.potential().update_atomic_potential();
        unit_cell_.generate_radial_functions();
        unit_cell_.generate_radial_integrals();
    }
    
    /* map local potential to a coarse grid */
    if (ctx_.full_potential()) {
        hamiltonian__.local_op().prepare(hamiltonian__.potential(), ctx_.step_function());
    } else {
        hamiltonian__.local_op().prepare(hamiltonian__.potential());
        /* prepare non-local operators */
        if (ctx_.gamma_point() && (ctx_.so_correction() == false)) {
            hamiltonian__.prepare<double>();
        } else {
            hamiltonian__.prepare<double_complex>();
        }
    }

    if (ctx_.comm().rank() == 0 && ctx_.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    int num_dav_iter{0};
    /* solve secular equation and generate wave functions */
    for (int ikloc = 0; ikloc < kset__.spl_num_kpoints().local_size(); ikloc++) {
        int ik  = kset__.spl_num_kpoints(ikloc);
        auto kp = kset__[ik];

        if (ctx_.full_potential()) {
            solve_full_potential(*kp, hamiltonian__);
        } else {
            if (ctx_.gamma_point() && (ctx_.so_correction() == false)) {
                num_dav_iter += solve_pseudo_potential<double>(*kp, hamiltonian__);
            } else {
                num_dav_iter += solve_pseudo_potential<double_complex>(*kp, hamiltonian__);
            }
        }
    }
    kset__.comm().allreduce(&num_dav_iter, 1);
    if (ctx_.comm().rank() == 0 && !ctx_.full_potential()) {
        printf("Average number of iterations: %12.6f\n", static_cast<double>(num_dav_iter) / kset__.num_kpoints());
    }

    hamiltonian__.local_op().dismiss();
    if (!ctx_.full_potential()) {
        hamiltonian__.dismiss();
    }

    /* synchronize eigen-values */
    kset__.sync_band_energies();

    if (ctx_.control().verbosity_ >= 1 && ctx_.comm().rank() == 0) {
        printf("Lowest band energies\n");
        for (int ik = 0; ik < kset__.num_kpoints(); ik++) {
            printf("ik : %2i, ", ik);
            for (int j = 0; j < std::min(ctx_.control().num_bands_to_print_, ctx_.num_bands()); j++) {
                printf("%12.6f", kset__[ik]->band_energy(j, 0));
            }
            if (ctx_.num_mag_dims() == 1) {
                printf("\n         ");
                for (int j = 0; j < std::min(ctx_.control().num_bands_to_print_, ctx_.num_bands()); j++) {
                    printf("%12.6f", kset__[ik]->band_energy(j, 1));
                }
            }
            printf("\n");
        }
    }
}
