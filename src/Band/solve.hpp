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

/** \file solve.hpp
 *   
 *   \brief Contains interfaces to the sirius::Band solvers.
 */

inline void Band::solve_with_second_variation(K_point& kp__, Potential& potential__) const
{
    /* solve non-magnetic Hamiltonian (so-called first variation) */
    auto& itso = ctx_.iterative_solver_input();
    if (itso.type_ == "exact") {
        diag_fv_exact(&kp__, potential__);
    } else if (itso.type_ == "davidson") {
        diag_fv_davidson(&kp__);
    }
    /* generate first-variational states */
    kp__.generate_fv_states();
    /* solve magnetic Hamiltonian */
    diag_sv(&kp__, potential__);
    /* generate spinor wave-functions */
    kp__.generate_spinor_wave_functions();
}

inline void Band::solve_with_single_variation(K_point& kp__, Potential& potential__) const
{
    switch (ctx_.esm_type()) {
        case electronic_structure_method_t::pseudopotential: {
            if (ctx_.gamma_point()) {
                diag_pseudo_potential<double>(&kp__);
            } else {
                diag_pseudo_potential<double_complex>(&kp__);
            }
            break;
        }
        default: {
            TERMINATE_NOT_IMPLEMENTED
        }
    }
    //== if (kp->num_ranks() > 1 && !parameters_.gen_evp_solver()->parallel())
    //==     error_local(__FILE__, __LINE__, "eigen-value solver is not parallel");

    //== mdarray<double_complex, 2> h(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
    //== mdarray<double_complex, 2> o(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
    //== 
    //== set_o(kp, o);

    //== std::vector<double> eval(parameters_.num_bands());
    //== mdarray<double_complex, 2>& fd_evec = kp->fd_eigen_vectors();

    //== if (parameters_.num_mag_dims() == 0)
    //== {
    //==     assert(kp->gklo_basis_size() >= parameters_.num_fv_states());
    //==     set_h<nm>(kp, effective_potential, effective_magnetic_field, h);
    //==    
    //==     Timer t2("sirius::Band::solve_fd|diag");
    //==     parameters_.gen_evp_solver()->solve(kp->gklo_basis_size(), kp->gklo_basis_size_row(), kp->gklo_basis_size_col(), 
    //==                                         parameters_.num_fv_states(), h.ptr(), h.ld(), o.ptr(), o.ld(), 
    //==                                         &eval[0], fd_evec.ptr(), fd_evec.ld());
    //== }
    //== 
    //== if (parameters_.num_mag_dims() == 1)
    //== {
    //==     assert(kp->gklo_basis_size() >= parameters_.num_fv_states());

    //==     mdarray<double_complex, 2> o1(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
    //==     memcpy(&o1(0, 0), &o(0, 0), o.size() * sizeof(double_complex));

    //==     set_h<uu>(kp, effective_potential, effective_magnetic_field, h);
    //==    
    //==     Timer t2("sirius::Band::solve_fd|diag");
    //==     parameters_.gen_evp_solver()->solve(kp->gklo_basis_size(), kp->gklo_basis_size_row(), kp->gklo_basis_size_col(), 
    //==                                         parameters_.num_fv_states(), h.ptr(), h.ld(), o.ptr(), o.ld(), 
    //==                                         &eval[0], &fd_evec(0, 0), fd_evec.ld());
    //==     t2.stop();

    //==     set_h<dd>(kp, effective_potential, effective_magnetic_field, h);
    //==     
    //==     t2.start();
    //==     parameters_.gen_evp_solver()->solve(kp->gklo_basis_size(), kp->gklo_basis_size_row(), kp->gklo_basis_size_col(), 
    //==                                         parameters_.num_fv_states(), h.ptr(), h.ld(), o1.ptr(), o1.ld(), 
    //==                                         &eval[parameters_.num_fv_states()], 
    //==                                         &fd_evec(0, parameters_.spl_fv_states().local_size()), fd_evec.ld());
    //==     t2.stop();
    //== }

    //== kp->set_band_energies(&eval[0]);
}

inline void Band::solve_for_kset(K_point_set& kset__,
                                 Potential& potential__,
                                 bool precompute__) const
{
    PROFILE("sirius::Band::solve_for_kset");

    if (precompute__ && ctx_.full_potential()) {
        potential__.generate_pw_coefs();
        potential__.update_atomic_potential();
        unit_cell_.generate_radial_functions();
        unit_cell_.generate_radial_integrals();
    }

    if (ctx_.full_potential()) {
        local_op_->prepare(ctx_.gvec_coarse(), ctx_.num_mag_dims(), potential__, ctx_.step_function());
    } else {
        local_op_->prepare(ctx_.gvec_coarse(), ctx_.num_mag_dims(), potential__);
    }

    if (ctx_.comm().rank() == 0 && ctx_.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    /* solve secular equation and generate wave functions */
    for (int ikloc = 0; ikloc < kset__.spl_num_kpoints().local_size(); ikloc++) {
        int ik = kset__.spl_num_kpoints(ikloc);
        auto kp = kset__[ik];

        if (ctx_.full_potential() && use_second_variation) {
            solve_with_second_variation(*kp, potential__);
        } else {
            solve_with_single_variation(*kp, potential__);
        }
    }

    local_op_->dismiss();

    /* synchronize eigen-values */
    kset__.sync_band_energies();

    if (ctx_.control().verbosity_ >= 1 && ctx_.comm().rank() == 0) {
        printf("Lowest band energies\n");
        for (int ik = 0; ik < kset__.num_kpoints(); ik++) {
            printf("ik : %2i, ", ik);
            if (ctx_.num_mag_dims() != 1) {
                for (int j = 0; j < std::min(ctx_.control().num_bands_to_print_, ctx_.num_bands()); j++) {
                    printf("%12.6f", kset__[ik]->band_energy(j));
                }
            } else {
                for (int j = 0; j < std::min(ctx_.control().num_bands_to_print_, ctx_.num_fv_states()); j++) {
                    printf("%12.6f", kset__[ik]->band_energy(j));
                }
                printf("\n         ");
                for (int j = 0; j < std::min(ctx_.control().num_bands_to_print_, ctx_.num_fv_states()); j++) {
                    printf("%12.6f", kset__[ik]->band_energy(ctx_.num_fv_states() + j));
                }
            }
            printf("\n");
        }
    }
}

