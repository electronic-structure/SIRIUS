// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

inline void Band::solve_fv(K_point* kp__,
                           Periodic_function<double>* effective_potential__) const
{
    if (kp__->gklo_basis_size() < ctx_.num_fv_states()) {
        TERMINATE("basis size is too small");
    }

    switch (ctx_.esm_type()) {
        case electronic_structure_method_t::full_potential_pwlo:
        case electronic_structure_method_t::full_potential_lapwlo: {
            diag_fv_full_potential(kp__, effective_potential__);
            break;
        }
        default: {
            TERMINATE_NOT_IMPLEMENTED
        }
    }
}

inline void Band::solve_fd(K_point* kp__,
                           Periodic_function<double>* effective_potential__, 
                           Periodic_function<double>* effective_magnetic_field__[3]) const
{
    switch (ctx_.esm_type()) {
        case electronic_structure_method_t::pseudopotential: {
            if (ctx_.gamma_point()) {
                diag_pseudo_potential<double>(kp__, effective_potential__, effective_magnetic_field__);
            } else {
                diag_pseudo_potential<double_complex>(kp__, effective_potential__, effective_magnetic_field__);
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

inline void Band::solve_sv(K_point* kp,
                           Periodic_function<double>* effective_magnetic_field[3]) const
{
    PROFILE_WITH_TIMER("sirius::Band::solve_sv");

    if (!ctx_.need_sv()) {
        kp->bypass_sv();
        return;
    }

    if (kp->num_ranks() > 1 && !std_evp_solver().parallel()) {
        TERMINATE("eigen-value solver is not parallel");
    }

    std::vector<double> band_energies(ctx_.num_bands());

    /* product of the second-variational Hamiltonian and a first-variational wave-function */
    std::vector<wave_functions> hpsi;
    for (int i = 0; i < ctx_.num_mag_comp(); i++) {
        hpsi.push_back(std::move(wave_functions(ctx_,
                                                kp->comm(),
                                                kp->gkvec(),
                                                unit_cell_.num_atoms(),
                                                [this](int ia)
                                                {
                                                    return unit_cell_.atom(ia).mt_basis_size();
                                                },
                                                ctx_.num_fv_states())));
    }

    /* compute product of magnetic field and wave-function */
    if (ctx_.num_spins() == 2) {
        apply_magnetic_field(kp->fv_states(), kp->gkvec(), effective_magnetic_field, hpsi);
    }
    else {
        hpsi[0].pw_coeffs().prime().zero();
        hpsi[0].mt_coeffs().prime().zero();
    }

    //== if (ctx_.uj_correction())
    //== {
    //==     apply_uj_correction<uu>(kp->fv_states_col(), hpsi);
    //==     if (ctx_.num_mag_dims() != 0) apply_uj_correction<dd>(kp->fv_states_col(), hpsi);
    //==     if (ctx_.num_mag_dims() == 3) 
    //==     {
    //==         apply_uj_correction<ud>(kp->fv_states_col(), hpsi);
    //==         if (ctx_.std_evp_solver()->parallel()) apply_uj_correction<du>(kp->fv_states_col(), hpsi);
    //==     }
    //== }

    //== if (ctx_.so_correction()) apply_so_correction(kp->fv_states_col(), hpsi);

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU && kp->num_ranks() == 1) {
        STOP();
        //kp->fv_states<true>().coeffs().allocate(memory_t::device);
        //kp->fv_states<true>().coeffs().copy_to_device();
    }
    #endif
 
    int nfv = ctx_.num_fv_states();
    int bs = ctx_.cyclic_block_size();

    if (ctx_.num_mag_dims() != 3) {
        dmatrix<double_complex> h(nfv, nfv, ctx_.blacs_grid(), bs, bs);

        if (ctx_.processing_unit() == GPU && kp->num_ranks() == 1) {
            #ifdef __GPU
            STOP();
            //h.allocate(memory_t::device);
            #endif
        }

        /* perform one or two consecutive diagonalizations */
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            inner(kp->fv_states(), 0, nfv, hpsi[ispn], 0, nfv, h, 0, 0);
            //if (ctx_.processing_unit() == GPU && kp->num_ranks() == 1) {
            //    #ifdef __GPU
            //    runtime::Timer t4("sirius::Band::solve_sv|zgemm");
            //    hpsi[ispn]->coeffs().allocate(memory_t::device);
            //    hpsi[ispn]->coeffs().copy_to_device();
            //    linalg<GPU>::gemm(2, 0, nfv, nfv, fvsz,
            //                      &alpha, 
            //                      kp->fv_states<true>().coeffs().at<GPU>(), kp->fv_states<true>().coeffs().ld(),
            //                      hpsi[ispn]->coeffs().at<GPU>(), hpsi[ispn]->coeffs().ld(),
            //                      &beta,
            //                      h.at<GPU>(), h.ld());
            //    h.copy_to_host();
            //    hpsi[ispn]->coeffs().deallocate_on_device();
            //    double tval = t4.stop();
            //    DUMP("effective zgemm performance: %12.6f GFlops", 
            //         8e-9 * ctx_.num_fv_states() * ctx_.num_fv_states() * fvsz / tval);
            //    #else
            //    TERMINATE_NO_GPU
            //    #endif
            //} else {
            //    STOP();
            //    ///* compute <wf_i | h * wf_j> for up-up or dn-dn block */
            //    //linalg<CPU>::gemm(2, 0, nfv, nfv, fvsz, complex_one, kp->fv_states<true>().prime(), hpsi[ispn]->prime(),
            //    //                  complex_zero, h);
            //}
            
            for (int i = 0; i < nfv; i++) {
                h.add(i, i, kp->fv_eigen_value(i));
            }
        
            runtime::Timer t1("sirius::Band::solve_sv|stdevp");
            std_evp_solver().solve(nfv, h.at<CPU>(), h.ld(), &band_energies[ispn * nfv],
                                   kp->sv_eigen_vectors(ispn).at<CPU>(), kp->sv_eigen_vectors(ispn).ld());
        }
    } else {
        int nb = ctx_.num_bands();
        dmatrix<double_complex> h(nb, nb, ctx_.blacs_grid(), bs, bs);

        /* compute <wf_i | h * wf_j> for up-up block */
        inner(kp->fv_states(), 0, nfv, hpsi[0], 0, nfv, h, 0, 0);
        /* compute <wf_i | h * wf_j> for dn-dn block */
        inner(kp->fv_states(), 0, nfv, hpsi[1], 0, nfv, h, nfv, nfv);
        /* compute <wf_i | h * wf_j> for up-dn block */
        inner(kp->fv_states(), 0, nfv, hpsi[2], 0, nfv, h, 0, nfv);

        if (kp->comm().size() == 1) {
            for (int i = 0; i < nfv; i++) {
                for (int j = 0; j < nfv; j++) {
                    h(nfv + j, i) = std::conj(h(i, nfv + j));
                }
            }
        } else {
            linalg<CPU>::tranc(nfv, nfv, h, 0, nfv, h, nfv, 0);
        }
        
        for (int i = 0; i < nfv; i++) {
            h.add(i,       i,       kp->fv_eigen_value(i));
            h.add(i + nfv, i + nfv, kp->fv_eigen_value(i));
        }
        runtime::Timer t1("sirius::Band::solve_sv|stdevp");
        std_evp_solver().solve(nb, h.at<CPU>(), h.ld(), &band_energies[0],
                               kp->sv_eigen_vectors(0).at<CPU>(), kp->sv_eigen_vectors(0).ld());
    }
 
    if (ctx_.processing_unit() == GPU && kp->num_ranks() == 1) {
        #ifdef __GPU
        STOP();
        //kp->fv_states<true>().coeffs().deallocate_on_device();
        #endif
    }

    kp->set_band_energies(&band_energies[0]);
}

inline void Band::solve_for_kset(K_set& kset, Potential& potential, bool precompute) const
{
    PROFILE_WITH_TIMER("sirius::Band::solve_for_kset");

    if (precompute && ctx_.full_potential()) {
        potential.generate_pw_coefs();
        potential.update_atomic_potential();
        unit_cell_.generate_radial_functions();
        unit_cell_.generate_radial_integrals();
    }

    // TODO: mapping to coarse effective potential is k-point independent

    /* solve secular equation and generate wave functions */
    for (int ikloc = 0; ikloc < kset.spl_num_kpoints().local_size(); ikloc++) {
        int ik = kset.spl_num_kpoints(ikloc);

        if (use_second_variation && ctx_.full_potential()) {
            /* solve non-magnetic Hamiltonian (so-called first variation) */
            solve_fv(kset.k_point(ik), potential.effective_potential());
            /* generate first-variational states */
            kset.k_point(ik)->generate_fv_states();
            /* solve magnetic Hamiltonian */
            solve_sv(kset.k_point(ik), potential.effective_magnetic_field());
            /* generate spinor wave-functions */
            kset.k_point(ik)->generate_spinor_wave_functions();
        } else {
            solve_fd(kset.k_point(ik), potential.effective_potential(), potential.effective_magnetic_field());
        }
    }

    /* synchronize eigen-values */
    kset.sync_band_energies();

    if (ctx_.control().verbosity_ > 0 && ctx_.comm().rank() == 0) {
        printf("Lowest band energies\n");
        for (int ik = 0; ik < kset.num_kpoints(); ik++) {
            printf("ik : %2i, ", ik);
            if (ctx_.num_mag_dims() != 1) {
                for (int j = 0; j < std::min(ctx_.control().num_bands_to_print_, ctx_.num_bands()); j++) {
                    printf("%12.6f", kset.k_point(ik)->band_energy(j));
                }
            } else {
                for (int j = 0; j < std::min(ctx_.control().num_bands_to_print_, ctx_.num_fv_states()); j++) {
                    printf("%12.6f", kset.k_point(ik)->band_energy(j));
                }
                printf("\n         ");
                for (int j = 0; j < std::min(ctx_.control().num_bands_to_print_, ctx_.num_fv_states()); j++) {
                    printf("%12.6f", kset.k_point(ik)->band_energy(ctx_.num_fv_states() + j));
                }
            }
            printf("\n");
        }

        //== FILE* fout = fopen("eval.txt", "w");
        //== for (int ik = 0; ik < num_kpoints(); ik++)
        //== {
        //==     fprintf(fout, "ik : %2i\n", ik);
        //==     for (int j = 0; j < ctx_.num_bands(); j++)
        //==         fprintf(fout, "%4i: %18.10f\n", j, kpoints_[ik]->band_energy(j));
        //== }
        //== fclose(fout);
    }
}

