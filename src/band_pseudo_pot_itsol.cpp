#include "band.h"

namespace sirius {

#ifdef _SCALAPACK_
void Band::diag_fv_pseudo_potential_parallel_chebyshev(K_point* kp__,
                                                       std::vector<double> const& veff_it_coarse__)
{
    log_function_enter(__func__);

    /* alias for wave-functions */
    dmatrix<double_complex>& psi = kp__->fv_states_panel();

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();

    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();

    auto uc = parameters_.unit_cell();

    //auto& beta_pw_panel = kp__->beta_pw_panel();
    //dmatrix<double_complex> S(uc->mt_basis_size(), uc->mt_basis_size(), kp__->blacs_grid());
    //linalg<CPU>::gemm(2, 0, uc->mt_basis_size(), uc->mt_basis_size(), kp__->num_gkvec(), complex_one,
    //                  beta_pw_panel, beta_pw_panel, complex_zero, S);
    //for (int ia = 0; ia < uc->num_atoms(); ia++)
    //{
    //    auto type = uc->atom(ia)->type();
    //    int nbf = type->mt_basis_size();
    //    int ofs = uc->atom(ia)->offset_lo();
    //    matrix<double_complex> qinv(nbf, nbf);
    //    type->uspp().q_mtrx >> qinv;
    //    linalg<CPU>::geinv(nbf, qinv);
    //    for (int i = 0; i < nbf; i++)
    //    {
    //        for (int j = 0; j < nbf; j++) S.add(ofs + j, ofs + i, qinv(j, i));
    //    }
    //}
    //linalg<CPU>::geinv(uc->mt_basis_size(), S);

    auto& itso = parameters_.iterative_solver_input_section_;

    /* maximum order of Chebyshev polynomial*/
    int order = itso.num_steps_ + 2;

    std::vector< dmatrix<double_complex> > phi(order);
    for (int i = 0; i < order; i++)
    {
        phi[i] = dmatrix<double_complex>(kp__->num_gkvec(), num_bands, kp__->blacs_grid());
        //phi[i].allocate_ata_buffer((int)kp__->spl_fv_states().local_size(0));
    }

    dmatrix<double_complex> hphi(kp__->num_gkvec(), num_bands, kp__->blacs_grid());
    hphi.allocate_ata_buffer((int)kp__->spl_fv_states().local_size(0));

    /* trial basis functions */
    psi.panel() >> phi[0].panel();

    //int num_atoms_in_block = std::min(uc->num_atoms(), 256);
    int num_bands_local = (int)kp__->spl_fv_states().local_size(0);
    int kappa_size = std::max(uc->max_mt_basis_size() * uc->beta_chunk(0).num_atoms_, 4 * num_bands_local);
    /* temporary array for <G+k|beta> */
    matrix<double_complex> kappa(kp__->num_gkvec_row(), kappa_size);
    if (kp__->comm().rank() == 0)
    {
        printf("size of kappa array: %f GB\n", 16 * double(kappa.size()) / 1073741824);
    }
    
    /* offset in the packed array of on-site matrices */
    mdarray<int, 1> packed_mtrx_offset(uc->num_atoms());
    int packed_mtrx_size = 0;
    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {   
        int nbf = uc->atom(ia)->mt_basis_size();
        packed_mtrx_offset(ia) = packed_mtrx_size;
        packed_mtrx_size += nbf * nbf;
    }
    
    /* pack D, Q and P matrices */
    mdarray<double_complex, 1> d_mtrx_packed(packed_mtrx_size);
    mdarray<double_complex, 1> q_mtrx_packed(packed_mtrx_size);
    mdarray<double_complex, 1> p_mtrx_packed(packed_mtrx_size);

    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {
        int nbf = uc->atom(ia)->mt_basis_size();
        int iat = uc->atom(ia)->type()->id();
        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                d_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = uc->atom(ia)->d_mtrx(xi1, xi2);
                q_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = uc->atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
                p_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = kp__->p_mtrx(xi1, xi2, iat);
            }
        }
    }
    
    /* copy G+k vectors */
    matrix<double> gkvec_row(3, kp__->num_gkvec_row());
    for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
    {
        for (int x = 0; x < 3; x++) gkvec_row(x, igk_row) = kp__->gklo_basis_descriptor_row(igk_row).gkvec[x];
    }

    //auto& beta_gk_t = kp__->beta_gk_t();

    //if (parameters_.processing_unit() == GPU)
    //{
    //    #ifdef _GPU_
    //    psi.allocate_on_device();
    //    for (int i = 0; i < order; i++) phi[i].allocate_on_device();
    //    kappa.allocate_on_device();
    //    d_mtrx_packed.allocate_on_device();
    //    d_mtrx_packed.copy_to_device();
    //    gkvec_row.allocate_on_device();
    //    gkvec_row.copy_to_device();
    //    beta_pw_t.allocate_on_device();
    //    beta_pw_t.copy_to_device();
    //    #else
    //    TERMINATE_NO_GPU
    //    #endif
    //}

    /* apply Hamiltonian to the basis functions */
    apply_h_parallel(kp__, veff_it_coarse__, pw_ekin, phi[0], hphi, kappa, gkvec_row, packed_mtrx_offset, d_mtrx_packed);

    /* compute Rayleight quotients */
    std::vector<double> e0(num_bands, 0.0);
    #pragma omp parallel for schedule(static)
    for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
    {
        int i = kp__->spl_fv_states(iloc);
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
        {
            e0[i] += real(conj(phi[0](igk_row, iloc)) * hphi(igk_row, iloc));
        }
    }
    kp__->comm().allreduce(e0);
    
    /* estimate low and upper bounds of the Chebyshev filter */
    double lambda0 = -1e10;
    for (int i = 0; i < num_bands; i++) lambda0 = std::max(lambda0, e0[i]);
    double lambda1 = 0.5 * std::pow(parameters_.gk_cutoff(), 2);

    double r = (lambda1 - lambda0) / 2.0;
    double c = (lambda1 + lambda0) / 2.0;

    //apply_oinv_parallel(kp__, phi[1], S);
    hphi.panel() >> phi[1].panel();
    add_non_local_contribution_parallel(kp__, hphi, phi[1], kappa, gkvec_row, packed_mtrx_offset, p_mtrx_packed,
                                        double_complex(-1, 0));

    /* compute \psi_1 = (S^{-1}H\psi_0 - c\psi_0) / r */
    #pragma omp parallel for schedule(static)
    for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
    {
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
        {
            phi[1](igk_row, iloc) = (phi[1](igk_row, iloc) - phi[0](igk_row, iloc) * c) / r;
        }
    }
    
    /* compute higher polinomial orders */
    for (int k = 2; k < order; k++)
    {
        apply_h_parallel(kp__, veff_it_coarse__, pw_ekin, phi[k - 1], hphi, kappa, gkvec_row, packed_mtrx_offset, d_mtrx_packed);
        
        //apply_oinv_parallel(kp__, phi[k], S);
        hphi.panel() >> phi[k].panel();
        add_non_local_contribution_parallel(kp__, hphi, phi[k], kappa, gkvec_row, packed_mtrx_offset, p_mtrx_packed,
                                            double_complex(-1, 0));

        #pragma omp parallel for schedule(static)
        for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
        {
            for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
            {
                phi[k](igk_row, iloc) = (phi[k](igk_row, iloc) - c * phi[k - 1](igk_row, iloc)) * 2.0 / r - phi[k - 2](igk_row, iloc);
            }
        }
    }

    /* apply Hamiltonian and overlap to the "filtered" basis functions */
    apply_h_o_parallel(kp__, veff_it_coarse__, pw_ekin, 0, num_bands, phi[order - 1], hphi, phi[0],
                       kappa, gkvec_row, packed_mtrx_offset, d_mtrx_packed, q_mtrx_packed);

    dmatrix<double_complex> hmlt(num_bands, num_bands, kp__->blacs_grid());
    dmatrix<double_complex> ovlp(num_bands, num_bands, kp__->blacs_grid());

    dmatrix<double_complex> evec(num_bands, num_bands, kp__->blacs_grid());
    std::vector<double> eval(num_bands);

    Timer t1("sirius::Band::diag_fv_pseudo_potential_parallel_chebyshev|set_h_o", kp__->comm());
    linalg<CPU>::gemm(2, 0, num_bands, num_bands, kp__->num_gkvec(), complex_one, phi[order - 1], hphi, complex_zero, hmlt);
    linalg<CPU>::gemm(2, 0, num_bands, num_bands, kp__->num_gkvec(), complex_one, phi[order - 1], phi[0], complex_zero, ovlp);
    double tval = t1.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("2x pzgemm with M, N, K: %6i %6i %6i: %12.4f sec, %12.4f GFlops/rank\n",
               num_bands, num_bands, kp__->num_gkvec(),
               tval, 2 * 8e-9 * num_bands * num_bands * kp__->num_gkvec() / tval / kp__->num_ranks());
    }
    
    Timer t2("sirius::Band::diag_fv_pseudo_potential_parallel_chebyshev|gen_evp");
    gen_evp_solver()->solve(num_bands, hmlt.num_rows_local(), hmlt.num_cols_local(), num_bands, 
                            hmlt.ptr(), hmlt.ld(), ovlp.ptr(), ovlp.ld(), 
                            &eval[0], evec.ptr(), evec.ld());
    t2.stop();
        
    if (kp__->comm().rank() == 0)
    {
        printf("eigen-values:\n");
        for (int i = 0; i < std::min(num_bands, 10); i++) printf("%18.12f ", eval[i]);
        printf("\n");
    }

    //generate_fv_states_pp(kp__, num_bands, evec, phi[order - 1], psi, kappa);
    //
    Timer t3("sirius::Band::diag_fv_pseudo_potential_parallel_chebyshev|psi");
    /* recompute wave-functions: \Psi_{i} = \phi_{mu} * Z_{mu, i} */
    linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, num_bands, complex_one, phi[order - 1], evec, complex_zero, psi); 
    t3.stop();
    //
    //#ifdef _GPU_
    //if (parameters_.processing_unit() == GPU)
    //{
    //    beta_pw_t.deallocate_on_device();
    //    psi.deallocate_on_device();
    //}
    //#endif

    kp__->set_fv_eigen_values(&eval[0]);
    log_function_exit(__func__);
}

void Band::diag_fv_pseudo_potential_parallel_davidson(K_point* kp__,
                                                      double v0__,
                                                      std::vector<double>& veff_it_coarse__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::diag_fv_uspp_gpu_parallel", kp__->comm());
    
    if (parameters_.esm_type() == norm_conserving_pseudopotential)
    {
        diag_fv_ncpp_parallel(kp__, v0__, veff_it_coarse__);
        return;
    }
    
    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();
    
    /* we need to apply overlap operator in case of ultrasoft pseudopotential */
    bool with_overlap = (parameters_.esm_type() == ultrasoft_pseudopotential);

    /* get diagonal elements for preconditioning */
    std::vector<double_complex> h_diag;
    std::vector<double_complex> o_diag;
    if (with_overlap) 
    {
        get_h_o_diag<true>(kp__, v0__, pw_ekin, h_diag, o_diag);
    } 
    else 
    {
        get_h_o_diag<false>(kp__, v0__, pw_ekin, h_diag, o_diag);
    }

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();

    auto& itso = parameters_.iterative_solver_input_section_;

    int num_phi = std::min(itso.subspace_size_ * num_bands, kp__->num_gkvec());

    dmatrix<double_complex> phi(kp__->num_gkvec(), num_phi, kp__->blacs_grid());
    dmatrix<double_complex> hphi(kp__->num_gkvec(), num_phi, kp__->blacs_grid());
    hphi.allocate_ata_buffer((int)kp__->spl_fv_states().local_size(0));

    dmatrix<double_complex> hmlt(num_phi, num_phi, kp__->blacs_grid());
    dmatrix<double_complex> ovlp(num_phi, num_phi, kp__->blacs_grid());
    dmatrix<double_complex> hmlt_old(num_phi, num_phi, kp__->blacs_grid());
    dmatrix<double_complex> ovlp_old(num_phi, num_phi, kp__->blacs_grid());

    dmatrix<double_complex> ophi;
    if (with_overlap) ophi = dmatrix<double_complex>(kp__->num_gkvec(), num_phi, kp__->blacs_grid());
    
    dmatrix<double_complex> evec(num_phi, num_bands, kp__->blacs_grid());
    std::vector<double> eval(num_bands);
    std::vector<double> eval_old(num_bands);

    /* alias for wave-functions */
    dmatrix<double_complex>& psi = kp__->fv_states_panel();
    
    dmatrix<double_complex> res(kp__->num_gkvec(), num_bands, kp__->blacs_grid());
    dmatrix<double_complex> hpsi(kp__->num_gkvec(), num_bands, kp__->blacs_grid());
    dmatrix<double_complex> opsi; 
    if (with_overlap) opsi = dmatrix<double_complex>(kp__->num_gkvec(), num_bands, kp__->blacs_grid());

    /* trial basis functions */
    assert(phi.num_rows_local() == psi.num_rows_local());
    memcpy(&phi(0, 0), &psi(0, 0), kp__->num_gkvec_row() * psi.num_cols_local() * sizeof(double_complex));

    std::vector<double> res_norm(num_bands);
    std::vector<double> res_rms(num_bands);

    auto uc = parameters_.unit_cell();

    int num_atoms_in_block = std::min(uc->num_atoms(), 256);
    int num_bands_local = (int)kp__->spl_fv_states().local_size(0);
    int kappa_size = std::max(uc->max_mt_basis_size() * num_atoms_in_block, 4 * num_bands_local);
    /* large temporary array for <G+k|beta>, hphi_tmp, ophi_tmp, hpsi_tmp, opsi_tmp */
    matrix<double_complex> kappa(kp__->num_gkvec_row(), kappa_size);
    if (kp__->comm().rank() == 0)
    {
        printf("size of kappa array: %f GB\n", 16 * double(kappa.size()) / 1073741824);
    }
    
    /* offset in the packed array of on-site matrices */
    mdarray<int, 1> packed_mtrx_offset(uc->num_atoms());
    int packed_mtrx_size = 0;
    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {   
        int nbf = uc->atom(ia)->mt_basis_size();
        packed_mtrx_offset(ia) = packed_mtrx_size;
        packed_mtrx_size += nbf * nbf;
    }
    
    /* pack Q and D matrices */
    mdarray<double_complex, 1> d_mtrx_packed(packed_mtrx_size);
    mdarray<double_complex, 1> q_mtrx_packed;
    if (with_overlap) q_mtrx_packed = mdarray<double_complex, 1>(packed_mtrx_size);

    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {
        int nbf = uc->atom(ia)->mt_basis_size();
        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                d_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = uc->atom(ia)->d_mtrx(xi1, xi2);
                if (with_overlap) q_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = uc->atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
            }
        }
    }
    
    /* copy G+k vectors to device */
    matrix<double> gkvec_row(3, kp__->num_gkvec_row());
    for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
    {
        for (int x = 0; x < 3; x++) gkvec_row(x, igk_row) = kp__->gklo_basis_descriptor_row(igk_row).gkvec[x];
    }

    auto& beta_gk_t = kp__->beta_gk_t();

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef _GPU_
        phi.allocate_on_device();
        if (!with_overlap) psi.allocate_on_device();
        res.allocate_on_device();
        hphi.allocate_on_device();
        hpsi.allocate_on_device();
        kappa.allocate_on_device();
        d_mtrx_packed.allocate_on_device();
        d_mtrx_packed.copy_to_device();
        if (with_overlap)
        {
            ophi.allocate_on_device();
            opsi.allocate_on_device();
            q_mtrx_packed.allocate_on_device();
            q_mtrx_packed.copy_to_device();
        }
        gkvec_row.allocate_on_device();
        gkvec_row.copy_to_device();
        beta_gk_t.allocate_on_device();
        beta_gk_t.copy_to_device();
        /* initial phi on GPU */
        cuda_copy_to_device(phi.at<GPU>(), psi.at<CPU>(), kp__->num_gkvec_row() * psi.num_cols_local() * sizeof(double_complex));
        #else
        TERMINATE_NO_GPU
        #endif
    }

    /* current diagonalziation subspace size */
    int N = 0;

    /* number of newly added basis functions */
    int n = num_bands;

    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++)
    {
        /* set H and O for the variational subspace */
        set_fv_h_o_uspp_gpu_parallel_v3(N, n, kp__, veff_it_coarse__, pw_ekin, phi, hphi, ophi, hmlt, ovlp, 
                                        hmlt_old, ovlp_old, num_atoms_in_block, kappa, beta_gk_t, gkvec_row,
                                        packed_mtrx_offset, d_mtrx_packed, q_mtrx_packed);
        /* increase size of the variation space */
        N += n;
    
        if (verbosity_level >= 6 && kp__->comm().rank() == 0)
        {
            printf("checking hermitian matrix\n");
        }

        if (verbosity_level >= 6 && kp__->comm().rank() == 0)
        {
            printf("iteration : %i, subspace size : %i\n", k, N);
        }

        {
        Timer t2("sirius::Band::diag_fv_uspp_cpu_parallel|solve_gevp");
        eval_old = eval;
        
        gen_evp_solver()->solve(N, hmlt.num_rows_local(), hmlt.num_cols_local(), num_bands, 
                                hmlt.ptr(), hmlt.ld(), ovlp.ptr(), ovlp.ld(), 
                                &eval[0], evec.ptr(), evec.ld());
        
        if (kp__->comm().rank() == 0)
        {
            printf("subspace size : %i, eigen-values:\n", N);
            for (int i = 0; i < std::min(num_bands, 10); i++) printf("%18.12f ", eval[i]);
            printf("\n");
        }
        }

        /* don't recompute residuals if we are going to exit on the last iteration */
        std::vector<int> res_list;
        if (k != itso.num_steps_ - 1)
        {
            if (with_overlap)
            {
                uspp_residuals_gpu_parallel(N, num_bands, kp__, eval, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag, 
                                            res_norm, kappa);
            }
            else
            {
                uspp_residuals_gpu_parallel(N, num_bands, kp__, eval, evec, hphi, phi, hpsi, psi, res, h_diag, o_diag, 
                                            res_norm, kappa);
            }

            for (int i = 0; i < num_bands; i++)
            {
                /* take the residual if it's norm is above the threshold */
                if (kp__->band_occupancy(i) > 1e-12 &&
                    (res_norm[i] > itso.tolerance_ || (res_norm[i] > itso.extra_tolerance_ && n != 0)))
                {
                    res_list.push_back(i);
                }
            }

            /* number of additional basis functions */
            n = (int)res_list.size();
        }

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (N + n > num_phi || n == 0 || k == (itso.num_steps_ - 1))
        {   
            Timer t3("sirius::Band::diag_fv_uspp_cpu_parallel|update_phi");

            #ifdef _GPU_
            if (parameters_.processing_unit() == GPU) phi.copy_cols_to_host(0, N);
            #endif

            /* recompute wave-functions: \Psi_{i} = \phi_{mu} * Z_{mu, i} */
            if (with_overlap) linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, complex_one, phi, evec, complex_zero, psi); 
            
            /* exit loop if the eigen-vectors are converged or this is the last iteration */
            if (n == 0 || k == (itso.num_steps_ - 1))
            {
                if (verbosity_level >= 6 && kp__->comm().rank() == 0)
                {
                    double demax = 0;
                    for (int i = 0; i < num_bands; i++)
                    {
                         if (kp__->band_occupancy(i) > 1e-12) demax = std::max(demax, std::abs(eval_old[i] - eval[i]));
                    }
                    if (k == 0) demax = 0.0;
                    printf("converged in %i iterations with maximum eigen-value error %18.12e\n", k, demax);
                }
                break;
            }

            STOP();
            // something has to be copied to GPU/CPU here
            for (int i = 0; i < psi.num_cols_local(); i++) 
            {
                /* update \phi */
                memcpy(&phi(0, i), &psi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));
                /* update H\phi */
                memcpy(&hphi(0, i), &hpsi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));
                /* update O\phi */
                memcpy(&ophi(0, i), &opsi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));
            }

            /* update H and O matrices. */
            hmlt_old.zero();
            ovlp_old.zero();
            for (int i = 0; i < num_bands; i++)
            {
                hmlt_old.set(i, i, eval[i]);
                ovlp_old.set(i, i, complex_one);
            }
            
            /* set new size of the variational space */
            N = num_bands;
        }
        
        if (parameters_.processing_unit() == CPU)
        {
            /* expand variational space with extra basis functions */
            for (int i = 0; i < n; i++)
            {
                dmatrix<double_complex>::copy_col<CPU>(res, res_list[i], phi, N + i);
            }
        }
        if (parameters_.processing_unit() == GPU)
        {
            #ifdef _GPU_
            #ifdef _GPU_DIRECT_
            /* expand variational space with extra basis functions */
            for (int i = 0; i < n; i++)
            {
                dmatrix<double_complex>::copy_col<GPU>(res, res_list[i], phi, N + i);
            }
            /* copy new phi to CPU */
            phi.copy_cols_to_host(N, N + n);
            #else
            res.panel().copy_to_host();
            for (int i = 0; i < n; i++)
            {
                dmatrix<double_complex>::copy_col<CPU>(res, res_list[i], phi, N + i);
            }
            phi.copy_cols_to_device(N, N + n);
            #endif
            #endif
        }
    }
    
    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        beta_gk_t.deallocate_on_device();
        if (!with_overlap) psi.deallocate_on_device();
    }
    #endif

    kp__->set_fv_eigen_values(&eval[0]);
    log_function_exit(__func__);
}
#endif

};
