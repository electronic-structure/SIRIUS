#include "band.h"
#include "debug.hpp"

namespace sirius {

#ifdef _GPU_
extern "C" void compute_inner_product_gpu(int num_gkvec_row,
                                          int n,
                                          cuDoubleComplex const* f1,
                                          cuDoubleComplex const* f2,
                                          double* prod);

extern "C" void compute_chebyshev_polynomial_gpu(int num_gkvec,
                                                 int n,
                                                 double c,
                                                 double r,
                                                 cuDoubleComplex* phi0,
                                                 cuDoubleComplex* phi1,
                                                 cuDoubleComplex* phi2);
#endif

#ifdef _SCALAPACK_
void Band::diag_fv_pseudo_potential_chebyshev_parallel(K_point* kp__,
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
    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
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

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef _GPU_
        for (int i = 0; i < order; i++) phi[i].allocate_on_device();
        hphi.allocate_on_device();
        kappa.allocate_on_device();
        d_mtrx_packed.allocate_on_device();
        d_mtrx_packed.copy_to_device();
        q_mtrx_packed.allocate_on_device();
        q_mtrx_packed.copy_to_device();
        p_mtrx_packed.allocate_on_device();
        p_mtrx_packed.copy_to_device();
        /* initial phi on GPU */
        phi[0].panel().copy_to_device();
        #else
        TERMINATE_NO_GPU
        #endif
    }

    /* apply Hamiltonian to the basis functions */
    apply_h_parallel(kp__, veff_it_coarse__, pw_ekin, 0, num_bands, phi[0], hphi, kappa, packed_mtrx_offset,
                     d_mtrx_packed);

    /* compute Rayleight quotients */
    std::vector<double> e0(num_bands, 0.0);
    if (parameters_.processing_unit() == CPU)
    {
        #pragma omp parallel for schedule(static)
        for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
        {
            int i = kp__->spl_fv_states(iloc);
            for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
            {
                e0[i] += real(conj(phi[0](igk_row, iloc)) * hphi(igk_row, iloc));
            }
        }
    }
    if (parameters_.processing_unit() == GPU)
    {
        #ifdef _GPU_
        mdarray<double, 1> e0_loc(kp__->spl_fv_states().local_size());
        e0_loc.allocate_on_device();
        e0_loc.zero_on_device();

        compute_inner_product_gpu(kp__->num_gkvec_row(),
                                  (int)kp__->spl_fv_states().local_size(),
                                  phi[0].at<GPU>(),
                                  hphi.at<GPU>(),
                                  e0_loc.at<GPU>());
        e0_loc.copy_to_host();
        for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
        {
            int i = kp__->spl_fv_states(iloc);
            e0[i] = e0_loc(iloc);
        }
        #endif
    }
    
    kp__->comm().allreduce(e0);
    
    /* estimate low and upper bounds of the Chebyshev filter */
    double lambda0 = -1e10;
    for (int i = 0; i < num_bands; i++) lambda0 = std::max(lambda0, e0[i]);
    lambda0 -= 0.1;
    double lambda1 = 0.5 * std::pow(parameters_.gk_cutoff(), 2);

    double r = (lambda1 - lambda0) / 2.0;
    double c = (lambda1 + lambda0) / 2.0;

    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            hphi.panel() >> phi[1].panel();
            break;
        }
        case GPU:
        {
            #ifdef _GPU_
            cuda_copy_device_to_device(phi[1].at<GPU>(), hphi.at<GPU>(), hphi.panel().size() * sizeof(double_complex));
            #endif
            break;
        }
    }

    //== add_non_local_contribution_parallel(kp__, hphi, phi[1], S, double_complex(-1, 0));
    add_non_local_contribution_parallel(kp__, 0, num_bands, hphi, phi[1], kappa, packed_mtrx_offset,
                                        p_mtrx_packed, double_complex(-1, 0));
    
    /* compute \psi_1 = (S^{-1}H\psi_0 - c\psi_0) / r */
    if (parameters_.processing_unit() == CPU)
    {
        #pragma omp parallel for schedule(static)
        for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
        {
            for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
            {
                phi[1](igk_row, iloc) = (phi[1](igk_row, iloc) - phi[0](igk_row, iloc) * c) / r;
            }
        }
    }
    if (parameters_.processing_unit() == GPU)
    {
        #ifdef _GPU_
        compute_chebyshev_polynomial_gpu(kp__->num_gkvec_row(), (int)kp__->spl_fv_states().local_size(), c, r,
                                         phi[0].at<GPU>(), phi[1].at<GPU>(), NULL);
        phi[1].panel().copy_to_host();
        #endif
    }

    /* compute higher polynomial orders */
    for (int k = 2; k < order; k++)
    {
        apply_h_parallel(kp__, veff_it_coarse__, pw_ekin, 0, num_bands, phi[k - 1], hphi, kappa, packed_mtrx_offset,
                         d_mtrx_packed);
        
        switch (parameters_.processing_unit())
        {
            case CPU:
            {
                hphi.panel() >> phi[k].panel();
                break;
            }
            case GPU:
            {
                #ifdef _GPU_
                cuda_copy_device_to_device(phi[k].at<GPU>(), hphi.at<GPU>(), hphi.panel().size() * sizeof(double_complex));
                #endif
                break;
            }
        }
        //add_non_local_contribution_parallel(kp__, hphi, phi[k], S, double_complex(-1, 0));
        add_non_local_contribution_parallel(kp__, 0, num_bands, hphi, phi[k], kappa, packed_mtrx_offset,
                                            p_mtrx_packed, double_complex(-1, 0));
        
        if (parameters_.processing_unit() == CPU)
        {
            #pragma omp parallel for schedule(static)
            for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
            {
                for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
                {
                    phi[k](igk_row, iloc) = (phi[k](igk_row, iloc) - c * phi[k - 1](igk_row, iloc)) * 2.0 / r -
                                            phi[k - 2](igk_row, iloc);
                }
            }
        }
        if (parameters_.processing_unit() == GPU)
        {
            #ifdef _GPU_
            compute_chebyshev_polynomial_gpu(kp__->num_gkvec_row(), (int)kp__->spl_fv_states().local_size(), c, r,
                                             phi[k - 2].at<GPU>(), phi[k - 1].at<GPU>(), phi[k].at<GPU>());
            phi[k].panel().copy_to_host();
            #endif
        }
    }

    /* apply Hamiltonian and overlap to the "filtered" basis functions */
    apply_h_o_parallel(kp__, veff_it_coarse__, pw_ekin, 0, num_bands, phi[order - 1], hphi, phi[0],
                       kappa, packed_mtrx_offset, d_mtrx_packed, q_mtrx_packed);

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef _GPU_
        hphi.panel().copy_to_host();
        phi[0].panel().copy_to_host();
        #endif
    }

    dmatrix<double_complex> hmlt(num_bands, num_bands, kp__->blacs_grid());
    dmatrix<double_complex> ovlp(num_bands, num_bands, kp__->blacs_grid());

    dmatrix<double_complex> evec(num_bands, num_bands, kp__->blacs_grid());
    std::vector<double> eval(num_bands);

    Timer t1("sirius::Band::diag_fv_pseudo_potential|set_h_o", kp__->comm());
    linalg<CPU>::gemm(2, 0, num_bands, num_bands, kp__->num_gkvec(), complex_one, phi[order - 1], hphi, complex_zero, hmlt);
    linalg<CPU>::gemm(2, 0, num_bands, num_bands, kp__->num_gkvec(), complex_one, phi[order - 1], phi[0], complex_zero, ovlp);
    double tval = t1.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("2x pzgemm with M, N, K: %6i %6i %6i: %12.4f sec, %12.4f GFlops/rank\n",
               num_bands, num_bands, kp__->num_gkvec(),
               tval, 2 * 8e-9 * num_bands * num_bands * kp__->num_gkvec() / tval / kp__->num_ranks());
    }
    
    Timer t2("sirius::Band::diag_fv_pseudo_potential|gen_evp");
    gen_evp_solver()->solve(num_bands, hmlt.num_rows_local(), hmlt.num_cols_local(), num_bands, 
                            hmlt.at<CPU>(), hmlt.ld(), ovlp.at<CPU>(), ovlp.ld(), 
                            &eval[0], evec.at<CPU>(), evec.ld());
    t2.stop();
        
    if (kp__->comm().rank() == 0)
    {
        printf("eigen-values:\n");
        for (int i = 0; i < std::min(num_bands, 10); i++) printf("%18.12f ", eval[i]);
        printf("\n");
    }

    //generate_fv_states_pp(kp__, num_bands, evec, phi[order - 1], psi, kappa);
    //
    Timer t3("sirius::Band::diag_fv_pseudo_potential|psi");
    /* recompute wave-functions: \Psi_{i} = \phi_{mu} * Z_{mu, i} */
    linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, num_bands, complex_one, phi[order - 1], evec, complex_zero, psi); 
    t3.stop();

    kp__->set_fv_eigen_values(&eval[0]);
    log_function_exit(__func__);
}

void Band::diag_fv_pseudo_potential_parallel_davidson(K_point* kp__,
                                                      double v0__,
                                                      std::vector<double>& veff_it_coarse__)
{
    Timer t("sirius::Band::diag_fv_pseudo_potential_parallel_davidson", kp__->comm());

    log_function_enter(__func__);
    
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

    auto& itso = kp__->iterative_solver_input_section_;

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
    for (int i = 0; i < num_bands; i++) eval[i] = kp__->band_energy(i);

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

    int num_bands_local = (int)kp__->spl_fv_states().local_size(0);
    int nbmax = 0;
    for (int ib = 0; ib < uc->num_beta_chunks(); ib++) nbmax = std::max(nbmax, uc->beta_chunk(ib).num_beta_);

    int kappa_size = std::max(nbmax, 4 * num_bands_local);
    /* large temporary array for <G+k|beta>, hphi_tmp, ophi_tmp, hpsi_tmp, opsi_tmp */
    matrix<double_complex> kappa(kp__->num_gkvec_row(), kappa_size);
    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
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
        set_fv_h_o_parallel(N, n, kp__, veff_it_coarse__, pw_ekin, phi, hphi, ophi, hmlt, ovlp, 
                            hmlt_old, ovlp_old, kappa, packed_mtrx_offset, d_mtrx_packed, q_mtrx_packed);

        /* increase size of the variation space */
        N += n;
    
        eval_old = eval;
        {
        Timer t1("sirius::Band::diag_fv_pseudo_potential|solve_gevp");

        gen_evp_solver()->solve(N, hmlt.num_rows_local(), hmlt.num_cols_local(), num_bands, 
                                hmlt.at<CPU>(), hmlt.ld(), ovlp.at<CPU>(), ovlp.ld(), 
                                &eval[0], evec.at<CPU>(), evec.ld());
        
        //== if (verbosity_level >= 6 && kp__->comm().rank() == 0)
        //== {
        //==     printf("subspace size : %i, eigen-values:\n", N);
        //==     for (int i = 0; i < std::min(num_bands, 10); i++) printf("%18.12f ", eval[i]);
        //==     printf("\n");
        //== }
        }

        /* don't recompute residuals if we are going to exit on the last iteration */
        std::vector<int> res_list;
        if (k != itso.num_steps_ - 1)
        {
            if (with_overlap)
            {
                residuals_parallel(N, num_bands, kp__, eval, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag, 
                                   res_norm, kappa);
            }
            else
            {
                residuals_parallel(N, num_bands, kp__, eval, evec, hphi, phi, hpsi, psi, res, h_diag, o_diag, 
                                   res_norm, kappa);
            }

            for (int i = 0; i < num_bands; i++)
            {
                /* take the residual if it's norm is above the threshold */
                //if ((kp__->band_occupancy(i) > 1e-12 && res_norm[i] > itso.tolerance_) ||
                //    (n != 0 &&  res_norm[i] > std::max(itso.tolerance_ / 2, itso.extra_tolerance_)))
                if (res_norm[i] > itso.tolerance_)
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
            Timer t2("sirius::Band::diag_fv_pseudo_potential|update_phi");

            #ifdef _GPU_
            if (parameters_.processing_unit() == GPU) phi.copy_cols_to_host(0, N);
            #endif

            /* recompute wave-functions: \Psi_{i} = \phi_{mu} * Z_{mu, i} */
            if (with_overlap) linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, complex_one, phi, evec, complex_zero, psi); 

            /* reduce the tolerance if residuals have converged before the last iteration */
            if (n == 0 && (k < itso.num_steps_ - 1))
            {
                itso.tolerance_ /= 2;
                itso.tolerance_ = std::max(itso.tolerance_, itso.extra_tolerance_);
            }
            
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
                    DUMP("exiting after %i iterations with maximum eigen-value error %18.12e\n", k + 1, demax);
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
        if (!with_overlap) psi.deallocate_on_device();
    }
    #endif

    kp__->set_fv_eigen_values(&eval[0]);
    log_function_exit(__func__);
}
#endif

void Band::diag_fv_pseudo_potential_serial_exact(K_point* kp__,
                                                 std::vector<double>& veff_it_coarse__)
{
    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();

    /* short notation for target wave-functions */
    mdarray<double_complex, 2>& psi = kp__->fv_states();

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();     

    int ngk = kp__->num_gkvec();

    mdarray<double_complex, 2> phi(ngk, ngk);
    mdarray<double_complex, 2> hphi(ngk, ngk);
    mdarray<double_complex, 2> ophi(ngk, ngk);
    matrix<double_complex> kappa(ngk, ngk);
    
    std::vector<double> eval(ngk);

    phi.zero();
    for (int i = 0; i < ngk; i++) phi(i, i) = complex_one;

    auto uc = parameters_.unit_cell();

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
    mdarray<double_complex, 1> q_mtrx_packed(packed_mtrx_size);

    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {
        int nbf = uc->atom(ia)->mt_basis_size();
        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                d_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = uc->atom(ia)->d_mtrx(xi1, xi2);
                q_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = uc->atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
            }
        }
    }
    
    apply_h_o_serial(kp__, veff_it_coarse__, pw_ekin, 0, ngk, phi, hphi, ophi, kappa, packed_mtrx_offset,
                         d_mtrx_packed, q_mtrx_packed);
        
    gen_evp_solver()->solve(ngk, num_bands, num_bands, num_bands, hphi.at<CPU>(), hphi.ld(), ophi.at<CPU>(), ophi.ld(), 
                            &eval[0], psi.at<CPU>(), psi.ld());

    kp__->set_fv_eigen_values(&eval[0]);
    kp__->fv_states_panel().scatter(psi);
}

void Band::diag_fv_pseudo_potential_serial_davidson(K_point* kp__,
                                                    double v0__,
                                                    std::vector<double>& veff_it_coarse__)
{
    Timer t("sirius::Band::diag_fv_pseudo_potential_serial_davidson");

    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();

    /* short notation for target wave-functions */
    mdarray<double_complex, 2>& psi = kp__->fv_states();

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();     

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

    auto& itso = kp__->iterative_solver_input_section_;
    
    bool converge_by_energy = (itso.converge_by_energy_ == 1);
    
    int num_phi = std::min(itso.subspace_size_ * num_bands, kp__->num_gkvec());

    mdarray<double_complex, 2> phi(kp__->num_gkvec(), num_phi);
    mdarray<double_complex, 2> hphi(kp__->num_gkvec(), num_phi);
    mdarray<double_complex, 2> ophi(kp__->num_gkvec(), num_phi);
    mdarray<double_complex, 2> hpsi(kp__->num_gkvec(), num_bands);
    mdarray<double_complex, 2> opsi(kp__->num_gkvec(), num_bands);

    mdarray<double_complex, 2> hmlt(num_phi, num_phi);
    mdarray<double_complex, 2> ovlp(num_phi, num_phi);
    mdarray<double_complex, 2> hmlt_old(num_phi, num_phi);
    mdarray<double_complex, 2> ovlp_old(num_phi, num_phi);
    mdarray<double_complex, 2> evec(num_phi, num_bands);
    mdarray<double_complex, 2> evec_tmp;

    std::vector<double> eval(num_bands);
    std::vector<double> eval_old(num_bands);
    for (int i = 0; i < num_bands; i++) eval[i] = kp__->band_energy(i);

    std::vector<double> eval_tmp(num_bands);
    
    /* residuals */
    mdarray<double_complex, 2> res(kp__->num_gkvec(), num_bands);

    /* norm of residuals */
    std::vector<double> res_norm(num_bands);

    auto uc = parameters_.unit_cell();

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

    bool economize_gpu_memory = true;
    matrix<double_complex> kappa;
    if (economize_gpu_memory) kappa = matrix<double_complex>(nullptr, kp__->num_gkvec(), std::max(uc->mt_basis_size(), num_phi) + num_bands);
    
    #ifdef _GPU_
    if (verbosity_level >= 6 && kp__->comm().rank() == 0 && parameters_.processing_unit() == GPU)
    {
        printf("size of kappa array: %f GB\n", 16 * double(kappa.size()) / 1073741824);
    }
    #endif

    /* trial basis functions */
    assert(phi.size(0) == psi.size(0));
    memcpy(&phi(0, 0), &psi(0, 0), kp__->num_gkvec() * num_bands * sizeof(double_complex));

    if (converge_by_energy) evec_tmp = mdarray<double_complex, 2>(num_phi, num_bands);
    
    if (parameters_.processing_unit() == GPU)
    {
        #ifdef _GPU_
        if (!economize_gpu_memory)
        {
            phi.allocate_on_device();
            psi.allocate_on_device();
            res.allocate_on_device();
            hphi.allocate_on_device();
            hpsi.allocate_on_device();
            kp__->beta_pw_panel().panel().allocate_on_device();
            kp__->beta_pw_panel().panel().copy_to_device();
            /* initial phi on GPU */
            cuda_copy_to_device(phi.at<GPU>(), psi.at<CPU>(), kp__->num_gkvec_row() * num_bands * sizeof(double_complex));
            if (with_overlap)
            {
                ophi.allocate_on_device();
                opsi.allocate_on_device();
            }
        }
        else
        {
            kappa.allocate_on_device();
        }
        d_mtrx_packed.allocate_on_device();
        d_mtrx_packed.copy_to_device();
        if (with_overlap)
        {
            q_mtrx_packed.allocate_on_device();
            q_mtrx_packed.copy_to_device();
        }
        hmlt.allocate_on_device();
        ovlp.allocate_on_device();
        evec.allocate_on_device();
        evec_tmp.allocate_on_device();
        #else
        TERMINATE_NO_GPU
        #endif
    }

    int N = 0; // current eigen-value problem size
    int n = num_bands; // number of new basis functions
    
    #ifdef _WRITE_OBJECTS_HASH_
    std::cout << "hash(beta_pw)       : " << kp__->beta_pw_panel().panel().hash() << std::endl;
    std::cout << "hash(d_mtrx_packed) : " << d_mtrx_packed.hash() << std::endl;
    std::cout << "hash(q_mtrx_packed) : " << q_mtrx_packed.hash() << std::endl;
    std::cout << "hash(v_eff_coarse)  : " << Utils::hash(&veff_it_coarse__[0], parameters_.fft_coarse()->size() * sizeof(double)) << std::endl;
    #endif

    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++)
    {
        /* stage 1: setup eigen-value problem.
         * N is the number of previous basis functions
         * n is the number of new basis functions
         */
        set_fv_h_o_serial(kp__, veff_it_coarse__, pw_ekin, N, n, phi, hphi, ophi, hmlt, ovlp, hmlt_old, ovlp_old,
                          kappa, packed_mtrx_offset, d_mtrx_packed, q_mtrx_packed);
 
        /* increase size of the variation space */
        N += n;
        
        for (int i = 0; i < N; i++)
        {
            if (imag(hmlt(i, i)) > 1e-10)
            {
                TERMINATE("wrong diagonal of H");
            }
            if (imag(ovlp(i, i)) > 1e-10)
            {
                TERMINATE("wrong diagonal of O");
            }
            hmlt(i, i) = real(hmlt(i, i));
            ovlp(i, i) = real(ovlp(i, i));
        }
        
        eval_old = eval;
        /* stage 2: solve generalized eigen-value problem with the size N */
        {
        Timer t1("sirius::Band::diag_fv_pseudo_potential|solve_gevp");
        gen_evp_solver()->solve(N, num_bands, num_bands, num_bands, hmlt.at<CPU>(), hmlt.ld(), ovlp.at<CPU>(), ovlp.ld(), 
                                &eval[0], evec.at<CPU>(), evec.ld());
        }
        
        /* copy eigen-vectors to GPU */
        #ifdef _GPU_
        if (parameters_.processing_unit() == GPU)
            cublas_set_matrix(N, num_bands, sizeof(double_complex), evec.at<CPU>(), evec.ld(), evec.at<GPU>(), evec.ld());
        #endif

        /* stage 3: compute residuals */
        /* don't compute residuals on last iteration */
        if (k != itso.num_steps_ - 1)
        {
            if (converge_by_energy)
            {
                /* main trick here: first estimate energy difference, and only then compute unconverged residuals */
                n = 0;
                for (int i = 0; i < num_bands; i++)
                {
                    if ((kp__->band_occupancy(i) > 1e-12 && std::abs(eval[i] - eval_old[i]) > itso.tolerance_) ||
                        (n != 0 && std::abs(eval[i] - eval_old[i]) > std::max(itso.tolerance_ / 2, itso.extra_tolerance_)))

                    {
                        memcpy(&evec_tmp(0, n), &evec(0, i), N * sizeof(double_complex));
                        eval_tmp[n] = eval[i];
                        n++;
                    }
                }
                #ifdef _GPU_
                if (parameters_.processing_unit() == GPU)
                    cublas_set_matrix(N, n, sizeof(double_complex), evec_tmp.at<CPU>(), evec_tmp.ld(), evec_tmp.at<GPU>(), evec_tmp.ld());
                #endif

                residuals_serial(kp__, N, n, eval_tmp, evec_tmp, hphi, ophi, hpsi, opsi, res, h_diag, o_diag, res_norm, kappa);

                #ifdef _GPU_
                if (parameters_.processing_unit() == GPU && economize_gpu_memory)
                {
                    /* copy residuals to CPU because the content of kappa array can be destroyed */
                    cublas_get_matrix(kp__->num_gkvec(), n, sizeof(double_complex), kappa.at<GPU>(0, 2 * n), kappa.ld(),
                                      res.at<CPU>(), res.ld());
                }
                #endif
            }
            else
            {
                residuals_serial(kp__, N, num_bands, eval, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag, res_norm, kappa);

                #ifdef _GPU_
                matrix<double_complex> res_tmp;
                if (parameters_.processing_unit() == GPU)
                {
                    if (economize_gpu_memory)
                    {
                        res_tmp = matrix<double_complex>(nullptr, kappa.at<GPU>(0, 2 * num_bands), kp__->num_gkvec(), num_bands);
                    }
                    else
                    {
                        res_tmp = matrix<double_complex>(nullptr, res.at<GPU>(), kp__->num_gkvec(), num_bands);
                    }
                }
                #endif
                
                Timer t1("sirius::Band::diag_fv_pseudo_potential|sort_res");

                n = 0;
                for (int i = 0; i < num_bands; i++)
                {
                    /* take the residual if it's norm is above the threshold */
                    //if ((kp__->band_occupancy(i) > 1e-12 && res_norm[i] > itso.tolerance_) ||
                    //    (n != 0 &&  res_norm[i] > std::max(itso.tolerance_ / 2, itso.extra_tolerance_)))
                    if (res_norm[i] > itso.tolerance_)
                    {
                        /* shift unconverged residuals to the beginning of array */
                        if (n != i)
                        {
                            switch (parameters_.processing_unit())
                            {
                                case CPU:
                                {
                                    memcpy(&res(0, n), &res(0, i), kp__->num_gkvec() * sizeof(double_complex));
                                    break;
                                }
                                case GPU:
                                {
                                    #ifdef _GPU_
                                    cuda_copy_device_to_device(res_tmp.at<GPU>(0, n), res_tmp.at<GPU>(0, i), kp__->num_gkvec() * sizeof(double_complex));
                                    #else
                                    TERMINATE_NO_GPU
                                    #endif
                                    break;
                                }
                            }
                        }
                        n++;
                    }
                }
                #ifdef _GPU_
                if (parameters_.processing_unit() == GPU && economize_gpu_memory)
                {
                    /* copy residuals to CPU because the content of kappa array will be destroyed */
                    cublas_get_matrix(kp__->num_gkvec(), n, sizeof(double_complex), res_tmp.at<GPU>(), res_tmp.ld(),
                                      res.at<CPU>(), res.ld());
                }
                #endif

                //== std::vector<int> nr(3, 0);
                //== for (int i = 0; i < num_bands; i++)
                //== {
                //==     if (res_norm[i] < itso.extra_tolerance_) nr[0]++;
                //==     if (res_norm[i] >= itso.extra_tolerance_ && res_norm[i] < itso.tolerance_) nr[1]++;
                //==     if (res_norm[i] >= itso.tolerance_) nr[2]++;
                //== }
                //== if (verbosity_level >= 6 && kp__->comm().rank() == 0)
                //== {
                //==     DUMP("residual statistics: %4.2f %4.2f %4.2f", double(nr[0]) / num_bands, double(nr[1]) / num_bands, double(nr[2]) / num_bands);
                //== }
            }
        }

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (N + n > num_phi || n == 0 || k == (itso.num_steps_ - 1))
        {   
            Timer t1("sirius::Band::diag_fv_pseudo_potential|update_phi");
            /* recompute wave-functions */
            /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
            switch (parameters_.processing_unit())
            {
                case CPU:
                {
                    linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), 
                                      &psi(0, 0), psi.ld());
                    break;
                }
                case GPU:
                {
                    #ifdef _GPU_
                    if (!economize_gpu_memory)
                    {
                        linalg<GPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, phi.at<GPU>(), phi.ld(), evec.at<GPU>(), evec.ld(), 
                                          psi.at<GPU>(), psi.ld());
                        psi.copy_to_host();
                    }
                    else
                    {
                        /* copy phi to device */
                        cublas_set_matrix(kp__->num_gkvec(), N, sizeof(double_complex), phi.at<CPU>(), phi.ld(),
                                          kappa.at<GPU>(), kappa.ld());
                        linalg<GPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, kappa.at<GPU>(), kappa.ld(), 
                                          evec.at<GPU>(), evec.ld(), kappa.at<GPU>(0, N), kappa.ld());
                        cublas_get_matrix(kp__->num_gkvec(), num_bands, sizeof(double_complex), kappa.at<GPU>(0, N), kappa.ld(),
                                          psi.at<CPU>(), psi.ld());
                    }
                    #else
                    TERMINATE_NO_GPU
                    #endif
                    break;
                }
            }

            /* reduce the tolerance if residuals have converged before the last iteration */
            if (n == 0 && (k < itso.num_steps_ - 1))
            {
                itso.tolerance_ /= 2;
                itso.tolerance_ = std::max(itso.tolerance_, itso.extra_tolerance_);
            }

            /* exit the loop if the eigen-vectors are converged or it's a last iteration */
            if (n == 0 || k == (itso.num_steps_ - 1))
            {
                if (verbosity_level >= 6 && kp__->comm().rank() == 0)
                {
                    double demax = 0;
                    for (int i = 0; i < num_bands; i++)
                    {
                         if (kp__->band_occupancy(i) > 1e-12) demax = std::max(demax, std::abs(eval_old[i] - eval[i]));
                    }
                    DUMP("exiting after %i iterations with maximum eigen-value error %18.12e\n", k + 1, demax);
                }
                //== if (verbosity_level >= 6 && kp__->comm().rank() == 0)
                //== {
                //==     DUMP("N = %i, n = %i, k = %i, tol = %18.14f", N, n, k, itso.tolerance_);
                //== }
                break;
            }
            else /* otherwise set Psi as a new trial basis */
            {
                hmlt_old.zero();
                ovlp_old.zero();
                for (int i = 0; i < num_bands; i++)
                {
                    hmlt_old(i, i) = eval[i];
                    ovlp_old(i, i) = complex_one;
                }
 
                /* set new basis functions */
                if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && economize_gpu_memory))
                {
                    memcpy(hphi.at<CPU>(), hpsi.at<CPU>(), num_bands * kp__->num_gkvec() * sizeof(double_complex));
                    memcpy(ophi.at<CPU>(), opsi.at<CPU>(), num_bands * kp__->num_gkvec() * sizeof(double_complex));
                }
                
                #ifdef _GPU_
                if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
                {
                    cuda_copy_device_to_device(hphi.at<GPU>(), hpsi.at<GPU>(), num_bands * kp__->num_gkvec() * sizeof(double_complex));
                    cuda_copy_device_to_device(ophi.at<GPU>(), opsi.at<GPU>(), num_bands * kp__->num_gkvec() * sizeof(double_complex));
                    cuda_copy_device_to_device( phi.at<GPU>(),  psi.at<GPU>(), num_bands * kp__->num_gkvec() * sizeof(double_complex));
                }
                #endif

                memcpy(phi.at<CPU>(), psi.at<CPU>(), num_bands * kp__->num_gkvec() * sizeof(double_complex));
                N = num_bands;
            }
        }
        /* expand variational subspace with new basis vectors obtatined from residuals */
        if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && economize_gpu_memory))
        {
            memcpy(&phi(0, N), &res(0, 0), n * kp__->num_gkvec() * sizeof(double_complex));
        }
        if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
        {
            #ifdef _GPU_
            cuda_copy_device_to_device(phi.at<GPU>(0, N), res.at<GPU>(), n * kp__->num_gkvec() * sizeof(double_complex));
            cuda_copy_to_host(phi.at<CPU>(0, N), phi.at<GPU>(0, N), n * kp__->num_gkvec() * sizeof(double_complex));
            #else
            TERMINATE_NO_GPU
            #endif
        }
    }

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef _GPU_
        if (!economize_gpu_memory)
        {
            kp__->beta_pw_panel().panel().deallocate_on_device();
            psi.deallocate_on_device();
        }
        #endif
    }

    kp__->set_fv_eigen_values(&eval[0]);
    kp__->fv_states_panel().scatter(psi);
}

void Band::diag_fv_pseudo_potential_chebyshev_serial(K_point* kp__,
                                                     std::vector<double> const& veff_it_coarse__)
{
    log_function_enter(__func__);

    /* alias for wave-functions */
    auto& psi = kp__->fv_states();

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

    std::vector< matrix<double_complex> > phi(order);
    for (int i = 0; i < order; i++)
    {
        phi[i] = matrix<double_complex>(kp__->num_gkvec(), num_bands);
    }

    matrix<double_complex> hphi(kp__->num_gkvec(), num_bands);

    /* trial basis functions */
    psi >> phi[0];

    int kappa_size = std::max(uc->mt_basis_size(), 4 * num_bands);
    /* temporary array for <G+k|beta> */
    matrix<double_complex> kappa(kp__->num_gkvec(), kappa_size);
    if (verbosity_level >= 6)
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

    //== if (parameters_.processing_unit() == GPU)
    //== {
    //==     #ifdef _GPU_
    //==     for (int i = 0; i < order; i++) phi[i].allocate_on_device();
    //==     hphi.allocate_on_device();
    //==     kappa.allocate_on_device();
    //==     d_mtrx_packed.allocate_on_device();
    //==     d_mtrx_packed.copy_to_device();
    //==     q_mtrx_packed.allocate_on_device();
    //==     q_mtrx_packed.copy_to_device();
    //==     p_mtrx_packed.allocate_on_device();
    //==     p_mtrx_packed.copy_to_device();
    //==     /* initial phi on GPU */
    //==     phi[0].panel().copy_to_device();
    //==     #else
    //==     TERMINATE_NO_GPU
    //==     #endif
    //== }

    /* apply Hamiltonian to the basis functions */
    apply_h_serial(kp__, veff_it_coarse__, pw_ekin, 0, num_bands, phi[0], hphi, kappa, packed_mtrx_offset,
                     d_mtrx_packed);

    /* compute Rayleight quotients */
    std::vector<double> e0(num_bands, 0.0);
    if (parameters_.processing_unit() == CPU)
    {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_bands; i++)
        {
            for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            {
                e0[i] += real(conj(phi[0](igk, i)) * hphi(igk, i));
            }
        }
    }
    //== if (parameters_.processing_unit() == GPU)
    //== {
    //==     #ifdef _GPU_
    //==     mdarray<double, 1> e0_loc(kp__->spl_fv_states().local_size());
    //==     e0_loc.allocate_on_device();
    //==     e0_loc.zero_on_device();

    //==     compute_inner_product_gpu(kp__->num_gkvec_row(),
    //==                               (int)kp__->spl_fv_states().local_size(),
    //==                               phi[0].at<GPU>(),
    //==                               hphi.at<GPU>(),
    //==                               e0_loc.at<GPU>());
    //==     e0_loc.copy_to_host();
    //==     for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
    //==     {
    //==         int i = kp__->spl_fv_states(iloc);
    //==         e0[i] = e0_loc(iloc);
    //==     }
    //==     #endif
    //== }
    //== 

    /* estimate low and upper bounds of the Chebyshev filter */
    double lambda0 = -1e10;
    //double emin = 1e100;
    for (int i = 0; i < num_bands; i++)
    {
        lambda0 = std::max(lambda0, e0[i]);
        //emin = std::min(emin, e0[i]);
    }
    double lambda1 = 0.5 * std::pow(parameters_.gk_cutoff(), 2);

    double r = (lambda1 - lambda0) / 2.0;
    double c = (lambda1 + lambda0) / 2.0;

    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            hphi >> phi[1];
            break;
        }
        case GPU:
        {
            #ifdef _GPU_
            cuda_copy_device_to_device(phi[1].at<GPU>(), hphi.at<GPU>(), hphi.size() * sizeof(double_complex));
            #endif
            break;
        }
    }

    add_non_local_contribution_serial(kp__, 0, num_bands, hphi, phi[1], kappa, packed_mtrx_offset,
                                      p_mtrx_packed, double_complex(-1, 0));
    
    /* compute \psi_1 = (S^{-1}H\psi_0 - c\psi_0) / r */
    if (parameters_.processing_unit() == CPU)
    {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_bands; i++)
        {
            for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            {
                phi[1](igk, i) = (phi[1](igk, i) - phi[0](igk, i) * c) / r;
            }
        }
    }
    //if (parameters_.processing_unit() == GPU)
    //{
    //    #ifdef _GPU_
    //    compute_chebyshev_polynomial_gpu(kp__->num_gkvec_row(), (int)kp__->spl_fv_states().local_size(), c, r,
    //                                     phi[0].at<GPU>(), phi[1].at<GPU>(), NULL);
    //    phi[1].panel().copy_to_host();
    //    #endif
    //}

    /* compute higher polynomial orders */
    for (int k = 2; k < order; k++)
    {
        apply_h_serial(kp__, veff_it_coarse__, pw_ekin, 0, num_bands, phi[k - 1], hphi, kappa, packed_mtrx_offset,
                       d_mtrx_packed);
        
        switch (parameters_.processing_unit())
        {
            case CPU:
            {
                hphi >> phi[k];
                break;
            }
            case GPU:
            {
                #ifdef _GPU_
                cuda_copy_device_to_device(phi[k].at<GPU>(), hphi.at<GPU>(), hphi.size() * sizeof(double_complex));
                #endif
                break;
            }
        }
        //add_non_local_contribution_parallel(kp__, hphi, phi[k], S, double_complex(-1, 0));
        add_non_local_contribution_serial(kp__, 0, num_bands, hphi, phi[k], kappa, packed_mtrx_offset,
                                          p_mtrx_packed, double_complex(-1, 0));
        
        if (parameters_.processing_unit() == CPU)
        {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_bands; i++)
            {
                for (int igk = 0; igk < kp__->num_gkvec(); igk++)
                {
                    phi[k](igk, i) = (phi[k](igk, i) - c * phi[k - 1](igk, i)) * 2.0 / r - phi[k - 2](igk, i);
                }
            }
        }
        if (parameters_.processing_unit() == GPU)
        {
            #ifdef _GPU_
            compute_chebyshev_polynomial_gpu(kp__->num_gkvec(), num_bands, c, r,
                                             phi[k - 2].at<GPU>(), phi[k - 1].at<GPU>(), phi[k].at<GPU>());
            phi[k].copy_to_host();
            #endif
        }
    }

    /* apply Hamiltonian and overlap to the "filtered" basis functions */
    apply_h_o_serial(kp__, veff_it_coarse__, pw_ekin, 0, num_bands, phi[order - 1], hphi, phi[0],
                     kappa, packed_mtrx_offset, d_mtrx_packed, q_mtrx_packed);

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef _GPU_
        hphi.copy_to_host();
        phi[0].copy_to_host();
        #endif
    }

    matrix<double_complex> hmlt(num_bands, num_bands);
    matrix<double_complex> ovlp(num_bands, num_bands);

    matrix<double_complex> evec(num_bands, num_bands);
    std::vector<double> eval(num_bands);

    Timer t1("sirius::Band::diag_fv_pseudo_potential|set_h_o");
    linalg<CPU>::gemm(2, 0, num_bands, num_bands, kp__->num_gkvec(), complex_one, phi[order - 1], hphi, complex_zero, hmlt);
    linalg<CPU>::gemm(2, 0, num_bands, num_bands, kp__->num_gkvec(), complex_one, phi[order - 1], phi[0], complex_zero, ovlp);
    t1.stop();
    
    Timer t2("sirius::Band::diag_fv_pseudo_potential|gen_evp");
    gen_evp_solver()->solve(num_bands, num_bands, num_bands, num_bands, 
                            hmlt.at<CPU>(), hmlt.ld(), ovlp.at<CPU>(), ovlp.ld(), 
                            &eval[0], evec.at<CPU>(), evec.ld());
    t2.stop();

    if (true)
    {
        std::vector<double_complex> h_diag;
        std::vector<double_complex> o_diag;
        get_h_o_diag<true>(kp__, 0, pw_ekin, h_diag, o_diag);

        mdarray<double_complex, 2> hpsi(kp__->num_gkvec(), num_bands);
        mdarray<double_complex, 2> opsi(kp__->num_gkvec(), num_bands);
        mdarray<double_complex, 2> res(kp__->num_gkvec(), num_bands);
        std::vector<double> res_norm(num_bands);
        
        residuals_serial(kp__, num_bands, num_bands, eval, evec, hphi, phi[0], hpsi, opsi, res, h_diag, o_diag, res_norm, kappa);

        for (int i = 0; i < num_bands; i++) std::cout << "band : " << i << " residual : " << res_norm[i] << std::endl;
    }
        
    Timer t3("sirius::Band::diag_fv_pseudo_potential|psi");
    /* recompute wave-functions: \Psi_{i} = \phi_{mu} * Z_{mu, i} */
    linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, num_bands, complex_one, phi[order - 1], evec, complex_zero, psi); 
    t3.stop();

    kp__->set_fv_eigen_values(&eval[0]);
    kp__->fv_states_panel().scatter(psi);
    log_function_exit(__func__);
}

};
