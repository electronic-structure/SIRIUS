#include "band.h"

namespace sirius {

#ifdef _SCALAPACK_
void Band::diag_fv_pseudo_potential_davidson_fast_parallel(K_point* kp__,
                                                           double v0__,
                                                           std::vector<double>& veff_it_coarse__)
{
    Timer t("sirius::Band::diag_fv_pseudo_potential_davidson_parallel", kp__->comm());

    log_function_enter(__func__);
    
    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();
    
    /* we need to apply overlap operator in case of ultrasoft pseudopotential */
    bool with_overlap = (parameters_.esm_type() == ultrasoft_pseudopotential);

    /* get diagonal elements for preconditioning */
    std::vector<double> h_diag;
    std::vector<double> o_diag;
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

    /* short notation for target wave-functions */
    auto& psi_slab = kp__->fv_states_slab();

    auto& psi_slice = kp__->fv_states();

    bool converge_by_energy = (itso.converge_by_energy_ == 1);

    /* number of auxiliary basis functions */
    int num_phi = std::min(itso.subspace_size_ * num_bands, kp__->num_gkvec());

    splindex<block> spl_bands(num_bands, kp__->comm().size(), kp__->comm().rank());

    int num_gkvec_loc = kp__->num_gkvec_loc();
    
    dmatrix<double_complex> hmlt(num_phi, num_phi, kp__->blacs_grid());
    dmatrix<double_complex> ovlp(num_phi, num_phi, kp__->blacs_grid());
    dmatrix<double_complex> hmlt_old(num_phi, num_phi, kp__->blacs_grid());
    dmatrix<double_complex> ovlp_old(num_phi, num_phi, kp__->blacs_grid());

    dmatrix<double_complex> evec(num_phi, num_bands, kp__->blacs_grid());

    matrix<double_complex> evec_full(num_phi, num_bands);
    matrix<double_complex> evec_full_tmp;
    if (converge_by_energy) evec_full_tmp = matrix<double_complex>(num_phi, num_bands);

    std::vector<double> eval(num_bands);
    for (int i = 0; i < num_bands; i++) eval[i] = kp__->band_energy(i);
    std::vector<double> eval_old(num_bands);
    std::vector<double> eval_tmp(num_bands);
    
    std::vector<double> res_norm(num_bands);

    auto uc = parameters_.unit_cell();

    /* find maximum number of beta-projectors across chunks */
    int nbmax = 0;
    for (int ib = 0; ib < uc->num_beta_chunks(); ib++) nbmax = std::max(nbmax, uc->beta_chunk(ib).num_beta_);
    
    /* size of <beta|phi>, D*<beta|phi> and <Gk|beta> */
    size_t kappa_size = 2 * nbmax * num_bands + num_gkvec_loc * nbmax;

    /* size of <phi|hphi> */ 
    kappa_size = std::max(kappa_size, size_t(num_phi * num_bands));

    if (parameters_.processing_unit() == CPU && itso.real_space_prj_)
    {
        auto rsp = parameters_.real_space_prj_;
        size_t size = 2 * rsp->fft()->size() * rsp->fft()->num_fft_threads();
        size += rsp->max_num_points_ * rsp->fft()->num_fft_threads();

        kappa_size = std::max(kappa_size, size);
    }

    matrix<double_complex> hphi_slice, ophi_slice;
    if (itso.real_space_prj_)
    {
        hphi_slice = matrix<double_complex>(kp__->num_gkvec(), spl_bands.local_size());
        ophi_slice = matrix<double_complex>(kp__->num_gkvec(), spl_bands.local_size());
    }

    /* large temporary array */
    mdarray<double_complex, 1> kappa(kappa_size);

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

    matrix<double_complex> phi_slab(num_gkvec_loc, num_phi);
    matrix<double_complex> hphi_slab(num_gkvec_loc, num_phi);
    matrix<double_complex> ophi_slab(num_gkvec_loc, num_phi);

    matrix<double_complex> res_slab(num_gkvec_loc, num_bands);
    matrix<double_complex> hpsi_slab(num_gkvec_loc, num_bands);
    matrix<double_complex> opsi_slab(num_gkvec_loc, num_bands);

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("total size slab arrays: %f GB\n",
               16 * double(phi_slab.size() + hpsi_slab.size() + ophi_slab.size() + psi_slab.size() +
                           res_slab.size() + hpsi_slab.size() + opsi_slab.size()) / (1 << 30)); 
    }

    /* set initial basis functions */
    memcpy(&phi_slab(0, 0), &psi_slab(0, 0), num_gkvec_loc * num_bands * sizeof(double_complex));

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef _GPU_
        phi_slab.allocate_on_device();
        hphi_slab.allocate_on_device();
        d_mtrx_packed.allocate_on_device();
        d_mtrx_packed.copy_to_device();
        kappa.allocate_on_device();
        evec_full.allocate_on_device();
        evec_full_tmp.allocate_on_device();
        hpsi_slab.allocate_on_device();
        opsi_slab.allocate_on_device();
        if (with_overlap)
        {
            ophi_slab.allocate_on_device();
            q_mtrx_packed.allocate_on_device();
            q_mtrx_packed.copy_to_device();
        }
        psi_slab.allocate_on_device();

        /* copy initial phi to GPU */
        cuda_copy_to_device(phi_slab.at<GPU>(), phi_slab.at<CPU>(), num_gkvec_loc * num_bands * sizeof(double_complex));
        #else
        TERMINATE_NO_GPU
        #endif
    }

    /* current subspace size */
    int N = 0;

    /* number of newly added basis functions */
    int n = num_bands;

    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++)
    {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        if (with_overlap)
        {
            if (!itso.real_space_prj_)
            {
                apply_h_o_fast_parallel(kp__, veff_it_coarse__, pw_ekin, N, n, psi_slice, phi_slab, hphi_slab, ophi_slab,
                                        packed_mtrx_offset, d_mtrx_packed, q_mtrx_packed, kappa);
            }
            else
            {
                apply_h_o_fast_parallel_rs(kp__, veff_it_coarse__, pw_ekin, N, n, psi_slice, hphi_slice, ophi_slice, 
                                           phi_slab, hphi_slab, ophi_slab, packed_mtrx_offset, d_mtrx_packed, q_mtrx_packed, kappa);
            }
        }
        else
        {
           // apply_h_parallel(kp__, veff_it_coarse__, pw_ekin, 0, n, phi, hphi, beta_gk,
           //                  packed_mtrx_offset, d_mtrx_packed);
        }

        /* set H and O for the variational subspace */
        set_fv_h_o_fast_parallel(N, n, kp__, phi_slab, hphi_slab, ophi_slab, hmlt, ovlp, hmlt_old, ovlp_old, kappa);

        /* increase size of the variation space */
        N += n;

        eval_old = eval;
        /* solve generalized eigen-value problem */
        {
        Timer t1("sirius::Band::diag_fv_pseudo_potential|solve_gevp", kp__->comm());

        gen_evp_solver()->solve(N, hmlt.num_rows_local(), hmlt.num_cols_local(), num_bands, 
                                hmlt.at<CPU>(), hmlt.ld(), ovlp.at<CPU>(), ovlp.ld(), 
                                &eval[0], evec.at<CPU>(), evec.ld());
        }
        if (kp__->comm().rank() == 0) DUMP("step: %i, eval: %18.12f %18.12f", k, eval[0], eval[num_bands - 1]);
        
        {
        Timer t1("sirius::Band::diag_fv_pseudo_potential|collect_evec");
        evec_full.zero();
        for (int i = 0; i < evec.num_cols_local(); i++)
        {
            for (int j = 0; j < evec.num_rows_local(); j++)
            {
                evec_full(evec.irow(j), evec.icol(i)) = evec(j, i);
            }
        }
        kp__->comm().allreduce(evec_full.at<CPU>(), (int)evec_full.size());
        }

        if (parameters_.processing_unit() == GPU)
        {
            #ifdef _GPU_
            cublas_set_matrix(N, num_bands, sizeof(double_complex), evec_full.at<CPU>(), evec_full.ld(),
                              evec_full.at<GPU>(), evec_full.ld());
            #endif
        }

        /* check for converged occupied bands */
        bool occ_band_converged = true;
        for (int i = 0; i < num_bands; i++)
        {
            if (kp__->band_occupancy(i) > 1e-2 && 
                std::abs(eval_old[i] - eval[i]) > parameters_.iterative_solver_input_section_.tolerance_ / 2) 
            {
                occ_band_converged = false;
            }
        }

        /* don't recompute residuals if we are going to exit on the last iteration */
        std::vector<int> res_list;
        if (k != itso.num_steps_ - 1 && !occ_band_converged)
        {
            if (converge_by_energy)
            {
                /* main trick here: first estimate energy difference, and only then compute unconverged residuals */
                double tol = parameters_.iterative_solver_input_section_.tolerance_ / 2;
                n = 0;
                for (int i = 0; i < num_bands; i++)
                {
                    if (kp__->band_occupancy(i) > 1e-10 && std::abs(eval[i] - eval_old[i]) > tol)
                    {
                        memcpy(&evec_full_tmp(0, n), &evec_full(0, i), num_phi * sizeof(double_complex));
                        eval_tmp[n] = eval[i];
                        res_list.push_back(n);
                        n++;
                    }
                }

                if (n != 0)
                {
                    if (parameters_.processing_unit() == GPU)
                    {
                        #ifdef _GPU_
                        cublas_set_matrix(N, n, sizeof(double_complex), evec_full_tmp.at<CPU>(), evec_full_tmp.ld(),
                                          evec_full_tmp.at<GPU>(), evec_full_tmp.ld());
                        #endif
                    }
                    residuals_fast_parallel(N, n, kp__, eval_tmp, evec_full_tmp, hphi_slab, ophi_slab, hpsi_slab, opsi_slab,
                                            res_slab, h_diag, o_diag, res_norm, kappa);
                }
                parameters_.work_load_ += n;
            }
            else
            {
                /* here we first compute all residuals, and only then estimate their norm */
                if (with_overlap)
                {
                    residuals_fast_parallel(N, num_bands, kp__, eval, evec_full, hphi_slab, ophi_slab, hpsi_slab, opsi_slab,
                                            res_slab, h_diag, o_diag, res_norm, kappa);
                }
                else
                {
                    residuals_fast_parallel(N, num_bands, kp__, eval, evec_full, hphi_slab, ophi_slab, hpsi_slab, opsi_slab,
                                            res_slab, h_diag, o_diag, res_norm, kappa);
                }

                for (int i = 0; i < num_bands; i++)
                {
                    /* take the residual if it's norm is above the threshold */
                    if (kp__->band_occupancy(i) > 1e-10 && res_norm[i] > itso.tolerance_)
                    {
                        res_list.push_back(i);
                    }
                }

                /* number of additional basis functions */
                n = (int)res_list.size();
            }
        }

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (N + n > num_phi || n == 0 || k == (itso.num_steps_ - 1) || occ_band_converged)
        {
            Timer t2("sirius::Band::diag_fv_pseudo_potential|update_phi");

            /* recompute wave-functions: \Psi_{i} = \phi_{mu} * Z_{mu, i} */
            if (with_overlap)
            {
                switch (parameters_.processing_unit())
                {
                    case CPU:
                    {
                        linalg<CPU>::gemm(0, 0, num_gkvec_loc, num_bands, N, phi_slab, evec_full, psi_slab); 
                        break;
                    }
                    case GPU:
                    {
                        #ifdef _GPU_
                        linalg<GPU>::gemm(0, 0, num_gkvec_loc, num_bands, N, phi_slab.at<GPU>(), phi_slab.ld(),
                                          evec_full.at<GPU>(), evec_full.ld(), psi_slab.at<GPU>(), psi_slab.ld()); 
                        #endif
                        break;
                    }
                }
            }

            /* exit loop if the eigen-vectors are converged or this is the last iteration */
            if (n == 0 || k == (itso.num_steps_ - 1) || occ_band_converged)
            {
                if (verbosity_level >= 6 && kp__->comm().rank() == 0)
                {
                    double demax = 0;
                    for (int i = 0; i < num_bands; i++)
                    {
                         if (kp__->band_occupancy(i) > 1e-12) demax = std::max(demax, std::abs(eval_old[i] - eval[i]));
                    }
                    DUMP("exiting after %i iterations with maximum eigen-value error %18.12e", k + 1, demax);
                }
                break;
            }

            if (converge_by_energy)
            {
                /* hpsi and opsi were computed only for part of the wave-functions,
                 * but we need all of them to update hphi and ophi
                 */
                switch (parameters_.processing_unit())
                {
                    case CPU:
                    {
                        linalg<CPU>::gemm(0, 0, num_gkvec_loc, num_bands, N, hphi_slab, evec_full, hpsi_slab);
                        linalg<CPU>::gemm(0, 0, num_gkvec_loc, num_bands, N, ophi_slab, evec_full, opsi_slab);
                        break;
                    }
                    case GPU:
                    {
                        #ifdef _GPU_
                        linalg<GPU>::gemm(0, 0, num_gkvec_loc, num_bands, N, hphi_slab.at<GPU>(), hphi_slab.ld(),
                                          evec_full.at<GPU>(), evec_full.ld(), hpsi_slab.at<GPU>(), hpsi_slab.ld()); 
                        linalg<GPU>::gemm(0, 0, num_gkvec_loc, num_bands, N, ophi_slab.at<GPU>(), ophi_slab.ld(),
                                          evec_full.at<GPU>(), evec_full.ld(), opsi_slab.at<GPU>(), opsi_slab.ld()); 
                        #endif
                        break;
                    }
                }
            }

            switch (parameters_.processing_unit())
            {
                case CPU:
                {
                    /* update \phi */
                    memcpy(&phi_slab(0, 0),  &psi_slab(0, 0),  num_gkvec_loc * num_bands * sizeof(double_complex));
                    /* update H\phi */
                    memcpy(&hphi_slab(0, 0), &hpsi_slab(0, 0), num_gkvec_loc * num_bands * sizeof(double_complex));
                    /* update O\phi */
                    memcpy(&ophi_slab(0, 0), &opsi_slab(0, 0), num_gkvec_loc * num_bands * sizeof(double_complex));
                    break;
                }
                case GPU:
                {
                    #ifdef _GPU_
                    cuda_copy_device_to_device(phi_slab.at<GPU>(),  psi_slab.at<GPU>(),  num_gkvec_loc * num_bands * sizeof(double_complex));
                    cuda_copy_device_to_device(hphi_slab.at<GPU>(), hpsi_slab.at<GPU>(), num_gkvec_loc * num_bands * sizeof(double_complex));
                    cuda_copy_device_to_device(ophi_slab.at<GPU>(), opsi_slab.at<GPU>(), num_gkvec_loc * num_bands * sizeof(double_complex));
                    #endif
                    break;
                }
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
       
        /* expand variational space with extra basis functions */
        for (int i = 0; i < n; i++)
        {
            memcpy(&phi_slab(0, N + i), &res_slab(0, res_list[i]), num_gkvec_loc * sizeof(double_complex));
        }

        if (parameters_.processing_unit() == GPU)
        {
            #ifdef _GPU_
            cuda_copy_to_device(phi_slab.at<GPU>(0, N), phi_slab.at<CPU>(0, N), n * num_gkvec_loc * sizeof(double_complex));
            #endif
        }
    }

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        //if (!with_overlap) psi.deallocate_on_device();
        psi_slab.copy_to_host();
        psi_slab.deallocate_on_device();
    }
    #endif

    //kp__->collect_all_gkvec(spl_bands, &psi_slab(0, 0), &psi_slice(0, 0)); 

    kp__->set_fv_eigen_values(&eval[0]);
    log_function_exit(__func__);
}
#endif

};
