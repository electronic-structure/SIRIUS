#include "band.h"

namespace sirius {

#ifdef __SCALAPACK
void Band::diag_fv_pseudo_potential_davidson_parallel(K_point* kp__,
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

    /* maximum local size of bands */
    int max_num_bands_local = (int)kp__->spl_fv_states().local_size(0);

    auto& itso = kp__->iterative_solver_input_section_;

    /* short notation for target wave-functions */
    dmatrix<double_complex>& psi = kp__->fv_states_panel();

    bool converge_by_energy = (itso.converge_by_energy_ == 1);

    /* number of auxiliary basis functions */
    int num_phi = std::min(itso.subspace_size_ * num_bands, kp__->num_gkvec());

    dmatrix<double_complex> phi(kp__->num_gkvec(), num_phi, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
    dmatrix<double_complex> hphi(kp__->num_gkvec(), num_phi, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());

    hphi.allocate_ata_buffer(max_num_bands_local);

    dmatrix<double_complex> ophi;
    if (with_overlap) ophi = dmatrix<double_complex>(kp__->num_gkvec(), num_phi, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());

    dmatrix<double_complex> hmlt(num_phi, num_phi, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
    dmatrix<double_complex> ovlp(num_phi, num_phi, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
    dmatrix<double_complex> hmlt_old(num_phi, num_phi, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
    dmatrix<double_complex> ovlp_old(num_phi, num_phi, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());

    dmatrix<double_complex> evec(num_phi, num_bands, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
    dmatrix<double_complex> evec_tmp;
    if (converge_by_energy) evec_tmp = dmatrix<double_complex>(num_phi, num_bands, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());

    std::vector<double> eval(num_bands);
    for (int i = 0; i < num_bands; i++) eval[i] = kp__->band_energy(i);
    std::vector<double> eval_old(num_bands);
    std::vector<double> eval_tmp(num_bands);
    
    dmatrix<double_complex> res(kp__->num_gkvec(), num_bands, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
    dmatrix<double_complex> hpsi(kp__->num_gkvec(), num_bands, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
    dmatrix<double_complex> opsi; 
    if (with_overlap) opsi = dmatrix<double_complex>(kp__->num_gkvec(), num_bands, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());

    /* trial basis functions */
    assert(phi.num_rows_local() == psi.num_rows_local());
    memcpy(&phi(0, 0), &psi(0, 0), kp__->num_gkvec_row() * psi.num_cols_local() * sizeof(double_complex));

    std::vector<double> res_norm(num_bands);

    /* find maximum number of beta-projectors across chunks */
    int nbmax = 0;
    for (int ib = 0; ib < unit_cell_.num_beta_chunks(); ib++) nbmax = std::max(nbmax, unit_cell_.beta_chunk(ib).num_beta_);

    int kappa_size = std::max(nbmax, 4 * max_num_bands_local);
    /* large temporary array for <G+k|beta>, hphi_tmp, ophi_tmp, hpsi_tmp, opsi_tmp */
    matrix<double_complex> kappa(kp__->num_gkvec_row(), kappa_size);
    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("size of kappa array: %f GB\n", 16 * double(kappa.size()) / 1073741824);
    }
    
    /* offset in the packed array of on-site matrices */
    mdarray<int, 1> packed_mtrx_offset(unit_cell_.num_atoms());
    int packed_mtrx_size = 0;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    {   
        int nbf = unit_cell_.atom(ia)->mt_basis_size();
        packed_mtrx_offset(ia) = packed_mtrx_size;
        packed_mtrx_size += nbf * nbf;
    }
    
    /* pack Q and D matrices */
    mdarray<double_complex, 1> d_mtrx_packed(packed_mtrx_size);
    mdarray<double_complex, 1> q_mtrx_packed;
    if (with_overlap) q_mtrx_packed = mdarray<double_complex, 1>(packed_mtrx_size);

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    {
        int nbf = unit_cell_.atom(ia)->mt_basis_size();
        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                d_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = unit_cell_.atom(ia)->d_mtrx(xi1, xi2);
                if (with_overlap) q_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = unit_cell_.atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
            }
        }
    }
    
    if (parameters_.processing_unit() == GPU)
    {
        #ifdef __GPU
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

    /* current subspace size */
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
        /* solve generalized eigen-value problem */
        {
        Timer t1("sirius::Band::diag_fv_pseudo_potential|solve_gevp");

        gen_evp_solver()->solve(N, hmlt.num_rows_local(), hmlt.num_cols_local(), num_bands, 
                                hmlt.at<CPU>(), hmlt.ld(), ovlp.at<CPU>(), ovlp.ld(), 
                                &eval[0], evec.at<CPU>(), evec.ld());
        }

        bool occ_band_converged = true;
        for (int i = 0; i < num_bands; i++)
        {
            if (kp__->band_occupancy(i) > 1e-2 && std::abs(eval_old[i] - eval[i]) > ctx_.iterative_solver_tolerance()) 
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
                double tol = ctx_.iterative_solver_tolerance();
                n = 0;
                for (int i = 0; i < num_bands; i++)
                {
                    if (kp__->band_occupancy(i) > 1e-10 && std::abs(eval[i] - eval_old[i]) > tol)
                    {
                        dmatrix<double_complex>::copy_col<CPU>(evec, i, evec_tmp, n);
                        eval_tmp[n] = eval[i];
                        res_list.push_back(n);
                        n++;
                    }
                }

                if (n != 0)
                {
                    residuals_parallel(N, n, kp__, eval_tmp, evec_tmp, hphi, ophi, hpsi, opsi, res, h_diag, o_diag, 
                                       res_norm, kappa);

                    ///* filter residuals by norm */
                    //for (int i = 0; i < n; i++)
                    //{
                    //    /* take the residual if it's norm is above the threshold */
                    //    if (res_norm[i] > 1e-6) res_list.push_back(i);
                    //}

                    ///* number of additional basis functions */
                    //n = (int)res_list.size();
                }
            }
            else
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
                    if (res_norm[i] > ctx_.iterative_solver_tolerance() && kp__->band_occupancy(i) > 1e-10)
                    {
                        res_list.push_back(i);
                    }
                }

                /* number of additional basis functions */
                n = (int)res_list.size();
            }
        }
        kp__->comm().barrier();

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (N + n > num_phi || n == 0 || k == (itso.num_steps_ - 1) || occ_band_converged)
        {
            Timer t2("sirius::Band::diag_fv_pseudo_potential|update_phi");

            #ifdef __GPU
            if (parameters_.processing_unit() == GPU) phi.copy_cols_to_host(0, N);
            #endif

            /* recompute wave-functions: \Psi_{i} = \phi_{mu} * Z_{mu, i} */
            if (with_overlap) linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, complex_one, phi, evec, complex_zero, psi); 

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

            // something has to be copied to GPU/CPU here
            if (parameters_.processing_unit() == GPU) STOP();

            if (converge_by_energy)
            {
                linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, complex_one, hphi, evec, complex_zero, hpsi); 
                linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, complex_one, ophi, evec, complex_zero, opsi); 
            }

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
            #ifdef __GPU
            #ifdef __GPU_DIRECT
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
    
    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        if (!with_overlap) psi.deallocate_on_device();
    }
    #endif

    kp__->set_fv_eigen_values(&eval[0]);
    log_function_exit(__func__);
}
#endif

};
