#include "band.h"

namespace sirius {

void Band::diag_fv_pseudo_potential_davidson_serial(K_point* kp__,
                                                    double v0__,
                                                    std::vector<double>& veff_it_coarse__)
{
    Timer t("sirius::Band::diag_fv_pseudo_potential_davidson_serial");

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

    /* short notation for number of G+k vectors */
    int ngk = kp__->num_gkvec();

    auto& itso = kp__->iterative_solver_input_section_;

    /* short notation for target wave-functions */
    matrix<double_complex>& psi = kp__->fv_states_slab();

    bool converge_by_energy = (itso.converge_by_energy_ == 1);

    /* number of auxiliary basis functions */
    int num_phi = std::min(itso.subspace_size_ * num_bands, ngk);

    matrix<double_complex> phi(ngk, num_phi);
    matrix<double_complex> hphi(ngk, num_phi);
    matrix<double_complex> ophi(ngk, num_phi);
    matrix<double_complex> hpsi(ngk, num_bands);
    matrix<double_complex> opsi(ngk, num_bands);

    matrix<double_complex> hmlt(num_phi, num_phi);
    matrix<double_complex> ovlp(num_phi, num_phi);
    matrix<double_complex> hmlt_old(num_phi, num_phi);
    matrix<double_complex> ovlp_old(num_phi, num_phi);

    matrix<double_complex> evec(num_phi, num_bands);
    matrix<double_complex> evec_tmp;
    if (converge_by_energy) evec_tmp = matrix<double_complex>(num_phi, num_bands);

    std::vector<double> eval(num_bands);
    for (int i = 0; i < num_bands; i++) eval[i] = kp__->band_energy(i);
    std::vector<double> eval_old(num_bands);
    std::vector<double> eval_tmp(num_bands);
    
    /* residuals */
    mdarray<double_complex, 2> res(ngk, num_bands);

    /* norm of residuals */
    std::vector<double> res_norm(num_bands);

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

    bool economize_gpu_memory = true; // TODO: move to user-controlled input
    
    mdarray<double_complex, 1> kappa;

    if (parameters_.processing_unit() == GPU && economize_gpu_memory)
    {
        size_t size = ngk * (std::max(unit_cell_.mt_basis_size(), num_phi) + num_bands);
        kappa = mdarray<double_complex, 1>(nullptr, size);
    }
    if (parameters_.processing_unit() == CPU && itso.real_space_prj_) 
    {
        auto rsp = ctx_.real_space_prj();
        size_t size = 2 * rsp->fft()->size() * rsp->fft()->num_fft_threads();
        size += rsp->max_num_points_ * rsp->fft()->num_fft_threads();

        kappa = mdarray<double_complex, 1>(size);
    }
    
    #ifdef __GPU
    if (verbosity_level >= 6 && kp__->comm().rank() == 0 && parameters_.processing_unit() == GPU)
    {
        printf("size of kappa array: %f GB\n", sizeof(double_complex) * double(kappa.size() >> 30));
    }
    #endif

    /* trial basis functions */
    assert(phi.size(0) == psi.size(0));
    memcpy(&phi(0, 0), &psi(0, 0), ngk * num_bands * sizeof(double_complex));
    
    if (parameters_.processing_unit() == GPU)
    {
        #ifdef __GPU
        if (!economize_gpu_memory)
        {
            phi.allocate_on_device();
            psi.allocate_on_device();
            res.allocate_on_device();
            hphi.allocate_on_device();
            hpsi.allocate_on_device();
            kp__->beta_gk().allocate_on_device();
            kp__->beta_gk().copy_to_device();
            /* initial phi on GPU */
            cuda_copy_to_device(phi.at<GPU>(), psi.at<CPU>(), ngk * num_bands * sizeof(double_complex));
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
        if (converge_by_energy) evec_tmp.allocate_on_device();
        #else
        TERMINATE_NO_GPU
        #endif
    }

    /* current subspace size */
    int N = 0;

    /* number of newly added basis functions */
    int n = num_bands;
    
    #ifdef __PRINT_OBJECT_HASH
    std::cout << "hash(beta_pw)       : " << kp__->beta_gk_panel().panel().hash() << std::endl;
    std::cout << "hash(d_mtrx_packed) : " << d_mtrx_packed.hash() << std::endl;
    std::cout << "hash(q_mtrx_packed) : " << q_mtrx_packed.hash() << std::endl;
    std::cout << "hash(v_eff_coarse)  : " << Utils::hash(&veff_it_coarse__[0], parameters_.fft_coarse()->size() * sizeof(double)) << std::endl;
    #endif

    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++)
    {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        apply_h_o_serial(kp__, veff_it_coarse__, pw_ekin, N, n, phi, hphi, ophi, kappa, packed_mtrx_offset,
                         d_mtrx_packed, q_mtrx_packed);
        
        /* setup eigen-value problem.
         * N is the number of previous basis functions
         * n is the number of new basis functions
         */
        set_fv_h_o_serial(kp__, N, n, phi, hphi, ophi, hmlt, ovlp, hmlt_old, ovlp_old, kappa);
 
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
                std::stringstream s;
                s << "wrong diagonal of O: " << ovlp(i, i);
                TERMINATE(s);
            }
            hmlt(i, i) = real(hmlt(i, i));
            ovlp(i, i) = real(ovlp(i, i));
        }
        
        eval_old = eval;
        /* solve generalized eigen-value problem with the size N */
        {
        Timer t1("sirius::Band::diag_fv_pseudo_potential|solve_gevp");
        gen_evp_solver()->solve(N, num_bands, num_bands, num_bands, hmlt.at<CPU>(), hmlt.ld(), ovlp.at<CPU>(), ovlp.ld(), 
                                &eval[0], evec.at<CPU>(), evec.ld());
        }

        bool occ_band_converged = true;
        for (int i = 0; i < num_bands; i++)
        {
            if (kp__->band_occupancy(i) > 1e-2 && std::abs(eval_old[i] - eval[i]) > ctx_.iterative_solver_tolerance()) 
                occ_band_converged = false;
        }

        /* copy eigen-vectors to GPU */
        #ifdef __GPU
        if (parameters_.processing_unit() == GPU)
            cublas_set_matrix(N, num_bands, sizeof(double_complex), evec.at<CPU>(), evec.ld(), evec.at<GPU>(), evec.ld());
        #endif

        /* stage 3: compute residuals */
        /* don't compute residuals on last iteration */
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
                        memcpy(&evec_tmp(0, n), &evec(0, i), N * sizeof(double_complex));
                        eval_tmp[n] = eval[i];
                        n++;
                    }
                }

                #ifdef __GPU
                if (parameters_.processing_unit() == GPU)
                    cublas_set_matrix(N, n, sizeof(double_complex), evec_tmp.at<CPU>(), evec_tmp.ld(), evec_tmp.at<GPU>(), evec_tmp.ld());
                #endif

                residuals_serial(kp__, N, n, eval_tmp, evec_tmp, hphi, ophi, hpsi, opsi, res, h_diag, o_diag, res_norm, kappa);

                #ifdef __GPU
                if (parameters_.processing_unit() == GPU && economize_gpu_memory)
                {
                    /* copy residuals to CPU because the content of kappa array can be destroyed */
                    cublas_get_matrix(ngk, n, sizeof(double_complex),
                                      kappa.at<GPU>(ngk * 2 * n), ngk,
                                      res.at<CPU>(), res.ld());
                }
                #endif
            }
            else
            {
                residuals_serial(kp__, N, num_bands, eval, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag, res_norm, kappa);

                #ifdef __GPU
                matrix<double_complex> res_tmp;
                if (parameters_.processing_unit() == GPU)
                {
                    if (economize_gpu_memory)
                    {
                        res_tmp = matrix<double_complex>(nullptr, kappa.at<GPU>(ngk * 2 * num_bands), ngk, num_bands);
                    }
                    else
                    {
                        res_tmp = matrix<double_complex>(nullptr, res.at<GPU>(), ngk, num_bands);
                    }
                }
                #endif
                
                Timer t1("sirius::Band::diag_fv_pseudo_potential|sort_res");

                n = 0;
                for (int i = 0; i < num_bands; i++)
                {
                    /* take the residual if it's norm is above the threshold */
                    if (res_norm[i] > ctx_.iterative_solver_tolerance() && kp__->band_occupancy(i) > 1e-10)
                    {
                        /* shift unconverged residuals to the beginning of array */
                        if (n != i)
                        {
                            switch (parameters_.processing_unit())
                            {
                                case CPU:
                                {
                                    memcpy(&res(0, n), &res(0, i), ngk * sizeof(double_complex));
                                    break;
                                }
                                case GPU:
                                {
                                    #ifdef __GPU
                                    cuda_copy_device_to_device(res_tmp.at<GPU>(0, n), res_tmp.at<GPU>(0, i), ngk * sizeof(double_complex));
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
                #ifdef __GPU
                if (parameters_.processing_unit() == GPU && economize_gpu_memory)
                {
                    /* copy residuals to CPU because the content of kappa array will be destroyed */
                    cublas_get_matrix(ngk, n, sizeof(double_complex), res_tmp.at<GPU>(), res_tmp.ld(),
                                      res.at<CPU>(), res.ld());
                }
                #endif
            }
        }

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (N + n > num_phi || n == 0 || k == (itso.num_steps_ - 1) || occ_band_converged)
        {   
            Timer t1("sirius::Band::diag_fv_pseudo_potential|update_phi");
            /* recompute wave-functions */
            /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
            switch (parameters_.processing_unit())
            {
                case CPU:
                {
                    linalg<CPU>::gemm(0, 0, ngk, num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), 
                                      &psi(0, 0), psi.ld());
                    break;
                }
                case GPU:
                {
                    #ifdef __GPU
                    if (!economize_gpu_memory)
                    {
                        linalg<GPU>::gemm(0, 0, ngk, num_bands, N, phi.at<GPU>(), phi.ld(), evec.at<GPU>(), evec.ld(), 
                                          psi.at<GPU>(), psi.ld());
                        psi.copy_to_host();
                    }
                    else
                    {
                        /* copy phi to device */
                        cublas_set_matrix(ngk, N, sizeof(double_complex), phi.at<CPU>(), phi.ld(),
                                          kappa.at<GPU>(), ngk);
                        linalg<GPU>::gemm(0, 0, ngk, num_bands, N, kappa.at<GPU>(), ngk, 
                                          evec.at<GPU>(), evec.ld(), kappa.at<GPU>(ngk * N), ngk);
                        cublas_get_matrix(ngk, num_bands, sizeof(double_complex),
                                          kappa.at<GPU>(ngk * N), ngk,
                                          psi.at<CPU>(), psi.ld());
                    }
                    #else
                    TERMINATE_NO_GPU
                    #endif
                    break;
                }
            }

            /* exit the loop if the eigen-vectors are converged or this is a last iteration */
            if (n == 0 || k == (itso.num_steps_ - 1) || occ_band_converged)
            {
                //if (verbosity_level >= 6 && kp__->comm().rank() == 0)
                //{
                //    double demax = 0;
                //    for (int i = 0; i < num_bands; i++)
                //    {
                //         if (kp__->band_occupancy(i) > 1e-12) demax = std::max(demax, std::abs(eval_old[i] - eval[i]));
                //    }
                //    DUMP("exiting after %i iterations with maximum eigen-value error %18.12f", k + 1, demax);
                //}
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

                /* need to recompute hpsi and opsi */
                if (converge_by_energy)
                {
                    switch (parameters_.processing_unit())
                    {
                        case CPU:
                        {
                            linalg<CPU>::gemm(0, 0, ngk, num_bands, N, &hphi(0, 0), hphi.ld(), &evec(0, 0), evec.ld(), 
                                              &hpsi(0, 0), hpsi.ld());
                            
                            linalg<CPU>::gemm(0, 0, ngk, num_bands, N, &ophi(0, 0), ophi.ld(), &evec(0, 0), evec.ld(), 
                                              &opsi(0, 0), opsi.ld());
                            break;
                        }
                        case GPU:
                        {
                            #ifdef __GPU
                            if (!economize_gpu_memory)
                            {
                                TERMINATE("implement this");
                            }
                            else
                            {
                                /* copy hphi to device */
                                cublas_set_matrix(ngk, N, sizeof(double_complex), hphi.at<CPU>(), hphi.ld(),
                                                  kappa.at<GPU>(), ngk);
                                linalg<GPU>::gemm(0, 0, ngk, num_bands, N, kappa.at<GPU>(), ngk, 
                                                  evec.at<GPU>(), evec.ld(), kappa.at<GPU>(ngk * N), ngk);
                                cublas_get_matrix(ngk, num_bands, sizeof(double_complex),
                                                  kappa.at<GPU>(ngk * N), ngk,
                                                  hpsi.at<CPU>(), hpsi.ld());
                                
                                /* copy ophi to device */
                                cublas_set_matrix(ngk, N, sizeof(double_complex), ophi.at<CPU>(), ophi.ld(),
                                                  kappa.at<GPU>(), ngk);
                                linalg<GPU>::gemm(0, 0, ngk, num_bands, N, kappa.at<GPU>(), ngk, 
                                                  evec.at<GPU>(), evec.ld(), kappa.at<GPU>(ngk * N), ngk);
                                cublas_get_matrix(ngk, num_bands, sizeof(double_complex),
                                                  kappa.at<GPU>(ngk * N), ngk,
                                                  opsi.at<CPU>(), opsi.ld());
                            }
                            #else
                            TERMINATE_NO_GPU
                            #endif
                            break;
                        }
                    }
                }
 
                /* update hphi and ophi */
                if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && economize_gpu_memory))
                {
                    memcpy(hphi.at<CPU>(), hpsi.at<CPU>(), num_bands * ngk * sizeof(double_complex));
                    memcpy(ophi.at<CPU>(), opsi.at<CPU>(), num_bands * ngk * sizeof(double_complex));
                }
                
                #ifdef __GPU
                if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
                {
                    cuda_copy_device_to_device(hphi.at<GPU>(), hpsi.at<GPU>(), num_bands * ngk * sizeof(double_complex));
                    cuda_copy_device_to_device(ophi.at<GPU>(), opsi.at<GPU>(), num_bands * ngk * sizeof(double_complex));
                    cuda_copy_device_to_device( phi.at<GPU>(),  psi.at<GPU>(), num_bands * ngk * sizeof(double_complex));
                }
                #endif
                
                /* update basis functions */
                memcpy(phi.at<CPU>(), psi.at<CPU>(), num_bands * ngk * sizeof(double_complex));
                N = num_bands;
            }
        }
        /* expand variational subspace with new basis vectors obtatined from residuals */
        if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && economize_gpu_memory))
        {
            memcpy(phi.at<CPU>(0, N), res.at<CPU>(), n * ngk * sizeof(double_complex));
        }
        if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
        {
            #ifdef __GPU
            cuda_copy_device_to_device(phi.at<GPU>(0, N), res.at<GPU>(), n * ngk * sizeof(double_complex));
            cuda_copy_to_host(phi.at<CPU>(0, N), phi.at<GPU>(0, N), n * ngk * sizeof(double_complex));
            #else
            TERMINATE_NO_GPU
            #endif
        }
    }

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef __GPU
        if (!economize_gpu_memory)
        {
            kp__->beta_gk().deallocate_on_device();
            psi.deallocate_on_device();
        }
        #endif
    }

    kp__->set_fv_eigen_values(&eval[0]);
}

};
