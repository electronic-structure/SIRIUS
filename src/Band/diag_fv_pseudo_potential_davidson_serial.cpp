#include "band.h"

namespace sirius {

void Band::diag_fv_pseudo_potential_davidson_serial(K_point* kp__,
                                                    double v0__,
                                                    std::vector<double>& veff_it_coarse__)
{
    PROFILE();

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

    auto pu = parameters_.processing_unit();

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();

    /* short notation for number of G+k vectors */
    int ngk = kp__->num_gkvec();

    auto& itso = kp__->iterative_solver_input_section_;

    /* short notation for target wave-functions */
    auto& psi = *kp__->fv_states();

    bool converge_by_energy = (itso.converge_by_energy_ == 1);
    
    assert(num_bands * 2 < ngk); // iterative subspace size can't be smaller than this

    /* number of auxiliary basis functions */
    int num_phi = std::min(itso.subspace_size_ * num_bands, ngk);

    /* allocate wave-functions */
    Wave_functions phi(num_phi, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
    Wave_functions hphi(num_phi, num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
    Wave_functions ophi(num_phi, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
    Wave_functions hpsi(num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
    Wave_functions opsi(num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);

    /* allocate Hamiltonian and overlap */
    matrix<double_complex> hmlt(num_phi, num_phi);
    matrix<double_complex> ovlp(num_phi, num_phi);
    matrix<double_complex> hmlt_old(num_phi, num_phi);
    matrix<double_complex> ovlp_old(num_phi, num_phi);

    matrix<double_complex> evec;
    if (converge_by_energy)
    {
        evec = matrix<double_complex>(num_phi, num_bands * 2);
    }
    else
    {
        evec = matrix<double_complex>(num_phi, num_bands);
    }

    int bs = parameters_.cyclic_block_size();

    dmatrix<double_complex> hmlt_dist;
    dmatrix<double_complex> ovlp_dist;
    dmatrix<double_complex> evec_dist;
    if (kp__->comm().size() == 1)
    {
        hmlt_dist = dmatrix<double_complex>(&hmlt(0, 0), num_phi, num_phi,   kp__->blacs_grid(), bs, bs);
        ovlp_dist = dmatrix<double_complex>(&ovlp(0, 0), num_phi, num_phi,   kp__->blacs_grid(), bs, bs);
        evec_dist = dmatrix<double_complex>(&evec(0, 0), num_phi, num_bands, kp__->blacs_grid(), bs, bs);
    }
    else
    {
        hmlt_dist = dmatrix<double_complex>(num_phi, num_phi,   kp__->blacs_grid(), bs, bs);
        ovlp_dist = dmatrix<double_complex>(num_phi, num_phi,   kp__->blacs_grid(), bs, bs);
        evec_dist = dmatrix<double_complex>(num_phi, num_bands, kp__->blacs_grid(), bs, bs);
    }

    std::vector<double> eval(num_bands);
    for (int i = 0; i < num_bands; i++) eval[i] = kp__->band_energy(i);
    std::vector<double> eval_old(num_bands);
    std::vector<double> eval_tmp(num_bands);
    
    /* residuals */
    Wave_functions res(num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);

    D_operator d_op(kp__->beta_projectors(), pu);
    Q_operator q_op(kp__->beta_projectors(), pu);
    Hloc_operator h_op(ctx_.fft_coarse_ctx(), kp__->gkvec(), pw_ekin, veff_it_coarse__);

    //bool economize_gpu_memory = true; // TODO: move to user-controlled input
    
    mdarray<double_complex, 1> kappa;

    //==if (parameters_.processing_unit() == GPU && economize_gpu_memory)
    //=={
    //==    size_t size = ngk * (std::max(unit_cell_.mt_basis_size(), num_phi) + num_bands);
    //==    kappa = mdarray<double_complex, 1>(nullptr, size);
    //==}
    //==if (parameters_.processing_unit() == CPU && itso.real_space_prj_) 
    //=={
    //==    auto rsp = ctx_.real_space_prj();
    //==    size_t size = 2 * rsp->fft()->size() * parameters_.num_fft_threads();
    //==    size += rsp->max_num_points_ * parameters_.num_fft_threads();

    //==    kappa = mdarray<double_complex, 1>(size);
    //==}
    //==
    //==#if defined(__GPU) && (__VERBOSITY > 1)
    //==if (kp__->comm().rank() == 0 && parameters_.processing_unit() == GPU)
    //=={
    //==    printf("size of kappa array: %f GB\n", sizeof(double_complex) * double(kappa.size() >> 30));
    //==}
    //==#endif

    /* trial basis functions */
    phi.copy_from(psi, 0, num_bands);

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        phi.allocate_on_device();
        phi.copy_to_device(0, num_bands);

        hphi.allocate_on_device();
        ophi.allocate_on_device();

    }
    #endif
    
    //if (parameters_.processing_unit() == GPU)
    //{
    //    #ifdef __GPU
    //    if (!economize_gpu_memory)
    //    {
    //        phi.allocate_on_device();
    //        psi.allocate_on_device();
    //        res.allocate_on_device();
    //        hphi.allocate_on_device();
    //        hpsi.allocate_on_device();
    //        kp__->beta_gk().allocate_on_device();
    //        kp__->beta_gk().copy_to_device();
    //        /* initial phi on GPU */
    //        cuda_copy_to_device(phi.at<GPU>(), psi.at<CPU>(), ngk * num_bands * sizeof(double_complex));
    //        if (with_overlap)
    //        {
    //            ophi.allocate_on_device();
    //            opsi.allocate_on_device();
    //        }
    //    }
    //    else
    //    {
    //        kappa.allocate_on_device();
    //    }
    //    d_mtrx_packed.allocate_on_device();
    //    d_mtrx_packed.copy_to_device();
    //    if (with_overlap)
    //    {
    //        q_mtrx_packed.allocate_on_device();
    //        q_mtrx_packed.copy_to_device();
    //    }
    //    hmlt.allocate_on_device();
    //    ovlp.allocate_on_device();
    //    evec.allocate_on_device();
    //    if (converge_by_energy) evec_tmp.allocate_on_device();
    //    #else
    //    TERMINATE_NO_GPU
    //    #endif
    //}

    /* current subspace size */
    int N = 0;

    /* number of newly added basis functions */
    int n = num_bands;
    
    #ifdef __PRINT_OBJECT_HASH
    //std::cout << "hash(beta_pw)       : " << kp__->beta_gk_panel().panel().hash() << std::endl;
    std::cout << "hash(d_mtrx_packed) : " << d_mtrx_packed.hash() << std::endl;
    std::cout << "hash(q_mtrx_packed) : " << q_mtrx_packed.hash() << std::endl;
    std::cout << "hash(v_eff_coarse)  : " << Utils::hash(&veff_it_coarse__[0], ctx_.fft_coarse()->size() * sizeof(double)) << std::endl;
    #endif

    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++)
    {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        apply_h_o(kp__, N, n, phi, hphi, ophi, kappa, h_op, d_op, q_op);
        
        /* setup eigen-value problem.
         * N is the number of previous basis functions
         * n is the number of new basis functions
         */
        set_fv_h_o_serial(kp__, N, n, phi, hphi, ophi, hmlt, ovlp, hmlt_old, ovlp_old, kappa);
 
        /* increase size of the variation space */
        N += n;

        eval_old = eval;

        /* solve generalized eigen-value problem with the size N */
        diag_h_o(kp__, N, num_bands, hmlt, ovlp, evec, hmlt_dist, ovlp_dist, evec_dist, eval);
    
        /* check if occupied bands have converged */
        bool occ_band_converged = true;
        for (int i = 0; i < num_bands; i++)
        {
            if (kp__->band_occupancy(i) > 1e-2 && std::abs(eval_old[i] - eval[i]) > ctx_.iterative_solver_tolerance()) 
                occ_band_converged = false;
        }

        for (int i = 0; i < num_bands; i++)
        {
            printf("eval[%i]=%f\n", i, eval[i]);
        }
        STOP();


        ///* copy eigen-vectors to GPU */
        //#ifdef __GPU
        //if (parameters_.processing_unit() == GPU)
        //    cublas_set_matrix(N, num_bands, sizeof(double_complex), evec.at<CPU>(), evec.ld(), evec.at<GPU>(), evec.ld());
        //#endif

        /* don't compute residuals on last iteration */
        if (k != itso.num_steps_ - 1 && !occ_band_converged)
        {
            /* get new preconditionined residuals, and also hpss and opsi as a by-product */
            n = residuals(kp__, N, num_bands, eval, eval_old, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag, kappa);
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
                    psi.transform_from(phi, N, evec, num_bands);
                    break;
                }
                case GPU:
                {
                    #ifdef __GPU
                    STOP();
                    //if (!economize_gpu_memory)
                    //{
                    //    linalg<GPU>::gemm(0, 0, ngk, num_bands, N, phi.at<GPU>(), phi.ld(), evec.at<GPU>(), evec.ld(), 
                    //                      psi.at<GPU>(), psi.ld());
                    //    psi.copy_to_host();
                    //}
                    //else
                    //{
                    //    /* copy phi to device */
                    //    cublas_set_matrix(ngk, N, sizeof(double_complex), phi.at<CPU>(), phi.ld(),
                    //                      kappa.at<GPU>(), ngk);
                    //    linalg<GPU>::gemm(0, 0, ngk, num_bands, N, kappa.at<GPU>(), ngk, 
                    //                      evec.at<GPU>(), evec.ld(), kappa.at<GPU>(ngk * N), ngk);
                    //    cublas_get_matrix(ngk, num_bands, sizeof(double_complex),
                    //                      kappa.at<GPU>(ngk * N), ngk,
                    //                      psi.at<CPU>(), psi.ld());
                    //}
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
                            hpsi.transform_from(hphi, N, evec, num_bands);
                            opsi.transform_from(ophi, N, evec, num_bands);
                            break;
                        }
                        case GPU:
                        {
                            #ifdef __GPU
                            STOP();
                            //if (!economize_gpu_memory)
                            //{
                            //    TERMINATE("implement this");
                            //}
                            //else
                            //{
                            //    /* copy hphi to device */
                            //    cublas_set_matrix(ngk, N, sizeof(double_complex), hphi.at<CPU>(), hphi.ld(),
                            //                      kappa.at<GPU>(), ngk);
                            //    linalg<GPU>::gemm(0, 0, ngk, num_bands, N, kappa.at<GPU>(), ngk, 
                            //                      evec.at<GPU>(), evec.ld(), kappa.at<GPU>(ngk * N), ngk);
                            //    cublas_get_matrix(ngk, num_bands, sizeof(double_complex),
                            //                      kappa.at<GPU>(ngk * N), ngk,
                            //                      hpsi.at<CPU>(), hpsi.ld());
                            //    
                            //    /* copy ophi to device */
                            //    cublas_set_matrix(ngk, N, sizeof(double_complex), ophi.at<CPU>(), ophi.ld(),
                            //                      kappa.at<GPU>(), ngk);
                            //    linalg<GPU>::gemm(0, 0, ngk, num_bands, N, kappa.at<GPU>(), ngk, 
                            //                      evec.at<GPU>(), evec.ld(), kappa.at<GPU>(ngk * N), ngk);
                            //    cublas_get_matrix(ngk, num_bands, sizeof(double_complex),
                            //                      kappa.at<GPU>(ngk * N), ngk,
                            //                      opsi.at<CPU>(), opsi.ld());
                            //}
                            #else
                            TERMINATE_NO_GPU
                            #endif
                            break;
                        }
                    }
                }
 
                ///* update hphi and ophi */
                //if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && economize_gpu_memory))
                //{
                //    hphi.copy_from(hpsi, 0, num_bands);
                //    ophi.copy_from(opsi, 0, num_bands);
                //    //std::memcpy(hphi.at<CPU>(), hpsi.at<CPU>(), num_bands * ngk * sizeof(double_complex));
                //    //std::memcpy(ophi.at<CPU>(), opsi.at<CPU>(), num_bands * ngk * sizeof(double_complex));
                //}
                
                #ifdef __GPU
                //STOP();
                //if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
                //{
                //    cuda_copy_device_to_device(hphi.at<GPU>(), hpsi.at<GPU>(), num_bands * ngk * sizeof(double_complex));
                //    cuda_copy_device_to_device(ophi.at<GPU>(), opsi.at<GPU>(), num_bands * ngk * sizeof(double_complex));
                //    cuda_copy_device_to_device( phi.at<GPU>(),  psi.at<GPU>(), num_bands * ngk * sizeof(double_complex));
                //}
                #endif
                
                /* update basis functions */
                phi.copy_from(psi, 0, num_bands);
                N = num_bands;
            }
        }
        ///* expand variational subspace with new basis vectors obtatined from residuals */
        //if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && economize_gpu_memory))
        //{
        //    phi.copy_from(res, 0, n, N);
        //}
        //if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
        //{
        //    #ifdef __GPU
        //    //STOP();
        //    //cuda_copy_device_to_device(phi.at<GPU>(0, N), res.at<GPU>(), n * ngk * sizeof(double_complex));
        //    //cuda_copy_to_host(phi.at<CPU>(0, N), phi.at<GPU>(0, N), n * ngk * sizeof(double_complex));
        //    #else
        //    TERMINATE_NO_GPU
        //    #endif
        //}
    }

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef __GPU
        //STOP();
        //if (!economize_gpu_memory)
        //{
        //    //kp__->beta_gk().deallocate_on_device();
        //    //psi.deallocate_on_device();
        //}
        #endif
    }

    kp__->set_fv_eigen_values(&eval[0]);
}

};
