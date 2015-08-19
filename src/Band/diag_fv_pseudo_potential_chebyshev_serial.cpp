#include "band.h"

#ifdef __GPU
extern "C" void compute_chebyshev_polynomial_gpu(int num_gkvec,
                                                 int n,
                                                 double c,
                                                 double r,
                                                 cuDoubleComplex* phi0,
                                                 cuDoubleComplex* phi1,
                                                 cuDoubleComplex* phi2);
#endif

namespace sirius {

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

    //auto& beta_pw_panel = kp__->beta_pw_panel();
    //dmatrix<double_complex> S(unit_cell_.mt_basis_size(), unit_cell_.mt_basis_size(), kp__->blacs_grid());
    //linalg<CPU>::gemm(2, 0, unit_cell_.mt_basis_size(), unit_cell_.mt_basis_size(), kp__->num_gkvec(), complex_one,
    //                  beta_pw_panel, beta_pw_panel, complex_zero, S);
    //for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    //{
    //    auto type = unit_cell_.atom(ia)->type();
    //    int nbf = type->mt_basis_size();
    //    int ofs = unit_cell_.atom(ia)->offset_lo();
    //    matrix<double_complex> qinv(nbf, nbf);
    //    type->uspp().q_mtrx >> qinv;
    //    linalg<CPU>::geinv(nbf, qinv);
    //    for (int i = 0; i < nbf; i++)
    //    {
    //        for (int j = 0; j < nbf; j++) S.add(ofs + j, ofs + i, qinv(j, i));
    //    }
    //}
    //linalg<CPU>::geinv(unit_cell_.mt_basis_size(), S);

    auto& itso = parameters_.iterative_solver_input_section();

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

    int kappa_size = std::max(unit_cell_.mt_basis_size(), 4 * num_bands);
    /* temporary array for <G+k|beta> */
    mdarray<double_complex, 1> kappa(kp__->num_gkvec() * kappa_size);
    if (verbosity_level >= 6)
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
    
    /* pack D, Q and P matrices */
    mdarray<double_complex, 1> d_mtrx_packed(packed_mtrx_size);
    mdarray<double_complex, 1> q_mtrx_packed(packed_mtrx_size);
    mdarray<double_complex, 1> p_mtrx_packed(packed_mtrx_size);

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    {
        int nbf = unit_cell_.atom(ia)->mt_basis_size();
        int iat = unit_cell_.atom(ia)->type()->id();
        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                d_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = unit_cell_.atom(ia)->d_mtrx(xi1, xi2);
                q_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = unit_cell_.atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
                p_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = kp__->p_mtrx(xi1, xi2, iat);
            }
        }
    }

    //== if (parameters_.processing_unit() == GPU)
    //== {
    //==     #ifdef __GPU
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
    //==     #ifdef __GPU
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
            #ifdef __GPU
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
    //    #ifdef __GPU
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
                #ifdef __GPU
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
            #ifdef __GPU
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
        #ifdef __GPU
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

    if (false)
    {
        std::vector<double> h_diag;
        std::vector<double> o_diag;
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
