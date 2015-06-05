#include "band.h"

namespace sirius {

void Band::diag_fv_pseudo_potential_rmm_diis_serial(K_point* kp__,
                                                    double v0__,
                                                    std::vector<double>& veff_it_coarse__)
{
    if (ctx_.iterative_solver_tolerance() > 1e-5)
    {
        diag_fv_pseudo_potential_davidson_serial(kp__, v0__, veff_it_coarse__);
        return;
    }

    Timer t("sirius::Band::diag_fv_pseudo_potential_rmm_diis_serial");

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
    matrix<double_complex>& psi = kp__->fv_states_slab();

    int niter = 4; //itso.num_steps_;

    generalized_evp_lapack evp_solver(0.0);

    std::vector< matrix<double_complex> > phi(niter);
    std::vector< matrix<double_complex> > res(niter);
    std::vector< matrix<double_complex> > ophi(niter);
    std::vector< matrix<double_complex> > hphi(niter);

    for (int i = 0; i < niter; i++)
    {
        phi[i] = matrix<double_complex>(kp__->num_gkvec(), num_bands);
        res[i] = matrix<double_complex>(kp__->num_gkvec(), num_bands);
        ophi[i] = matrix<double_complex>(kp__->num_gkvec(), num_bands);
        hphi[i] = matrix<double_complex>(kp__->num_gkvec(), num_bands);
    }

    matrix<double_complex> phi_tmp(kp__->num_gkvec(), num_bands);
    matrix<double_complex> hphi_tmp(kp__->num_gkvec(), num_bands);
    matrix<double_complex> ophi_tmp(kp__->num_gkvec(), num_bands);

    matrix<double_complex> hmlt(num_bands, num_bands);
    matrix<double_complex> ovlp(num_bands, num_bands);
    matrix<double_complex> hmlt_old(num_bands, num_bands);
    matrix<double_complex> ovlp_old(num_bands, num_bands);

    matrix<double_complex> evec(num_bands, num_bands);

    std::vector<double> eval(num_bands);
    for (int i = 0; i < num_bands; i++) eval[i] = kp__->band_energy(i);
    std::vector<double> eval_old(num_bands);

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

    bool economize_gpu_memory = true;
    mdarray<double_complex, 1> kappa;
    if (economize_gpu_memory) kappa = mdarray<double_complex, 1>(nullptr, kp__->num_gkvec() * (std::max(unit_cell_.mt_basis_size(), num_bands) + num_bands));
    
    //== #ifdef __GPU
    //== if (verbosity_level >= 6 && kp__->comm().rank() == 0 && parameters_.processing_unit() == GPU)
    //== {
    //==     printf("size of kappa array: %f GB\n", 16 * double(kappa.size()) / 1073741824);
    //== }
    //== #endif

    /* trial basis functions */
    memcpy(&phi[0](0, 0), &psi(0, 0), kp__->num_gkvec() * num_bands * sizeof(double_complex));

    
    //== if (parameters_.processing_unit() == GPU)
    //== {
    //==     #ifdef __GPU
    //==     if (!economize_gpu_memory)
    //==     {
    //==         phi.allocate_on_device();
    //==         psi.allocate_on_device();
    //==         res.allocate_on_device();
    //==         hphi.allocate_on_device();
    //==         hpsi.allocate_on_device();
    //==         kp__->beta_gk().allocate_on_device();
    //==         kp__->beta_gk().copy_to_device();
    //==         /* initial phi on GPU */
    //==         cuda_copy_to_device(phi.at<GPU>(), psi.at<CPU>(), kp__->num_gkvec_row() * num_bands * sizeof(double_complex));
    //==         if (with_overlap)
    //==         {
    //==             ophi.allocate_on_device();
    //==             opsi.allocate_on_device();
    //==         }
    //==     }
    //==     else
    //==     {
    //==         kappa.allocate_on_device();
    //==     }
    //==     d_mtrx_packed.allocate_on_device();
    //==     d_mtrx_packed.copy_to_device();
    //==     if (with_overlap)
    //==     {
    //==         q_mtrx_packed.allocate_on_device();
    //==         q_mtrx_packed.copy_to_device();
    //==     }
    //==     hmlt.allocate_on_device();
    //==     ovlp.allocate_on_device();
    //==     evec.allocate_on_device();
    //==     evec_tmp.allocate_on_device();
    //==     if (converge_by_energy) evec_tmp.allocate_on_device();
    //==     #else
    //==     TERMINATE_NO_GPU
    //==     #endif
    //== }

    //== #ifdef __PRINT_OBJECT_HASH
    //== std::cout << "hash(beta_pw)       : " << kp__->beta_gk_panel().panel().hash() << std::endl;
    //== std::cout << "hash(d_mtrx_packed) : " << d_mtrx_packed.hash() << std::endl;
    //== std::cout << "hash(q_mtrx_packed) : " << q_mtrx_packed.hash() << std::endl;
    //== std::cout << "hash(v_eff_coarse)  : " << Utils::hash(&veff_it_coarse__[0], parameters_.fft_coarse()->size() * sizeof(double)) << std::endl;
    //== #endif

    std::vector<int> last(num_bands, 0);
    std::vector<bool> conv_band(num_bands, false);
    std::vector<double> res_norm(num_bands);
    std::vector<double> res_norm_start(num_bands);
    std::vector<double> lambda(num_bands, 0);
    
    auto update_res = [kp__, num_bands, &phi, &res, &hphi, &ophi, &last, &conv_band, &eval]
                      (std::vector<double>& res_norm__) -> void
    {
        Timer t("sirius::Band::diag_fv_pseudo_potential_rmm_diis_serial|res");
        for (int i = 0; i < num_bands; i++)
        {
            if (!conv_band[i])
            {
                double e = 0;
                double d = 0;
                for (int igk = 0; igk < kp__->num_gkvec(); igk++)
                {
                    e += real(conj(phi[last[i]](igk, i)) * hphi[last[i]](igk, i));
                    d += real(conj(phi[last[i]](igk, i)) * ophi[last[i]](igk, i));
                }
                eval[i] = e / d;

                /* compute residual r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
                for (int igk = 0; igk < kp__->num_gkvec(); igk++)
                {
                    res[last[i]](igk, i) = hphi[last[i]](igk, i) - eval[i] * ophi[last[i]](igk, i);
                }

                double r = 0;
                for (int igk = 0; igk < kp__->num_gkvec(); igk++)
                {
                    r += real(conj(res[last[i]](igk, i)) * res[last[i]](igk, i));
                }
                res_norm__[i] = r; //std::sqrt(r);
            }
        }
    };

    auto apply_h_o = [this, kp__, num_bands, &phi, &phi_tmp, &hphi, &hphi_tmp, &ophi, &ophi_tmp, &conv_band, &last,
                      &veff_it_coarse__, &pw_ekin, &kappa, &packed_mtrx_offset, &d_mtrx_packed, &q_mtrx_packed]() -> int
    {
        Timer t("sirius::Band::diag_fv_pseudo_potential_rmm_diis_serial|h_o");
        int n = 0;
        for (int i = 0; i < num_bands; i++)
        {
            if (!conv_band[i])
            {
                memcpy(&phi_tmp(0, n), &phi[last[i]](0, i), kp__->num_gkvec() * sizeof(double_complex));
                n++;
            }
        }

        if (n == 0) return 0;

        apply_h_o_serial(kp__, veff_it_coarse__, pw_ekin, 0, n, phi_tmp, hphi_tmp, ophi_tmp, kappa, packed_mtrx_offset,
                         d_mtrx_packed, q_mtrx_packed);

        n = 0;
        for (int i = 0; i < num_bands; i++)
        {
            if (!conv_band[i])
            {
                memcpy(&hphi[last[i]](0, i), &hphi_tmp(0, n), kp__->num_gkvec() * sizeof(double_complex));
                memcpy(&ophi[last[i]](0, i), &ophi_tmp(0, n), kp__->num_gkvec() * sizeof(double_complex));
                n++;
            }
        }
        return n;
    };

    auto apply_preconditioner = [kp__, num_bands, &h_diag, &o_diag, &eval, &conv_band]
                                (std::vector<double> lambda,
                                 matrix<double_complex>& res__,
                                 double alpha,
                                 matrix<double_complex>& kres__) -> void
    {
        Timer t("sirius::Band::diag_fv_pseudo_potential_rmm_diis_serial|pre");
        #pragma omp parallel for
        for (int i = 0; i < num_bands; i++)
        {
            if (!conv_band[i])
            {
                for (int igk = 0; igk < kp__->num_gkvec(); igk++)
                {
                    double p = h_diag[igk] - eval[i] * o_diag[igk];

                    p *= 2; // QE formula is in Ry; here we convert to Ha
                    p = 0.25 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                    kres__(igk, i) = alpha * kres__(igk, i) + lambda[i] * res__(igk, i) / p;
                }
            }

            //== double Ekin = 0;
            //== double norm = 0;
            //== for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            //== {
            //==     Ekin += 0.5 * std::pow(std::abs(res__(igk, i)), 2) * std::pow(kp__->gkvec_cart(igk).length(), 2);
            //==     norm += std::pow(std::abs(res__(igk, i)), 2);
            //== }
            //== Ekin /= norm;
            //== for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            //== {
            //==     double x = std::pow(kp__->gkvec_cart(igk).length(), 2) / 3 / Ekin;
            //==     kres__(igk, i) = alpha * kres__(igk, i) + lambda[i] * res__(igk, i) * 
            //==         (4.0 / 3 / Ekin) * (27 + 18 * x + 12 * x * x + 8 * x * x * x) / (27 + 18 * x + 12 * x * x + 8 * x * x * x + 16 * x * x * x * x);
            //== }
        }
    };


    apply_h_o_serial(kp__, veff_it_coarse__, pw_ekin, 0, num_bands, phi[0], hphi[0], ophi[0], kappa, packed_mtrx_offset,
                     d_mtrx_packed, q_mtrx_packed);

    update_res(res_norm_start);

    bool conv = true;
    for (int i = 0; i < num_bands; i++)
    {
        if (kp__->band_occupancy(i) > 1e-10 && res_norm_start[i] > 1e-12) conv = false;
        //if (res_norm_start[i] < 1e-12)
        //{
        //    conv_band[i] = true;
        //}
        //else
        //{
        //    last[i] = 1;
        //}
    }
    if (conv) return;

    last = std::vector<int>(num_bands, 1);
    
    apply_preconditioner(std::vector<double>(num_bands, 1), res[0], 0.0, phi[1]);
    
    apply_h_o();

    /* estimate lambda */
    for (int i = 0; i < num_bands; i++)
    {
        if (!conv_band[i])
        {
            double f1(0), f2(0), f3(0), f4(0);
            for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            {
                f1 += real(conj(phi[1](igk, i)) * ophi[1](igk, i));        //  <KR_i | OKR_i>
                f2 += 2.0 * real(conj(phi[0](igk, i)) * ophi[1](igk, i));  // <phi_i | OKR_i> 
                f3 += real(conj(phi[1](igk, i)) * hphi[1](igk, i));        //  <KR_i | HKR_i>
                f4 += 2.0 * real(conj(phi[0](igk, i)) * hphi[1](igk, i));  // <phi_i | HKR_i>
            }
            
            double a = f1 * f4 - f2 * f3;
            double b = f3 - eval[i] * f1;
            double c = eval[i] * f2 - f4;

            lambda[i] = (b - std::sqrt(b * b - 4.0 * a * c)) / 2.0 / a;
            if (std::abs(lambda[i]) > 2.0) lambda[i] = 2.0 * lambda[i] / std::abs(lambda[i]);
            if (std::abs(lambda[i]) < 0.5) lambda[i] = 0.5 * lambda[i] / std::abs(lambda[i]);
            
            for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            {
                phi[1](igk, i) = phi[0](igk, i) + lambda[i] * phi[1](igk, i);
                hphi[1](igk, i) = hphi[0](igk, i) + lambda[i] * hphi[1](igk, i);
                ophi[1](igk, i) = ophi[0](igk, i) + lambda[i] * ophi[1](igk, i);
            }
        }
    }

    update_res(res_norm);
    for (int i = 0; i < num_bands; i++)
    {
        if (res_norm[i] < 1e-12)
        {
            conv_band[i] = true;
        }
    }



    mdarray<double_complex, 3> A(niter, niter, num_bands);
    mdarray<double_complex, 3> B(niter, niter, num_bands);
    mdarray<double_complex, 2> V(niter, num_bands);
    std::vector<double> ev(niter);

    for (int iter = 2; iter < niter; iter++)
    {
        Timer t1("sirius::Band::diag_fv_pseudo_potential_rmm_diis_serial|AB");
        A.zero();
        B.zero();
        for (int i = 0; i < num_bands; i++)
        {
            if (!conv_band[i])
            {
                for (int i1 = 0; i1 < iter; i1++)
                {
                    for (int i2 = 0; i2 < iter; i2++)
                    {
                        for (int igk = 0; igk < kp__->num_gkvec(); igk++)
                        {
                            A(i1, i2, i) += conj(res[i1](igk, i)) * res[i2](igk, i);
                            B(i1, i2, i) += conj(phi[i1](igk, i)) * ophi[i2](igk, i);
                        }
                    }
                }
            }
        }
        t1.stop();

        Timer t2("sirius::Band::diag_fv_pseudo_potential_rmm_diis_serial|phi");
        for (int i = 0; i < num_bands; i++)
        {
            if (!conv_band[i])
            {
                if (evp_solver.solve(iter, iter, iter, 1, &A(0, 0, i), A.ld(), &B(0, 0, i), B.ld(), &ev[0], &V(0, i), V.ld()) == 0)
                {
                    memset(&phi[iter](0, i), 0, kp__->num_gkvec() * sizeof(double_complex));
                    memset(&res[iter](0, i), 0, kp__->num_gkvec() * sizeof(double_complex));
                    for (int i1 = 0; i1 < iter; i1++)
                    {
                        for (int igk = 0; igk < kp__->num_gkvec(); igk++)
                        {
                            phi[iter](igk, i) += phi[i1](igk, i) * V(i1, i);
                            res[iter](igk, i) += res[i1](igk, i) * V(i1, i);
                        }
                    }
                    last[i] = iter;
                }
                else
                {
                    conv_band[i] = true;
                }
            }
        }
        t2.stop();
        
        apply_preconditioner(lambda, res[iter], 1.0, phi[iter]);

        int n = apply_h_o();
        if (n == 0) break;

        eval_old = eval;

        update_res(res_norm);

        double tol = ctx_.iterative_solver_tolerance();
        
        for (int i = 0; i < num_bands; i++)
        {
            if (!conv_band[i])
            {
                if ((kp__->band_occupancy(i) < 1e-10 && iter == 2) ||
                    (res_norm[i] / res_norm_start[i] < 0.7) ||
                    (std::abs(eval[i] - eval_old[i]) < tol))
                {
                    conv_band[i] = true;
                }
            }
        }
    }

    for (int i = 0; i < num_bands; i++)
    {
        memcpy(&phi_tmp(0, i), &phi[last[i]](0, i), kp__->num_gkvec() * sizeof(double_complex));
        memcpy(&hphi_tmp(0, i), &hphi[last[i]](0, i), kp__->num_gkvec() * sizeof(double_complex));
        memcpy(&ophi_tmp(0, i), &ophi[last[i]](0, i), kp__->num_gkvec() * sizeof(double_complex));
    }

    set_fv_h_o_serial(kp__, 0, num_bands, phi_tmp, hphi_tmp, ophi_tmp, hmlt, ovlp, hmlt_old, ovlp_old, kappa);
 
    Timer t1("sirius::Band::diag_fv_pseudo_potential|solve_gevp", kp__->comm());
    if (gen_evp_solver()->solve(num_bands, num_bands, num_bands, num_bands, hmlt.at<CPU>(), hmlt.ld(), ovlp.at<CPU>(), ovlp.ld(), 
                                &eval[0], evec.at<CPU>(), evec.ld()))
    {
        TERMINATE("error in gen_evp_solver");
    }
    t1.stop();

    linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, num_bands, &phi_tmp(0, 0), phi_tmp.ld(), &evec(0, 0), evec.ld(), 
                      &psi(0, 0), psi.ld());
 
    kp__->set_fv_eigen_values(&eval[0]);
}

};
