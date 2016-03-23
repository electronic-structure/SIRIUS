#include "band.h"

namespace sirius {

template <typename T>
void Band::diag_pseudo_potential_rmm_diis(K_point* kp__,
                                          int ispn__,
                                          Hloc_operator& h_op__,
                                          D_operator<T>& d_op__,
                                          Q_operator<T>& q_op__) const

{
    if (ctx_.iterative_solver_tolerance() > 1e-5)
    {
        diag_pseudo_potential_davidson(kp__, ispn__, h_op__, d_op__, q_op__);
        return;
    }

    PROFILE_WITH_TIMER("sirius::Band::diag_pseudo_potential_rmm_diis");

    /* get diagonal elements for preconditioning */
    auto h_diag = get_h_diag(kp__, ispn__, h_op__.v0(ispn__), d_op__);
    auto o_diag = get_o_diag(kp__, q_op__);

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_fv_states();

    ///auto& itso = ctx_.iterative_solver_input_section();

    auto pu = ctx_.processing_unit();

    /* short notation for target wave-functions */
    auto& psi = kp__->spinor_wave_functions<false>(ispn__);

    int niter = 4; //itso.num_steps_;

    Eigenproblem_lapack evp_solver(2 * linalg_base::dlamch('S'));

    std::vector< Wave_functions<false>* > phi(niter);
    std::vector< Wave_functions<false>* > res(niter);
    std::vector< Wave_functions<false>* > ophi(niter);
    std::vector< Wave_functions<false>* > hphi(niter);

    for (int i = 0; i < niter; i++)
    {
        phi[i] = new Wave_functions<false>(num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
        res[i] = new Wave_functions<false>(num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
        hphi[i] = new Wave_functions<false>(num_bands, num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
        ophi[i] = new Wave_functions<false>(num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
    }

    Wave_functions<false> phi_tmp(num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
    Wave_functions<false> hphi_tmp(num_bands, num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
    Wave_functions<false> ophi_tmp(num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);

    /* allocate Hamiltonian and overlap */
    matrix<T> hmlt(num_bands, num_bands);
    matrix<T> ovlp(num_bands, num_bands);
    matrix<T> hmlt_old(num_bands, num_bands);
    matrix<T> ovlp_old(num_bands, num_bands);

    #ifdef __GPU
    if (gen_evp_solver_->type() == ev_magma)
    {
        hmlt.pin_memory();
        ovlp.pin_memory();
    }
    #endif

    matrix<T> evec(num_bands, num_bands);

    int bs = ctx_.cyclic_block_size();

    dmatrix<T> hmlt_dist;
    dmatrix<T> ovlp_dist;
    dmatrix<T> evec_dist;
    if (kp__->comm().size() == 1)
    {
        hmlt_dist = dmatrix<T>(&hmlt(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
        ovlp_dist = dmatrix<T>(&ovlp(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
        evec_dist = dmatrix<T>(&evec(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
    }
    else
    {
        hmlt_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
        ovlp_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
        evec_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
    }

    std::vector<double> eval(num_bands);
    for (int i = 0; i < num_bands; i++) eval[i] = kp__->band_energy(i);
    std::vector<double> eval_old(num_bands);

    /* trial basis functions */
    phi[0]->copy_from(psi, 0, num_bands);

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
    
    auto update_res = [kp__, num_bands, &phi, &res, &hphi, &ophi, &last, &conv_band]
                      (std::vector<double>& res_norm__, std::vector<double>& eval__) -> void
    {
        runtime::Timer t("sirius::Band::diag_pseudo_potential_rmm_diis|res");
        for (int i = 0; i < num_bands; i++)
        {
            if (!conv_band[i])
            {
                double e = 0;
                double d = 0;
                for (int igk = 0; igk < kp__->num_gkvec(); igk++)
                {
                    e += std::real(std::conj((*phi[last[i]])(igk, i)) * (*hphi[last[i]])(igk, i));
                    d += std::real(std::conj((*phi[last[i]])(igk, i)) * (*ophi[last[i]])(igk, i));
                }
                eval__[i] = e / d;

                /* compute residual r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
                for (int igk = 0; igk < kp__->num_gkvec(); igk++)
                {
                    (*res[last[i]])(igk, i) = (*hphi[last[i]])(igk, i) - eval__[i] * (*ophi[last[i]])(igk, i);
                }

                double r = 0;
                for (int igk = 0; igk < kp__->num_gkvec(); igk++)
                {
                    r += std::real(std::conj((*res[last[i]])(igk, i)) * (*res[last[i]])(igk, i));
                }
                res_norm__[i] = r; //std::sqrt(r);
            }
        }
    };

    auto apply_h_o = [this, kp__, num_bands, &phi, &phi_tmp, &hphi, &hphi_tmp, &ophi, &ophi_tmp, &conv_band, &last,
                      &h_op__, &d_op__, &q_op__, ispn__]() -> int
    {
        runtime::Timer t("sirius::Band::diag_pseudo_potential_rmm_diis|h_o");
        int n = 0;
        for (int i = 0; i < num_bands; i++)
        {
            if (!conv_band[i])
            {
                std::memcpy(&phi_tmp(0, n), &(*phi[last[i]])(0, i), kp__->num_gkvec() * sizeof(double_complex));
                n++;
            }
        }

        if (n == 0) return 0;
        
        /* apply Hamiltonian and overlap operators to the initial basis functions */
        this->apply_h_o<T>(kp__, ispn__, 0, n, phi_tmp, hphi_tmp, ophi_tmp, h_op__, d_op__, q_op__);


        n = 0;
        for (int i = 0; i < num_bands; i++)
        {
            if (!conv_band[i])
            {
                std::memcpy(&(*hphi[last[i]])(0, i), &hphi_tmp(0, n), kp__->num_gkvec() * sizeof(double_complex));
                std::memcpy(&(*ophi[last[i]])(0, i), &ophi_tmp(0, n), kp__->num_gkvec() * sizeof(double_complex));
                n++;
            }
        }
        return n;
    };

    auto apply_preconditioner = [kp__, num_bands, &h_diag, &o_diag, &eval, &conv_band]
                                (std::vector<double> lambda,
                                 Wave_functions<false>& res__,
                                 double alpha,
                                 Wave_functions<false>& kres__) -> void
    {
        runtime::Timer t("sirius::Band::diag_pseudo_potential_rmm_diis|pre");
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

    /* apply Hamiltonian and overlap operators to the initial basis functions */
    this->apply_h_o<T>(kp__, ispn__, 0, num_bands, *phi[0], *hphi[0], *ophi[0], h_op__, d_op__, q_op__);
    
    /* compute initial residuals */
    update_res(res_norm_start, eval);

    bool conv = true;
    for (int i = 0; i < num_bands; i++)
    {
        if (kp__->band_occupancy(i) > 1e-2 && res_norm_start[i] > 1e-12) conv = false;
        //if (res_norm_start[i] < 1e-12 || kp__->band_occupancy(i) <= 1e-2)
        //{
        //    conv_band[i] = true;
        //}
    }
    if (conv) DUMP("all bands are converged at stage#0");
    if (conv) return;

    last = std::vector<int>(num_bands, 1);
    
    /* apply preconditioner to the initial residuals */
    apply_preconditioner(std::vector<double>(num_bands, 1), *res[0], 0.0, *phi[1]);
    
    /* apply H and O to the preconditioned residuals */
    apply_h_o();

    /* estimate lambda */
    for (int i = 0; i < num_bands; i++)
    {
        if (!conv_band[i])
        {
            double f1(0), f2(0), f3(0), f4(0);
            for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            {
                f1 += std::real(std::conj((*phi[1])(igk, i)) * (*ophi[1])(igk, i));      //  <KR_i | OKR_i>
                f2 += std::real(std::conj((*phi[0])(igk, i)) * (*ophi[1])(igk, i)) * 2;  // <phi_i | OKR_i> 
                f3 += std::real(std::conj((*phi[1])(igk, i)) * (*hphi[1])(igk, i));      //  <KR_i | HKR_i>
                f4 += std::real(std::conj((*phi[0])(igk, i)) * (*hphi[1])(igk, i)) * 2;  // <phi_i | HKR_i>
            }
            
            double a = f1 * f4 - f2 * f3;
            double b = f3 - eval[i] * f1;
            double c = eval[i] * f2 - f4;

            lambda[i] = (b - std::sqrt(b * b - 4.0 * a * c)) / 2.0 / a;
            if (std::abs(lambda[i]) > 2.0) lambda[i] = 2.0 * lambda[i] / std::abs(lambda[i]);
            if (std::abs(lambda[i]) < 0.5) lambda[i] = 0.5 * lambda[i] / std::abs(lambda[i]);
            
            /* construct new basis functions */
            for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            {
                 (*phi[1])(igk, i) =  (*phi[0])(igk, i) + lambda[i] *  (*phi[1])(igk, i);
                (*hphi[1])(igk, i) = (*hphi[0])(igk, i) + lambda[i] * (*hphi[1])(igk, i);
                (*ophi[1])(igk, i) = (*ophi[0])(igk, i) + lambda[i] * (*ophi[1])(igk, i);
            }
        }
    }
    /* compute new residuals */
    update_res(res_norm, eval);
    /* check which bands have converged */
    for (int i = 0; i < num_bands; i++)
    {
        if (kp__->band_occupancy(i) <= 1e-2 || res_norm[i] < 1e-12)
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
        runtime::Timer t1("sirius::Band::diag_pseudo_potential_rmm_diis|AB");
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
                            A(i1, i2, i) += std::conj((*res[i1])(igk, i)) * (*res[i2])(igk, i);
                            B(i1, i2, i) += std::conj((*phi[i1])(igk, i)) * (*ophi[i2])(igk, i);
                        }
                    }
                }
            }
        }
        t1.stop();

        runtime::Timer t2("sirius::Band::diag_pseudo_potential_rmm_diis|phi");
        for (int i = 0; i < num_bands; i++)
        {
            if (!conv_band[i])
            {
                if (evp_solver.solve(iter, 1, &A(0, 0, i), A.ld(), &B(0, 0, i), B.ld(), &ev[0], &V(0, i), V.ld()) == 0)
                {
                    std::memset(&(*phi[iter])(0, i), 0, kp__->num_gkvec() * sizeof(double_complex));
                    std::memset(&(*res[iter])(0, i), 0, kp__->num_gkvec() * sizeof(double_complex));
                    for (int i1 = 0; i1 < iter; i1++)
                    {
                        for (int igk = 0; igk < kp__->num_gkvec(); igk++)
                        {
                            (*phi[iter])(igk, i) += (*phi[i1])(igk, i) * V(i1, i);
                            (*res[iter])(igk, i) += (*res[i1])(igk, i) * V(i1, i);
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
        
        apply_preconditioner(lambda, *res[iter], 1.0, *phi[iter]);

        int n = apply_h_o();
        if (n == 0) break;

        eval_old = eval;

        update_res(res_norm, eval);

        double tol = ctx_.iterative_solver_tolerance();
        
        for (int i = 0; i < num_bands; i++)
        {
            if (!conv_band[i])
            {
                if (kp__->band_occupancy(i) <= 1e-2 ||
                    res_norm[i] / res_norm_start[i] < 0.7 ||
                    (kp__->band_occupancy(i) > 1e-2 && std::abs(eval[i] - eval_old[i]) < tol))
                {
                    conv_band[i] = true;
                }
            }
        }
    }

    for (int i = 0; i < num_bands; i++)
    {
        std::memcpy(&phi_tmp(0, i),  &(*phi[last[i]])(0, i),  kp__->num_gkvec() * sizeof(double_complex));
        std::memcpy(&hphi_tmp(0, i), &(*hphi[last[i]])(0, i), kp__->num_gkvec() * sizeof(double_complex));
        std::memcpy(&ophi_tmp(0, i), &(*ophi[last[i]])(0, i), kp__->num_gkvec() * sizeof(double_complex));
    }

    set_h_o<T>(kp__, 0, num_bands, phi_tmp, hphi_tmp, ophi_tmp, hmlt, ovlp, hmlt_old, ovlp_old);
    
    /* solve generalized eigen-value problem with the size N */
    diag_h_o<T>(kp__, num_bands, num_bands, hmlt, ovlp, evec, hmlt_dist, ovlp_dist, evec_dist, eval);
    
    /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
    psi.transform_from<T>(phi_tmp, num_bands, evec, num_bands);
    
    for (int j = 0; j < ctx_.num_fv_states(); j++)
        kp__->band_energy(j + ispn__ * ctx_.num_fv_states()) = eval[j];

    for (int i = 0; i < niter; i++)
    {
        delete phi[i];
        delete res[i];
        delete hphi[i];
        delete ophi[i];
    }
}

/* explicit instantiation for general k-point solver */
template void Band::diag_pseudo_potential_rmm_diis<double_complex>(K_point* kp__,
                                                                   int ispn__,
                                                                   Hloc_operator& h_op__,
                                                                   D_operator<double_complex>& d_op__,
                                                                   Q_operator<double_complex>& q_op__) const;
/* explicit instantiation for gamma-point solver */
template void Band::diag_pseudo_potential_rmm_diis<double>(K_point* kp__,
                                                           int ispn__,
                                                           Hloc_operator& h_op__,
                                                           D_operator<double>& d_op__,
                                                           Q_operator<double>& q_op__) const;

};
