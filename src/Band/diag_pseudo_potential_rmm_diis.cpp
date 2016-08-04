#include <algorithm>
#include "band.h"

namespace sirius {

template <typename T>
inline T inner_local(K_point* kp__, Wave_functions<false>& a, int ia, Wave_functions<false>& b, int ib);

template<>
inline double inner_local<double>(K_point* kp__, Wave_functions<false>& a, int ia, Wave_functions<false>& b, int ib)
{
    double result{0};
    double* a_tmp = reinterpret_cast<double*>(&a(0, ia));
    double* b_tmp = reinterpret_cast<double*>(&b(0, ib));
    for (int igk = 0; igk < 2 * kp__->num_gkvec_loc(); igk++) {
        result += a_tmp[igk] * b_tmp[igk];
    }

    if (kp__->comm().rank() == 0) {
        result = 2 * result - a_tmp[0] * b_tmp[0];
    } else {
        result *= 2;
    }

    return result;
}

template<>
inline double_complex inner_local<double_complex>(K_point* kp__, Wave_functions<false>& a, int ia, Wave_functions<false>& b, int ib)
{
    double_complex result{0, 0};
    for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
        result += std::conj(a(igk, ia)) * b(igk, ib);
    }
    return result;
}

template <typename T>
void Band::diag_pseudo_potential_rmm_diis(K_point* kp__,
                                          int ispn__,
                                          Hloc_operator& h_op__,
                                          D_operator<T>& d_op__,
                                          Q_operator<T>& q_op__) const

{
    auto& itso = ctx_.iterative_solver_input_section();
    double tol = ctx_.iterative_solver_tolerance();

    if (tol > 1e-4) {
        diag_pseudo_potential_davidson(kp__, ispn__, h_op__, d_op__, q_op__);
        return;
    }

    PROFILE_WITH_TIMER("sirius::Band::diag_pseudo_potential_rmm_diis");

    /* get diagonal elements for preconditioning */
    auto h_diag = get_h_diag(kp__, ispn__, h_op__.v0(ispn__), d_op__);
    auto o_diag = get_o_diag(kp__, q_op__);

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_fv_states();

    auto pu = ctx_.processing_unit();

    /* short notation for target wave-functions */
    auto& psi = kp__->spinor_wave_functions<false>(ispn__);

    int niter = itso.num_steps_;

    Eigenproblem_lapack evp_solver(2 * linalg_base::dlamch('S'));

    std::vector< Wave_functions<false>* > phi(niter);
    std::vector< Wave_functions<false>* > res(niter);
    std::vector< Wave_functions<false>* > ophi(niter);
    std::vector< Wave_functions<false>* > hphi(niter);

    for (int i = 0; i < niter; i++) {
        phi[i]  = new Wave_functions<false>(kp__->num_gkvec_loc(), num_bands, pu);
        res[i]  = new Wave_functions<false>(kp__->num_gkvec_loc(), num_bands, pu);
        hphi[i] = new Wave_functions<false>(kp__->num_gkvec_loc(), num_bands, pu);
        ophi[i] = new Wave_functions<false>(kp__->num_gkvec_loc(), num_bands, pu);
    }

    Wave_functions<false>  phi_tmp(kp__->num_gkvec_loc(), num_bands, pu);
    Wave_functions<false> hphi_tmp(kp__->num_gkvec_loc(), num_bands, pu);
    Wave_functions<false> ophi_tmp(kp__->num_gkvec_loc(), num_bands, pu);

    /* allocate Hamiltonian and overlap */
    matrix<T> hmlt(num_bands, num_bands);
    matrix<T> ovlp(num_bands, num_bands);
    matrix<T> hmlt_old;
    matrix<T> ovlp_old;

    #ifdef __GPU
    if (gen_evp_solver_->type() == ev_magma) {
        hmlt.pin_memory();
        ovlp.pin_memory();
    }
    #endif

    matrix<T> evec(num_bands, num_bands);

    int bs = ctx_.cyclic_block_size();

    dmatrix<T> hmlt_dist;
    dmatrix<T> ovlp_dist;
    dmatrix<T> evec_dist;
    if (kp__->comm().size() == 1) {
        hmlt_dist = dmatrix<T>(&hmlt(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
        ovlp_dist = dmatrix<T>(&ovlp(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
        evec_dist = dmatrix<T>(&evec(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
    } else {
        hmlt_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
        ovlp_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
        evec_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
    }

    std::vector<double> eval(num_bands);
    for (int i = 0; i < num_bands; i++) {
        eval[i] = kp__->band_energy(i);
    }
    std::vector<double> eval_old(num_bands);

    /* trial basis functions */
    phi[0]->copy_from(psi, 0, num_bands);

    std::vector<int> last(num_bands, 0);
    std::vector<bool> conv_band(num_bands, false);
    std::vector<double> res_norm(num_bands);
    std::vector<double> res_norm_start(num_bands);
    std::vector<double> lambda(num_bands, 0);
    
    auto update_res = [kp__, num_bands, &phi, &res, &hphi, &ophi, &last, &conv_band]
                      (std::vector<double>& res_norm__, std::vector<double>& eval__) -> void
    {
        runtime::Timer t("sirius::Band::diag_pseudo_potential_rmm_diis|res");
        std::vector<double> e_tmp(num_bands, 0), d_tmp(num_bands, 0);

        #pragma omp parallel for
        for (int i = 0; i < num_bands; i++) {
            if (!conv_band[i]) {
                e_tmp[i] = std::real(inner_local<T>(kp__, *phi[last[i]], i, *hphi[last[i]], i));
                d_tmp[i] = std::real(inner_local<T>(kp__, *phi[last[i]], i, *ophi[last[i]], i));
            }
        }
        kp__->comm().allreduce(e_tmp);
        kp__->comm().allreduce(d_tmp);
        
        res_norm__ = std::vector<double>(num_bands, 0);
        #pragma omp parallel for
        for (int i = 0; i < num_bands; i++) {
            if (!conv_band[i]) {
                eval__[i] = e_tmp[i] / d_tmp[i];

                /* compute residual r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
                for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
                    (*res[last[i]])(igk, i) = (*hphi[last[i]])(igk, i) - eval__[i] * (*ophi[last[i]])(igk, i);
                }
                res_norm__[i] = std::real(inner_local<T>(kp__, *res[last[i]], i, *res[last[i]], i));
            }
        }
        kp__->comm().allreduce(res_norm__);
        for (int i = 0; i < num_bands; i++) {
            if (!conv_band[i]) {
                res_norm__[i] = std::sqrt(res_norm__[i]);
            }
        }
    };

    auto apply_h_o = [this, kp__, num_bands, &phi, &phi_tmp, &hphi, &hphi_tmp, &ophi, &ophi_tmp, &conv_band, &last,
                      &h_op__, &d_op__, &q_op__, ispn__]() -> int
    {
        runtime::Timer t("sirius::Band::diag_pseudo_potential_rmm_diis|h_o");
        int n{0};
        for (int i = 0; i < num_bands; i++) {
            if (!conv_band[i]) {
                std::memcpy(&phi_tmp(0, n), &(*phi[last[i]])(0, i), kp__->num_gkvec_loc() * sizeof(double_complex));
                n++;
            }
        }

        if (n == 0) {
            return 0;
        }
        
        /* apply Hamiltonian and overlap operators to the initial basis functions */
        this->apply_h_o<T>(kp__, ispn__, 0, n, phi_tmp, hphi_tmp, ophi_tmp, h_op__, d_op__, q_op__);

        n = 0;
        for (int i = 0; i < num_bands; i++) {
            if (!conv_band[i]) {
                std::memcpy(&(*hphi[last[i]])(0, i), &hphi_tmp(0, n), kp__->num_gkvec_loc() * sizeof(double_complex));
                std::memcpy(&(*ophi[last[i]])(0, i), &ophi_tmp(0, n), kp__->num_gkvec_loc() * sizeof(double_complex));
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
        for (int i = 0; i < num_bands; i++) {
            if (!conv_band[i]) {
                for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
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
    for (int i = 0; i < num_bands; i++) {
        if (kp__->band_occupancy(i) > 1e-2 && res_norm_start[i] > itso.residual_tolerance_) {
            conv = false;
        }
    }
    if (conv) {
        DUMP("all bands are converged at stage#0");
        return;
    }

    last = std::vector<int>(num_bands, 1);
    
    /* apply preconditioner to the initial residuals */
    apply_preconditioner(std::vector<double>(num_bands, 1), *res[0], 0.0, *phi[1]);
    
    /* apply H and O to the preconditioned residuals */
    apply_h_o();

    /* estimate lambda */
    std::vector<double> f1(num_bands, 0);
    std::vector<double> f2(num_bands, 0);
    std::vector<double> f3(num_bands, 0);
    std::vector<double> f4(num_bands, 0);

    #pragma omp parallel for
    for (int i = 0; i < num_bands; i++) {
        if (!conv_band[i]) {
            f1[i] = std::real(inner_local<T>(kp__, *phi[1], i, *ophi[1], i));     //  <KR_i | OKR_i>
            f2[i] = std::real(inner_local<T>(kp__, *phi[0], i, *ophi[1], i)) * 2; // <phi_i | OKR_i>
            f3[i] = std::real(inner_local<T>(kp__, *phi[1], i, *hphi[1], i));     //  <KR_i | HKR_i>
            f4[i] = std::real(inner_local<T>(kp__, *phi[0], i, *hphi[1], i)) * 2; // <phi_i | HKR_i>
        }
    }
    kp__->comm().allreduce(f1);
    kp__->comm().allreduce(f2);
    kp__->comm().allreduce(f3);
    kp__->comm().allreduce(f4);

    #pragma omp parallel for
    for (int i = 0; i < num_bands; i++) {
        if (!conv_band[i]) {
            double a = f1[i] * f4[i] - f2[i] * f3[i];
            double b = f3[i] - eval[i] * f1[i];
            double c = eval[i] * f2[i] - f4[i];

            lambda[i] = (b - std::sqrt(b * b - 4.0 * a * c)) / 2.0 / a;
            if (std::abs(lambda[i]) > 2.0) {
                lambda[i] = 2.0 * Utils::sign(lambda[i]);
            }
            if (std::abs(lambda[i]) < 0.5) {
                lambda[i] = 0.5 * Utils::sign(lambda[i]);
            }
            
            /* construct new basis functions */
            for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
                 (*phi[1])(igk, i) =  (*phi[0])(igk, i) + lambda[i] *  (*phi[1])(igk, i);
                (*hphi[1])(igk, i) = (*hphi[0])(igk, i) + lambda[i] * (*hphi[1])(igk, i);
                (*ophi[1])(igk, i) = (*ophi[0])(igk, i) + lambda[i] * (*ophi[1])(igk, i);
            }
        }
    }
    /* compute new residuals */
    update_res(res_norm, eval);
    /* check which bands have converged */
    for (int i = 0; i < num_bands; i++) {
        if (kp__->band_occupancy(i) <= 1e-2 || res_norm[i] < itso.residual_tolerance_) {
            conv_band[i] = true;
        }
    }

    mdarray<T, 3> A(niter, niter, num_bands);
    mdarray<T, 3> B(niter, niter, num_bands);
    mdarray<T, 2> V(niter, num_bands);
    std::vector<double> ev(niter);

    for (int iter = 2; iter < niter; iter++) {
        runtime::Timer t1("sirius::Band::diag_pseudo_potential_rmm_diis|AB");
        A.zero();
        B.zero();
        for (int i = 0; i < num_bands; i++) {
            if (!conv_band[i]) {
                for (int i1 = 0; i1 < iter; i1++) {
                    for (int i2 = 0; i2 < iter; i2++) {
                        A(i1, i2, i) = inner_local<T>(kp__, *res[i1], i, *res[i2], i);
                        B(i1, i2, i) = inner_local<T>(kp__, *phi[i1], i, *ophi[i2], i);
                    }
                }
            }
        }
        kp__->comm().allreduce(A.template at<CPU>(), (int)A.size());
        kp__->comm().allreduce(B.template at<CPU>(), (int)B.size());
        t1.stop();

        runtime::Timer t2("sirius::Band::diag_pseudo_potential_rmm_diis|phi");
        for (int i = 0; i < num_bands; i++) {
            if (!conv_band[i]) {
                if (evp_solver.solve(iter, 1, &A(0, 0, i), A.ld(), &B(0, 0, i), B.ld(), &ev[0], &V(0, i), V.ld()) == 0) {
                    std::memset(&(*phi[iter])(0, i), 0, kp__->num_gkvec_loc() * sizeof(double_complex));
                    std::memset(&(*res[iter])(0, i), 0, kp__->num_gkvec_loc() * sizeof(double_complex));
                    for (int i1 = 0; i1 < iter; i1++) {
                        for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
                            (*phi[iter])(igk, i) += (*phi[i1])(igk, i) * V(i1, i);
                            (*res[iter])(igk, i) += (*res[i1])(igk, i) * V(i1, i);
                        }
                    }
                    last[i] = iter;
                } else {
                    conv_band[i] = true;
                }
            }
        }
        t2.stop();
        
        apply_preconditioner(lambda, *res[iter], 1.0, *phi[iter]);

        apply_h_o();

        eval_old = eval;

        update_res(res_norm, eval);
        
        for (int i = 0; i < num_bands; i++) {
            if (!conv_band[i]) {
                if (kp__->band_occupancy(i) <= 1e-2) {
                    conv_band[i] = true;
                }
                if (kp__->band_occupancy(i) > 1e-2 && std::abs(eval[i] - eval_old[i]) < tol) {
                    conv_band[i] = true;
                }
                if (kp__->band_occupancy(i) > 1e-2 && res_norm[i] < itso.residual_tolerance_) {
                    conv_band[i] = true;
                }
                //if (kp__->band_occupancy(i) <= 1e-2 ||
                //    res_norm[i] / res_norm_start[i] < 0.7 ||
                //    (kp__->band_occupancy(i) > 1e-2 && std::abs(eval[i] - eval_old[i]) < tol)) {
                //    conv_band[i] = true;
                //}
            }
        }
        if (std::all_of(conv_band.begin(), conv_band.end(), [](bool e){return e;})) {
            std::cout << "early exit from the diis loop" << std::endl;
            break;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < num_bands; i++) {
        std::memcpy(&phi_tmp(0, i),  &(*phi[last[i]])(0, i),  kp__->num_gkvec_loc() * sizeof(double_complex));
        std::memcpy(&hphi_tmp(0, i), &(*hphi[last[i]])(0, i), kp__->num_gkvec_loc() * sizeof(double_complex));
        std::memcpy(&ophi_tmp(0, i), &(*ophi[last[i]])(0, i), kp__->num_gkvec_loc() * sizeof(double_complex));
    }


    orthogonalize<T>(kp__, 0, num_bands, phi_tmp, hphi_tmp, ophi_tmp, ovlp);

    set_h_o<T>(kp__, 0, num_bands, phi_tmp, hphi_tmp, ophi_tmp, hmlt, ovlp, hmlt_old, ovlp_old);
    
    /* solve generalized eigen-value problem with the size N */
    diag_h_o<T>(kp__, num_bands, num_bands, hmlt, ovlp, evec, hmlt_dist, ovlp_dist, evec_dist, eval);
    
    /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
    psi.transform_from<T>(phi_tmp, num_bands, evec, num_bands);
    
    for (int j = 0; j < ctx_.num_fv_states(); j++) {
        kp__->band_energy(j + ispn__ * ctx_.num_fv_states()) = eval[j];
    }

    for (int i = 0; i < niter; i++) {
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
