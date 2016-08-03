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

template <typename T>
void Band::diag_pseudo_potential_chebyshev(K_point* kp__,
                                           int ispn__,
                                           Hloc_operator& h_op__,
                                           D_operator<T>& d_op__,
                                           Q_operator<T>& q_op__,
                                           P_operator<T>& p_op__) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_pseudo_potential_chebyshev");

    auto pu = ctx_.processing_unit();

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_fv_states();

    auto& itso = ctx_.iterative_solver_input_section();

    /* short notation for target wave-functions */
    auto& psi = kp__->spinor_wave_functions<false>(ispn__);

//== 
//==     //auto& beta_pw_panel = kp__->beta_pw_panel();
//==     //dmatrix<double_complex> S(unit_cell_.mt_basis_size(), unit_cell_.mt_basis_size(), kp__->blacs_grid());
//==     //linalg<CPU>::gemm(2, 0, unit_cell_.mt_basis_size(), unit_cell_.mt_basis_size(), kp__->num_gkvec(), complex_one,
//==     //                  beta_pw_panel, beta_pw_panel, complex_zero, S);
//==     //for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==     //{
//==     //    auto type = unit_cell_.atom(ia)->type();
//==     //    int nbf = type->mt_basis_size();
//==     //    int ofs = unit_cell_.atom(ia)->offset_lo();
//==     //    matrix<double_complex> qinv(nbf, nbf);
//==     //    type->uspp().q_mtrx >> qinv;
//==     //    linalg<CPU>::geinv(nbf, qinv);
//==     //    for (int i = 0; i < nbf; i++)
//==     //    {
//==     //        for (int j = 0; j < nbf; j++) S.add(ofs + j, ofs + i, qinv(j, i));
//==     //    }
//==     //}
//==     //linalg<CPU>::geinv(unit_cell_.mt_basis_size(), S);
//== 
//== 
    /* maximum order of Chebyshev polynomial*/
    int order = itso.num_steps_ + 2;

    std::vector< Wave_functions<false>* > phi(order);
    for (int i = 0; i < order; i++) {
        phi[i] = new Wave_functions<false>(kp__->num_gkvec_loc(), num_bands, pu);
    }

    Wave_functions<false> hphi(kp__->num_gkvec_loc(), num_bands, pu);

    /* trial basis functions */
    phi[0]->copy_from(psi, 0, num_bands);

    /* apply Hamiltonian to the basis functions */
    apply_h<T>(kp__, ispn__, 0, num_bands, *phi[0], hphi, h_op__, d_op__);

    /* compute Rayleight quotients */
    std::vector<double> e0(num_bands, 0.0);
    if (pu == CPU) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_bands; i++) {
            for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
                e0[i] += std::real(std::conj((*phi[0])(igk, i)) * hphi(igk, i));
            }
        }
    }
    kp__->comm().allreduce(e0);

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
    double lambda1 = 0.5 * std::pow(ctx_.gk_cutoff(), 2);

    double r = (lambda1 - lambda0) / 2.0;
    double c = (lambda1 + lambda0) / 2.0;

    auto apply_p = [kp__, &p_op__, num_bands](Wave_functions<false>& phi, Wave_functions<false>& op_phi) {
        op_phi.copy_from(phi, 0, num_bands);
        //for (int i = 0; i < kp__->beta_projectors().num_beta_chunks(); i++) {
        //    kp__->beta_projectors().generate(i);

        //    kp__->beta_projectors().inner<T>(i, phi, 0, num_bands);

        //    p_op__.apply(i, 0, op_phi, 0, num_bands);
        //}
    };

    apply_p(hphi, *phi[1]);
    
    /* compute \psi_1 = (S^{-1}H\psi_0 - c\psi_0) / r */
    if (pu == CPU) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_bands; i++) {
            for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
                (*phi[1])(igk, i) = ((*phi[1])(igk, i) - (*phi[0])(igk, i) * c) / r;
            }
        }
    }
//==     //if (parameters_.processing_unit() == GPU)
//==     //{
//==     //    #ifdef __GPU
//==     //    compute_chebyshev_polynomial_gpu(kp__->num_gkvec_row(), (int)kp__->spl_fv_states().local_size(), c, r,
//==     //                                     phi[0].at<GPU>(), phi[1].at<GPU>(), NULL);
//==     //    phi[1].panel().copy_to_host();
//==     //    #endif
//==     //}
//== 

    /* compute higher polynomial orders */
    for (int k = 2; k < order; k++) {

        apply_h<T>(kp__, ispn__, 0, num_bands, *phi[k - 1], hphi, h_op__, d_op__);

        apply_p(hphi, *phi[k]);

        if (pu == CPU) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_bands; i++) {
                for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
                    (*phi[k])(igk, i) = ((*phi[k])(igk, i) - c * (*phi[k - 1])(igk, i)) * 2.0 / r - (*phi[k - 2])(igk, i);
                }
            }
        }
        //== if (parameters_.processing_unit() == GPU)
        //== {
        //==     #ifdef __GPU
        //==     compute_chebyshev_polynomial_gpu(kp__->num_gkvec(), num_bands, c, r,
        //==                                      phi[k - 2].at<GPU>(), phi[k - 1].at<GPU>(), phi[k].at<GPU>());
        //==     phi[k].copy_to_host();
        //==     #endif
        //== }
    }

    /* allocate Hamiltonian and overlap */
    matrix<T> hmlt(num_bands, num_bands);
    matrix<T> ovlp(num_bands, num_bands);
    matrix<T> evec(num_bands, num_bands);
    matrix<T> hmlt_old;
    matrix<T> ovlp_old;

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

    /* apply Hamiltonian and overlap operators to the new basis functions */
    apply_h_o<T>(kp__, ispn__, 0, num_bands, *phi[order - 1], hphi, *phi[0], h_op__, d_op__, q_op__);
    
    //orthogonalize<T>(kp__, N, n, phi, hphi, ophi, ovlp);

    /* setup eigen-value problem */
    set_h_o<T>(kp__, 0, num_bands, *phi[order - 1], hphi, *phi[0], hmlt, ovlp, hmlt_old, ovlp_old);

    /* solve generalized eigen-value problem with the size N */
    diag_h_o<T>(kp__, num_bands, num_bands, hmlt, ovlp, evec, hmlt_dist, ovlp_dist, evec_dist, eval);

    /* recompute wave-functions */
    /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
    psi.transform_from<T>(*phi[order - 1], num_bands, evec, num_bands);

    for (int j = 0; j < ctx_.num_fv_states(); j++) {
        kp__->band_energy(j + ispn__ * ctx_.num_fv_states()) = eval[j];
    }

    for (int i = 0; i < order; i++) {
        delete phi[i];
    }
}

template void Band::diag_pseudo_potential_chebyshev<double>(K_point* kp__,
                                                            int ispn__,
                                                            Hloc_operator& h_op__,
                                                            D_operator<double>& d_op__,
                                                            Q_operator<double>& q_op__,
                                                            P_operator<double>& p_op__) const;

template void Band::diag_pseudo_potential_chebyshev<double_complex>(K_point* kp__,
                                                                    int ispn__,
                                                                    Hloc_operator& h_op__,
                                                                    D_operator<double_complex>& d_op__,
                                                                    Q_operator<double_complex>& q_op__,
                                                                    P_operator<double_complex>& p_op__) const;

};
