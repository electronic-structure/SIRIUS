#include "band.h"

namespace sirius {

void Band::solve_sv_pp(K_point* kp__, Periodic_function<double>* effective_magnetic_field__[3]) const
{
    PROFILE_WITH_TIMER("sirius::Band::solve_sv_pp");

    STOP();

//    auto fft_coarse = ctx_.fft_coarse();
//    auto& gv = ctx_.gvec();
//    auto& gvc = ctx_.gvec_coarse();
//
//    /* map effective magnetic field to a corase grid */
//    mdarray<double, 2> beff_r_coarse(fft_coarse->local_size(), ctx_.num_mag_dims());
//    std::vector<double_complex> beff_pw_coarse(gvc.num_gvec_fft());
//
//    for (int j = 0; j < ctx_.num_mag_dims(); j++)
//    {
//        effective_magnetic_field__[j]->fft_transform(-1);
//        for (int ig = 0; ig < gvc.num_gvec_fft(); ig++)
//        {
//            auto G = gvc[ig + gvc.offset_gvec_fft()];
//            beff_pw_coarse[ig] = effective_magnetic_field__[j]->f_pw(gv.index_by_gvec(G));
//        }
//        fft_coarse->transform<1>(gvc, &beff_pw_coarse[0]);
//        fft_coarse->output(&beff_r_coarse(0, j));
//    }
//
//    /* number of h|\psi> components */
//    int nhpsi = (ctx_.num_mag_dims() == 3) ? 3 : ctx_.num_spins();
//
//    std::vector<double> band_energies(ctx_.num_bands());
//
//    /* product of the second-variational Hamiltonian and a first-variational wave-function */
//    std::vector<Wave_functions<false>*> hpsi;
//    for (int i = 0; i < nhpsi; i++)
//    {
//        hpsi.push_back(new Wave_functions<false>(ctx_.num_fv_states(), ctx_.num_fv_states(), kp__->gkvec(),
//                                                 ctx_.mpi_grid_fft(), ctx_.processing_unit()));
//    }
//
//    hpsi[0]->copy_from(kp__->fv_states<false>(), 0, ctx_.num_fv_states());
//    hpsi[0]->swap_forward(0, ctx_.num_fv_states());
//    
//    /* save omp_nested flag */
//    int nested = omp_get_nested();
//    omp_set_nested(1);
//    #pragma omp parallel num_threads(ctx_.fft_ctx().num_fft_streams())
//    {
//        int thread_id = omp_get_thread_num();
//
//        #pragma omp for schedule(dynamic, 1)
//        for (int i = 0; i < hpsi[0]->spl_num_swapped().local_size(); i++)
//        {
//            /* phi(G) -> phi(r) */
//            ctx_.fft_coarse_ctx().fft(thread_id)->transform<1>(kp__->gkvec(), (*hpsi[0])[i]);
//            /* multiply by effective magnetic field potential */
//            if (ctx_.fft_coarse_ctx().fft(thread_id)->hybrid())
//            {
//                STOP();
//                //#ifdef __GPU
//                //scale_matrix_rows_gpu(fft_ctx_.fft(thread_id)->local_size(), 1,
//                //                      fft_ctx_.fft(thread_id)->buffer<GPU>(), veff_.at<GPU>());
//
//                //#else
//                //TERMINATE_NO_GPU
//                //#endif
//            }
//            else
//            {
//                for (int ir = 0; ir < ctx_.fft_coarse_ctx().fft(thread_id)->local_size(); ir++)
//                    ctx_.fft_coarse_ctx().fft(thread_id)->buffer(ir) *= beff_r_coarse(ir, 0);
//            }
//            /* B(r)phi(r) -> [B*phi](G) */
//            ctx_.fft_coarse_ctx().fft(thread_id)->transform<-1>(kp__->gkvec(), (*hpsi[0])[i]);
//        }
//    }
//    /* restore the nested flag */
//    omp_set_nested(nested);
//
//    hpsi[0]->swap_backward(0, ctx_.num_fv_states());
//    for (int i = 0; i < ctx_.num_fv_states(); i++)
//    {
//        for (int j = 0; j < hpsi[0]->num_gvec_loc(); j++) (*hpsi[1])(j, i) = -(*hpsi[0])(j, i);
//    }
//
//
//
//    int bs = ctx_.cyclic_block_size();
//    int nfv = ctx_.num_fv_states();
//    if (ctx_.num_mag_dims() != 3)
//    {
//        matrix<double_complex> h(nfv, nfv);
//        dmatrix<double_complex> h_dist(nfv, nfv, kp__->blacs_grid(), bs, bs);
//
//        /* perform one or two consecutive diagonalizations */
//        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
//        {
//            kp__->fv_states<false>().inner(0, nfv, (*hpsi[ispn]), 0, nfv, h, 0, 0);
//            for (int i = 0; i < nfv; i++) h(i, i) += kp__->fv_eigen_value(i);
//
//            if (kp__->comm().size() > 1 && std_evp_solver()->parallel())
//            {
//                for (int jloc = 0; jloc < h_dist.num_cols_local(); jloc++)
//                {
//                    int j = h_dist.icol(jloc);
//                    for (int iloc = 0; iloc < h_dist.num_rows_local(); iloc++)
//                    {
//                        int i = h_dist.irow(iloc);
//                        h_dist(iloc, jloc) = (i > j) ? std::conj(h(j, i)) : h(i, j);
//                    }
//                }
//            }
//
//            if (std_evp_solver()->parallel())
//            {
//                std_evp_solver()->solve(nfv, h_dist.at<CPU>(), h_dist.ld(), &band_energies[ispn * nfv],
//                                        kp__->sv_eigen_vectors(ispn).at<CPU>(), kp__->sv_eigen_vectors(ispn).ld());
//
//            }
//            else
//            {
//                std_evp_solver()->solve(nfv, h.at<CPU>(), h.ld(), &band_energies[ispn * nfv],
//                                        kp__->sv_eigen_vectors(ispn).at<CPU>(), kp__->sv_eigen_vectors(ispn).ld());
//            }
//        }
//    }
//
//    for (auto e: hpsi) delete e;
//    kp__->set_band_energies(&band_energies[0]);
}

void Band::solve_sv(K_point* kp, Periodic_function<double>* effective_magnetic_field[3]) const
{
    PROFILE_WITH_TIMER("sirius::Band::solve_sv");

    if (!ctx_.need_sv())
    {
        kp->bypass_sv();
        return;
    }

    if (!ctx_.full_potential())
    {
        solve_sv_pp(kp, effective_magnetic_field);
        return;
    }
    
    if (kp->num_ranks() > 1 && !std_evp_solver()->parallel()) TERMINATE("eigen-value solver is not parallel");

    /* number of h|\psi> components */
    int nhpsi = (ctx_.num_mag_dims() == 3) ? 3 : ctx_.num_spins();

    /* size of the first-variational state */
    int fvsz = kp->wf_size();

    std::vector<double> band_energies(ctx_.num_bands());

    /* product of the second-variational Hamiltonian and a first-variational wave-function */
    std::vector<Wave_functions<true>*> hpsi;
    for (int i = 0; i < nhpsi; i++)
    {
        hpsi.push_back(new Wave_functions<true>(kp->wf_size(), ctx_.num_fv_states(), ctx_.cyclic_block_size(),
                                                ctx_.blacs_grid(), ctx_.blacs_grid_slice()));
    }

    /* compute product of magnetic field and wave-function */
    if (ctx_.num_spins() == 2)
    {
        apply_magnetic_field(kp->fv_states<true>(), kp->gkvec(), effective_magnetic_field, hpsi);
    }
    else
    {
        hpsi[0]->set_num_swapped(ctx_.num_fv_states());
        std::memset((*hpsi[0])[0], 0, kp->wf_size() * hpsi[0]->spl_num_swapped().local_size() * sizeof(double_complex));
    }

    //== if (ctx_.uj_correction())
    //== {
    //==     apply_uj_correction<uu>(kp->fv_states_col(), hpsi);
    //==     if (ctx_.num_mag_dims() != 0) apply_uj_correction<dd>(kp->fv_states_col(), hpsi);
    //==     if (ctx_.num_mag_dims() == 3) 
    //==     {
    //==         apply_uj_correction<ud>(kp->fv_states_col(), hpsi);
    //==         if (ctx_.std_evp_solver()->parallel()) apply_uj_correction<du>(kp->fv_states_col(), hpsi);
    //==     }
    //== }

    //== if (ctx_.so_correction()) apply_so_correction(kp->fv_states_col(), hpsi);

    for (auto e: hpsi) e->swap_backward(0, ctx_.num_fv_states());

//==     if (ctx_.processing_unit() == GPU && kp->num_ranks() == 1)
//==     {
//==         #ifdef __GPU
//==         STOP();
//==         //kp->fv_states().allocate_on_device();
//==         //kp->fv_states().copy_to_device();
//==         #endif
//==     }
//== 
//==     #ifdef __GPU
//==     //double_complex alpha = complex_one;
//==     //double_complex beta = complex_zero;
//==     #endif
//==

    int nfv = ctx_.num_fv_states();
    int bs = ctx_.cyclic_block_size();

    if (ctx_.num_mag_dims() != 3)
    {
        dmatrix<double_complex> h(nfv, nfv, ctx_.blacs_grid(), bs, bs);

        //if (ctx_.processing_unit() == GPU && kp->num_ranks() == 1)
        //{
        //    #ifdef __GPU
        //    h.allocate_on_device();
        //    #endif
        //}

        /* perform one or two consecutive diagonalizations */
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
        {
            if (ctx_.processing_unit() == GPU && kp->num_ranks() == 1)
            {
                #ifdef __GPU
                STOP();
                //Timer t4("sirius::Band::solve_sv|zgemm");
                //hpsi[ispn].allocate_on_device();
                //hpsi[ispn].copy_to_device();
                //linalg<GPU>::gemm(2, 0, ctx_.num_fv_states(), ctx_.num_fv_states(), fvsz, &alpha, 
                //                  kp->fv_states().at<GPU>(), kp->fv_states().ld(),
                //                  hpsi[ispn].at<GPU>(), hpsi[ispn].ld(), &beta, h.at<GPU>(), h.ld());
                //h.copy_to_host();
                //hpsi[ispn].deallocate_on_device();
                //double tval = t4.stop();
                //DUMP("effective zgemm performance: %12.6f GFlops", 
                //     8e-9 * ctx_.num_fv_states() * ctx_.num_fv_states() * fvsz / tval);
                #else
                TERMINATE_NO_GPU
                #endif
            }
            else
            {
                /* compute <wf_i | h * wf_j> for up-up or dn-dn block */
                linalg<CPU>::gemm(2, 0, nfv, nfv, fvsz, complex_one, kp->fv_states<true>().coeffs(), hpsi[ispn]->coeffs(),
                                  complex_zero, h);
            }
            
            for (int i = 0; i < nfv; i++) h.add(i, i, kp->fv_eigen_value(i));
        
            runtime::Timer t1("sirius::Band::solve_sv|stdevp");
            std_evp_solver()->solve(nfv, h.at<CPU>(), h.ld(), &band_energies[ispn * nfv],
                                    kp->sv_eigen_vectors(ispn).at<CPU>(), kp->sv_eigen_vectors(ispn).ld());
        }
    }
    else
    {
        int nb = ctx_.num_bands();
        dmatrix<double_complex> h(nb, nb, ctx_.blacs_grid(), bs, bs);

        /* compute <wf_i | h * wf_j> for up-up block */
        linalg<CPU>::gemm(2, 0, nfv, nfv, fvsz, complex_one, kp->fv_states<true>().coeffs(), 0, 0, hpsi[0]->coeffs(), 0, 0,
                          complex_zero, h, 0, 0);
        /* compute <wf_i | h * wf_j> for dn-dn block */
        linalg<CPU>::gemm(2, 0, nfv, nfv, fvsz, complex_one, kp->fv_states<true>().coeffs(), 0, 0, hpsi[1]->coeffs(), 0, 0,
                          complex_zero, h, nfv, nfv);
        /* compute <wf_i | h * wf_j> for up-dn block */
        linalg<CPU>::gemm(2, 0, nfv, nfv, fvsz, complex_one, kp->fv_states<true>().coeffs(), 0, 0, hpsi[2]->coeffs(), 0, 0,
                          complex_zero, h, 0, nfv);
        
        for (int i = 0; i < nfv; i++)
        {
            h.add(i, i, kp->fv_eigen_value(i));
            h.add(i + nfv, i + nfv, kp->fv_eigen_value(i));
        }
        runtime::Timer t1("sirius::Band::solve_sv|stdevp");
        std_evp_solver()->solve(nb, h.at<CPU>(), h.ld(), &band_energies[0],
                                kp->sv_eigen_vectors(0).at<CPU>(), kp->sv_eigen_vectors(0).ld());
        
    }
//== 
//==     if (ctx_.processing_unit() == GPU && kp->num_ranks() == 1)
//==     {
//==         #ifdef __GPU
//==         //kp->fv_states().deallocate_on_device();
//==         #endif
//==     }
//== 
    for (auto e: hpsi) delete e;

    kp->set_band_energies(&band_energies[0]);
}

};
