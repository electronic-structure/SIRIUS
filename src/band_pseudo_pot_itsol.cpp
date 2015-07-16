#include "band.h"
#include "debug.hpp"

namespace sirius {

#ifdef __GPU
extern "C" void compute_inner_product_gpu(int num_gkvec_row,
                                          int n,
                                          cuDoubleComplex const* f1,
                                          cuDoubleComplex const* f2,
                                          double* prod);

extern "C" void compute_chebyshev_polynomial_gpu(int num_gkvec,
                                                 int n,
                                                 double c,
                                                 double r,
                                                 cuDoubleComplex* phi0,
                                                 cuDoubleComplex* phi1,
                                                 cuDoubleComplex* phi2);
#endif

#ifdef __SCALAPACK
void Band::diag_fv_pseudo_potential_chebyshev_parallel(K_point* kp__,
                                                       std::vector<double> const& veff_it_coarse__)
{
    STOP();

//==     log_function_enter(__func__);
//== 
//==     /* alias for wave-functions */
//==     dmatrix<double_complex>& psi = kp__->fv_states_panel();
//== 
//==     /* short notation for number of target wave-functions */
//==     int num_bands = parameters_.num_fv_states();
//== 
//==     /* cache kinetic energy */
//==     std::vector<double> pw_ekin = kp__->get_pw_ekin();
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
//==     auto& itso = parameters_.iterative_solver_input_section();
//== 
//==     /* maximum order of Chebyshev polynomial*/
//==     int order = itso.num_steps_ + 2;
//== 
//==     std::vector< dmatrix<double_complex> > phi(order);
//==     for (int i = 0; i < order; i++)
//==     {
//==         phi[i] = dmatrix<double_complex>(kp__->num_gkvec(), num_bands, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
//==     }
//== 
//==     dmatrix<double_complex> hphi(kp__->num_gkvec(), num_bands, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
//==     hphi.allocate_ata_buffer((int)kp__->spl_fv_states().local_size(0));
//== 
//==     /* trial basis functions */
//==     psi.panel() >> phi[0].panel();
//== 
//==     //int num_atoms_in_block = std::min(unit_cell_.num_atoms(), 256);
//==     int num_bands_local = (int)kp__->spl_fv_states().local_size(0);
//==     int kappa_size = std::max(unit_cell_.max_mt_basis_size() * unit_cell_.beta_chunk(0).num_atoms_, 4 * num_bands_local);
//==     /* temporary array for <G+k|beta> */
//==     matrix<double_complex> kappa(kp__->num_gkvec_row(), kappa_size);
//==     if (verbosity_level >= 6 && kp__->comm().rank() == 0)
//==     {
//==         printf("size of kappa array: %f GB\n", 16 * double(kappa.size()) / 1073741824);
//==     }
//==     
//==     /* offset in the packed array of on-site matrices */
//==     mdarray<int, 1> packed_mtrx_offset(unit_cell_.num_atoms());
//==     int packed_mtrx_size = 0;
//==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==     {   
//==         int nbf = unit_cell_.atom(ia)->mt_basis_size();
//==         packed_mtrx_offset(ia) = packed_mtrx_size;
//==         packed_mtrx_size += nbf * nbf;
//==     }
//==     
//==     /* pack D, Q and P matrices */
//==     mdarray<double_complex, 1> d_mtrx_packed(packed_mtrx_size);
//==     mdarray<double_complex, 1> q_mtrx_packed(packed_mtrx_size);
//==     mdarray<double_complex, 1> p_mtrx_packed(packed_mtrx_size);
//== 
//==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==     {
//==         int nbf = unit_cell_.atom(ia)->mt_basis_size();
//==         int iat = unit_cell_.atom(ia)->type()->id();
//==         for (int xi2 = 0; xi2 < nbf; xi2++)
//==         {
//==             for (int xi1 = 0; xi1 < nbf; xi1++)
//==             {
//==                 d_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = unit_cell_.atom(ia)->d_mtrx(xi1, xi2);
//==                 q_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = unit_cell_.atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
//==                 p_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = kp__->p_mtrx(xi1, xi2, iat);
//==             }
//==         }
//==     }
//== 
//==     if (parameters_.processing_unit() == GPU)
//==     {
//==         #ifdef __GPU
//==         for (int i = 0; i < order; i++) phi[i].allocate_on_device();
//==         hphi.allocate_on_device();
//==         kappa.allocate_on_device();
//==         d_mtrx_packed.allocate_on_device();
//==         d_mtrx_packed.copy_to_device();
//==         q_mtrx_packed.allocate_on_device();
//==         q_mtrx_packed.copy_to_device();
//==         p_mtrx_packed.allocate_on_device();
//==         p_mtrx_packed.copy_to_device();
//==         /* initial phi on GPU */
//==         phi[0].panel().copy_to_device();
//==         #else
//==         TERMINATE_NO_GPU
//==         #endif
//==     }
//== 
//==     /* apply Hamiltonian to the basis functions */
//==     apply_h_parallel(kp__, veff_it_coarse__, pw_ekin, 0, num_bands, phi[0], hphi, kappa, packed_mtrx_offset,
//==                      d_mtrx_packed);
//== 
//==     /* compute Rayleight quotients */
//==     std::vector<double> e0(num_bands, 0.0);
//==     if (parameters_.processing_unit() == CPU)
//==     {
//==         #pragma omp parallel for schedule(static)
//==         for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
//==         {
//==             int i = kp__->spl_fv_states(iloc);
//==             for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
//==             {
//==                 e0[i] += real(conj(phi[0](igk_row, iloc)) * hphi(igk_row, iloc));
//==             }
//==         }
//==     }
//==     if (parameters_.processing_unit() == GPU)
//==     {
//==         #ifdef __GPU
//==         mdarray<double, 1> e0_loc(kp__->spl_fv_states().local_size());
//==         e0_loc.allocate_on_device();
//==         e0_loc.zero_on_device();
//== 
//==         compute_inner_product_gpu(kp__->num_gkvec_row(),
//==                                   (int)kp__->spl_fv_states().local_size(),
//==                                   phi[0].at<GPU>(),
//==                                   hphi.at<GPU>(),
//==                                   e0_loc.at<GPU>());
//==         e0_loc.copy_to_host();
//==         for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
//==         {
//==             int i = kp__->spl_fv_states(iloc);
//==             e0[i] = e0_loc(iloc);
//==         }
//==         #endif
//==     }
//==     
//==     kp__->comm().allreduce(e0);
//==     
//==     /* estimate low and upper bounds of the Chebyshev filter */
//==     double lambda0 = -1e10;
//==     for (int i = 0; i < num_bands; i++) lambda0 = std::max(lambda0, e0[i]);
//==     double lambda1 = 0.5 * std::pow(parameters_.gk_cutoff(), 2);
//== 
//==     double r = (lambda1 - lambda0) / 2.0;
//==     double c = (lambda1 + lambda0) / 2.0;
//== 
//==     switch (parameters_.processing_unit())
//==     {
//==         case CPU:
//==         {
//==             hphi.panel() >> phi[1].panel();
//==             break;
//==         }
//==         case GPU:
//==         {
//==             #ifdef __GPU
//==             cuda_copy_device_to_device(phi[1].at<GPU>(), hphi.at<GPU>(), hphi.panel().size() * sizeof(double_complex));
//==             #endif
//==             break;
//==         }
//==     }
//== 
//==     //== add_non_local_contribution_parallel(kp__, hphi, phi[1], S, double_complex(-1, 0));
//==     add_non_local_contribution_parallel(kp__, 0, num_bands, hphi, phi[1], kappa, packed_mtrx_offset,
//==                                         p_mtrx_packed, double_complex(-1, 0));
//==     
//==     /* compute \psi_1 = (S^{-1}H\psi_0 - c\psi_0) / r */
//==     if (parameters_.processing_unit() == CPU)
//==     {
//==         #pragma omp parallel for schedule(static)
//==         for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
//==         {
//==             for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
//==             {
//==                 phi[1](igk_row, iloc) = (phi[1](igk_row, iloc) - phi[0](igk_row, iloc) * c) / r;
//==             }
//==         }
//==     }
//==     if (parameters_.processing_unit() == GPU)
//==     {
//==         #ifdef __GPU
//==         compute_chebyshev_polynomial_gpu(kp__->num_gkvec_row(), (int)kp__->spl_fv_states().local_size(), c, r,
//==                                          phi[0].at<GPU>(), phi[1].at<GPU>(), NULL);
//==         phi[1].panel().copy_to_host();
//==         #endif
//==     }
//== 
//==     /* compute higher polynomial orders */
//==     for (int k = 2; k < order; k++)
//==     {
//==         apply_h_parallel(kp__, veff_it_coarse__, pw_ekin, 0, num_bands, phi[k - 1], hphi, kappa, packed_mtrx_offset,
//==                          d_mtrx_packed);
//==         
//==         switch (parameters_.processing_unit())
//==         {
//==             case CPU:
//==             {
//==                 hphi.panel() >> phi[k].panel();
//==                 break;
//==             }
//==             case GPU:
//==             {
//==                 #ifdef __GPU
//==                 cuda_copy_device_to_device(phi[k].at<GPU>(), hphi.at<GPU>(), hphi.panel().size() * sizeof(double_complex));
//==                 #endif
//==                 break;
//==             }
//==         }
//==         //add_non_local_contribution_parallel(kp__, hphi, phi[k], S, double_complex(-1, 0));
//==         add_non_local_contribution_parallel(kp__, 0, num_bands, hphi, phi[k], kappa, packed_mtrx_offset,
//==                                             p_mtrx_packed, double_complex(-1, 0));
//==         
//==         if (parameters_.processing_unit() == CPU)
//==         {
//==             #pragma omp parallel for schedule(static)
//==             for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
//==             {
//==                 for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
//==                 {
//==                     phi[k](igk_row, iloc) = (phi[k](igk_row, iloc) - c * phi[k - 1](igk_row, iloc)) * 2.0 / r -
//==                                             phi[k - 2](igk_row, iloc);
//==                 }
//==             }
//==         }
//==         if (parameters_.processing_unit() == GPU)
//==         {
//==             #ifdef __GPU
//==             compute_chebyshev_polynomial_gpu(kp__->num_gkvec_row(), (int)kp__->spl_fv_states().local_size(), c, r,
//==                                              phi[k - 2].at<GPU>(), phi[k - 1].at<GPU>(), phi[k].at<GPU>());
//==             phi[k].panel().copy_to_host();
//==             #endif
//==         }
//==     }
//== 
//==     /* apply Hamiltonian and overlap to the "filtered" basis functions */
//==     apply_h_o_parallel(kp__, veff_it_coarse__, pw_ekin, 0, num_bands, phi[order - 1], hphi, phi[0],
//==                        kappa, packed_mtrx_offset, d_mtrx_packed, q_mtrx_packed);
//== 
//==     if (parameters_.processing_unit() == GPU)
//==     {
//==         #ifdef __GPU
//==         hphi.panel().copy_to_host();
//==         phi[0].panel().copy_to_host();
//==         #endif
//==     }
//== 
//==     dmatrix<double_complex> hmlt(num_bands, num_bands, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
//==     dmatrix<double_complex> ovlp(num_bands, num_bands, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
//== 
//==     dmatrix<double_complex> evec(num_bands, num_bands, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
//==     std::vector<double> eval(num_bands);
//== 
//==     Timer t1("sirius::Band::diag_fv_pseudo_potential|set_h_o", kp__->comm());
//==     linalg<CPU>::gemm(2, 0, num_bands, num_bands, kp__->num_gkvec(), complex_one, phi[order - 1], hphi, complex_zero, hmlt);
//==     linalg<CPU>::gemm(2, 0, num_bands, num_bands, kp__->num_gkvec(), complex_one, phi[order - 1], phi[0], complex_zero, ovlp);
//==     double tval = t1.stop();
//== 
//==     if (verbosity_level >= 6 && kp__->comm().rank() == 0)
//==     {
//==         printf("2x pzgemm with M, N, K: %6i %6i %6i: %12.4f sec, %12.4f GFlops/rank\n",
//==                num_bands, num_bands, kp__->num_gkvec(),
//==                tval, 2 * 8e-9 * num_bands * num_bands * kp__->num_gkvec() / tval / kp__->num_ranks());
//==     }
//==     
//==     Timer t2("sirius::Band::diag_fv_pseudo_potential|gen_evp");
//==     gen_evp_solver()->solve(num_bands, hmlt.num_rows_local(), hmlt.num_cols_local(), num_bands, 
//==                             hmlt.at<CPU>(), hmlt.ld(), ovlp.at<CPU>(), ovlp.ld(), 
//==                             &eval[0], evec.at<CPU>(), evec.ld());
//==     t2.stop();
//==         
//==     if (kp__->comm().rank() == 0)
//==     {
//==         printf("eigen-values:\n");
//==         for (int i = 0; i < std::min(num_bands, 10); i++) printf("%18.12f ", eval[i]);
//==         printf("\n");
//==     }
//== 
//==     //generate_fv_states_pp(kp__, num_bands, evec, phi[order - 1], psi, kappa);
//==     //
//==     Timer t3("sirius::Band::diag_fv_pseudo_potential|psi");
//==     /* recompute wave-functions: \Psi_{i} = \phi_{mu} * Z_{mu, i} */
//==     linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, num_bands, complex_one, phi[order - 1], evec, complex_zero, psi); 
//==     t3.stop();
//== 
//==     kp__->set_fv_eigen_values(&eval[0]);
//==     log_function_exit(__func__);
}
#endif

void Band::diag_fv_pseudo_potential_serial_exact(K_point* kp__,
                                                 std::vector<double>& veff_it_coarse__)
{
    STOP();

//    /* cache kinetic energy */
//    std::vector<double> pw_ekin = kp__->get_pw_ekin();
//
//    /* short notation for target wave-functions */
//    mdarray<double_complex, 2>& psi = kp__->fv_states_slab();
//
//    /* short notation for number of target wave-functions */
//    int num_bands = parameters_.num_fv_states();     
//
//    int ngk = kp__->num_gkvec();
//
//    mdarray<double_complex, 2> phi(ngk, ngk);
//    mdarray<double_complex, 2> hphi(ngk, ngk);
//    mdarray<double_complex, 2> ophi(ngk, ngk);
//    mdarray<double_complex, 1> kappa(ngk * ngk);
//    
//    std::vector<double> eval(ngk);
//
//    phi.zero();
//    for (int i = 0; i < ngk; i++) phi(i, i) = complex_one;
//
//    /* offset in the packed array of on-site matrices */
//    mdarray<int, 1> packed_mtrx_offset(unit_cell_.num_atoms());
//    int packed_mtrx_size = 0;
//    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//    {   
//        int nbf = unit_cell_.atom(ia)->mt_basis_size();
//        packed_mtrx_offset(ia) = packed_mtrx_size;
//        packed_mtrx_size += nbf * nbf;
//    }
//    
//    /* pack Q and D matrices */
//    mdarray<double_complex, 1> d_mtrx_packed(packed_mtrx_size);
//    mdarray<double_complex, 1> q_mtrx_packed(packed_mtrx_size);
//
//    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//    {
//        int nbf = unit_cell_.atom(ia)->mt_basis_size();
//        for (int xi2 = 0; xi2 < nbf; xi2++)
//        {
//            for (int xi1 = 0; xi1 < nbf; xi1++)
//            {
//                d_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = unit_cell_.atom(ia)->d_mtrx(xi1, xi2);
//                q_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = unit_cell_.atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
//            }
//        }
//    }
//    
//    apply_h_o_serial(kp__, veff_it_coarse__, pw_ekin, 0, ngk, phi, hphi, ophi, kappa, packed_mtrx_offset,
//                         d_mtrx_packed, q_mtrx_packed);
//        
//    gen_evp_solver()->solve(ngk, num_bands, num_bands, num_bands, hphi.at<CPU>(), hphi.ld(), ophi.at<CPU>(), ophi.ld(), 
//                            &eval[0], psi.at<CPU>(), psi.ld());
//
//    kp__->set_fv_eigen_values(&eval[0]);
}


};
