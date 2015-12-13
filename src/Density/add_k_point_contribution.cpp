#include "density.h"

namespace sirius {

template <>
void Density::add_k_point_contribution<full_potential_lapwlo>(K_point* kp__,
                                                              occupied_bands_descriptor const& occupied_bands__,
                                                              mdarray<double_complex, 4>& density_matrix__)
{
    PROFILE();

    STOP();

    //Timer t("sirius::Density::add_k_point_contribution");

    //int nbnd = occupied_bands__.num_occupied_bands_local();
    //
    //if (!nbnd) return;
   
    //mdarray<double_complex, 3> wf1(unit_cell_.max_mt_basis_size(), nbnd, parameters_.num_spins());
    //mdarray<double_complex, 3> wf2(unit_cell_.max_mt_basis_size(), nbnd, parameters_.num_spins());

    //for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    //{
    //    int offset_wf = unit_cell_.atom(ia)->offset_wf();
    //    int mt_basis_size = unit_cell_.atom(ia)->type()->mt_basis_size();

    //    if (parameters_.num_mag_dims() == 3)
    //    {
    //        for (int i = 0; i < nbnd; i++)
    //        {   
    //            int ibnd_loc = occupied_bands__.idx_bnd_loc[i];
    //            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
    //            {
    //                for (int xi = 0; xi < mt_basis_size; xi++)
    //                {
    //                    wf1(xi, i, ispn) = std::conj(*kp__->spinor_wave_functions(ispn).at<CPU>(offset_wf + xi, ibnd_loc));
    //                    wf2(xi, i, ispn) = (*kp__->spinor_wave_functions(ispn).at<CPU>(offset_wf + xi, ibnd_loc)) * occupied_bands__.weight[i];
    //                }
    //            }
    //        }
    //    }
    //    else
    //    {
    //        for (int i = 0; i < nbnd; i++)
    //        {   
    //            int ibnd_loc = occupied_bands__.idx_bnd_loc[i];
    //            int ibnd = occupied_bands__.idx_bnd_glob[i];
    //            int ispn = (ibnd < parameters_.num_fv_states()) ? 0 : 1;
    //            for (int xi = 0; xi < mt_basis_size; xi++)
    //            {
    //                wf1(xi, i, ispn) = std::conj(*kp__->spinor_wave_functions(ispn).at<CPU>(offset_wf + xi, ibnd_loc));
    //                wf2(xi, i, ispn) = (*kp__->spinor_wave_functions(ispn).at<CPU>(offset_wf + xi, ibnd_loc)) * occupied_bands__.weight[i];
    //            }
    //        }

    //    }

    //    for (int j = 0; j < (int)density_matrix__.size(2); j++)
    //    {
    //        linalg<CPU>::gemm(0, 1, mt_basis_size, mt_basis_size, nbnd, complex_one, 
    //                          &wf1(0, 0, dmat_spins_[j].first), wf1.ld(), 
    //                          &wf2(0, 0, dmat_spins_[j].second), wf2.ld(), complex_one, 
    //                          density_matrix__.at<CPU>(0, 0, j, ia), density_matrix__.ld());
    //    }
    //}

}

template <>
void Density::add_k_point_contribution<ultrasoft_pseudopotential>(K_point* kp__,
                                                                  occupied_bands_descriptor const& occupied_bands__,
                                                                  mdarray<double_complex, 4>& density_matrix__)
{
    PROFILE();

    Timer t("sirius::Density::add_k_point_contribution");

    int nbnd = occupied_bands__.num_occupied_bands();
    int nbnd_loc = occupied_bands__.num_occupied_bands_local();

    if (!nbnd) return;

    kp__->beta_projectors().allocate_workspace();
    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        kp__->fv_states<false>().allocate_on_device();
        kp__->fv_states<false>().copy_to_device(0, nbnd);
    }
    #endif

    for (int chunk = 0; chunk < kp__->beta_projectors().num_beta_chunks(); chunk++)
    {
        kp__->beta_projectors().generate(chunk);
        kp__->beta_projectors().inner(chunk, kp__->fv_states<false>(), 0, nbnd);
        int nbeta = kp__->beta_projectors().beta_chunk(chunk).num_beta_;

        mdarray<double_complex, 2> beta_psi(const_cast<double_complex*>(kp__->beta_projectors().beta_phi().at<CPU>()), nbeta, nbnd);

        if (nbnd_loc) // TODO: this part can also be moved to GPU
        {
            #pragma omp parallel
            {
                /* auxiliary arrays */
                mdarray<double_complex, 2> bp1(nbeta, nbnd_loc);
                mdarray<double_complex, 2> bp2(nbeta, nbnd_loc);
                #pragma omp for
                for (int ia = 0; ia < kp__->beta_projectors().beta_chunk(chunk).num_atoms_; ia++)
                {
                    int nbf = kp__->beta_projectors().beta_chunk(chunk).desc_(0, ia);
                    int offs = kp__->beta_projectors().beta_chunk(chunk).desc_(1, ia);
                    int ja = kp__->beta_projectors().beta_chunk(chunk).desc_(3, ia);

                    for (int i = 0; i < nbnd_loc; i++)
                    {
                        int j = occupied_bands__.idx_bnd_glob[i];
                        for (int xi = 0; xi < nbf; xi++)
                        {
                            bp1(xi, i) = beta_psi(offs + xi, j);
                            bp2(xi, i) = std::conj(bp1(xi, i)) * kp__->band_occupancy(j) * kp__->weight();
                        }
                    }

                    linalg<CPU>::gemm(0, 1, nbf, nbf, nbnd_loc, complex_one, &bp1(0, 0), bp1.ld(),
                                      &bp2(0, 0), bp2.ld(), complex_one, &density_matrix__(0, 0, 0, ja), 
                                      density_matrix__.ld());
                }
            }
        }

    }
    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        kp__->fv_states<false>().deallocate_on_device();
    }
    #endif
    kp__->beta_projectors().deallocate_workspace();
}

//#ifdef __GPU

//== extern "C" void copy_beta_psi_gpu(int nbf,
//==                                   int nloc,
//==                                   double_complex const* beta_psi,
//==                                   int beta_psi_ld,
//==                                   double const* wo,
//==                                   double_complex* beta_psi_wo,
//==                                   int beta_psi_wo_ld,
//==                                   int stream_id);

//== template <>
//== void Density::add_k_point_contribution<GPU, ultrasoft_pseudopotential>(K_point* kp__,
//==                                                                        occupied_bands_descriptor const& occupied_bands__,
//==                                                                        mdarray<double_complex, 4>& density_matrix__)
//== {
//==     Timer t("sirius::Density::add_k_point_contribution");
//== 
//==     int nbnd = num_occupied_bands(kp__);
//== 
//==     if (nbnd == 0) return;
//== 
//==     /* compute <beta|Psi> */
//==     Timer t1("sirius::Density::add_k_point_contribution|beta_psi");
//==     
//==     matrix<double_complex> beta_psi(unit_cell_.mt_basis_size(), nbnd);
//==     beta_psi.allocate_on_device();
//== 
//==     kp__->beta_gk().allocate_on_device();
//==     kp__->beta_gk().copy_to_device();
//== 
//==     kp__->fv_states_slab().allocate_on_device();
//==     kp__->fv_states_slab().copy_to_device();
//== 
//==     linalg<GPU>::gemm(2, 0, unit_cell_.mt_basis_size(), nbnd, kp__->num_gkvec_loc(),
//==                       kp__->beta_gk().at<GPU>(), kp__->beta_gk().ld(),
//==                       kp__->fv_states_slab().at<GPU>(), kp__->fv_states_slab().ld(),
//==                       beta_psi.at<GPU>(), beta_psi.ld()); 
//== 
//==     beta_psi.copy_to_host();
//==     kp__->comm().allreduce(&beta_psi(0, 0), (int)beta_psi.size());
//==     t1.stop();
//==     
//==     kp__->beta_gk().deallocate_on_device();
//==     kp__->fv_states_slab().deallocate_on_device();
//== 
//==     splindex<block> spl_bands(nbnd, kp__->comm().size(), kp__->comm().rank());
//== 
//==     if (spl_bands.local_size())
//==     {
//==         #pragma omp parallel
//==         {
//==             /* auxiliary arrays */
//==             mdarray<double_complex, 2> bp1(unit_cell_.max_mt_basis_size(), spl_bands.local_size());
//==             mdarray<double_complex, 2> bp2(unit_cell_.max_mt_basis_size(), spl_bands.local_size());
//==             #pragma omp for
//==             for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==             {   
//==                 /* number of beta functions for a given atom */
//==                 int nbf = unit_cell_.atom(ia)->mt_basis_size();
//== 
//==                 for (int i = 0; i < (int)spl_bands.local_size(); i++)
//==                 {
//==                     int j = (int)spl_bands[i];
//==                     for (int xi = 0; xi < nbf; xi++)
//==                     {
//==                         bp1(xi, i) = beta_psi(unit_cell_.atom(ia)->offset_lo() + xi, j);
//==                         bp2(xi, i) = conj(bp1(xi, i)) * kp__->band_occupancy(j) * kp__->weight();
//==                     }
//==                 }
//== 
//==                 linalg<CPU>::gemm(0, 1, nbf, nbf, (int)spl_bands.local_size(), complex_one, &bp1(0, 0), bp1.ld(),
//==                                   &bp2(0, 0), bp2.ld(), complex_one, &density_matrix__(0, 0, 0, ia), 
//==                                   density_matrix__.ld());
//==             }
//==         }
//==     }
//== 
//== 
//== 
//== 
//== 
//== 
//== 
//== 
//== 
//== 
//== 
//== 
//== 
//== 
//== 
//== 
//== 
//== 
//==     //== auto& psi = kp__->fv_states_panel();
//==     //== 
//==     //== mdarray<double, 1> wo(psi.num_cols_local());
//==     //== int nloc = 0;
//==     //== for (int jloc = 0; jloc < psi.num_cols_local(); jloc++)
//==     //== {
//==     //==     int j = psi.icol(jloc);
//==     //==     double d = kp__->band_occupancy(j) * kp__->weight();
//==     //==     if (d > 1e-14) wo(nloc++) = d;
//==     //== }
//==     //== if (!nloc) return;
//== 
//==     //== wo.allocate_on_device();
//==     //== wo.copy_to_device();
//== 
//==     //== auto uc = parameters_.unit_cell();
//==     //== 
//==     //== /* allocate space for <beta|psi> array */
//==     //== int nbf_max = 0;
//==     //== for (int ib = 0; ib < unit_cell_.num_beta_chunks(); ib++)
//==     //==     nbf_max  = std::max(nbf_max, unit_cell_.beta_chunk(ib).num_beta_);
//== 
//==     //== mdarray<double_complex, 1> beta_psi_tmp(nbf_max * nloc);
//==     //== beta_psi_tmp.allocate_on_device();
//== 
//==     //== matrix<double_complex> beta_gk(nullptr, kp__->num_gkvec_row(), nbf_max);
//==     //== beta_gk.allocate_on_device();
//== 
//==     //== matrix<double_complex> psi_occ(&psi(0, 0), kp__->num_gkvec_row(), nloc);
//==     //== psi_occ.allocate_on_device();
//==     //== psi_occ.copy_to_device();
//== 
//==     //== mdarray<double_complex, 3> tmp(nullptr, unit_cell_.max_mt_basis_size(), nloc, Platform::max_num_threads());
//==     //== tmp.allocate_on_device();
//== 
//==     //== for (int ib = 0; ib < unit_cell_.num_beta_chunks(); ib++)
//==     //== {
//==     //==     /* wrapper for <beta|psi> with required dimensions */
//==     //==     matrix<double_complex> beta_psi(beta_psi_tmp.at<CPU>(), beta_psi_tmp.at<GPU>(), unit_cell_.beta_chunk(ib).num_beta_, nloc);
//== 
//==     //==     kp__->generate_beta_gk(unit_cell_.beta_chunk(ib).num_atoms_, unit_cell_.beta_chunk(ib).atom_pos_, unit_cell_.beta_chunk(ib).desc_, beta_gk);
//==     //==     kp__->generate_beta_phi(unit_cell_.beta_chunk(ib).num_beta_, psi_occ, nloc, 0, beta_gk, beta_psi);
//== 
//==     //==     double_complex alpha(1, 0);
//== 
//==     //==     #pragma omp parallel for
//==     //==     for (int i = 0; i < unit_cell_.beta_chunk(ib).num_atoms_; i++)
//==     //==     {
//==     //==         /* number of beta functions for a given atom */
//==     //==         int nbf = unit_cell_.beta_chunk(ib).desc_(0, i);
//==     //==         int ofs = unit_cell_.beta_chunk(ib).desc_(1, i);
//==     //==         int ia = unit_cell_.beta_chunk(ib).desc_(3, i);
//==     //==         int thread_id = Platform::thread_id();
//==     //==         
//==     //==         copy_beta_psi_gpu(nbf,
//==     //==                           nloc,
//==     //==                           beta_psi.at<GPU>(ofs, 0),
//==     //==                           beta_psi.ld(),
//==     //==                           wo.at<GPU>(),
//==     //==                           tmp.at<GPU>(0, 0, thread_id),
//==     //==                           tmp.ld(),
//==     //==                           thread_id);
//==     //==         
//==     //==         linalg<GPU>::gemm(0, 1, nbf, nbf, nloc, &alpha, beta_psi.at<GPU>(ofs, 0), beta_psi.ld(),
//==     //==                           tmp.at<GPU>(0, 0, thread_id), tmp.ld(), &alpha, 
//==     //==                           pp_complex_density_matrix__.at<GPU>(0, 0, 0, ia), pp_complex_density_matrix__.ld(), thread_id);
//==     //==     }
//==     //==     cuda_device_synchronize();
//==     //== }
//== 
//==     //== tmp.deallocate_on_device();
//==     //== psi_occ.deallocate_on_device();
//== }
//== #endif

};
