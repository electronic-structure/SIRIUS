#include "density.h"

namespace sirius {

template <>
void Density::add_k_point_contribution<CPU, full_potential_lapwlo>(K_point* kp__,
                                                                   std::vector< std::pair<int, double> >& occupied_bands__,
                                                                   mdarray<double_complex, 4>& density_matrix__)
{
    Timer t("sirius::Density::add_k_point_contribution");
    
    if (occupied_bands__.size() == 0) return;
   
    mdarray<double_complex, 3> wf1(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands__.size(), parameters_.num_spins());
    mdarray<double_complex, 3> wf2(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands__.size(), parameters_.num_spins());

    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        int offset_wf = parameters_.unit_cell()->atom(ia)->offset_wf();
        int mt_basis_size = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
        
        for (int i = 0; i < (int)occupied_bands__.size(); i++)
        {
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                for (int j = 0; j < mt_basis_size; j++)
                {
                    wf1(j, i, ispn) = conj(kp__->spinor_wave_function(offset_wf + j, occupied_bands__[i].first, ispn));
                    wf2(j, i, ispn) = kp__->spinor_wave_function(offset_wf + j, occupied_bands__[i].first, ispn) * occupied_bands__[i].second;
                }
            }
        }

        for (int j = 0; j < (int)density_matrix__.size(2); j++)
        {
            linalg<CPU>::gemm(0, 1, mt_basis_size, mt_basis_size, (int)occupied_bands__.size(), complex_one, 
                              &wf1(0, 0, dmat_spins_[j].first), wf1.ld(), 
                              &wf2(0, 0, dmat_spins_[j].second), wf2.ld(), complex_one, 
                              density_matrix__.at<CPU>(0, 0, j, ia), density_matrix__.ld());
        }
    }

}

template <>
void Density::add_k_point_contribution<CPU, ultrasoft_pseudopotential>(K_point* kp__,
                                                                       std::vector< std::pair<int, double> >& occupied_bands__,
                                                                       mdarray<double_complex, 4>& density_matrix__)
{
    Timer t("sirius::Density::add_k_point_contribution");

    int nbnd = num_occupied_bands(kp__);

    if (nbnd == 0) return;

    auto uc = parameters_.unit_cell();

    /* compute <beta|Psi> */
    Timer t1("sirius::Density::add_k_point_contribution|beta_psi");
    matrix<double_complex> beta_psi(uc->mt_basis_size(), nbnd);
    linalg<CPU>::gemm(2, 0, uc->mt_basis_size(), nbnd, kp__->num_gkvec_loc(), complex_one, 
                      kp__->beta_gk(), kp__->fv_states_slab(), complex_zero, beta_psi);
    kp__->comm().allreduce(&beta_psi(0, 0), (int)beta_psi.size());
    t1.stop();

    splindex<block> spl_bands(nbnd, kp__->comm().size(), kp__->comm().rank());

    if (spl_bands.local_size())
    {
        #pragma omp parallel
        {
            /* auxiliary arrays */
            mdarray<double_complex, 2> bp1(uc->max_mt_basis_size(), spl_bands.local_size());
            mdarray<double_complex, 2> bp2(uc->max_mt_basis_size(), spl_bands.local_size());
            #pragma omp for
            for (int ia = 0; ia < uc->num_atoms(); ia++)
            {   
                /* number of beta functions for a given atom */
                int nbf = uc->atom(ia)->mt_basis_size();

                for (int i = 0; i < (int)spl_bands.local_size(); i++)
                {
                    int j = (int)spl_bands[i];
                    for (int xi = 0; xi < nbf; xi++)
                    {
                        bp1(xi, i) = beta_psi(uc->atom(ia)->offset_lo() + xi, j);
                        bp2(xi, i) = conj(bp1(xi, i)) * kp__->band_occupancy(j) * kp__->weight();
                    }
                }

                linalg<CPU>::gemm(0, 1, nbf, nbf, (int)spl_bands.local_size(), complex_one, &bp1(0, 0), bp1.ld(),
                                  &bp2(0, 0), bp2.ld(), complex_one, &density_matrix__(0, 0, 0, ia), 
                                  density_matrix__.ld());
            }
        }
    }
}

#ifdef _GPU_
extern "C" void copy_beta_psi_gpu(int nbf,
                                  int nloc,
                                  double_complex const* beta_psi,
                                  int beta_psi_ld,
                                  double const* wo,
                                  double_complex* beta_psi_wo,
                                  int beta_psi_wo_ld,
                                  int stream_id);

template <>
void Density::add_k_point_contribution<GPU, ultrasoft_pseudopotential>(K_point* kp__,
                                                                       std::vector< std::pair<int, double> >& occupied_bands__,
                                                                       mdarray<double_complex, 4>& density_matrix__)
{
    Timer t("sirius::Density::add_k_point_contribution");

    int nbnd = num_occupied_bands(kp__);

    if (nbnd == 0) return;

    auto uc = parameters_.unit_cell();

    /* compute <beta|Psi> */
    Timer t1("sirius::Density::add_k_point_contribution|beta_psi");
    
    matrix<double_complex> beta_psi(uc->mt_basis_size(), nbnd);
    beta_psi.allocate_on_device();

    kp__->beta_gk().allocate_on_device();
    kp__->beta_gk().copy_to_device();

    kp__->fv_states_slab().allocate_on_device();
    kp__->fv_states_slab().copy_to_device();

    linalg<GPU>::gemm(2, 0, uc->mt_basis_size(), nbnd, kp__->num_gkvec_loc(),
                      kp__->beta_gk().at<GPU>(), kp__->beta_gk().ld(),
                      kp__->fv_states_slab().at<GPU>(), kp__->fv_states_slab().ld(),
                      beta_psi.at<GPU>(), beta_psi.ld()); 

    beta_psi.copy_to_host();
    kp__->comm().allreduce(&beta_psi(0, 0), (int)beta_psi.size());
    t1.stop();
    
    kp__->beta_gk().deallocate_on_device();
    kp__->fv_states_slab().deallocate_on_device();

    splindex<block> spl_bands(nbnd, kp__->comm().size(), kp__->comm().rank());

    if (spl_bands.local_size())
    {
        #pragma omp parallel
        {
            /* auxiliary arrays */
            mdarray<double_complex, 2> bp1(uc->max_mt_basis_size(), spl_bands.local_size());
            mdarray<double_complex, 2> bp2(uc->max_mt_basis_size(), spl_bands.local_size());
            #pragma omp for
            for (int ia = 0; ia < uc->num_atoms(); ia++)
            {   
                /* number of beta functions for a given atom */
                int nbf = uc->atom(ia)->mt_basis_size();

                for (int i = 0; i < (int)spl_bands.local_size(); i++)
                {
                    int j = (int)spl_bands[i];
                    for (int xi = 0; xi < nbf; xi++)
                    {
                        bp1(xi, i) = beta_psi(uc->atom(ia)->offset_lo() + xi, j);
                        bp2(xi, i) = conj(bp1(xi, i)) * kp__->band_occupancy(j) * kp__->weight();
                    }
                }

                linalg<CPU>::gemm(0, 1, nbf, nbf, (int)spl_bands.local_size(), complex_one, &bp1(0, 0), bp1.ld(),
                                  &bp2(0, 0), bp2.ld(), complex_one, &density_matrix__(0, 0, 0, ia), 
                                  density_matrix__.ld());
            }
        }
    }


















    //== auto& psi = kp__->fv_states_panel();
    //== 
    //== mdarray<double, 1> wo(psi.num_cols_local());
    //== int nloc = 0;
    //== for (int jloc = 0; jloc < psi.num_cols_local(); jloc++)
    //== {
    //==     int j = psi.icol(jloc);
    //==     double d = kp__->band_occupancy(j) * kp__->weight();
    //==     if (d > 1e-14) wo(nloc++) = d;
    //== }
    //== if (!nloc) return;

    //== wo.allocate_on_device();
    //== wo.copy_to_device();

    //== auto uc = parameters_.unit_cell();
    //== 
    //== /* allocate space for <beta|psi> array */
    //== int nbf_max = 0;
    //== for (int ib = 0; ib < uc->num_beta_chunks(); ib++)
    //==     nbf_max  = std::max(nbf_max, uc->beta_chunk(ib).num_beta_);

    //== mdarray<double_complex, 1> beta_psi_tmp(nbf_max * nloc);
    //== beta_psi_tmp.allocate_on_device();

    //== matrix<double_complex> beta_gk(nullptr, kp__->num_gkvec_row(), nbf_max);
    //== beta_gk.allocate_on_device();

    //== matrix<double_complex> psi_occ(&psi(0, 0), kp__->num_gkvec_row(), nloc);
    //== psi_occ.allocate_on_device();
    //== psi_occ.copy_to_device();

    //== mdarray<double_complex, 3> tmp(nullptr, uc->max_mt_basis_size(), nloc, Platform::max_num_threads());
    //== tmp.allocate_on_device();

    //== for (int ib = 0; ib < uc->num_beta_chunks(); ib++)
    //== {
    //==     /* wrapper for <beta|psi> with required dimensions */
    //==     matrix<double_complex> beta_psi(beta_psi_tmp.at<CPU>(), beta_psi_tmp.at<GPU>(), uc->beta_chunk(ib).num_beta_, nloc);

    //==     kp__->generate_beta_gk(uc->beta_chunk(ib).num_atoms_, uc->beta_chunk(ib).atom_pos_, uc->beta_chunk(ib).desc_, beta_gk);
    //==     kp__->generate_beta_phi(uc->beta_chunk(ib).num_beta_, psi_occ, nloc, 0, beta_gk, beta_psi);

    //==     double_complex alpha(1, 0);

    //==     #pragma omp parallel for
    //==     for (int i = 0; i < uc->beta_chunk(ib).num_atoms_; i++)
    //==     {
    //==         /* number of beta functions for a given atom */
    //==         int nbf = uc->beta_chunk(ib).desc_(0, i);
    //==         int ofs = uc->beta_chunk(ib).desc_(1, i);
    //==         int ia = uc->beta_chunk(ib).desc_(3, i);
    //==         int thread_id = Platform::thread_id();
    //==         
    //==         copy_beta_psi_gpu(nbf,
    //==                           nloc,
    //==                           beta_psi.at<GPU>(ofs, 0),
    //==                           beta_psi.ld(),
    //==                           wo.at<GPU>(),
    //==                           tmp.at<GPU>(0, 0, thread_id),
    //==                           tmp.ld(),
    //==                           thread_id);
    //==         
    //==         linalg<GPU>::gemm(0, 1, nbf, nbf, nloc, &alpha, beta_psi.at<GPU>(ofs, 0), beta_psi.ld(),
    //==                           tmp.at<GPU>(0, 0, thread_id), tmp.ld(), &alpha, 
    //==                           pp_complex_density_matrix__.at<GPU>(0, 0, 0, ia), pp_complex_density_matrix__.ld(), thread_id);
    //==     }
    //==     cuda_device_synchronize();
    //== }

    //== tmp.deallocate_on_device();
    //== psi_occ.deallocate_on_device();
}
#endif

};
