#include "band.h"

namespace sirius {

void Band::apply_h_local_parallel(K_point* kp__,
                                  std::vector<double> const& effective_potential__,
                                  std::vector<double> const& pw_ekin__,
                                  dmatrix<double_complex>& phi__,
                                  dmatrix<double_complex>& hphi__)
{
    splindex<block> spl_gkvec(kp__->num_gkvec(), kp__->comm_row().size(), kp__->comm_row().rank());

    auto a2a = kp__->comm_row().map_alltoall(spl_gkvec.counts(), kp__->pgkvec().counts());
    std::vector<double_complex> buf(kp__->pgkvec().num_gvec_loc());

    int offs = ctx_.pfft_coarse()->size(0) * ctx_.pfft_coarse()->size(1) * ctx_.pfft_coarse()->offset_z();

    for (int i = 0; i < phi__.num_cols_local(); i++)
    {
        kp__->comm_row().alltoall(&phi__(0, i), &a2a.sendcounts[0], &a2a.sdispls[0], &buf[0],
                                  &a2a.recvcounts[0], &a2a.rdispls[0]);
        ctx_.pfft_coarse()->input(kp__->pgkvec().num_gvec_loc(), kp__->pgkvec().index_map(), &buf[0]);
        ctx_.pfft_coarse()->transform(1);
        for (int ir = 0; ir < ctx_.pfft_coarse()->local_size(); ir++) ctx_.pfft_coarse()->buffer(ir) *= effective_potential__[offs + ir];
        ctx_.pfft_coarse()->transform(-1);
        ctx_.pfft_coarse()->output(kp__->pgkvec().num_gvec_loc(), kp__->pgkvec().index_map(), &buf[0]);
        
        kp__->comm_row().alltoall(&buf[0], &a2a.recvcounts[0], &a2a.rdispls[0], &hphi__(0, i),
                                  &a2a.sendcounts[0], &a2a.sdispls[0]);
        
        for (int igk = 0; igk < (int)spl_gkvec.local_size(); igk++) hphi__(igk, i) += phi__(igk, i) * pw_ekin__[spl_gkvec[igk]];
    }
}



void Band::apply_h_o_fast_parallel(K_point* kp__,
                                   std::vector<double> const& effective_potential__,
                                   std::vector<double> const& pw_ekin__,
                                   int N__,
                                   int n__,
                                   matrix<double_complex>& phi_slice__,
                                   matrix<double_complex>& phi_slab__,
                                   matrix<double_complex>& hphi_slab__,
                                   matrix<double_complex>& ophi_slab__,
                                   mdarray<int, 1>& packed_mtrx_offset__,
                                   mdarray<double_complex, 1>& d_mtrx_packed__,
                                   mdarray<double_complex, 1>& q_mtrx_packed__,
                                   mdarray<double_complex, 1>& kappa__)
{
    PROFILE();

    Timer t("sirius::Band::apply_h_o_fast_parallel", kp__->comm());

    //== splindex<block> spl_phi(n__, kp__->comm().size(), kp__->comm().rank());

    //== kp__->collect_all_gkvec(spl_phi, &phi_slab__(0, N__), &phi_slice__(0, 0));
    //== 
    //== if (spl_phi.local_size())
    //==     apply_h_local_slice(kp__, effective_potential__, pw_ekin__, (int)spl_phi.local_size(), phi_slice__, phi_slice__);

    //== kp__->collect_all_bands(spl_phi, &phi_slice__(0, 0),  &hphi_slab__(0, N__));

    int bs1 = (int)splindex_base::block_size(kp__->num_gkvec(), kp__->num_ranks());
    int bs2 = (int)splindex_base::block_size(n__, kp__->num_ranks());
    dmatrix<double_complex> phi1(&phi_slab__(0, N__), kp__->num_gkvec(), n__, kp__->blacs_grid_slab(), bs1, 1);
    dmatrix<double_complex> phi2(&phi_slice__(0, 0), kp__->num_gkvec(), n__, kp__->blacs_grid_slice(), 1, bs2);
    dmatrix<double_complex> hphi1(&hphi_slab__(0, N__), kp__->num_gkvec(), n__, kp__->blacs_grid_slab(), bs1, 1);

    linalg<CPU>::gemr2d(kp__->num_gkvec(), n__, phi1, 0, 0, phi2, 0, 0, kp__->blacs_grid().context());
    if (phi2.num_cols_local())
        apply_h_local_slice(kp__, effective_potential__, pw_ekin__, phi2.num_cols_local(), phi_slice__, phi_slice__);
    linalg<CPU>::gemr2d(kp__->num_gkvec(), n__, phi2, 0, 0, hphi1, 0, 0, kp__->blacs_grid().context());

    //== int bs1 = (int)splindex_base::block_size(kp__->num_gkvec(), kp__->num_ranks());

    //== dmatrix<double_complex> phi1(&phi_slab__(0, N__), kp__->num_gkvec(), n__, kp__->blacs_grid_slab(), bs1, 1);
    //== dmatrix<double_complex> hphi1(&hphi_slab__(0, N__), kp__->num_gkvec(), n__, kp__->blacs_grid_slab(), bs1, 1);

    //== int bs2 = (int)splindex_base::block_size(kp__->num_gkvec(), kp__->num_ranks_row());
    //== dmatrix<double_complex> phi2(kp__->num_gkvec(), n__, kp__->blacs_grid(), bs2, 1);
    //== 
    //== dmatrix<double_complex> hphi2(kp__->num_gkvec(), n__, kp__->blacs_grid(), bs2, 1);


    //== linalg<CPU>::gemr2d(kp__->num_gkvec(), n__, phi1, 0, 0, phi2, 0, 0, kp__->blacs_grid().context());

    //== apply_h_local_parallel(kp__, effective_potential__, pw_ekin__, phi2, hphi2);

    //== linalg<CPU>::gemr2d(kp__->num_gkvec(), n__, hphi2, 0, 0, hphi1, 0, 0, kp__->blacs_grid().context());

    if (parameters_.processing_unit() == CPU)
    {
        /* set intial ophi */
        memcpy(&ophi_slab__(0, N__), &phi_slab__(0, N__), kp__->num_gkvec_loc() * n__ * sizeof(double_complex));
    }

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        /* copy hphi do device */
        cuda_copy_to_device(hphi_slab__.at<GPU>(0, N__), hphi_slab__.at<CPU>(0, N__),
                            kp__->num_gkvec_loc() * n__ * sizeof(double_complex));

        /* set intial ophi */
        cuda_copy_device_to_device(ophi_slab__.at<GPU>(0, N__), phi_slab__.at<GPU>(0, N__), 
                                   kp__->num_gkvec_loc() * n__ * sizeof(double_complex));
    }
    #endif

    int offs = 0;
    for (int ib = 0; ib < unit_cell_.num_beta_chunks(); ib++)
    {
        /* number of beta-projectors in the current chunk */
        int nbeta =  unit_cell_.beta_chunk(ib).num_beta_;
        int natoms = unit_cell_.beta_chunk(ib).num_atoms_;

        /* wrapper for <beta|phi> with required dimensions */
        matrix<double_complex> beta_gk;
        matrix<double_complex> beta_phi;
        matrix<double_complex> work;
        switch (parameters_.processing_unit())
        {
            case CPU:
            {
                beta_phi = matrix<double_complex>(kappa__.at<CPU>(),            nbeta, n__);
                work     = matrix<double_complex>(kappa__.at<CPU>(nbeta * n__), nbeta, n__);
                beta_gk  = matrix<double_complex>(kp__->beta_gk().at<CPU>(0, offs), kp__->num_gkvec_loc(), nbeta);
                break;
            }
            case GPU:
            {
                #ifdef __GPU
                beta_phi = matrix<double_complex>(kappa__.at<CPU>(),            kappa__.at<GPU>(),            nbeta, n__);
                work     = matrix<double_complex>(kappa__.at<CPU>(nbeta * n__), kappa__.at<GPU>(nbeta * n__), nbeta, n__);
                beta_gk  = matrix<double_complex>(kp__->beta_gk().at<CPU>(0, offs), kappa__.at<GPU>(2 * nbeta * n__), kp__->num_gkvec_loc(), nbeta);
                beta_gk.copy_to_device();
                #endif
                break;
            }
        }

        kp__->generate_beta_phi(nbeta, phi_slab__, n__, N__, beta_gk, beta_phi);

        kp__->add_non_local_contribution(natoms, nbeta, unit_cell_.beta_chunk(ib).desc_, beta_gk, d_mtrx_packed__,
                                         packed_mtrx_offset__, beta_phi, hphi_slab__, n__, N__, complex_one, work);
        
        kp__->add_non_local_contribution(natoms, nbeta, unit_cell_.beta_chunk(ib).desc_, beta_gk, q_mtrx_packed__,
                                         packed_mtrx_offset__, beta_phi, ophi_slab__, n__, N__, complex_one, work);
        
        offs += nbeta;
    }

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif
}

};
