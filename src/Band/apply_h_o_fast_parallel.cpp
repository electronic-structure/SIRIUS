#include "band.h"

namespace sirius {

void Band::apply_h_local_parallel(K_point* kp__,
                                  std::vector<double> const& effective_potential__,
                                  std::vector<double> const& pw_ekin__,
                                  int nst__,
                                  dmatrix<double_complex>& phi__,
                                  dmatrix<double_complex>& hphi__)
{
    PROFILE();

    Timer t("sirius::Band::apply_h_local_parallel");

    splindex<block> spl_gkvec(kp__->num_gkvec(), kp__->num_ranks_row(), kp__->rank_row());

    assert(phi__.num_rows_local() == (int)spl_gkvec.local_size());
    assert(hphi__.num_rows_local() == (int)spl_gkvec.local_size());

    auto a2a = kp__->comm_row().map_alltoall(spl_gkvec.counts(), kp__->pgkvec().counts());
    std::vector<double_complex> buf(kp__->pgkvec().num_gvec_loc());
    std::vector<double_complex> htmp(hphi__.num_rows_local());

    int offs = ctx_.pfft_coarse()->size(0) * ctx_.pfft_coarse()->size(1) * ctx_.pfft_coarse()->offset_z();

    for (int i = 0; i < nst__; i++)
    {
        /* redistribute plane-wave coefficients between slabs of FFT buffer */
        kp__->comm_row().alltoall(&phi__(0, i), &a2a.sendcounts[0], &a2a.sdispls[0], &buf[0],
                                  &a2a.recvcounts[0], &a2a.rdispls[0]);
        /* load local part of coefficients into local part of FFT buffer */
        ctx_.pfft_coarse()->input(kp__->pgkvec().num_gvec_loc(), kp__->pgkvec().index_map(), &buf[0]);
        /* transform to real space */
        ctx_.pfft_coarse()->transform(1);
        /* multiply by effective potential */
        for (int ir = 0; ir < ctx_.pfft_coarse()->local_size(); ir++) ctx_.pfft_coarse()->buffer(ir) *= effective_potential__[offs + ir];
        /* transform back to reciprocal space */
        ctx_.pfft_coarse()->transform(-1);
        /* gather pw coefficients in the temporary buffer */
        ctx_.pfft_coarse()->output(kp__->pgkvec().num_gvec_loc(), kp__->pgkvec().index_map(), &buf[0]);
        /* redistribute uniformly local sets of coefficients */
        kp__->comm_row().alltoall(&buf[0], &a2a.recvcounts[0], &a2a.rdispls[0], &htmp[0], &a2a.sendcounts[0], &a2a.sdispls[0]);
        
        for (int igk = 0; igk < (int)spl_gkvec.local_size(); igk++) hphi__(igk, i) = htmp[igk] + phi__(igk, i) * pw_ekin__[spl_gkvec[igk]];
    }
}



void Band::apply_h_o_fast_parallel(K_point* kp__,
                                   std::vector<double> const& effective_potential__,
                                   std::vector<double> const& pw_ekin__,
                                   int N__,
                                   int n__,
                                   dmatrix<double_complex>& phi_slice__,
                                   dmatrix<double_complex>& phi_slab__,
                                   dmatrix<double_complex>& hphi_slab__,
                                   dmatrix<double_complex>& ophi_slab__,
                                   mdarray<int, 1>& packed_mtrx_offset__,
                                   mdarray<double_complex, 1>& d_mtrx_packed__,
                                   mdarray<double_complex, 1>& q_mtrx_packed__,
                                   mdarray<double_complex, 1>& kappa__)
{
    PROFILE();

    Timer t("sirius::Band::apply_h_o_fast_parallel", kp__->comm());

    if (ctx_.pfft_coarse()->parallel())
    {
        int bs = (int)splindex_base::block_size(kp__->num_gkvec(), kp__->num_ranks_row());
        dmatrix<double_complex> phi_tmp(kp__->num_gkvec(), n__, kp__->blacs_grid(), bs, 1);
        
        linalg<CPU>::gemr2d(kp__->num_gkvec(), n__, phi_slab__, 0, N__, phi_tmp, 0, 0, kp__->blacs_grid().context());
        /* this is how n wave-functions are distributed */
        splindex<block_cyclic> spl_phi(n__, kp__->num_ranks_col(), kp__->rank_col(), 1);
        if (spl_phi.local_size())
        {
            apply_h_local_parallel(kp__, effective_potential__, pw_ekin__, (int)spl_phi.local_size(), phi_tmp, phi_tmp);
        }
        linalg<CPU>::gemr2d(kp__->num_gkvec(), n__, phi_tmp, 0, 0, hphi_slab__, 0, N__, kp__->blacs_grid().context());
    }
    else
    {
        /* this is how n wave-functions are distributed */
        splindex<block_cyclic> spl_phi(n__, kp__->num_ranks(), kp__->rank(), 1);
        /* collect slices of wave-functions */
        linalg<CPU>::gemr2d(kp__->num_gkvec(), n__, phi_slab__, 0, N__, phi_slice__, 0, 0, kp__->blacs_grid().context());
        if (spl_phi.local_size())
        {
            apply_h_local_slice(kp__, effective_potential__, pw_ekin__, (int)spl_phi.local_size(), phi_slice__.panel(), phi_slice__.panel());
        }
        /* collect slabs of hphi */
        linalg<CPU>::gemr2d(kp__->num_gkvec(), n__, phi_slice__, 0, 0, hphi_slab__, 0, N__, kp__->blacs_grid().context());
    }

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

        kp__->generate_beta_phi(nbeta, phi_slab__.panel(), n__, N__, beta_gk, beta_phi);

        kp__->add_non_local_contribution(natoms, nbeta, unit_cell_.beta_chunk(ib).desc_, beta_gk, d_mtrx_packed__,
                                         packed_mtrx_offset__, beta_phi, hphi_slab__.panel(), n__, N__, complex_one, work);
        
        kp__->add_non_local_contribution(natoms, nbeta, unit_cell_.beta_chunk(ib).desc_, beta_gk, q_mtrx_packed__,
                                         packed_mtrx_offset__, beta_phi, ophi_slab__.panel(), n__, N__, complex_one, work);
        
        offs += nbeta;
    }

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif
}

};
