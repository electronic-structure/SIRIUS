#include "band.h"

namespace sirius {

#ifdef __SCALAPACK
/** \param [in] phi Input wave-functions [storage: CPU || GPU].
 *  \param [out] op_phi Result of application of operator to the wave-functions [storage: CPU || GPU].
 */
void Band::add_non_local_contribution_parallel(K_point* kp__,
                                               int N__,
                                               int n__,
                                               dmatrix<double_complex>& phi__,
                                               dmatrix<double_complex>& op_phi__, 
                                               matrix<double_complex>& beta_gk__,
                                               mdarray<int, 1> const& packed_mtrx_offset__,
                                               mdarray<double_complex, 1>& op_mtrx_packed__,
                                               double_complex alpha)
{
    PROFILE();

    Timer t("sirius::Band::add_non_local_contribution_parallel");

    /* beginning of the band index */
    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    /* end of the band index */
    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());

    /* local number of states to which Hamiltonian has to be applied */
    int nloc = static_cast<int>(s1.local_size() - s0.local_size());

    if (!nloc) return;

    /* allocate space for <beta|phi> array */
    int nbf_max = unit_cell_.max_mt_basis_size() * unit_cell_.beta_chunk(0).num_atoms_;
    mdarray<double_complex, 1> beta_phi_tmp(nbf_max * nloc);

    /* result of atom-block-diagonal operator multiplied by <beta|phi> */
    matrix<double_complex> tmp(nbf_max, nloc);

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        beta_phi_tmp.allocate_on_device();
        tmp.allocate_on_device();
    }
    #endif

    for (int ib = 0; ib < unit_cell_.num_beta_chunks(); ib++)
    {
        /* number of beta-projectors in the current chunk */
        int nbeta =  unit_cell_.beta_chunk(ib).num_beta_;
        int natoms = unit_cell_.beta_chunk(ib).num_atoms_;

        /* wrapper for <beta|phi> with required dimensions */
        matrix<double_complex> beta_phi;
        switch (parameters_.processing_unit())
        {
            case CPU:
            {
                beta_phi = matrix<double_complex>(beta_phi_tmp.at<CPU>(), nbeta, nloc);
                break;
            }
            case GPU:
            {
                beta_phi = matrix<double_complex>(beta_phi_tmp.at<CPU>(), beta_phi_tmp.at<GPU>(), nbeta, nloc);
                break;
            }
        }

        //Timer t1("sirius::Band::add_non_local_contribution_parallel|beta_phi", kp__->comm_row());
        kp__->generate_beta_gk(natoms, unit_cell_.beta_chunk(ib).atom_pos_, unit_cell_.beta_chunk(ib).desc_, beta_gk__);
        kp__->generate_beta_phi(nbeta, phi__.panel(), nloc, (int)s0.local_size(), beta_gk__, beta_phi);
        //double tval = t1.stop();

        //if (verbosity_level >= 6 && kp__->comm().rank() == 0)
        //{
        //    printf("<beta|phi> effective zgemm with M, N, K: %6i %6i %6i, %12.4f sec, %12.4f GFlops/rank\n",
        //           nbeta, nloc, kp__->num_gkvec(),
        //           tval, 8e-9 * nbeta * nloc * kp__->num_gkvec() / tval / kp__->num_ranks_row());
        //}

        kp__->add_non_local_contribution(natoms, nbeta, unit_cell_.beta_chunk(ib).desc_, beta_gk__, op_mtrx_packed__,
                                         packed_mtrx_offset__, beta_phi, op_phi__.panel(), nloc, (int)s0.local_size(),
                                         alpha, tmp);
    }
    #ifdef __GPU
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif
}
#endif // __SCALAPACK

};

