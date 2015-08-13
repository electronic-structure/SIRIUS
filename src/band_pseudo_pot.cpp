#include <thread>
#include <mutex>
#include "band.h"
#include "debug.hpp"

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

/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 */
void Band::apply_h_serial(K_point* kp__, 
                          std::vector<double> const& effective_potential__, 
                          std::vector<double> const& pw_ekin__, 
                          int N__,
                          int n__,
                          matrix<double_complex>& phi__,
                          matrix<double_complex>& hphi__,
                          mdarray<double_complex, 1>& kappa__,
                          mdarray<int, 1>& packed_mtrx_offset__,
                          mdarray<double_complex, 1>& d_mtrx_packed__)
{
    Timer t("sirius::Band::apply_h_serial");

    matrix<double_complex> phi, hphi;
    
    /* if temporary array is allocated, this would be the only big array on GPU */
    bool economize_gpu_memory = (kappa__.size() != 0);

    if (parameters_.processing_unit() == CPU)
    {
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    {
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__),  phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), hphi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (parameters_.processing_unit() == GPU && economize_gpu_memory)
    {
        double_complex* gpu_ptr = kappa__.at<GPU>(kp__->num_gkvec() * unit_cell_.mt_basis_size());
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
    }
    
    /* apply local part of Hamiltonian */
    apply_h_local_serial(kp__, effective_potential__, pw_ekin__, n__, phi, hphi);

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    {
        /* copy hphi do device */
        hphi.copy_to_device();
    }
    #endif

    add_non_local_contribution_serial(kp__, N__, n__, phi__, hphi__, kappa__, packed_mtrx_offset__, d_mtrx_packed__, complex_one);
}

/** \param [in] phi Input wave-functions [storage: CPU || GPU].
 *  \param [out] op_phi Result of application of operator to the wave-functions [storage: CPU || GPU].
 */
void Band::add_non_local_contribution_serial(K_point* kp__,
                                             int N__,
                                             int n__,
                                             matrix<double_complex>& phi__,
                                             matrix<double_complex>& op_phi__, 
                                             mdarray<double_complex, 1>& kappa__,
                                             mdarray<int, 1> const& packed_mtrx_offset__,
                                             mdarray<double_complex, 1>& op_mtrx_packed__,
                                             double_complex alpha)
{
    PROFILE();

    Timer t("sirius::Band::add_non_local_contribution_serial");

    matrix<double_complex> phi, op_phi, beta_gk;
    
    /* if temporary array is allocated, this would be the only big array on GPU */
    bool economize_gpu_memory = (kappa__.size() != 0);

    if (parameters_.processing_unit() == CPU)
    {
        phi    = matrix<double_complex>(   phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
        op_phi = matrix<double_complex>(op_phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    {
        phi    = matrix<double_complex>(   phi__.at<CPU>(0, N__),    phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
        op_phi = matrix<double_complex>(op_phi__.at<CPU>(0, N__), op_phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (parameters_.processing_unit() == GPU && economize_gpu_memory)
    {
        double_complex* gpu_ptr = kappa__.at<GPU>(kp__->num_gkvec() * unit_cell_.mt_basis_size());
        phi    = matrix<double_complex>(   phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
        op_phi = matrix<double_complex>(op_phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
        beta_gk = matrix<double_complex>(nullptr, kappa__.at<GPU>(), kp__->num_gkvec(), unit_cell_.mt_basis_size());
    }
    
    /* <\beta_{\xi}^{\alpha}|\phi_j> */
    matrix<double_complex> beta_phi(unit_cell_.mt_lo_basis_size(), n__);

    /* operator multiplied by <\beta_{\xi}^{\alpha}|\phi_j> */
    matrix<double_complex> work(unit_cell_.mt_lo_basis_size(), n__);

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        beta_phi.allocate_on_device();
        work.allocate_on_device();
    }
    #endif

    if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && !economize_gpu_memory))
    {
        kp__->generate_beta_phi(unit_cell_.mt_lo_basis_size(), phi, n__, 0, kp__->beta_gk(), beta_phi);

        kp__->add_non_local_contribution(unit_cell_.num_atoms(), unit_cell_.mt_lo_basis_size(), unit_cell_.beta_chunk(0).desc_,
                                         kp__->beta_gk(), op_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         op_phi, n__, 0, alpha, work);
    }
    else
    {
        #ifdef __GPU
        kp__->generate_beta_gk(unit_cell_.num_atoms(), unit_cell_.beta_chunk(0).atom_pos_, unit_cell_.beta_chunk(0).desc_, beta_gk);
        phi.copy_to_device();
        kp__->generate_beta_phi(unit_cell_.mt_basis_size(), phi, n__, 0, beta_gk, beta_phi);
        
        op_phi.copy_to_device();
        kp__->add_non_local_contribution(unit_cell_.num_atoms(), unit_cell_.mt_basis_size(), unit_cell_.beta_chunk(0).desc_,
                                         beta_gk, op_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         op_phi, n__, 0, alpha, work);
        op_phi.copy_to_host();
        #else
        TERMINATE_NO_GPU
        #endif
    }
    #ifdef __GPU
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif
}

};

