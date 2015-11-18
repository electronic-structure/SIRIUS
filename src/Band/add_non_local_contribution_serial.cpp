#include "band.h"

namespace sirius {

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

    STOP();

    //matrix<double_complex> phi, op_phi, beta_gk;
    //
    ///* if temporary array is allocated, this would be the only big array on GPU */
    //bool economize_gpu_memory = (kappa__.size() != 0);

    //if (parameters_.processing_unit() == CPU)
    //{
    //    phi    = matrix<double_complex>(   phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
    //    op_phi = matrix<double_complex>(op_phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
    //}
    //if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    //{
    //    phi    = matrix<double_complex>(   phi__.at<CPU>(0, N__),    phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
    //    op_phi = matrix<double_complex>(op_phi__.at<CPU>(0, N__), op_phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
    //}
    //if (parameters_.processing_unit() == GPU && economize_gpu_memory)
    //{
    //    double_complex* gpu_ptr = kappa__.at<GPU>(kp__->num_gkvec() * unit_cell_.mt_basis_size());
    //    phi    = matrix<double_complex>(   phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
    //    op_phi = matrix<double_complex>(op_phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
    //    beta_gk = matrix<double_complex>(nullptr, kappa__.at<GPU>(), kp__->num_gkvec(), unit_cell_.mt_basis_size());
    //}
    //
    ///* <\beta_{\xi}^{\alpha}|\phi_j> */
    //matrix<double_complex> beta_phi(unit_cell_.mt_lo_basis_size(), n__);

    ///* operator multiplied by <\beta_{\xi}^{\alpha}|\phi_j> */
    //matrix<double_complex> work(unit_cell_.mt_lo_basis_size(), n__);

    //#ifdef __GPU
    //if (parameters_.processing_unit() == GPU)
    //{
    //    beta_phi.allocate_on_device();
    //    work.allocate_on_device();
    //}
    //#endif

    //if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && !economize_gpu_memory))
    //{
    //    kp__->generate_beta_phi(unit_cell_.mt_lo_basis_size(), phi, n__, 0, kp__->beta_gk(), beta_phi);

    //    kp__->add_non_local_contribution(unit_cell_.num_atoms(), unit_cell_.mt_lo_basis_size(), unit_cell_.beta_chunk(0).desc_,
    //                                     kp__->beta_gk(), op_mtrx_packed__, packed_mtrx_offset__, beta_phi,
    //                                     op_phi, n__, 0, alpha, work);
    //}
    //else
    //{
    //    #ifdef __GPU
    //    kp__->generate_beta_gk(unit_cell_.num_atoms(), unit_cell_.beta_chunk(0).atom_pos_, unit_cell_.beta_chunk(0).desc_, beta_gk);
    //    phi.copy_to_device();
    //    kp__->generate_beta_phi(unit_cell_.mt_basis_size(), phi, n__, 0, beta_gk, beta_phi);
    //    
    //    op_phi.copy_to_device();
    //    kp__->add_non_local_contribution(unit_cell_.num_atoms(), unit_cell_.mt_basis_size(), unit_cell_.beta_chunk(0).desc_,
    //                                     beta_gk, op_mtrx_packed__, packed_mtrx_offset__, beta_phi,
    //                                     op_phi, n__, 0, alpha, work);
    //    op_phi.copy_to_host();
    //    #else
    //    TERMINATE_NO_GPU
    //    #endif
    //}
    //#ifdef __GPU
    //if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    //#endif
}

};
