#include "band.h"

namespace sirius {

// TODO: merge with apply_h_o_serial

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
                          mdarray<double_complex, 1>& d_mtrx_packed__) const
{
    runtime::Timer t("sirius::Band::apply_h_serial");

    matrix<double_complex> phi, hphi;
    
    /* if temporary array is allocated, this would be the only big array on GPU */
    bool economize_gpu_memory = (kappa__.size() != 0);

    if (ctx_.processing_unit() == CPU)
    {
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (ctx_.processing_unit() == GPU && !economize_gpu_memory)
    {
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__),  phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), hphi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (ctx_.processing_unit() == GPU && economize_gpu_memory)
    {
        double_complex* gpu_ptr = kappa__.at<GPU>(kp__->num_gkvec() * unit_cell_.mt_basis_size());
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
    }
    
    /* apply local part of Hamiltonian */
    STOP();

    //apply_h_local_serial(kp__, effective_potential__, pw_ekin__, n__, phi, hphi);

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU && !economize_gpu_memory)
    {
        /* copy hphi do device */
        hphi.copy_to_device();
    }
    #endif

    STOP();
    //add_non_local_contribution_serial(kp__, N__, n__, phi__, hphi__, kappa__, packed_mtrx_offset__, d_mtrx_packed__, complex_one);
}

};
