#include "band.h"

namespace sirius {

/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 *  \param [out] ophi Overlap operator, applied to wave-functions [storage: CPU || GPU].
 */
void Band::apply_h_o_serial(K_point* kp__, 
                            std::vector<double> const& effective_potential__, 
                            std::vector<double> const& pw_ekin__, 
                            int N__,
                            int n__,
                            matrix<double_complex>& phi__,
                            matrix<double_complex>& hphi__,
                            matrix<double_complex>& ophi__,
                            mdarray<double_complex, 1>& kappa__,
                            mdarray<int, 1>& packed_mtrx_offset__,
                            mdarray<double_complex, 1>& d_mtrx_packed__,
                            mdarray<double_complex, 1>& q_mtrx_packed__)
{
    LOG_FUNC_BEGIN();

    Timer t("sirius::Band::apply_h_o_serial");

    auto uc = parameters_.unit_cell();

    matrix<double_complex> phi, hphi, ophi, beta_gk;
    
    /* if temporary array is allocated, this would be the only big array on GPU */
    bool economize_gpu_memory = (kappa__.size() != 0);

    if (parameters_.processing_unit() == CPU)
    {
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
        ophi = matrix<double_complex>(ophi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
        beta_gk = matrix<double_complex>(kp__->beta_gk().at<CPU>(), kp__->num_gkvec(), uc->mt_basis_size());
    }
    if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    {
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__),  phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), hphi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
        ophi = matrix<double_complex>(ophi__.at<CPU>(0, N__), ophi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
        beta_gk = matrix<double_complex>(kp__->beta_gk().at<CPU>(), kp__->beta_gk().at<GPU>(), kp__->num_gkvec(), uc->mt_basis_size());
    }
    if (parameters_.processing_unit() == GPU && economize_gpu_memory)
    {
        beta_gk = matrix<double_complex>(nullptr, kappa__.at<GPU>(), kp__->num_gkvec(), uc->mt_basis_size());
        double_complex* gpu_ptr = kappa__.at<GPU>(beta_gk.size());
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
        ophi = matrix<double_complex>(ophi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
    }
    
    /* apply local part of Hamiltonian */
    apply_h_local_slice(kp__, effective_potential__, pw_ekin__, n__, phi, hphi);
    
    /* set intial ophi */
    if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && economize_gpu_memory)) 
        phi >> ophi;

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    {
        /* copy hphi do device */
        hphi.copy_to_device();

        /* set intial ophi */
        cuda_copy_device_to_device(ophi.at<GPU>(), phi.at<GPU>(), kp__->num_gkvec() * n__ * sizeof(double_complex));
    }
    #endif

    if (!kp__->iterative_solver_input_section_.real_space_prj_)
    {
        add_nl_h_o_pw(kp__, n__, phi, hphi, ophi, beta_gk, packed_mtrx_offset__, d_mtrx_packed__, q_mtrx_packed__);
    }
    else
    {
        add_nl_h_o_rs(kp__, n__, phi, hphi, ophi, packed_mtrx_offset__, d_mtrx_packed__, q_mtrx_packed__, kappa__);
    }

    LOG_FUNC_END();
}

};
