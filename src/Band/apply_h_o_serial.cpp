#include "band.h"

namespace sirius {

/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 *  \param [out] ophi Overlap operator, applied to wave-functions [storage: CPU || GPU].
 */
void Band::apply_h_o_serial(K_point* kp__, 
                            int N__,
                            int n__,
                            Wave_functions& phi__,
                            Wave_functions& hphi__,
                            Wave_functions& ophi__,
                            mdarray<double_complex, 1>& kappa__,
                            Hloc_operator& h_op,
                            D_operator& d_op,
                            Q_operator& q_op)
{
    PROFILE();

    Timer t("sirius::Band::apply_h_o_serial");

    matrix<double_complex> phi, hphi, ophi, beta_gk;
    
    /* if temporary array is allocated, this would be the only big array on GPU */
    //bool economize_gpu_memory = (kappa__.size() != 0);

    //if (parameters_.processing_unit() == CPU)
    //{
    //    phi  = matrix<double_complex>( phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
    //    hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
    //    ophi = matrix<double_complex>(ophi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
    //    beta_gk = matrix<double_complex>(kp__->beta_gk().at<CPU>(), kp__->num_gkvec(), unit_cell_.mt_basis_size());
    //}
    //if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    //{
    //    phi  = matrix<double_complex>( phi__.at<CPU>(0, N__),  phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
    //    hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), hphi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
    //    ophi = matrix<double_complex>(ophi__.at<CPU>(0, N__), ophi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
    //    beta_gk = matrix<double_complex>(kp__->beta_gk().at<CPU>(), kp__->beta_gk().at<GPU>(), kp__->num_gkvec(), unit_cell_.mt_basis_size());
    //}
    //if (parameters_.processing_unit() == GPU && economize_gpu_memory)
    //{
    //    beta_gk = matrix<double_complex>(nullptr, kappa__.at<GPU>(), kp__->num_gkvec(), unit_cell_.mt_basis_size());
    //    double_complex* gpu_ptr = kappa__.at<GPU>(beta_gk.size());
    //    phi  = matrix<double_complex>( phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
    //    hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
    //    ophi = matrix<double_complex>(ophi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
    //}

    //phi__.swap_forward(N__, n__);
    //
    ///* apply local part of Hamiltonian */
    //apply_h_local_serial(kp__, effective_potential__, pw_ekin__, phi__.local_size(), phi__, hphi__);

    ////for (int i = 0; i < phi__.local_size(); i++)
    ////{
    ////    std::memcpy(hphi__[i], phi__[i], kp__->num_gkvec() * sizeof(double_complex));
    ////}
    //
    //hphi__.swap_backward(N__, n__);
    //Utils::write_matrix("hloc_phi.txt", true, hphi__.primary_data_storage_as_matrix());
    hphi__.copy_from(phi__, N__, n__);
    h_op.apply(hphi__, N__, n__);

    /* set intial ophi */
    ophi__.copy_from(phi__, N__, n__);
    
    #ifdef __GPU
    //if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    //{
    //    /* copy hphi do device */
    //    hphi.copy_to_device();

    //    /* set intial ophi */
    //    cuda_copy_device_to_device(ophi.at<GPU>(), phi.at<GPU>(), kp__->num_gkvec() * n__ * sizeof(double_complex));
    //}
    STOP();
    #endif
    
    kp__->beta_projectors().inner(phi__, N__, n__);

    if (!kp__->iterative_solver_input_section_.real_space_prj_)
    {
        d_op.apply(hphi__, N__, n__);
        q_op.apply(ophi__, N__, n__);
    }
    else
    {
        STOP();
        //add_nl_h_o_rs(kp__, n__, phi, hphi, ophi, packed_mtrx_offset__, d_mtrx_packed__, q_mtrx_packed__, kappa__);
    }
}

};
