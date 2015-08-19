#include "band.h"

namespace sirius {

void Band::add_nl_h_o_pw(K_point* kp__,
                         int n__,
                         matrix<double_complex>& phi__,
                         matrix<double_complex>& hphi__,
                         matrix<double_complex>& ophi__,
                         matrix<double_complex>& beta_gk__,
                         mdarray<int, 1>& packed_mtrx_offset__,
                         mdarray<double_complex, 1>& d_mtrx_packed__,
                         mdarray<double_complex, 1>& q_mtrx_packed__)
{
    bool economize_gpu_memory = true;

    /* <\beta_{\xi}^{\alpha}|\phi_j> */
    matrix<double_complex> beta_phi(unit_cell_.mt_basis_size(), n__);

    /* Q or D multiplied by <\beta_{\xi}^{\alpha}|\phi_j> */
    matrix<double_complex> work(unit_cell_.mt_basis_size(), n__);

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        beta_phi.allocate_on_device();
        work.allocate_on_device();
    }
    #endif

    if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && !economize_gpu_memory))
    {
        /* compute <beta|phi> */
        kp__->generate_beta_phi(unit_cell_.mt_basis_size(), phi__, n__, 0, beta_gk__, beta_phi);
        
        #ifdef __PRINT_OBJECT_CHECKSUM
        auto c1 = beta_phi.checksum();
        DUMP("checksum(beta_phi) : %18.10f %18.10f", std::real(c1), std::imag(c1));
        #endif
       
        /* add |beta>D<beta|phi> to |hphi> */
        kp__->add_non_local_contribution(unit_cell_.num_atoms(), unit_cell_.mt_basis_size(), unit_cell_.beta_chunk(0).desc_,
                                         beta_gk__, d_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         hphi__, n__, 0, complex_one, work);
            
        /* add |beta>Q<beta|phi> to |ophi> */
        kp__->add_non_local_contribution(unit_cell_.num_atoms(), unit_cell_.mt_basis_size(), unit_cell_.beta_chunk(0).desc_,
                                         beta_gk__, q_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         ophi__, n__, 0, complex_one, work);
    }
    else
    {
        #ifdef __GPU
        kp__->generate_beta_gk(unit_cell_.num_atoms(), unit_cell_.beta_chunk(0).atom_pos_, unit_cell_.beta_chunk(0).desc_, beta_gk__);
        
        phi__.copy_to_device();
        kp__->generate_beta_phi(unit_cell_.mt_basis_size(), phi__, n__, 0, beta_gk__, beta_phi);
        
        hphi__.copy_to_device();
        kp__->add_non_local_contribution(unit_cell_.num_atoms(), unit_cell_.mt_basis_size(), unit_cell_.beta_chunk(0).desc_,
                                         beta_gk__, d_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         hphi__, n__, 0, complex_one, work);
        hphi__.copy_to_host();
        
        ophi__.copy_to_device();    
        kp__->add_non_local_contribution(unit_cell_.num_atoms(), unit_cell_.mt_basis_size(), unit_cell_.beta_chunk(0).desc_,
                                         beta_gk__, q_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         ophi__, n__, 0, complex_one, work);
        ophi__.copy_to_host();
        #else
        TERMINATE_NO_GPU
        #endif
    }
    #ifdef __GPU
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif
}

};
