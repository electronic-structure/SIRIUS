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
    auto uc = parameters_.unit_cell();

    bool economize_gpu_memory = true;

    /* <\beta_{\xi}^{\alpha}|\phi_j> */
    matrix<double_complex> beta_phi(uc->mt_basis_size(), n__);

    /* Q or D multiplied by <\beta_{\xi}^{\alpha}|\phi_j> */
    matrix<double_complex> work(uc->mt_basis_size(), n__);

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        beta_phi.allocate_on_device();
        work.allocate_on_device();
    }
    #endif

    if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && !economize_gpu_memory))
    {
        /* compute <beta|phi> */
        kp__->generate_beta_phi(uc->mt_basis_size(), phi__, n__, 0, beta_gk__, beta_phi);
       
        /* add |beta>D<beta|phi> to |hphi> */
        kp__->add_non_local_contribution(uc->num_atoms(), uc->mt_basis_size(), uc->beta_chunk(0).desc_,
                                         beta_gk__, d_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         hphi__, n__, 0, complex_one, work);
            
        /* add |beta>Q<beta|phi> to |ophi> */
        kp__->add_non_local_contribution(uc->num_atoms(), uc->mt_basis_size(), uc->beta_chunk(0).desc_,
                                         beta_gk__, q_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         ophi__, n__, 0, complex_one, work);
    }
    else
    {
        #ifdef _GPU_
        kp__->generate_beta_gk(uc->num_atoms(), uc->beta_chunk(0).atom_pos_, uc->beta_chunk(0).desc_, beta_gk__);
        
        phi__.copy_to_device();
        kp__->generate_beta_phi(uc->mt_basis_size(), phi__, n__, 0, beta_gk__, beta_phi);
        
        hphi__.copy_to_device();
        kp__->add_non_local_contribution(uc->num_atoms(), uc->mt_basis_size(), uc->beta_chunk(0).desc_,
                                         beta_gk__, d_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         hphi__, n__, 0, complex_one, work);
        hphi__.copy_to_host();
        
        ophi__.copy_to_device();    
        kp__->add_non_local_contribution(uc->num_atoms(), uc->mt_basis_size(), uc->beta_chunk(0).desc_,
                                         beta_gk__, q_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         ophi__, n__, 0, complex_one, work);
        ophi__.copy_to_host();
        #else
        TERMINATE_NO_GPU
        #endif
    }
    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif
}

};
