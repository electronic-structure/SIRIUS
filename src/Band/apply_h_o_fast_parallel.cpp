#include "band.h"

namespace sirius {

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
    LOG_FUNC_BEGIN();

    Timer t("sirius::Band::apply_h_o_fast_parallel", kp__->comm());

    splindex<block> spl_phi(n__, kp__->comm().size(), kp__->comm().rank());

    kp__->collect_all_gkvec(spl_phi, &phi_slab__(0, N__), &phi_slice__(0, 0));
    
    if (spl_phi.local_size())
        apply_h_local_slice(kp__, effective_potential__, pw_ekin__, (int)spl_phi.local_size(), phi_slice__, phi_slice__);

    kp__->collect_all_bands(spl_phi, &phi_slice__(0, 0),  &hphi_slab__(0, N__));

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

    LOG_FUNC_END();
}

};
